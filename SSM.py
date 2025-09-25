import torch
import torch.nn as nn
import math
from einops import rearrange
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class VI2DSSM(nn.Module):
    def __init__(
        self, d_inner: int, state_size: int,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001, dt_max: float = 0.1, dt_init: str = "random",
        dt_scale: float = 1.0, dt_init_floor: float = 1e-4, bias: bool = False,
        device: str = 'cuda', dtype: torch.dtype = torch.float32, pool_type='sum'
    ):
        super().__init__()
        self.d_inner = d_inner
        self.state_size = state_size
        self.pool_type = pool_type

        self.dt_rank = math.ceil(d_inner / 16) if dt_rank == "auto" else dt_rank
        factory_kwargs = {"device": device, "dtype": dtype}

        # SSM parameters (horizontal)
        self.x_proj_hh = nn.Linear(d_inner, self.dt_rank + state_size * 2, bias=bias, **factory_kwargs)
        self.dt_proj_hh = nn.Linear(self.dt_rank, d_inner, bias=True, **factory_kwargs)
        A_hh = torch.arange(1, state_size + 1, dtype=torch.float32, device=device).repeat(d_inner, 1)
        self.A_log_hh = nn.Parameter(torch.log(A_hh))
        self.D_hh = nn.Parameter(torch.ones(d_inner, device=device))

        # SSM parameters (vertical)
        self.x_proj_hv = nn.Linear(d_inner, self.dt_rank + state_size * 2, bias=bias, **factory_kwargs)
        self.dt_proj_hv = nn.Linear(self.dt_rank, d_inner, bias=True, **factory_kwargs)
        A_hv = torch.arange(1, state_size + 1, dtype=torch.float32, device=device).repeat(d_inner, 1)
        self.A_log_hv = nn.Parameter(torch.log(A_hv))
        self.D_hv = nn.Parameter(torch.ones(d_inner, device=device))

        # Bias init
        for dt_proj in [self.dt_proj_hh, self.dt_proj_hv]:
            dt_init_std = self.dt_rank**-0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError(f"Invalid dt_init: {dt_init}")
            dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            dt_proj.bias._no_reinit = True

        # summary
        self.W_v = nn.Linear(1, d_inner, bias=False)
        self.Psi_proj_hh = nn.Linear(d_inner, state_size, bias=False)
        self.Psi_proj_hv = nn.Linear(d_inner, state_size, bias=False)
        self.X_proj_hv = nn.Linear(d_inner, state_size, bias=False)

        # attention aggregation
        if self.pool_type == 'attention': 
            attn_dim = min(d_inner, 64)
            self.pool = InvariantAttnPool(d_inner, attn_dim)

        # output proj
        self.C_h = nn.Linear(d_inner, d_inner, bias=False)
        self.C_v = nn.Linear(d_inner, d_inner, bias=False)

    def aggregation(self, h_v, pool_type='mean'):
        # h_v: [B, C, L]
        # z = h_v.sum(dim=1, keepdim=True) / h_v.shape[1] # [B, 1, L]
        # proj = self.W_v(z.permute(0, 2, 1)).permute(0, 2, 1)
        # return proj
        if self.pool_type == 'mean':
            z = h_v.mean(dim=1, keepdim=True)
            proj = self.W_v(z.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.pool_type == 'sum':
            z = h_v.sum(dim=1, keepdim=True) / h_v.shape[1] # scaling
            proj = self.W_v(z.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.pool_type == 'attention': 
            proj = self.pool(h_v)
        else:
            raise NotImplementedError("pooling error")
        return proj

    def selective_scan_with_psi(self, u, psi_seq, dt, A, B, C, D, psi_proj, delta_bias=None, x_proj=None, x_extra=None):
        psi_B = psi_proj(psi_seq.permute(0, 2, 1)).permute(0, 2, 1)  # [B*, N, L]
        B_mod = B + psi_B
        C_mod = C 

        if x_proj is not None and x_extra is not None:
            x_B = x_proj(x_extra.permute(0, 2, 1)).permute(0, 2, 1)  # [B*, N, L]
            B_mod = B_mod + x_B
            C_mod = C_mod + x_B

        y = selective_scan_fn(
            u=u, delta=dt, A=A, B=B_mod, C=C_mod, D=D, z=None,
            delta_bias=delta_bias, delta_softplus=True,
        )
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        return: [B, C, L]
        """
        B, C, L = x.shape

        psi_seq = self.aggregation(x)  # [B, L]

        # h_h
        x_reshaped = rearrange(x, "b c l -> (b l) c")
        x_proj_out_hh = self.x_proj_hh(x_reshaped)
        dt_inter_hh, B_hh, C_hh = torch.split(x_proj_out_hh, [self.dt_rank, self.state_size, self.state_size], dim=-1)
        dt_hh = self.dt_proj_hh.weight @ dt_inter_hh.t()
        dt_hh = rearrange(dt_hh, "c (b l) -> b c l", l=L)
        A_hh = -torch.exp(self.A_log_hh.float())
        B_proj_hh = rearrange(B_hh, "(b l) n -> b n l", l=L)
        C_proj_hh = rearrange(C_hh, "(b l) n -> b n l", l=L)
        D_param_hh = self.D_hh.float().contiguous()
        delta_bias_hh = self.dt_proj_hh.bias.float().contiguous() if self.dt_proj_hh.bias is not None else None

        h_h = self.selective_scan_with_psi(
            u=x,
            psi_seq=psi_seq,
            dt=dt_hh,
            A=A_hh,
            B=B_proj_hh,
            C=C_proj_hh,
            D=D_param_hh,
            psi_proj=self.Psi_proj_hh,
            delta_bias=delta_bias_hh,
        )  # [B, C, L]

        h_h_bc = h_h.reshape(B, C, L)
        x_proj_out_hv = self.x_proj_hv(rearrange(h_h_bc, "b c l -> (b l) c"))
        dt_inter_hv, B_hv, C_hv = torch.split(x_proj_out_hv, [self.dt_rank, self.state_size, self.state_size], dim=-1)
        dt_hv = self.dt_proj_hv.weight @ dt_inter_hv.t()
        dt_hv = rearrange(dt_hv, "c (b l) -> b c l", l=L)
        A_hv = -torch.exp(self.A_log_hv.float())
        B_proj_hv = rearrange(B_hv, "(b l) n -> b n l", l=L)
        C_proj_hv = rearrange(C_hv, "(b l) n -> b n l", l=L)
        D_param_hv = self.D_hv.float().contiguous()
        delta_bias_hv = self.dt_proj_hv.bias.float().contiguous() if self.dt_proj_hv.bias is not None else None

        h_v = self.selective_scan_with_psi(
            u=h_h_bc,
            psi_seq=psi_seq,
            dt=dt_hv,
            A=A_hv,
            B=B_proj_hv,
            C=C_proj_hv,
            D=D_param_hv,
            psi_proj=self.Psi_proj_hv,
            delta_bias=delta_bias_hv,
            x_proj=self.X_proj_hv,
            x_extra=x,
        )  # [B, D, L]
        h_v = h_v.view(B, C, L)

        # output proj
        y = self.C_h(h_h.permute(0, 2, 1)) + self.C_v(h_v.permute(0, 2, 1))   # [B, C, L]
        return y.permute(0, 2, 1)


class InvariantAttnPool(nn.Module):
    """
    Permutation-INVARIANT attention pooling over variables.
    h_v: [B, C, L]  -> psi: [B, d_inner, L]
    """
    def __init__(self, d_inner: int, att_dim: int = 64):
        super().__init__()
        self.d_inner = d_inner
        self.att_dim = min(d_inner, att_dim)
        self.W_k = nn.Linear(1, self.att_dim, bias=False)
        self.W_v = nn.Linear(1, self.att_dim, bias=False)
        self.W_q = nn.Linear(1, self.att_dim, bias=False)
        self.psi_out = nn.Linear(self.att_dim, d_inner, bias=False)

    def forward(self, h_v: torch.Tensor) -> torch.Tensor:
        B, C, L = h_v.shape
        x = h_v.unsqueeze(-1)                   # [B,C,L,1]
        k = self.W_k(x)                         # [B,C,L,H]
        v = self.W_v(x)                         # [B,C,L,H]
        s = h_v.mean(dim=1, keepdim=True).unsqueeze(-1)  # [B,1,L,1]  (대칭)
        q = self.W_q(s)                         # [B,1,L,H]
        logits = (q * k).sum(-1) / math.sqrt(self.att_dim)  # [B,C,L]
        alpha = torch.softmax(logits, dim=1)    # [B,C,L]
        psi_latent = (alpha.unsqueeze(-1) * v).sum(dim=1)   # [B,L,H]
        psi = self.psi_out(psi_latent).transpose(1, 2).contiguous()  # [B,d_inner,L]
        return psi
