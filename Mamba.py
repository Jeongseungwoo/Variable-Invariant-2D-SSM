import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from causal_conv1d import causal_conv1d_fn
from SSM import VI2DSSM

from fft import dft, idft

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm_x

class VIMamba(nn.Module):
    def __init__(self, d_model, state_size, variable_dim, expand=2, d_conv=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model        # Hidden dim
        self.d_inner = d_model * expand 
        self.state_size = state_size
        self.variable_dim = variable_dim
        self.d_conv = d_conv

        self.norm_in = RMSNorm(self.d_model) # B, C, L
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=False)


        n_variable = self.variable_dim
        self.conv_weight = nn.Parameter(torch.empty(n_variable, d_conv))
        self.conv_bias = nn.Parameter(torch.zeros(n_variable))
        nn.init.kaiming_normal_(self.conv_weight)
        nn.init.zeros_(self.conv_bias)

        # Multi-Scale branches
        self.long_branch = VI2DSSM(n_variable, state_size, dt_min=0.1, dt_max=0.5) # [0.001, 0.1]
        self.short_branch = VI2DSSM(n_variable, state_size, dt_min=0.01, dt_max=0.05) # [0.001, 0.05]
        self.freq_branch = VI2DSSM(n_variable, state_size, dt_min=0.001, dt_max=0.01) # [0.001, 0.01]

        self.norm_long = nn.LayerNorm(self.d_inner)
        self.norm_short = nn.LayerNorm(self.d_inner)
        self.norm_freq = nn.LayerNorm(self.d_inner)

        self.alpha_long = nn.Parameter(torch.tensor(0.2))
        self.alpha_short = nn.Parameter(torch.tensor(0.2))
        self.alpha_freq = nn.Parameter(torch.tensor(0.2))

        self.gate_fc = nn.Sequential(
            nn.Linear(3 * self.d_inner, 3),
            nn.Softmax(dim=-1)
        )
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.norm_out = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, L]
        B, C, L = x.shape
        residual = x
        # RMSNorm & input projection
        x = self.norm_in(x)           # [B, C, L]
        x = self.in_proj(x) # [B, C, L]

        # causal conv1d: (B, C, L)
        x_conv = causal_conv1d_fn(
            x, self.conv_weight, self.conv_bias, activation='silu'
        )

        x_conv_freq = dft(x_conv.permute(0, 2, 1)).permute(0, 2, 1)

        # Multi-Scale branches
        x_branches = torch.cat([x_conv, x_conv, x_conv_freq], dim=0)  # [3B, C, L]
        branch_outputs = []

        for branch, x_in in zip([self.long_branch, self.short_branch, self.freq_branch],
                                torch.chunk(x_branches, 3, dim=0)):
            branch_outputs.append(branch(x_in))

        y_long, y_short, y_freq = branch_outputs

        y_long = self.alpha_long*self.norm_long(y_long) + x_conv
        y_short = self.alpha_short*self.norm_short(y_short) + x_conv
        y_freq = self.alpha_freq*self.norm_freq(y_freq) + x_conv_freq


        pooled_long = y_long.mean(dim=1)  # [B, L]
        pooled_short = y_short.mean(dim=1)
        pooled_freq = y_freq.mean(dim=1)

        gate_in = torch.cat([pooled_long, pooled_short, pooled_freq], dim=-1)  # [B, 3*L]
        gate = self.gate_fc(gate_in)                             # [B, 3]
        gate = gate.unsqueeze(-1).unsqueeze(-1)    # [B, 3, 1, 1]
        y_stack = torch.stack([y_long, y_short, y_freq], dim=1)          # [B, 3, C, L]
        out = (gate * y_stack).sum(dim=1)                        # [B, C, L]

        out = self.dropout(self.norm_out(self.out_proj(out) + residual))
        return out
