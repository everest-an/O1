"""
mt_lnn_layer.py — Microtubule-Enhanced Liquid Neural Network Layer

Replaces the standard Transformer FFN with a 13-protofilament parallel
closed-form LTC (Liquid Time-Constant) network, with:
  - Lateral coupling between protofilaments (13×13 matrix)
  - GTP hydrolysis gating on lateral signal
  - MAP protein gating (small MLP stabiliser per protofilament)
  - Multi-scale resonance (3 time-scales blended per protofilament)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import MTLNNConfig


# ---------------------------------------------------------------------------
# ProtofilamentLTC — single protofilament, closed-form LTC
# ---------------------------------------------------------------------------

class ProtofilamentLTC(nn.Module):
    """
    Closed-form Liquid Time-Constant neuron for one protofilament.

    Equation (no ODE iteration needed):
        decay = exp(-dt / τ)
        A     = σ(W_in @ x + b)
        h_new = h_prev * decay + A * (1 - decay)

    In parallel (training) mode h_prev=0 → h_new = A * (1 - decay),
    which is a nonlinear gated projection — still expressive and fast.
    """

    def __init__(self, d_proto: int, tau_init: float, tau_min: float, tau_max: float, dt: float):
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dt = dt

        self.W_in = nn.Linear(d_proto, d_proto, bias=True)
        # log_tau initialised so that softplus(log_tau) + tau_min ≈ tau_init
        log_tau_init = math.log(math.expm1(max(tau_init - tau_min, 1e-6)))
        self.log_tau = nn.Parameter(torch.full((), log_tau_init))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        x, h_prev: (B, T, d_proto)
        Returns:   (B, T, d_proto)
        """
        tau = F.softplus(self.log_tau) + self.tau_min
        tau = tau.clamp(self.tau_min, self.tau_max)
        decay = torch.exp(-self.dt / tau)           # scalar
        A = torch.sigmoid(self.W_in(x))             # (B,T,d_proto)
        return h_prev * decay + A * (1.0 - decay)


# ---------------------------------------------------------------------------
# MultiScaleResonance — blend of 3 protofilament LTCs at different τ
# ---------------------------------------------------------------------------

class MultiScaleResonance(nn.Module):
    """
    Three parallel ProtofilamentLTC instances at τ_fast / τ_mid / τ_slow.
    Their outputs are blended by learned softmax weights.
    Mimics multi-scale resonance of biological microtubules.
    """

    def __init__(self, config: MTLNNConfig):
        super().__init__()
        d = config.d_proto
        self.ltcs = nn.ModuleList([
            ProtofilamentLTC(d, tau, config.tau_min, config.tau_max, config.dt)
            for tau in config.resonance_freqs
        ])
        # Blend weights: softmax(zeros) = uniform at init
        self.blend_weights = nn.Parameter(torch.zeros(config.n_time_scales))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        x, h_prev: (B, T, d_proto)
        Returns:   (B, T, d_proto)
        """
        w = F.softmax(self.blend_weights, dim=0)    # (n_time_scales,)
        out = sum(w[i] * ltc(x, h_prev) for i, ltc in enumerate(self.ltcs))
        return out


# ---------------------------------------------------------------------------
# LateralCoupling — content-aware coupling across 13 protofilaments (RMC-style)
# ---------------------------------------------------------------------------

class LateralCoupling(nn.Module):
    """
    Relational Memory Core (RMC) style coupling: instead of a fixed 13×13 matrix,
    treat the 13 protofilaments as memory slots and let them mix via a single
    attention head.

        Coupling(H) = softmax(Q K^T / √d) V

    where Q, K, V are linear projections of the per-protofilament state H.

    A static W_lat residual term is kept (initialised to identity) so the layer
    starts as a no-op and the attention contribution is added in gradually via
    a learned scalar gate.

    h: (B, T, P=13, d_proto)
    """

    def __init__(self, n_protofilaments: int, d_proto: int):
        super().__init__()
        self.n_protofilaments = n_protofilaments
        self.d_proto = d_proto
        self.scale = math.sqrt(d_proto)

        # RMC-style content-aware mixing
        self.q_proj = nn.Linear(d_proto, d_proto, bias=False)
        self.k_proj = nn.Linear(d_proto, d_proto, bias=False)
        self.v_proj = nn.Linear(d_proto, d_proto, bias=False)
        self.out_proj = nn.Linear(d_proto, d_proto, bias=False)

        # Learned identity-residual matrix (kept for stability at init)
        self.W_lat = nn.Parameter(torch.eye(n_protofilaments))

        # Gate that controls how much RMC contributes vs identity residual.
        # Init at 0 → layer starts as pure identity coupling, gradually mixes in.
        self.rmc_gate = nn.Parameter(torch.zeros(()))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, T, P, d_proto)  — P = 13 protofilaments treated as memory slots
        """
        # Identity-style residual: static W_lat across protofilament dim
        residual = torch.einsum("btpd,pq->btqd", h, self.W_lat)

        # RMC: per-(B,T) attention over P slots
        B, T, P, D = h.shape
        h_flat = h.reshape(B * T, P, D)                              # treat (B,T) as batch
        q = self.q_proj(h_flat)
        k = self.k_proj(h_flat)
        v = self.v_proj(h_flat)
        # SDPA: shape (B*T, P, D) → SDPA wants (..., L, S) so add a head dim
        q = q.unsqueeze(1)                                            # (B*T, 1, P, D)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        attn_out = F.scaled_dot_product_attention(q, k, v)            # (B*T, 1, P, D)
        attn_out = attn_out.squeeze(1)                                # (B*T, P, D)
        rmc = self.out_proj(attn_out).reshape(B, T, P, D)

        # Gated mix: gate sigmoid keeps it in [0,1]
        g = torch.sigmoid(self.rmc_gate)
        return residual + g * rmc


# ---------------------------------------------------------------------------
# MAPGate — MAP protein stabilisation (small MLP → scalar gate s ∈ [0,1])
# ---------------------------------------------------------------------------

class MAPGate(nn.Module):
    """
    Mimics MAP (Microtubule-Associated Protein) stabilisation:
    a small MLP takes [h_p; x_p] and outputs a scalar gate s ∈ [0,1].
    s≈1 → stabilise (slow decay), s≈0 → destabilise (fast decay / pruning).
    """

    def __init__(self, d_proto: int, map_hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(d_proto * 2, map_hidden_dim)
        self.fc2 = nn.Linear(map_hidden_dim, 1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        h, x: (B, T, d_proto)
        Returns: (B, T, 1) gate values in [0, 1]
        """
        z = torch.cat([h, x], dim=-1)               # (B,T,2*d_proto)
        return torch.sigmoid(self.fc2(F.relu(self.fc1(z))))


# ---------------------------------------------------------------------------
# MTLNNLayer — full 13-protofilament layer replacing FFN
# ---------------------------------------------------------------------------

class MTLNNLayer(nn.Module):
    """
    Full Microtubule LNN layer.

    Forward pipeline:
      1. in_proj: d_model → d_proto_total, split into 13 protofilaments
      2. 13× MultiScaleResonance (each proto processes its slice independently)
      3. LateralCoupling: mix information across protofilaments
      4. GTP hydrolysis gate on lateral contribution (temporal decay)
      5. 13× MAPGate: learned per-protofilament stabilisation
      6. out_proj: d_proto_total → d_model

    Returns (output, h_last):
      output: (B, T, d_model)
      h_last: (B, n_protofilaments, d_proto) — last-step hidden for recurrence
    """

    def __init__(self, config: MTLNNConfig):
        super().__init__()
        self.n_proto = config.n_protofilaments
        self.d_proto = config.d_proto
        self.d_proto_total = config.d_proto_total

        self.in_proj = nn.Linear(config.d_model, config.d_proto_total, bias=True)
        self.out_proj = nn.Linear(config.d_proto_total, config.d_model, bias=True)

        self.protofilaments = nn.ModuleList([
            MultiScaleResonance(config) for _ in range(config.n_protofilaments)
        ])

        self.lateral = LateralCoupling(config.n_protofilaments, config.d_proto)

        # GTP hydrolysis: scalar γ for lateral gating; temporal decay exp(-γ*t)
        self.gtp_gamma = nn.Parameter(torch.tensor(config.gamma_init))

        self.map_gates = nn.ModuleList([
            MAPGate(config.d_proto, config.map_hidden_dim)
            for _ in range(config.n_protofilaments)
        ])

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,                      # (B, T, d_model)
        h_prev: torch.Tensor = None,           # (B, n_protofilaments, d_proto) or None
        position_offset: int = 0,              # absolute position of x[:, 0, :]
    ):
        B, T, _ = x.shape
        P, D = self.n_proto, self.d_proto

        # 1. Project and split into protofilament slices
        x_proto = self.in_proj(x)                                      # (B,T,d_proto_total)
        x_split = x_proto.view(B, T, P, D)                            # (B,T,P,d_proto)

        # 2. Initialise h_prev
        if h_prev is None:
            h_prev = torch.zeros(B, P, D, device=x.device, dtype=x.dtype)
        # Expand h_prev across time (parallel mode: same h_prev for all positions)
        h_prev_t = h_prev.unsqueeze(1).expand(B, T, P, D)             # (B,T,P,d_proto)

        # 3. Per-protofilament MultiScaleResonance
        proto_outs = []
        for p in range(P):
            out_p = self.protofilaments[p](
                x_split[:, :, p, :],       # (B,T,d_proto) as input x
                h_prev_t[:, :, p, :],      # (B,T,d_proto) as h_prev
            )
            proto_outs.append(out_p)
        h_stack = torch.stack(proto_outs, dim=2)                       # (B,T,P,d_proto)

        # 4. Lateral coupling with GTP hydrolysis temporal gate
        #    gate[t_abs] = exp(-γ * t_abs), uses absolute position so the cached
        #    and full-forward paths produce identical outputs.
        t_idx = torch.arange(position_offset, position_offset + T,
                             device=x.device, dtype=x.dtype)
        gtp_scale = torch.exp(-self.gtp_gamma.clamp(min=1e-4) * t_idx)  # (T,)
        gtp_scale = gtp_scale.view(1, T, 1, 1)                        # (1,T,1,1)

        h_lateral = self.lateral(h_stack)                              # (B,T,P,d_proto)
        h_coupled = h_stack + gtp_scale * (h_lateral - h_stack)       # residual blend

        # 5. MAP gating (per protofilament)
        gated_parts = []
        for p in range(P):
            s = self.map_gates[p](h_coupled[:, :, p, :], x_split[:, :, p, :])  # (B,T,1)
            gated_parts.append(h_coupled[:, :, p, :] * s)
        h_gated = torch.stack(gated_parts, dim=2)                     # (B,T,P,d_proto)

        # 6. Output projection
        h_flat = h_gated.reshape(B, T, P * D)                         # (B,T,d_proto_total)
        out = self.dropout(self.out_proj(h_flat))                      # (B,T,d_model)

        # 7. Return last hidden state for recurrent chaining at inference
        h_last = h_gated[:, -1, :, :]                                 # (B,P,d_proto)
        return out, h_last
