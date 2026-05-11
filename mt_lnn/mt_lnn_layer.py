"""
mt_lnn_layer.py — Microtubule-Enhanced Liquid Neural Network Layer.

Replaces the Transformer FFN with a P-protofilament parallel closed-form LTC
network with lateral coupling, GTP gating, MAP-protein gating, and multi-scale
resonance.

This implementation is **fully vectorised** over the protofilament dimension P:
no Python `for p in range(P)` loops in the forward path. Increasing P from
the biological 13 up to 32 / 64 / 128 is essentially free on GPU.

Components:
  - VectorizedMultiScaleResonance: P × S closed-form LTC banks computed in one
    einsum + decay update.
  - LateralCoupling: identity residual + RMC-style content-aware self-attention
    across the P slots.
  - VectorizedMAPGate: P MAP-protein MLPs computed via batched einsum.
  - MTLNNLayer: assembles the pipeline.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import MTLNNConfig


# ---------------------------------------------------------------------------
# VectorizedMultiScaleResonance
# ---------------------------------------------------------------------------

class VectorizedMultiScaleResonance(nn.Module):
    """
    All P protofilaments × S time-scales of closed-form LTC, run in one shot.

    Per (proto p, scale s):
        decay_{p,s} = exp(-dt / τ_{p,s})
        A_{b,t,p,s} = σ(W_in[p,s] x_{b,t,p} + b_in[p,s])
        h^{p,s}_{new} = h_prev_{b,t,p} * decay_{p,s}  +  A_{b,t,p,s} * (1 - decay_{p,s})

    Then blend across scales with softmax(blend_weights[p]):
        h_blended_{b,t,p} = Σ_s w_{p,s} * h^{p,s}_{new}
    """

    def __init__(self, config: MTLNNConfig):
        super().__init__()
        P = config.n_protofilaments
        S = config.n_time_scales
        D = config.d_proto
        self.P, self.S, self.D = P, S, D
        self.tau_min = config.tau_min
        self.tau_max = config.tau_max
        self.dt = config.dt

        # Shared weights: (P, S, D, D) — independent W_in per (proto, scale)
        self.W_in = nn.Parameter(torch.empty(P, S, D, D))
        self.b_in = nn.Parameter(torch.zeros(P, S, D))
        nn.init.normal_(self.W_in, mean=0.0, std=0.02)

        # log_tau in raw space (softplus-parameterised); shape (P, S).
        # Init from config.resonance_freqs so each scale starts at a different τ.
        log_tau_init = torch.empty(P, S)
        for s, tau_s in enumerate(config.resonance_freqs[:S]):
            v = math.log(math.expm1(max(tau_s - config.tau_min, 1e-6)))
            log_tau_init[:, s] = v
        self.log_tau = nn.Parameter(log_tau_init)

        # blend_weights: (P, S) — softmax → uniform at init
        self.blend_weights = nn.Parameter(torch.zeros(P, S))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        x, h_prev: (B, T, P, D)
        Returns:    (B, T, P, D)
        """
        B, T, P, D = x.shape
        S = self.S

        # 1. Per-(proto, scale) input projection: (B,T,P,D) × (P,S,D,D) → (B,T,P,S,D)
        A = torch.einsum("btpd,psde->btpse", x, self.W_in)            # (B,T,P,S,D)
        A = A + self.b_in                                              # broadcast (P,S,D)
        A = torch.sigmoid(A)

        # 2. Decay per (proto, scale)
        tau = F.softplus(self.log_tau) + self.tau_min                 # (P,S)
        tau = tau.clamp(self.tau_min, self.tau_max)
        decay = torch.exp(-self.dt / tau)                              # (P,S)
        decay = decay.view(1, 1, P, S, 1)                              # (1,1,P,S,1)

        # 3. State update — all (P × S) banks at once
        h_prev_e = h_prev.unsqueeze(3).expand(B, T, P, S, D)          # (B,T,P,S,D)
        h_per_scale = h_prev_e * decay + A * (1.0 - decay)            # (B,T,P,S,D)

        # 4. Blend across scales with softmax(blend_weights)
        w = F.softmax(self.blend_weights, dim=-1)                     # (P,S)
        w = w.view(1, 1, P, S, 1)                                      # broadcast
        return (h_per_scale * w).sum(dim=3)                           # (B,T,P,D)


# ---------------------------------------------------------------------------
# LateralCoupling — RMC content-aware mixing across protofilaments
# ---------------------------------------------------------------------------

class LateralCoupling(nn.Module):
    """
    Content-aware coupling across protofilaments via SDPA, with a learned
    identity residual (W_lat) gated by sigmoid(rmc_gate).

    h: (B, T, P, d_proto)  → (B, T, P, d_proto)
    """

    def __init__(self, n_protofilaments: int, d_proto: int):
        super().__init__()
        self.n_protofilaments = n_protofilaments
        self.d_proto = d_proto

        self.q_proj = nn.Linear(d_proto, d_proto, bias=False)
        self.k_proj = nn.Linear(d_proto, d_proto, bias=False)
        self.v_proj = nn.Linear(d_proto, d_proto, bias=False)
        self.out_proj = nn.Linear(d_proto, d_proto, bias=False)

        self.W_lat = nn.Parameter(torch.eye(n_protofilaments))
        # Init from utils.init_mt_params: rmc_gate = -3 → sigmoid(-3) ≈ 0.05
        self.rmc_gate = nn.Parameter(torch.zeros(()))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Identity / static-matrix residual
        residual = torch.einsum("btpd,pq->btqd", h, self.W_lat)

        # RMC self-attention over the P "slots", treating (B*T) as batch
        B, T, P, D = h.shape
        h_flat = h.reshape(B * T, P, D)
        q = self.q_proj(h_flat).unsqueeze(1)                          # (B*T,1,P,D)
        k = self.k_proj(h_flat).unsqueeze(1)
        v = self.v_proj(h_flat).unsqueeze(1)
        attn = F.scaled_dot_product_attention(q, k, v).squeeze(1)     # (B*T,P,D)
        rmc = self.out_proj(attn).reshape(B, T, P, D)

        return residual + torch.sigmoid(self.rmc_gate) * rmc


# ---------------------------------------------------------------------------
# VectorizedMAPGate — all P MAP gates in one shot
# ---------------------------------------------------------------------------

class VectorizedMAPGate(nn.Module):
    """
    P parallel 2-layer MLPs producing per-protofilament stabilisation gates s ∈ [0,1].
        z_p = ReLU(W1_p [h_p; x_p] + b1_p)
        s_p = σ(W2_p z_p + b2_p)

    Implemented as two batched einsums, one weight tensor per layer with leading
    P dimension.
    """

    def __init__(self, n_protofilaments: int, d_proto: int, map_hidden_dim: int):
        super().__init__()
        P = n_protofilaments
        in_dim = 2 * d_proto
        H = map_hidden_dim
        self.fc1_weight = nn.Parameter(torch.empty(P, in_dim, H))
        self.fc1_bias = nn.Parameter(torch.zeros(P, H))
        self.fc2_weight = nn.Parameter(torch.empty(P, H, 1))
        self.fc2_bias = nn.Parameter(torch.zeros(P, 1))
        nn.init.normal_(self.fc1_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2_weight, mean=0.0, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        h, x: (B, T, P, D)
        Returns: (B, T, P, 1)
        """
        z = torch.cat([h, x], dim=-1)                                  # (B,T,P,2D)
        z = torch.einsum("btpi,pih->btph", z, self.fc1_weight) + self.fc1_bias
        z = F.relu(z)
        z = torch.einsum("btph,pho->btpo", z, self.fc2_weight) + self.fc2_bias
        return torch.sigmoid(z)


# ---------------------------------------------------------------------------
# MTLNNLayer — fully vectorized over P
# ---------------------------------------------------------------------------

class MTLNNLayer(nn.Module):
    """
    The full Microtubule LNN layer.

    Pipeline:
      1. in_proj: (B,T,d_model) → (B,T,d_proto_total) → reshape to (B,T,P,D)
      2. VectorizedMultiScaleResonance: (B,T,P,D) → (B,T,P,D)
      3. LateralCoupling (identity + RMC) with GTP temporal gate
      4. VectorizedMAPGate per protofilament
      5. out_proj: (B,T,P*D) → (B,T,d_model)

    Returns (output, h_last), where h_last is (B,P,D) — the recurrent state
    for the next call when streaming.
    """

    def __init__(self, config: MTLNNConfig):
        super().__init__()
        self.n_proto = config.n_protofilaments
        self.d_proto = config.d_proto
        self.d_proto_total = config.d_proto_total

        self.in_proj = nn.Linear(config.d_model, config.d_proto_total, bias=True)
        self.out_proj = nn.Linear(config.d_proto_total, config.d_model, bias=True)

        self.resonance = VectorizedMultiScaleResonance(config)
        self.lateral = LateralCoupling(config.n_protofilaments, config.d_proto)
        self.map_gates = VectorizedMAPGate(
            config.n_protofilaments, config.d_proto, config.map_hidden_dim
        )

        # GTP hydrolysis on lateral signal — temporal decay
        self.gtp_gamma = nn.Parameter(torch.tensor(config.gamma_init))
        # GTP cap renewal period: lateral coupling refreshes every `gtp_period`
        # tokens so long contexts don't silently kill mixing. Stored as a buffer
        # (non-learned, fixed at config time).
        self.register_buffer("gtp_period",
                             torch.tensor(float(config.gtp_period)),
                             persistent=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, d_model)
        h_prev: torch.Tensor = None,           # (B, P, D) or None
        position_offset: int = 0,
    ):
        B, T, _ = x.shape
        P, D = self.n_proto, self.d_proto

        # 1. Project and split into protofilaments
        x_proto = self.in_proj(x)                                      # (B,T,d_proto_total)
        x_split = x_proto.view(B, T, P, D)                            # (B,T,P,D)

        # 2. Initialise h_prev
        if h_prev is None:
            h_prev = torch.zeros(B, P, D, device=x.device, dtype=x.dtype)
        h_prev_t = h_prev.unsqueeze(1).expand(B, T, P, D)             # (B,T,P,D)

        # 3. ALL protofilaments × scales in one shot
        h_stack = self.resonance(x_split, h_prev_t)                    # (B,T,P,D)

        # 4. Lateral coupling with GTP temporal gate.
        # Originally used absolute position t_abs which makes exp(-γ·t_abs) → 0
        # for large contexts (e.g. t=4096) and lateral coupling silently
        # disappears. We now use a **periodic local clock**: t mod period.
        # This mimics microtubule GTP-cap renewal: a fresh cap forms every
        # `gtp_period` steps, lateral coupling refreshes, and the model never
        # loses lateral mixing in long sequences.
        period = self.gtp_period
        t_idx = torch.arange(position_offset, position_offset + T,
                             device=x.device, dtype=x.dtype)
        t_local = t_idx % period                                       # (T,)
        gtp_scale = torch.exp(-self.gtp_gamma.clamp(min=1e-4) * t_local)
        gtp_scale = gtp_scale.view(1, T, 1, 1)
        h_lateral = self.lateral(h_stack)                              # (B,T,P,D)
        h_coupled = h_stack + gtp_scale * (h_lateral - h_stack)

        # 5. ALL MAP gates in one shot
        s = self.map_gates(h_coupled, x_split)                         # (B,T,P,1)
        h_gated = h_coupled * s                                         # (B,T,P,D)

        # 6. Output projection
        h_flat = h_gated.reshape(B, T, P * D)
        out = self.dropout(self.out_proj(h_flat))                      # (B,T,d_model)

        h_last = h_gated[:, -1, :, :]                                  # (B,P,D)
        return out, h_last


# ---------------------------------------------------------------------------
# Backward-compat aliases (so existing __init__ exports still resolve)
# ---------------------------------------------------------------------------

# Keep exporting these names for compatibility with the package __init__.
MultiScaleResonance = VectorizedMultiScaleResonance
MAPGate = VectorizedMAPGate


class ProtofilamentLTC(nn.Module):
    """Single-protofilament LTC — kept only for backward-compat tests / docs.

    Not used by MTLNNLayer anymore (replaced by VectorizedMultiScaleResonance).
    """

    def __init__(self, d_proto: int, tau_init: float, tau_min: float,
                 tau_max: float, dt: float):
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dt = dt
        self.W_in = nn.Linear(d_proto, d_proto, bias=True)
        log_tau_init = math.log(math.expm1(max(tau_init - tau_min, 1e-6)))
        self.log_tau = nn.Parameter(torch.full((), log_tau_init))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        tau = F.softplus(self.log_tau) + self.tau_min
        tau = tau.clamp(self.tau_min, self.tau_max)
        decay = torch.exp(-self.dt / tau)
        A = torch.sigmoid(self.W_in(x))
        return h_prev * decay + A * (1.0 - decay)
