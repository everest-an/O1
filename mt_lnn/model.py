"""
model.py — MTLNNModel: full Microtubule-Enhanced Liquid Neural Network.

Dual-state inference cache:
  - Attention sub-layer: KV cache (past_kv = (K, V)) per layer
  - LNN sub-layer:       recurrent hidden state h_prev per layer

A `LayerCache` per layer is the tuple (kv, h_prev). The full cache is a list
of LayerCache, one per block. Pass `use_cache=True` to enable.
"""

from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MTLNNConfig
from .embedding import MTLNNEmbedding
from .mt_attention import MicrotubuleAttention
from .mt_lnn_layer import MTLNNLayer
from .global_coherence import GlobalCoherenceLayer
from .gwtb import GWTBLayer
from .utils import init_weights, init_mt_params


# Type aliases for clarity.
# LayerCache is now a 3-tuple: (attention KV, LNN recurrent state, per-block GWTB KV).
# The third slot is None when gwtb_per_block=False so the cache structure stays
# identical for both modes — code reads cache.layers[i][2] unconditionally.
KVCache    = Tuple[torch.Tensor, torch.Tensor]                          # (K, V)
LayerCache = Tuple[
    Optional[KVCache],          # 0: attention KV
    Optional[torch.Tensor],     # 1: MT-DL recurrent h_prev
    Optional[KVCache],          # 2: per-block GWTB KV (None if not per-block)
]


class ModelCacheStruct:
    """Full inference cache for incremental decoding."""
    def __init__(self):
        self.layers: List[LayerCache] = []
        self.gwtb_kv: Optional[KVCache] = None
        self.coherence_kv: Optional[KVCache] = None


class MTLNNBlock(nn.Module):
    """One transformer-style block with microtubule attention + LNN layer.

    If `config.gwtb_per_block` is True, the block additionally contains a
    GWTBLayer that runs after the LNN sub-layer — matching the paper §4
    architecture where every block ignites its own workspace.
    """

    def __init__(self, config: MTLNNConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.has_gwtb = config.gwtb_per_block

        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn: MicrotubuleAttention   # set by parent after embedding init
        self.lnn_norm = nn.LayerNorm(config.d_model)
        self.lnn = MTLNNLayer(config)

        if self.has_gwtb:
            self.gwtb_norm = nn.LayerNorm(config.d_model)
            self.gwtb = GWTBLayer(config)
        else:
            self.gwtb_norm = None
            self.gwtb = None

    def forward(
        self,
        x: torch.Tensor,                                  # (B, T_new, d_model)
        layer_cache: Optional[LayerCache] = None,
        pad_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        use_cache: bool = False,
        use_lnn_recurrence: bool = True,
    ) -> Tuple[torch.Tensor, Optional[LayerCache]]:

        past_kv      = layer_cache[0] if layer_cache is not None else None
        h_prev       = (layer_cache[1] if (layer_cache is not None and use_lnn_recurrence)
                        else None)
        past_gwtb_kv = (layer_cache[2] if (layer_cache is not None and len(layer_cache) > 2)
                        else None)

        # Attention sub-layer (pre-norm)
        attn_out, new_kv = self.attn(
            self.attn_norm(x),
            pad_mask=pad_mask,
            past_kv=past_kv,
            position_offset=position_offset,
            use_cache=use_cache,
        )
        x = x + attn_out

        # LNN sub-layer (pre-norm).
        # use_lnn_recurrence drives BOTH (a) whether to pull h_prev from the
        # cache or use zeros, and (b) whether the resonance bank runs a real
        # parallel scan (True) or the legacy broadcast-h_prev parallel mode
        # (False). The two together preserve cache parity in both modes.
        lnn_out, h_last = self.lnn(
            self.lnn_norm(x), h_prev,
            position_offset=position_offset,
            use_scan=use_lnn_recurrence,
        )
        x = x + lnn_out

        # Per-block GWTB (pre-norm, gated residual already inside GWTBLayer)
        new_gwtb_kv = None
        if self.has_gwtb:
            x, new_gwtb_kv = self.gwtb(
                self.gwtb_norm(x),
                past_kv=past_gwtb_kv,
                position_offset=position_offset,
                use_cache=use_cache,
            )

        new_cache: Optional[LayerCache] = (
            (new_kv, h_last, new_gwtb_kv) if use_cache else None
        )
        return x, new_cache


class MTLNNModel(nn.Module):
    """Full MT-LNN language model (~125M params)."""

    def __init__(self, config: MTLNNConfig):
        super().__init__()
        self.config = config

        self.embedding = MTLNNEmbedding(config)

        self.blocks = nn.ModuleList()
        for i in range(config.n_layers):
            block = MTLNNBlock(config, layer_idx=i)
            block.attn = MicrotubuleAttention(config, rope=self.embedding.rope)
            self.blocks.append(block)

        # Global Workspace Theory Bottleneck.
        # If gwtb_per_block=True the bottleneck lives inside every block (paper
        # §4 default). Otherwise it's applied once after the entire stack.
        # Mutually exclusive to avoid double-broadcasting the workspace.
        self.gwtb_per_block = config.gwtb_per_block
        if not config.gwtb_per_block:
            self.gwtb = GWTBLayer(config)
        else:
            self.gwtb = None

        # Orch-OR collapse layer (complementary to GWTB)
        self.coherence = GlobalCoherenceLayer(config)
        self.final_norm = nn.LayerNorm(config.d_model)

        # LM head (no bias, optionally weight-tied)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.token_embed.weight

        # Weight initialisation
        self.apply(lambda m: init_weights(m, config))
        init_mt_params(self, config)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,                              # (B, T_new)
        cache: Optional[ModelCacheStruct] = None,
        pad_mask: Optional[torch.Tensor] = None,              # (B, T_total) bool
        labels: Optional[torch.Tensor] = None,                # (B, T_new) for causal LM loss
        use_cache: bool = False,
        position_offset: Optional[int] = None,
        use_lnn_recurrence: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
          use_cache: build/extend the KV+coherence cache for incremental decoding.
          use_lnn_recurrence: if True (default for inference), thread h_prev across
            steps so the LNN behaves as a true RNN. Set False to match the
            training-time parallel mode (h_prev=0 each step) — this is what makes
            cached vs full-forward outputs bit-exact equal.
          position_offset: if None, infer from cache (length of cached K) or 0.

        Returns dict with keys:
          - logits:   (B, T_new, vocab_size)
          - loss:     scalar (only if labels provided)
          - cache:    new ModelCacheStruct (only if use_cache=True)
        """
        # Infer absolute position offset
        if position_offset is None:
            if (cache is not None and len(cache.layers) > 0
                and cache.layers[0] is not None and cache.layers[0][0] is not None):
                position_offset = cache.layers[0][0][0].shape[2]   # T_past from first layer's K
            else:
                position_offset = 0

        x = self.embedding(input_ids)                         # (B, T_new, d_model)

        new_cache = ModelCacheStruct() if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_cache = (cache.layers[i] if (cache is not None and i < len(cache.layers)) else None)
            x, new_layer_cache = block(
                x,
                layer_cache=layer_cache,
                pad_mask=pad_mask,
                position_offset=position_offset,
                use_cache=use_cache,
                use_lnn_recurrence=use_lnn_recurrence,
            )
            if use_cache:
                new_cache.layers.append(new_layer_cache)

        # Top-level GWTB (only if gwtb_per_block=False).
        if self.gwtb is not None:
            gwtb_past = cache.gwtb_kv if cache is not None else None
            x, gwtb_new_kv = self.gwtb(
                x, past_kv=gwtb_past, position_offset=position_offset,
                use_cache=use_cache,
            )
            if use_cache:
                new_cache.gwtb_kv = gwtb_new_kv

        coh_past = cache.coherence_kv if cache is not None else None
        x, coh_new_kv = self.coherence(
            x, past_kv=coh_past, position_offset=position_offset, use_cache=use_cache
        )
        if use_cache:
            new_cache.coherence_kv = coh_new_kv

        x = self.final_norm(x)
        logits = self.lm_head(x)                              # (B, T_new, vocab_size)

        result: Dict[str, torch.Tensor] = {"logits": logits}
        if use_cache:
            result["cache"] = new_cache

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_num_params(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.embedding.token_embed.weight.numel()
        return total

    @classmethod
    def from_config(cls, config: MTLNNConfig) -> "MTLNNModel":
        return cls(config)

    def get_mt_diagnostics(self) -> Dict[str, float]:
        """Return a dict of MT health metrics for monitoring during training."""
        diag: Dict[str, float] = {}
        tau_vals, gamma_vals, pol_vals = [], [], []
        lat_norms, rmc_gates = [], []

        for block in self.blocks:
            # τ tensor for this block's resonance bank: shape (P, S)
            tau = (F.softplus(block.lnn.resonance.log_tau)
                   + self.config.tau_min).clamp(self.config.tau_min,
                                                self.config.tau_max)
            tau_vals.extend(tau.detach().flatten().cpu().tolist())

            gamma_vals.append(block.lnn.gtp_gamma.item())
            pol_vals.extend(block.attn.polarity_direction.detach().cpu().tolist())
            W = block.lnn.lateral.W_lat
            off_diag = W - torch.eye(W.shape[0], device=W.device)
            lat_norms.append(off_diag.norm().item())
            rmc_gates.append(torch.sigmoid(block.lnn.lateral.rmc_gate).item())

        if tau_vals:
            t = torch.tensor(tau_vals)
            diag["tau_mean"] = t.mean().item()
            diag["tau_std"] = t.std().item()
            diag["tau_min"] = t.min().item()
            diag["tau_max"] = t.max().item()
        if gamma_vals:
            diag["gamma_mean"] = sum(gamma_vals) / len(gamma_vals)
        if pol_vals:
            p = torch.tensor(pol_vals)
            diag["polarity_mean"] = p.mean().item()
            diag["polarity_std"] = p.std().item()
        if lat_norms:
            diag["lat_coupling_mean_off_diag_norm"] = sum(lat_norms) / len(lat_norms)
        if rmc_gates:
            diag["rmc_gate_mean"] = sum(rmc_gates) / len(rmc_gates)

        diag["coherence_scale"] = self.coherence.coherence_scale.item()
        diag["collapse_threshold"] = self.coherence.collapse_threshold.item()
        diag["collapse_gate_last"] = self.coherence.last_gate.item()

        if self.gwtb is not None:
            # Single top-level GWTB
            diag["gwtb_broadcast_gate"] = self.gwtb.broadcast_gate.item()
            diag["gwtb_d_gw"] = float(self.gwtb.d_gw)
        else:
            # Per-block GWTB — report mean / spread across blocks
            gates = [b.gwtb.broadcast_gate.item() for b in self.blocks if b.has_gwtb]
            if gates:
                gates_t = torch.tensor(gates)
                diag["gwtb_broadcast_gate_mean"] = gates_t.mean().item()
                diag["gwtb_broadcast_gate_std"]  = gates_t.std().item()
                diag["gwtb_d_gw"] = float(self.blocks[0].gwtb.d_gw)
        return diag

    def get_mt_histograms(self) -> Dict[str, torch.Tensor]:
        """Return raw tensors for W&B histogram logging."""
        taus, gammas, pols = [], [], []
        for block in self.blocks:
            tau = F.softplus(block.lnn.resonance.log_tau).detach() + self.config.tau_min
            taus.append(tau.flatten())
            gammas.append(block.lnn.gtp_gamma.detach().reshape(-1))
            pols.append(block.attn.polarity_direction.detach())
        return {
            "tau": torch.cat(taus).cpu().float(),
            "gamma": torch.cat(gammas).cpu().float(),
            "polarity": torch.cat(pols).cpu().float(),
        }
