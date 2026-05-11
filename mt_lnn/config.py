import math
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class MTLNNConfig:
    # Vocabulary and sequence
    vocab_size: int = 50257          # GPT-2 BPE default
    max_seq_len: int = 1024
    pad_token_id: int = 0

    # Model dimensions
    d_model: int = 1024
    n_layers: int = 12               # 12 layers → ~125M params
    n_heads: int = 16
    n_kv_heads: int = 4              # GQA: 4 KV heads, 16 Q heads → 4× KV-cache savings
    d_head: int = 64                 # d_model // n_heads

    # Microtubule protofilament settings
    # Biological constant is 13; vectorised forward scales near-flat so going
    # higher (32 / 64 / 128) only costs more parameters, not more wall-clock.
    n_protofilaments: int = 13
    map_hidden_dim: int = 64

    # LTC / ODE parameters
    tau_init: float = 1.0
    tau_min: float = 0.01
    tau_max: float = 10.0
    dt: float = 1.0

    # GTP hydrolysis (lateral coupling in MTLNNLayer)
    gamma_init: float = 0.1
    # GTP cap renewal period — lateral coupling refreshes every gtp_period
    # positions. Without this, the exp(-γ·t) gate vanishes at large t and
    # microtubule mixing silently dies in long contexts.
    gtp_period: int = 256

    # Continuous τ spectrum. With n_time_scales > 3 the resonance frequencies
    # are chosen as a geometric sweep spanning tau_min → tau_max.
    n_time_scales: int = 5
    resonance_freqs: Optional[Tuple[float, ...]] = None

    # Global coherence
    coherence_sparsity: float = 0.1  # keep top 10% of attention scores
    coherence_heads: int = 4

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Misc
    tie_embeddings: bool = True

    # Derived (set in __post_init__)
    d_proto: int = field(init=False)
    d_proto_total: int = field(init=False)

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.d_head == self.d_model // self.n_heads, "d_head must equal d_model // n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads (GQA)"
        # Pad to next multiple of n_protofilaments so each proto gets equal width
        self.d_proto = math.ceil(self.d_model / self.n_protofilaments)
        self.d_proto_total = self.d_proto * self.n_protofilaments
        # e.g. d_model=1024, P=13: d_proto=79, d_proto_total=1027

        # Continuous τ spectrum: geometric sweep tau_min → tau_max.
        # Each scale s in [0, n_time_scales) gets τ_s = tau_min * (tau_max/tau_min)^(s/(S-1))
        if self.resonance_freqs is None:
            if self.n_time_scales == 1:
                self.resonance_freqs = (self.tau_init,)
            else:
                ratio = self.tau_max / self.tau_min
                freqs = tuple(
                    self.tau_min * (ratio ** (s / (self.n_time_scales - 1)))
                    for s in range(self.n_time_scales)
                )
                self.resonance_freqs = freqs
        else:
            assert len(self.resonance_freqs) == self.n_time_scales, \
                f"resonance_freqs length {len(self.resonance_freqs)} != n_time_scales {self.n_time_scales}"
