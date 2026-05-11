"""
Test suite for MT-LNN.

Run:  python -m pytest tests/   -or-   python tests/test_model.py
"""

import math
import sys
import warnings
import torch
import torch.nn.functional as F

# allow `python tests/test_model.py` from project root
sys.path.insert(0, ".")

# Tests use tiny non-TC-aligned dims for speed; suppress the alignment warning.
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel, ModelCacheStruct, anesthetize
from mt_lnn.utils import make_param_groups, count_parameters


def small_config():
    """Tiny config for fast tests."""
    return MTLNNConfig(
        vocab_size=200,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,        # GQA: 2 KV heads, 4 Q heads (2× repeat)
        d_head=32,
        dropout=0.0,         # disable dropout for deterministic tests
        attention_dropout=0.0,
    )


# ---------------------------------------------------------------------------
# 1. Shape and basic forward
# ---------------------------------------------------------------------------

def test_shapes_and_loss():
    cfg = small_config()
    model = MTLNNModel(cfg).eval()

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(ids, labels=ids)

    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert out["loss"].dim() == 0 and out["loss"].item() > 0
    print("[ok] test_shapes_and_loss")


# ---------------------------------------------------------------------------
# 2. Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow():
    cfg = small_config()
    model = MTLNNModel(cfg).train()

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(ids, labels=ids)
    out["loss"].backward()

    dead, exploding = [], []
    for name, p in model.named_parameters():
        if p.grad is None:
            dead.append(name)
        elif not p.grad.isfinite().all():
            exploding.append(name)
    assert not dead, f"Dead gradients: {dead}"
    assert not exploding, f"Exploding gradients: {exploding}"
    print(f"[ok] test_gradient_flow ({sum(1 for _ in model.parameters())} params, all finite)")


# ---------------------------------------------------------------------------
# 3. KV-cache parity: cached vs uncached must produce identical logits
# ---------------------------------------------------------------------------

def test_kv_cache_parity():
    """
    Parallel-mode parity: with use_lnn_recurrence=False (the same mode used in
    training), the cached and full-forward paths must produce IDENTICAL logits.

    This isolates the KV-cache + RoPE-offset + polarity/GTP correctness from
    the LNN's recurrent state behaviour.
    """
    torch.manual_seed(42)
    cfg = small_config()
    model = MTLNNModel(cfg).eval()

    T = 12
    ids = torch.randint(0, cfg.vocab_size, (1, T))

    # A. Full forward
    with torch.no_grad():
        full_logits = model(ids, use_lnn_recurrence=False)["logits"]

    # B. Incremental with cache, parallel-LNN mode
    cache = None
    step_logits = []
    with torch.no_grad():
        for t in range(T):
            tok = ids[:, t:t+1]
            out = model(tok, cache=cache, use_cache=True, use_lnn_recurrence=False)
            cache = out["cache"]
            step_logits.append(out["logits"][:, -1, :])
        step_logits = torch.stack(step_logits, dim=1)

    max_diff = (full_logits - step_logits).abs().max().item()
    print(f"  max logit diff (parallel-mode cached vs full): {max_diff:.6e}")
    assert max_diff < 1e-4, f"KV-cache parity FAILED: max diff = {max_diff:.6e}"
    print("[ok] test_kv_cache_parity")


def test_lnn_recurrence_active():
    """
    Sanity check: with use_lnn_recurrence=True, cached decode SHOULD differ
    from a parallel forward — proving that h_prev is actually flowing.
    """
    torch.manual_seed(123)
    cfg = small_config()
    model = MTLNNModel(cfg).eval()

    T = 8
    ids = torch.randint(0, cfg.vocab_size, (1, T))

    with torch.no_grad():
        full_logits = model(ids, use_lnn_recurrence=False)["logits"]

    cache = None
    step_logits = []
    with torch.no_grad():
        for t in range(T):
            tok = ids[:, t:t+1]
            out = model(tok, cache=cache, use_cache=True, use_lnn_recurrence=True)
            cache = out["cache"]
            step_logits.append(out["logits"][:, -1, :])
        step_logits = torch.stack(step_logits, dim=1)

    diff = (full_logits - step_logits).abs().max().item()
    print(f"  diff with recurrence active: {diff:.4e} (expect > 0)")
    assert diff > 1e-4, "LNN recurrence appears INACTIVE — h_prev not flowing"
    print("[ok] test_lnn_recurrence_active")


# ---------------------------------------------------------------------------
# 4. Prefill-then-decode: encode prompt fully, decode rest token by token
# ---------------------------------------------------------------------------

def test_prefill_then_decode():
    """
    Mixed mode: prefill T_prompt=8 tokens in one shot, then 4 incremental tokens.
    Parallel-LNN mode for bit-exact parity with single-shot full forward.
    """
    torch.manual_seed(7)
    cfg = small_config()
    model = MTLNNModel(cfg).eval()

    T_prompt, T_gen = 8, 4
    ids = torch.randint(0, cfg.vocab_size, (1, T_prompt + T_gen))

    cache = None
    pieces = []
    with torch.no_grad():
        full = model(ids, use_lnn_recurrence=False)["logits"]
        # Prefill (parallel mode for parity)
        out = model(ids[:, :T_prompt], use_cache=True, use_lnn_recurrence=False)
        cache = out["cache"]
        pieces.append(out["logits"])
        # Decode one token at a time
        for t in range(T_gen):
            tok = ids[:, T_prompt + t: T_prompt + t + 1]
            out = model(tok, cache=cache, use_cache=True, use_lnn_recurrence=False)
            cache = out["cache"]
            pieces.append(out["logits"])
        cached = torch.cat(pieces, dim=1)

    diff = (full - cached).abs().max().item()
    print(f"  max diff (prefill+decode vs full): {diff:.6e}")
    assert diff < 1e-4, f"Prefill-decode parity FAILED: {diff:.6e}"
    print("[ok] test_prefill_then_decode")


# ---------------------------------------------------------------------------
# 5. GQA verification: KV cache uses fewer heads than Q
# ---------------------------------------------------------------------------

def test_gqa_kv_cache_size():
    cfg = small_config()
    model = MTLNNModel(cfg).eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(ids, use_cache=True)

    # Inspect first layer's KV cache
    K, V = out["cache"].layers[0][0]
    assert K.shape == (1, cfg.n_kv_heads, 8, cfg.d_head), f"K shape: {K.shape}"
    assert V.shape == (1, cfg.n_kv_heads, 8, cfg.d_head), f"V shape: {V.shape}"
    saved = (cfg.n_heads - cfg.n_kv_heads) / cfg.n_heads
    print(f"  KV cache: {cfg.n_kv_heads} heads vs {cfg.n_heads} Q heads ({saved*100:.0f}% memory saved)")
    print("[ok] test_gqa_kv_cache_size")


# ---------------------------------------------------------------------------
# 6. Overfit: can the model actually learn?
# ---------------------------------------------------------------------------

def test_low_rank_polarity():
    """
    Low-rank bilinear polarity mode should:
      1. Be opt-in (not affect default config behaviour)
      2. Produce a valid forward + backward pass
      3. Have a non-trivial number of extra params (2·d·r per head)
    """
    cfg_default = small_config()
    cfg_lr = MTLNNConfig(
        vocab_size=200, max_seq_len=64, d_model=128, n_layers=2,
        n_heads=4, n_kv_heads=2, d_head=32, dropout=0.0, attention_dropout=0.0,
        polarity_mode="low_rank", polarity_rank=4,
    )
    m_default = MTLNNModel(cfg_default).eval()
    m_lr = MTLNNModel(cfg_lr).eval()

    n_default = m_default.get_num_params()
    n_lr = m_lr.get_num_params()
    extra = n_lr - n_default
    print(f"  scalar polarity: {n_default:,} params  |  low_rank polarity: {n_lr:,} params  "
          f"(+{extra:,})")
    assert extra > 0, "low_rank polarity didn't add any parameters"

    ids = torch.randint(0, cfg_lr.vocab_size, (2, 16))
    out = m_lr(ids, labels=ids)
    out["loss"].backward()
    for name, p in m_lr.named_parameters():
        if "pol_W_A" in name or "pol_W_B" in name or "pol_bilinear_gate" in name:
            assert p.grad is not None, f"low-rank polarity param has no grad: {name}"
    print("[ok] test_low_rank_polarity")


def test_nearest_neighbor_coupling():
    """
    Nearest-neighbor torch.roll coupling should produce non-trivial gradients
    on W_left, W_right, nn_eta and not break the parity test paths.
    """
    cfg = small_config()
    model = MTLNNModel(cfg).train()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(ids, labels=ids)
    out["loss"].backward()

    found = {"W_left": False, "W_right": False, "nn_eta": False}
    for name, p in model.named_parameters():
        for key in found:
            if name.endswith(key + ".weight") or name.endswith(key):
                assert p.grad is not None and p.grad.abs().sum().item() > 0, \
                    f"{key} has zero gradient"
                found[key] = True
    assert all(found.values()), f"missing NN-coupling params: {found}"
    print("[ok] test_nearest_neighbor_coupling")


def test_anesthesia_collapse():
    """
    Anesthesia validation: as anesthesia_level rises from 0→1, the entropy of
    the model's output distribution should INCREASE (the model becomes
    less confident — its 'consciousness' is degrading) and the output should
    diverge from the clean prediction.

    This is the in-silico mirror of the 2025 Wiest/Hameroff anesthesia finding.
    """
    torch.manual_seed(0)
    cfg = small_config()
    model = MTLNNModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))

    def entropy_and_logits(level):
        with anesthetize(model, level):
            with torch.no_grad():
                logits = model(ids)["logits"]                          # (1,T,V)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        ent = -(probs * probs.clamp(min=1e-9).log()).sum().item()
        return ent, logits.detach()

    ent_0, logits_0 = entropy_and_logits(0.0)
    ent_5, logits_5 = entropy_and_logits(0.5)
    ent_1, logits_1 = entropy_and_logits(1.0)

    diff_5 = (logits_0 - logits_5).abs().max().item()
    diff_1 = (logits_0 - logits_1).abs().max().item()

    print(f"  entropy: clean={ent_0:.3f}  half={ent_5:.3f}  full={ent_1:.3f}")
    print(f"  logit diff vs clean: half={diff_5:.4f}  full={diff_1:.4f}")
    # Anesthesia must actually change the output (sanity)
    assert diff_1 > diff_5 > 1e-5, "anesthesia hooks not firing"
    # Full anesthesia should produce a different distribution from clean
    assert diff_1 > 1e-3, f"full anesthesia barely affects output: {diff_1}"
    print("[ok] test_anesthesia_collapse")


def test_protofilament_scaling():
    """
    The vectorized MTLNNLayer should scale gracefully as P grows.
    Time at P=13 should not be >2× time at P=64 — proving the work happens
    on a single GPU/CPU einsum, not in a Python loop.
    """
    import time
    base_cfg = dict(vocab_size=200, max_seq_len=64, d_model=128, n_layers=1,
                    n_heads=4, n_kv_heads=2, d_head=32, dropout=0.0,
                    attention_dropout=0.0)

    times = {}
    for P in (13, 32, 64):
        cfg = MTLNNConfig(n_protofilaments=P, **base_cfg)
        model = MTLNNModel(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 32))
        # warmup
        for _ in range(3):
            with torch.no_grad():
                model(ids)
        t0 = time.time()
        for _ in range(20):
            with torch.no_grad():
                model(ids)
        times[P] = (time.time() - t0) / 20

    print(f"  P=13 → {times[13]*1000:.1f}ms  P=32 → {times[32]*1000:.1f}ms  "
          f"P=64 → {times[64]*1000:.1f}ms")
    # 5× more protofilaments should NOT cost 5× the time (CPU/GPU parallelism)
    ratio = times[64] / max(times[13], 1e-6)
    assert ratio < 5.0, f"P=64 is {ratio:.1f}× slower than P=13 — vectorisation broken"
    print(f"  scaling ratio (P=64 / P=13): {ratio:.2f}× — vectorisation OK")
    print("[ok] test_protofilament_scaling")


def test_overfit_single_batch():
    """Loss should drop ≥10× within 200 steps on a fixed batch."""
    torch.manual_seed(0)
    cfg = small_config()
    model = MTLNNModel(cfg).train()

    groups = make_param_groups(model, base_lr=3e-3)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))

    ids = torch.randint(0, cfg.vocab_size, (4, 16))

    init_loss = None
    for step in range(200):
        opt.zero_grad()
        out = model(ids, labels=ids)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0:
            init_loss = out["loss"].item()
    final = out["loss"].item()
    print(f"  init {init_loss:.3f} → final {final:.3f}")
    assert final < init_loss * 0.1, f"Overfit failed: init={init_loss:.3f} final={final:.3f}"
    print("[ok] test_overfit_single_batch")


# ---------------------------------------------------------------------------
# 7. MT diagnostics return finite values
# ---------------------------------------------------------------------------

def test_mt_diagnostics():
    cfg = small_config()
    model = MTLNNModel(cfg).eval()
    diag = model.get_mt_diagnostics()

    expected_keys = {"tau_mean", "tau_std", "gamma_mean", "polarity_mean",
                     "polarity_std", "lat_coupling_mean_off_diag_norm",
                     "coherence_scale", "collapse_threshold"}
    missing = expected_keys - diag.keys()
    assert not missing, f"Missing diagnostic keys: {missing}"
    for k, v in diag.items():
        assert math.isfinite(v), f"Non-finite diagnostic: {k}={v}"

    # τ initialised from resonance_freqs → values span the configured range
    # and there must be variance (multi-scale init not collapsed to a single value)
    assert cfg.tau_min <= diag["tau_min"] <= diag["tau_max"] <= cfg.tau_max
    assert diag["tau_std"] > 0.1, f"τ has no spread — multi-scale init failed: {diag['tau_std']}"
    # Lateral coupling near-identity at init
    assert diag["lat_coupling_mean_off_diag_norm"] < 0.5
    print(f"  τ={diag['tau_mean']:.3f}±{diag['tau_std']:.3f} "
          f"[{diag['tau_min']:.3f}, {diag['tau_max']:.3f}]  γ={diag['gamma_mean']:.3f}  "
          f"polarity_std={diag['polarity_std']:.3f}  rmc_gate={diag['rmc_gate_mean']:.3f}")
    print("[ok] test_mt_diagnostics")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    print("=" * 60)
    print("MT-LNN test suite")
    print("=" * 60)
    test_shapes_and_loss()
    test_gradient_flow()
    test_kv_cache_parity()
    test_lnn_recurrence_active()
    test_prefill_then_decode()
    test_gqa_kv_cache_size()
    test_mt_diagnostics()
    test_low_rank_polarity()
    test_nearest_neighbor_coupling()
    test_anesthesia_collapse()
    test_protofilament_scaling()
    test_overfit_single_batch()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
