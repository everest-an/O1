# Sparse Resonance Ablation

Compares dense tau-scale computation against optional sparse top-k resonance.

| Mode | Top-k | Selected scale ratio | Mean time (s) | Min time (s) | Std (s) | Tok/s | Mean div vs dense | Max div vs dense |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dense` | 5 | 1.000 | 0.0153 | 0.0149 | 0.0003 | 6528.0 | 0 | 0 |
| `sparse` | 1 | 0.200 | 0.0135 | 0.0130 | 0.0003 | 7390.0 | 0.00261 | 0.107 |
| `sparse` | 2 | 0.400 | 0.0152 | 0.0139 | 0.0015 | 6586.2 | 0.00331 | 0.0796 |
| `sparse` | 3 | 0.600 | 0.0147 | 0.0142 | 0.0004 | 6780.4 | 0.00171 | 0.0699 |

## Caveats

- Sparse top-k selection is intentionally approximate and can change outputs.
- The current implementation uses PyTorch tensor indexing and top-k over tau scales, not a custom CUDA/Triton kernel.
- Speedups depend on sequence length, device, and whether top-k selection overhead dominates the skipped scale work.
