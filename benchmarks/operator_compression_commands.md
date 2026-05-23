# Operator Compression Commands

## State-Only Streaming Demo

```powershell
python demo.py --ckpt checkpoints/final.pt --prompt "Hello" --state_only --session_id demo_session
python demo.py --ckpt checkpoints/final.pt --prompt "Continue from memory" --state_only --session_id demo_session
```

## Direct Target Demo

```powershell
python demo.py --ckpt checkpoints/final.pt --prompt "copy targets:" --direct_target_len 4
```

## Benchmark

```powershell
python benchmarks/state_only_streaming.py --steps 100 1000
```

## Report

```powershell
python benchmarks/operator_compression_report.py --steps 100 1000
```

Notes:

- State-only mode keeps recurrent `h_prev` and drops historical KV tensors.
- Direct target mode is useful only after the direct head has task-specific supervision.
- Scale-gate masking is currently blend-level masking, not a custom sparse kernel.

