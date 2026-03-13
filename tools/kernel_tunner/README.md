# SYCLTLA MoE Tuner

This tool benchmarks and tunes grouped GEMM policy pairs for MoE kernels on XPU.

Script:

- `tools/kernel_tunner/sycltla_moe_tunner.py`

## What It Does

- Benchmark mode: measure kernel latency for selected batch sizes.
- Tune mode: search policy pairs for w13/w2 GEMMs and write tuned policy JSON.

## Quick Start

Run from repo root (`vllm-xpu-kernels`).

Benchmark with an existing policy template:

```bash
python tools/kernel_tunner/sycltla_moe_tunner.py \
	--model Qwen/Qwen3-30B-A3B-Instruct-2507 \
	--dtype-key bf16 \
    --tp-size 4 \
	--policy-config tools/kernel_tunner/default_moe_policy.json \
	--batch-size 80 8192
```

Tune policy pairs and save results:

```bash
python tools/kernel_tunner/sycltla_moe_tunner.py \
	--model Qwen/Qwen3-30B-A3B-Instruct-2507 \
	--dtype-key bf16 \
    --tp-size 4 \
	--policy-config tools/kernel_tunner/default_moe_policy.json \
    --batch-size 80 8192 \
	--tune
```

## Important Arguments

- `--dtype-key`: `bf16`, `fp8_w8a16`, `mxfp4_w4a16`
- `--policy-config`: base policy JSON template used during benchmark/tuning
- `--batch-size`: one or more batch sizes
- `--tune`: enable policy search mode
- `--token-stats-dir`: optional data directory containing num_rows_per_expert
- `--tp-size`: tensor parallel size
- `--enable-expert-parallel`: enable expert parallel path

## Outputs

When `--tune` is enabled, two files are generated in `--save-dir`:

- `<model_prefix>_sycltla_policy_tuning_<dtype_key>.json`
	- Raw best pair per batch size.
- `<model_prefix>_default_moe_policy_tuned_<dtype_key>.json`
	- Runtime policy JSON (global + overrides) built from tuned results.

