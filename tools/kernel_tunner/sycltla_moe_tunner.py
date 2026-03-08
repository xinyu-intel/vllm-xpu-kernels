# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import copy
import contextlib
import json
import os
import re
import tempfile
import time
from datetime import datetime
from typing import Any
from tqdm import tqdm

import torch

from vllm.model_executor.layers.fused_moe.fused_moe import *
from vllm.transformers_utils.config import get_config
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed

try:
    from . import _C  # noqa: F401
    from . import _xpu_C  # noqa: F401

    FUSEDMOE_UNAVAILABLE_REASON = None
    FUSEDMOE_AVAILABLE = True
except ImportError as e:
    FUSEDMOE_UNAVAILABLE_REASON = str(e)
    FUSEDMOE_AVAILABLE = False

from vllm_xpu_kernels.fused_moe_interface import cutlass_grouped_gemm_xe2

POLICY_IDS = [
    "wg_256_128_32_sg_8_2_1",
    "wg_128_256_32_sg_4_8_1",
    "wg_32_64_32_sg_1_4_1",
    "wg_16_64_32_sg_1_4_1",
    "wg_8_64_32_sg_1_4_1",
    "wg_128_64_32_sg_4_2_1",
    "wg_128_128_32_sg_4_2_1",
    "wg_256_64_32_sg_8_2_1",
    "wg_256_256_32_sg_8_4_1",
]
DEFAULT_POLICY_NAME = POLICY_IDS[0]
DEFAULT_AVG_M_THRESHOLDS = [8, 16, 32]


def vlog(message: str):
    if args.verbose:
        print(f"[verbose] {message}")


def get_all_policy_pairs() -> list[dict[str, str]]:
    """Cartesian product search space for w13 and w2 grouped GEMMs."""
    return [
        {"w13_policy": w13_policy, "w2_policy": w2_policy}
        for w13_policy in POLICY_IDS
        for w2_policy in POLICY_IDS
    ]


def _set_policy_block(
    policy_block: dict[str, Any],
    policy_name: str,
    avg_m_max: int | None = None,
) -> None:
    policy_block["default_policy"] = policy_name
    if avg_m_max is not None:
        policy_block["rules"] = [
            {"avg_m_max": int(avg_m_max), "policy": policy_name}
        ]
        return

    rules = policy_block.get("rules", [])
    if isinstance(rules, list) and len(rules) > 0:
        for rule in rules:
            if isinstance(rule, dict):
                rule["policy"] = policy_name
    else:
        policy_block["rules"] = [
            {"avg_m_max": max_m, "policy": policy_name}
            for max_m in DEFAULT_AVG_M_THRESHOLDS
        ]


def _collect_avg_m_thresholds(configs: dict[int, dict[str, Any]]) -> list[int]:
    thresholds = {
        max(1, int(round(float(entry.get("avg_m", -1)))))
        for entry in configs.values()
        if float(entry.get("avg_m", -1)) > 0
    }
    if not thresholds:
        return DEFAULT_AVG_M_THRESHOLDS
    return sorted(thresholds)


def _build_rules_from_tuned_configs(
    configs: dict[int, dict[str, Any]],
    policy_key: str,
    thresholds: list[int],
) -> list[dict[str, Any]]:
    """Create rules using a shared avg_m_max grid across all GEMMs."""
    point_to_policy: dict[int, str] = {}
    for entry in configs.values():
        if policy_key not in entry:
            continue
        avg_m = float(entry.get("avg_m", -1))
        if avg_m <= 0:
            continue
        avg_m_max = max(1, int(round(avg_m)))
        point_to_policy[avg_m_max] = str(entry[policy_key])

    if not point_to_policy:
        return [
            {"avg_m_max": t, "policy": DEFAULT_POLICY_NAME} for t in thresholds
        ]

    sorted_points = sorted(point_to_policy.keys())
    rules: list[dict[str, Any]] = []
    last_policy = point_to_policy[sorted_points[0]]
    for t in thresholds:
        # Use policy of nearest tuned point <= threshold, else smallest point.
        chosen_point = sorted_points[0]
        for p in sorted_points:
            if p <= t:
                chosen_point = p
            else:
                break
        policy = point_to_policy[chosen_point]
        rules.append({"avg_m_max": t, "policy": policy})
        last_policy = policy

    return rules


def _make_override(
    dtype_key: str,
    n: int,
    k: int,
    tag: str,
    policy_name: str,
    avg_m_max: int | None = None,
) -> dict[str, Any]:
    if avg_m_max is not None:
        rules = [{"avg_m_max": int(avg_m_max), "policy": policy_name}]
    else:
        rules = [
            {"avg_m_max": max_m, "policy": policy_name}
            for max_m in DEFAULT_AVG_M_THRESHOLDS
        ]

    policies = {
        "default_policy": policy_name,
        "rules": rules,
    }
    return {
        "dtype": dtype_key,
        "n": n,
        "k": k,
        "tag": tag,
        "policies": policies,
    }


def _extract_template_defaults(
    policy_template_path: str,
    dtype_key: str,
    w13_n: int,
    w13_k: int,
    w2_n: int,
    w2_k: int,
) -> dict[str, str]:
    """Extract default_policy values from template for global/w13/w2 scopes."""
    with open(policy_template_path) as f:
        payload = json.load(f)

    out: dict[str, str] = {}

    global_block = payload.get("global", {}).get(dtype_key, {})
    if isinstance(global_block, dict) and "default_policy" in global_block:
        out["global"] = str(global_block["default_policy"])

    overrides = payload.get("overrides", [])
    if isinstance(overrides, list):
        for ov in overrides:
            if not isinstance(ov, dict):
                continue
            tag = str(ov.get("tag", ""))
            ov_dtype = str(ov.get("dtype", "")).lower()
            ov_n = ov.get("n")
            ov_k = ov.get("k")
            policies = ov.get("policies", {})
            if (
                not isinstance(policies, dict)
                or "default_policy" not in policies
            ):
                continue

            if ov_dtype == dtype_key and ov_n == w13_n and ov_k == w13_k:
                out["w13"] = str(policies["default_policy"])
            elif ov_dtype == dtype_key and ov_n == w2_n and ov_k == w2_k:
                out["w2"] = str(policies["default_policy"])
            # Fallback for templates that rely on tag readability rather than
            # exact dtype/n/k shape values.
            elif "w13" not in out and tag == "w13":
                out["w13"] = str(policies["default_policy"])
            elif "w2" not in out and tag == "w2":
                out["w2"] = str(policies["default_policy"])

    return out


def build_trial_policy_json(
    policy_template_path: str,
    dtype_key: str,
    w13_n: int,
    w13_k: int,
    w2_n: int,
    w2_k: int,
    w13_policy: str,
    w2_policy: str,
    avg_m_max: int | None = None,
) -> dict[str, Any]:
    with open(policy_template_path) as f:
        payload = json.load(f)

    trial = copy.deepcopy(payload)

    if "global" not in trial:
        trial["global"] = {}
    if dtype_key not in trial["global"]:
        trial["global"][dtype_key] = {
            "default_policy": DEFAULT_POLICY_NAME,
            "rules": [
                {"avg_m_max": max_m, "policy": DEFAULT_POLICY_NAME}
                for max_m in DEFAULT_AVG_M_THRESHOLDS
            ],
        }

    # Keep global untouched by default; tuning focus is w13/w2 override pair.
    overrides = trial.get("overrides", [])
    if not isinstance(overrides, list):
        raise ValueError("Invalid policy JSON: 'overrides' must be a list")

    w13_found = False
    w2_found = False
    w13_tag_idx: int | None = None
    w2_tag_idx: int | None = None
    for idx, ov in enumerate(overrides):
        if not isinstance(ov, dict):
            continue
        tag = str(ov.get("tag", ""))
        ov_dtype = str(ov.get("dtype", "")).lower()
        ov_n = ov.get("n")
        ov_k = ov.get("k")

        if tag == "w13" and w13_tag_idx is None:
            w13_tag_idx = idx
        elif tag == "w2" and w2_tag_idx is None:
            w2_tag_idx = idx

        if ov_dtype == dtype_key and ov_n == w13_n and ov_k == w13_k:
            policies = ov.setdefault("policies", {})
            _set_policy_block(policies, w13_policy, avg_m_max)
            w13_found = True
        elif ov_dtype == dtype_key and ov_n == w2_n and ov_k == w2_k:
            policies = ov.setdefault("policies", {})
            _set_policy_block(policies, w2_policy, avg_m_max)
            w2_found = True

    # If exact keys are missing, first try to reuse a readable-tag override by
    # updating its dtype/n/k, otherwise append a new override entry.
    if not w13_found:
        if w13_tag_idx is not None:
            ov = overrides[w13_tag_idx]
            ov["dtype"] = dtype_key
            ov["n"] = w13_n
            ov["k"] = w13_k
            policies = ov.setdefault("policies", {})
            _set_policy_block(policies, w13_policy, avg_m_max)
        else:
            overrides.append(
                _make_override(
                    dtype_key,
                    w13_n,
                    w13_k,
                    "w13",
                    w13_policy,
                    avg_m_max,
                )
            )

    if not w2_found:
        if w2_tag_idx is not None:
            ov = overrides[w2_tag_idx]
            ov["dtype"] = dtype_key
            ov["n"] = w2_n
            ov["k"] = w2_k
            policies = ov.setdefault("policies", {})
            _set_policy_block(policies, w2_policy, avg_m_max)
        else:
            overrides.append(
                _make_override(
                    dtype_key,
                    w2_n,
                    w2_k,
                    "w2",
                    w2_policy,
                    avg_m_max,
                )
            )

    return trial


class PolicyConfigContext(contextlib.AbstractContextManager[Any]):
    def __init__(self, policy_json: dict[str, Any]):
        self.policy_json = policy_json
        self.prev_policy_path: str | None = None
        self.temp_file: tempfile.NamedTemporaryFile | None = None

    def __enter__(self):
        self.prev_policy_path = os.environ.get("VLLM_XPU_MOE_POLICY_PATH")
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(self.policy_json, self.temp_file, indent=2)
        self.temp_file.write("\n")
        self.temp_file.flush()
        self.temp_file.close()
        os.environ["VLLM_XPU_MOE_POLICY_PATH"] = self.temp_file.name
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.prev_policy_path is None:
            os.environ.pop("VLLM_XPU_MOE_POLICY_PATH", None)
        else:
            os.environ["VLLM_XPU_MOE_POLICY_PATH"] = self.prev_policy_path
        if self.temp_file is not None:
            try:
                os.unlink(self.temp_file.name)
            except FileNotFoundError:
                pass
        return False


def ensure_divisibility(numerator, denominator, text):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, (
        "{} {} is not divisible by tp {}.".format(text, numerator, denominator)
    )


def load_token_stats(
    token_stats_dir: str,
    i: int,
    num_layers: int,
    dense_layers: int,
    batch_size: int,
) -> torch.Tensor:
    """Load dumped token_stats following sglang naming convention."""
    moe_layers = num_layers - dense_layers
    if moe_layers <= 0:
        raise ValueError(
            f"Invalid layer config: num_layers={num_layers}, dense_layers={dense_layers}"
        )
    layer_id = i % moe_layers + dense_layers
    req_idx = i // moe_layers
    bs_path = os.path.join(
        token_stats_dir,
        f"token_stats_bs{batch_size}_layer{layer_id}_idx{req_idx}.pt",
    )
    if os.path.exists(bs_path):
        return torch.load(bs_path, map_location="xpu")
    else:
        raise FileNotFoundError(f"TopK IDs file not found: {bs_path}")


def load_token_stats_list(
    token_stats_dir: str,
    count: int,
    num_layers: int,
    dense_layers: int,
    batch_size: int,
) -> list[torch.Tensor]:
    return [
        load_token_stats(
            token_stats_dir,
            i,
            num_layers=num_layers,
            dense_layers=dense_layers,
            batch_size=batch_size,
        )
        for i in range(count)
    ]


def get_token_stats_file_count(
    token_stats_dir: str,
    batch_size: int,
    num_layers: int,
    dense_layers: int,
) -> tuple[int, bool]:
    """Return (file_count, has_bs_specific_files_for_batch)."""
    moe_layers = num_layers - dense_layers
    if moe_layers <= 0:
        return 0, False

    bs_pat = re.compile(rf"^token_stats_bs{batch_size}_layer\d+_idx\d+\.pt$")
    legacy_pat = re.compile(r"^token_stats_layer\d+_idx\d+\.pt$")

    names = os.listdir(token_stats_dir)
    bs_count = sum(1 for n in names if bs_pat.match(n))
    if bs_count > 0:
        return bs_count, True

    legacy_count = sum(1 for n in names if legacy_pat.match(n))
    return legacy_count, False


def scan_token_stats_dir(token_stats_dir: str) -> tuple[dict[int, int], int]:
    """Return (bs_specific_counts, legacy_count) from token_stats directory."""
    bs_specific_counts: dict[int, int] = {}
    legacy_count = 0
    bs_pat = re.compile(r"^token_stats_bs(\d+)_layer\d+_idx\d+\.pt$")
    legacy_pat = re.compile(r"^token_stats_layer\d+_idx\d+\.pt$")
    for name in os.listdir(token_stats_dir):
        bs_match = bs_pat.match(name)
        if bs_match:
            bs = int(bs_match.group(1))
            bs_specific_counts[bs] = bs_specific_counts.get(bs, 0) + 1
            continue
        if legacy_pat.match(name):
            legacy_count += 1
    return bs_specific_counts, legacy_count


def init_rows_for_experts(tokens, topk, num_rows_per_expert):
    if num_rows_per_expert.shape[0] == 1:
        num_rows_per_expert[0] = tokens * topk
        return
    n_experts = num_rows_per_expert.numel()
    rand = torch.rand(tokens, n_experts, device=num_rows_per_expert.device)
    topk_idx = torch.topk(rand, topk, dim=1).indices  # [tokens, topk]
    flat_idx = topk_idx.flatten()
    num_rows_per_expert += torch.bincount(flat_idx, minlength=n_experts)


def get_act_weight_dtype(dtype_key: str) -> torch.dtype:
    if dtype_key == "bf16":
        act_dtype = torch.bfloat16
        weight_dtype = torch.bfloat16
    elif dtype_key == "fp8_w8a16":
        act_dtype = torch.bfloat16
        weight_dtype = torch.float8_e4m3fn
    elif dtype_key == "mxfp4_w4a16":
        act_dtype = torch.bfloat16
        weight_dtype = torch.uint8
    else:
        raise ValueError(f"Unsupported dtype_key: {dtype_key}")
    return act_dtype, weight_dtype


def benchmark_config(
    config: contextlib.AbstractContextManager[Any],
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    dtype_key: str,
    token_stats_list: list[torch.Tensor] | None = None,
    token_stats_stride: int = 1,
    num_iters: int = 100,
    block_quant_shape: list[int] = None,
) -> float:
    act_dtype, weight_dtype = get_act_weight_dtype(dtype_key)
    x1 = torch.randn(
        num_tokens * topk, hidden_size, dtype=act_dtype, device="xpu"
    )
    x2 = torch.randn(
        (topk * num_tokens, shard_intermediate_size // 2),
        dtype=act_dtype,
        device="xpu",
    )
    if dtype_key == "mxfp4_w4a16":
        w1 = torch.randint(
            0,
            0xFF,
            (num_experts, shard_intermediate_size, hidden_size // 2),
            dtype=weight_dtype,
            device="xpu",
        )
        w2 = torch.randint(
            0,
            0xFF,
            (num_experts, hidden_size, shard_intermediate_size // 4),
            dtype=weight_dtype,
            device="xpu",
        )
    else:
        w1 = torch.randn(
            num_experts,
            hidden_size,
            shard_intermediate_size,
            dtype=torch.bfloat16,
            device="xpu",
        ).to(weight_dtype)
        w2 = torch.randn(
            num_experts,
            shard_intermediate_size // 2,
            hidden_size,
            dtype=torch.bfloat16,
            device="xpu",
        ).to(weight_dtype)
    use_external_token_stats = (
        token_stats_list is not None and len(token_stats_list) > 0
    )

    if dtype_key == "fp8_w8a16":
        if block_quant_shape:
            block_n, block_k = block_quant_shape[0], block_quant_shape[1]
            E = num_experts
            N = shard_intermediate_size // 2
            K = hidden_size
            factor_for_scale = 1e-2
            n_tiles_w1 = (2 * N + block_n - 1) // block_n
            n_tiles_w2 = (K + block_n - 1) // block_n
            k_tiles_w1 = (K + block_k - 1) // block_k
            k_tiles_w2 = (N + block_k - 1) // block_k
            w1_scale = (
                torch.rand((E, n_tiles_w1, k_tiles_w1), dtype=torch.float32)
                * factor_for_scale
            )
            w2_scale = (
                torch.rand((E, n_tiles_w2, k_tiles_w2), dtype=torch.float32)
                * factor_for_scale
            )
        else:
            w1_scale = torch.randn(num_experts, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, dtype=torch.float32)
    elif dtype_key == "mxfp4_w4a16":
        group_size = 32
        w1_scale = torch.randint(
            0,
            0x7F,
            (num_experts, shard_intermediate_size, hidden_size // group_size),
            dtype=torch.uint8,
        )
        w2_scale = torch.randint(
            0,
            0x7F,
            (num_experts, hidden_size, shard_intermediate_size // 2 // group_size),
            dtype=torch.uint8,
        )
    else:
        w1_scale = None
        w2_scale = None

    gemm1_output = torch.empty(
        (topk * num_tokens, shard_intermediate_size), dtype=dtype, device="xpu"
    )
    gemm2_output = torch.empty(
        (topk * num_tokens, hidden_size), dtype=dtype, device="xpu"
    )
    num_rows_per_expert = torch.zeros(
        num_experts, device="xpu", dtype=torch.int32
    )
    init_rows_for_experts(num_tokens, topk, num_rows_per_expert)

    @torch.compile
    def run(num_rows_per_expert):
        with config:
            cutlass_grouped_gemm_xe2(
                x1,
                w1,
                w1_scale,
                None,
                gemm1_output,
                num_rows_per_expert,
                shard_intermediate_size,
                hidden_size,
                num_experts,
                False,
                dtype_key == "mxfp4_w4a16",
            )
            cutlass_grouped_gemm_xe2(
                x2,
                w2,
                w2_scale,
                None,
                gemm2_output,
                num_rows_per_expert,
                hidden_size,
                shard_intermediate_size // 2,
                num_experts,
                False,
                dtype_key == "mxfp4_w4a16",
            )

    # JIT compilation & warmup
    for _ in range(10):
        run(num_rows_per_expert)
    torch.accelerator.synchronize()

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)

    start_event.record()
    for i in range(num_iters):
        num_rows_per_expert = (
            token_stats_list[
                (i * max(1, token_stats_stride)) % len(token_stats_list)
            ]
            if use_external_token_stats
            else num_rows_per_expert
        )
        run(num_rows_per_expert)
    end_event.record()
    end_event.synchronize()
    avg = start_event.elapsed_time(end_event) / (num_iters) * 1000  # us
    return avg


def merge_unique_dicts(list1, list2):
    result = []
    combined_list = list1.copy()
    combined_list.extend(list2)
    for dictionary in combined_list:
        if dictionary not in result:
            result.append(dictionary)
    return result


class BenchmarkWorker:
    def __init__(self, seed: int) -> None:
        torch.set_default_device("xpu")
        set_random_seed(seed)
        self.seed = seed

    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        dtype_key: str,
        policy_template_path: str,
        token_stats_list: list[torch.Tensor] | None = None,
        token_stats_stride: int = 1,
        num_iters: int = 100,
        block_quant_shape: list[int] = None,
    ) -> tuple[dict[str, int], float]:

        set_random_seed(self.seed)
        with open(policy_template_path) as f:
            payload = json.load(f)
        trial = copy.deepcopy(payload)
        config = PolicyConfigContext(trial)
        kernel_time = benchmark_config(
            config,
            num_tokens,
            num_experts,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            dtype_key,
            token_stats_list=token_stats_list,
            token_stats_stride=token_stats_stride,
            num_iters=num_iters,
            block_quant_shape=block_quant_shape,
        )
        return config, kernel_time

    def tune(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        search_space: list[dict[str, str]],
        token_stats_list: list[torch.Tensor] | None,
        token_stats_stride: int,
        num_iters: int,
        block_quant_shape: list[int],
        policy_template_path: str,
        dtype_key: str,
        w13_n: int,
        w13_k: int,
        w2_n: int,
        w2_k: int,
    ) -> dict[str, Any]:
        best_config: dict[str, Any] | None = None
        best_time = float("inf")
        avg_m = (num_tokens * topk) / num_experts
        avg_m_max = max(1, int(round(avg_m)))

        for _, config in enumerate(tqdm(search_space)):
            try:
                trial_policy_json = build_trial_policy_json(
                    policy_template_path=policy_template_path,
                    dtype_key=dtype_key,
                    w13_n=w13_n,
                    w13_k=w13_k,
                    w2_n=w2_n,
                    w2_k=w2_k,
                    w13_policy=config["w13_policy"],
                    w2_policy=config["w2_policy"],
                    avg_m_max=avg_m_max,
                )
                kernel_time = benchmark_config(
                    PolicyConfigContext(trial_policy_json),
                    num_tokens,
                    num_experts,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    dtype_key,
                    token_stats_list=token_stats_list,
                    token_stats_stride=token_stats_stride,
                    num_iters=num_iters,
                    block_quant_shape=block_quant_shape,
                )
                vlog(
                    f"num_tokens={num_tokens}, Config: {config}, Kernel time: {kernel_time:.2f} us"
                )
            except triton.runtime.autotuner.OutOfResources:
                # Some configurations may be invalid and fail to compile.
                continue

            if kernel_time < best_time:
                best_time = kernel_time
                best_config = {
                    "w13_policy": config["w13_policy"],
                    "w2_policy": config["w2_policy"],
                    "kernel_time_us": kernel_time,
                    "avg_m": avg_m,
                }

        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        assert best_config is not None
        return best_config


def save_tuned_policy_pairs(
    configs: dict[int, dict[str, Any]],
    save_dir: str,
    model_name_prefix: str,
    dtype_key: str,
    policy_template_path: str,
    w13_n: int,
    w13_k: int,
    w2_n: int,
    w2_k: int,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(
        save_dir, f"{model_name_prefix}_sycltla_policy_tuning_{dtype_key}.json"
    )
    print(f"Writing tuned policy pairs to {filename}...")
    with open(filename, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")

    # Build a formal runtime policy JSON from all tuned batch sizes so policy
    # can vary across avg_m regions.
    if not configs:
        return filename

    template_defaults = _extract_template_defaults(
        policy_template_path=policy_template_path,
        dtype_key=dtype_key,
        w13_n=w13_n,
        w13_k=w13_k,
        w2_n=w2_n,
        w2_k=w2_k,
    )

    best_batch, best_entry = min(
        configs.items(),
        key=lambda kv: float(kv[1].get("kernel_time_us", float("inf"))),
    )
    formal_json = build_trial_policy_json(
        policy_template_path=policy_template_path,
        dtype_key=dtype_key,
        w13_n=w13_n,
        w13_k=w13_k,
        w2_n=w2_n,
        w2_k=w2_k,
        w13_policy=best_entry["w13_policy"],
        w2_policy=best_entry["w2_policy"],
    )

    shared_thresholds = _collect_avg_m_thresholds(configs)
    w13_rules = _build_rules_from_tuned_configs(
        configs, "w13_policy", shared_thresholds
    )
    w2_rules = _build_rules_from_tuned_configs(
        configs, "w2_policy", shared_thresholds
    )

    for ov in formal_json.get("overrides", []):
        if not isinstance(ov, dict):
            continue
        ov_dtype = str(ov.get("dtype", "")).lower()
        ov_n = ov.get("n")
        ov_k = ov.get("k")
        if ov_dtype != dtype_key:
            continue

        policies = ov.setdefault("policies", {})
        if ov_n == w13_n and ov_k == w13_k:
            if "w13" in template_defaults:
                policies["default_policy"] = template_defaults["w13"]
            policies["rules"] = w13_rules
        elif ov_n == w2_n and ov_k == w2_k:
            if "w2" in template_defaults:
                policies["default_policy"] = template_defaults["w2"]
            policies["rules"] = w2_rules

    global_block = formal_json.get("global", {}).get(dtype_key, {})
    if (
        isinstance(global_block, dict)
        and "global" in template_defaults
        and "default_policy" in global_block
    ):
        global_block["default_policy"] = template_defaults["global"]

    formal_json.setdefault("context", {})
    formal_json["context"]["tuning_selected_batch_size"] = int(best_batch)
    formal_json["context"]["tuning_selected_kernel_time_us"] = float(
        best_entry["kernel_time_us"]
    )

    formal_filename = os.path.join(
        save_dir,
        f"{model_name_prefix}_default_moe_policy_tuned_{dtype_key}.json",
    )
    print(f"Writing formal tuned policy config to {formal_filename}...")
    with open(formal_filename, "w") as f:
        json.dump(formal_json, f, indent=2)
        f.write("\n")

    return filename


def get_compressed_tensors_block_structure(config, default_value=None):
    config_groups = config.get("config_groups", {})
    if len(config_groups) != 1:
        return default_value
    group = next(iter(config_groups.values()))
    weights = group.get("weights", {})
    block_structure = weights.get("block_structure", default_value)
    return block_structure


def get_weight_block_size_safety(config, default_value=None):
    quantization_config = getattr(config, "quantization_config", {})
    if isinstance(quantization_config, dict):
        if "weight_block_size" in quantization_config:
            return quantization_config["weight_block_size"]
        return get_compressed_tensors_block_structure(
            quantization_config, default_value
        )
    return default_value


def get_model_params(config):
    if config.architectures[0] == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
        hidden_size = config.hidden_size
    elif config.architectures[0] == "JambaForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        hidden_size = config.hidden_size
    elif config.architectures[0] in (
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "GlmMoeDsaForCausalLM",
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "NemotronHForCausalLM",
        "MistralLarge3ForCausalLM",
    ):
        E = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        hidden_size = config.hidden_size
    elif config.architectures[0] in (
        "Qwen2MoeForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3NextForCausalLM",
    ):
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        hidden_size = config.hidden_size
    elif config.architectures[0] == "Qwen3VLMoeForConditionalGeneration":
        text_config = config.get_text_config()
        E = text_config.num_experts
        topk = text_config.num_experts_per_tok
        intermediate_size = text_config.moe_intermediate_size
        hidden_size = text_config.hidden_size
    elif config.architectures[0] == "HunYuanMoEV1ForCausalLM":
        E = config.num_experts
        topk = config.moe_topk[0]
        intermediate_size = config.moe_intermediate_size[0]
        hidden_size = config.hidden_size
    elif config.architectures[0] == "Qwen3OmniMoeForConditionalGeneration":
        E = config.thinker_config.text_config.num_experts
        topk = config.thinker_config.text_config.num_experts_per_tok
        intermediate_size = (
            config.thinker_config.text_config.moe_intermediate_size
        )
        hidden_size = config.thinker_config.text_config.hidden_size
    elif config.architectures[0] == "PixtralForConditionalGeneration":
        # Pixtral can contain different LLM architectures,
        # recurse to get their parameters
        return get_model_params(config.get_text_config())
    else:
        # Support for llama4
        config = config.get_text_config()
        # Default: Mixtral.
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        hidden_size = config.hidden_size
    return E, topk, intermediate_size, hidden_size


def get_quantization_group_size(config) -> int | None:
    """Extract the quantization group size from the HF model config.

    This reads directly from the HuggingFace config object (as returned by
    ``get_config()``), not from vLLM's quantization config classes.

    Supports AWQ/GPTQ-style configs (direct 'group_size' key) and
    compressed-tensors configs (nested inside 'config_groups').
    """
    quantization_config = getattr(config, "quantization_config", {})
    if not isinstance(quantization_config, dict):
        return None
    # AWQ / GPTQ style: group_size is a top-level key
    gs = quantization_config.get("group_size")
    if gs is not None:
        return gs
    # compressed-tensors style: group_size is nested in config_groups
    config_groups = quantization_config.get("config_groups", {})
    if not isinstance(config_groups, dict):
        return None
    for group_cfg in config_groups.values():
        if not isinstance(group_cfg, dict):
            continue
        weights = group_cfg.get("weights", {})
        if not isinstance(weights, dict):
            continue
        gs = weights.get("group_size")
        if gs is not None:
            return gs
    return None


def get_token_stats_layer_mapping(config) -> tuple[int, int]:
    """Return (num_layers, dense_layers) for token_stats filename mapping.

    num_layers: prefer num_hidden_layers, fallback to text_config.num_hidden_layers.
    dense_layers: prefer first_k_dense_replace, fallback to
    text_config.first_k_dense_replace, else 0.
    """
    num_layers = getattr(config, "num_hidden_layers", None)
    dense_layers = getattr(config, "first_k_dense_replace", 0)

    if (num_layers is None or int(num_layers) <= 0) and hasattr(
        config, "get_text_config"
    ):
        text_config = config.get_text_config()
        num_layers = getattr(text_config, "num_hidden_layers", num_layers)
        dense_layers = getattr(
            text_config, "first_k_dense_replace", dense_layers
        )

    if num_layers is None or int(num_layers) <= 0:
        raise ValueError(
            "Cannot infer num_hidden_layers from model config for token_stats mapping."
        )

    return int(num_layers), int(dense_layers or 0)


def main(args: argparse.Namespace):
    print(args)

    config = get_config(
        model=args.model, trust_remote_code=args.trust_remote_code
    )
    model_name_prefix = args.model.replace("/", "_").replace("-", "_")
    if args.model_prefix:
        config = getattr(config, args.model_prefix)
    E, topk, intermediate_size, hidden_size = get_model_params(config)
    enable_ep = bool(args.enable_expert_parallel)
    if enable_ep:
        ensure_divisibility(E, args.tp_size, "Number of experts")
        E = E // args.tp_size
        shard_intermediate_size = 2 * intermediate_size
    else:
        ensure_divisibility(
            intermediate_size, args.tp_size, "intermediate_size"
        )
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    dtype = config.dtype
    dtype_key = args.dtype_key
    w13_n = shard_intermediate_size
    w13_k = hidden_size
    w2_n = hidden_size
    w2_k = shard_intermediate_size // 2
    block_quant_shape = get_weight_block_size_safety(config)
    num_layers, dense_layers = get_token_stats_layer_mapping(config)
    moe_layers = num_layers - dense_layers
    if args.token_stats_dir:
        bs_specific_counts, legacy_count = scan_token_stats_dir(
            args.token_stats_dir
        )
        total_bs_specific = sum(bs_specific_counts.values())
        print(
            "token_stats summary: "
            f"dir={args.token_stats_dir}, bs_specific_files={total_bs_specific}, "
            f"legacy_files={legacy_count}, bs_buckets={len(bs_specific_counts)}, "
            f"moe_layers={moe_layers}"
        )
        vlog(f"bs buckets detail: {dict(sorted(bs_specific_counts.items()))}")

    if args.batch_size is None:
        batch_sizes = [
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
            8192,
        ]
    else:
        batch_sizes = args.batch_size

    worker = BenchmarkWorker(seed=args.seed)

    if args.tune:
        search_space = get_all_policy_pairs()

        print(f"Start tuning over {len(search_space)} configurations...")
        start = time.time()
        configs = []
        for batch_size in batch_sizes:
            token_stats_list_bs: list[torch.Tensor] | None = None
            token_stats_stride = 1
            tune_iters = 20
            if args.token_stats_dir:
                file_count, has_bs_specific = get_token_stats_file_count(
                    args.token_stats_dir,
                    int(batch_size),
                    num_layers,
                    dense_layers,
                )
                replay_iters = file_count // moe_layers if moe_layers > 0 else 0
                if replay_iters > 0:
                    token_stats_list_bs = load_token_stats_list(
                        token_stats_dir=args.token_stats_dir,
                        count=replay_iters * moe_layers,
                        num_layers=num_layers,
                        dense_layers=dense_layers,
                        batch_size=int(batch_size) if has_bs_specific else None,
                    )
                    token_stats_stride = moe_layers
                    tune_iters = replay_iters * moe_layers
                    mode = "bs-specific" if has_bs_specific else "legacy"
                    vlog(
                        f"tune bs={batch_size} mode={mode}, files={file_count}, "
                        f"replay_iters={replay_iters}, stride={token_stats_stride}"
                    )
                else:
                    vlog(
                        f"No valid external token_stats for bs={batch_size}; "
                        "fall back to random fused_topk."
                    )

            configs.append(
                worker.tune(
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    search_space,
                    token_stats_list_bs,
                    token_stats_stride,
                    tune_iters,
                    block_quant_shape,
                    args.policy_config,
                    dtype_key,
                    w13_n,
                    w13_k,
                    w2_n,
                    w2_k,
                )
            )
        best_configs = {M: config for M, config in zip(batch_sizes, configs)}
        save_tuned_policy_pairs(
            best_configs,
            args.save_dir,
            model_name_prefix,
            dtype_key,
            args.policy_config,
            w13_n,
            w13_k,
            w2_n,
            w2_k,
        )
        end = time.time()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        outputs = []
        for batch_size in batch_sizes:
            token_stats_list_bs: list[torch.Tensor] | None = None
            token_stats_stride = 1
            bench_iters = 100
            if args.token_stats_dir:
                file_count, has_bs_specific = get_token_stats_file_count(
                    args.token_stats_dir,
                    int(batch_size),
                    num_layers,
                    dense_layers,
                )
                replay_iters = file_count // moe_layers if moe_layers > 0 else 0
                if replay_iters > 0:
                    token_stats_list_bs = load_token_stats_list(
                        token_stats_dir=args.token_stats_dir,
                        count=replay_iters * moe_layers,
                        num_layers=num_layers,
                        dense_layers=dense_layers,
                        batch_size=int(batch_size) if has_bs_specific else None,
                    )
                    token_stats_stride = moe_layers
                    bench_iters = replay_iters * moe_layers
                    mode = "bs-specific" if has_bs_specific else "legacy"
                    vlog(
                        f"benchmark bs={batch_size} mode={mode}, files={file_count}, "
                        f"replay_iters={replay_iters}, stride={token_stats_stride}"
                    )
                else:
                    print(
                        f"No valid external token_stats for bs={batch_size}; "
                        "fall back to random fused_topk."
                    )

            outputs.append(
                worker.benchmark(
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    dtype_key,
                    args.policy_config,
                    token_stats_list_bs,
                    token_stats_stride,
                    bench_iters,
                    block_quant_shape,
                )
            )

        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"Batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507"
    )
    parser.add_argument(
        "--tp-size", "-tp", "--tensor-parallel-size", type=int, default=1
    )
    parser.add_argument(
        "--enable-expert-parallel", "-enable-ep", action="store_true"
    )
    parser.add_argument(
        "--dtype-key",
        type=str,
        choices=["bf16", "fp8_w8a16", "mxfp4_w4a16"],
        default="bf16",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./",
        help="Directory to save tuned results",
    )
    parser.add_argument(
        "--policy-config",
        type=str,
        default="default_moe_policy.json",
        help="Base policy JSON template used to materialize each tuning trial.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, nargs="+", required=False)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--model-prefix", type=str, required=False)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for per-batch token_stats replay details.",
    )
    parser.add_argument(
        "--token-stats-dir",
        type=str,
        required=False,
        help=(
            "Directory of dumped token_stats .pt files. Prefer bs-specific naming "
            "token_stats_bs{B}_layer{L}_idx{I}.pt; fallback to "
            "token_stats_layer{L}_idx{I}.pt if bs-specific files for that batch "
            "are unavailable."
        ),
    )
    args = parser.parse_args()

    main(args)
