#pragma once

#include <nlohmann/json.hpp>
#include <torch/all.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace MoE {

enum class MoePolicyId {
  kBASE = 0,
  kWG_256_128_32_SG_8_2_1,
  kWG_8_64_32_SG_1_4_1,
  kWG_16_64_32_SG_1_4_1,
  kWG_32_64_32_SG_1_4_1,
  kWG_128_256_32_SG_4_8_1,
  kWG_128_64_32_SG_4_2_1,
  kWG_128_128_32_SG_4_2_1,
  kWG_256_64_32_SG_8_2_1,
  kWG_256_256_32_SG_8_4_1
};

struct MoePolicyRule {
  int max_avg_m;
  MoePolicyId policy;
};

struct MoePolicyConfig {
  MoePolicyId default_policy = MoePolicyId::kWG_256_128_32_SG_8_2_1;
  std::vector<MoePolicyRule> rules;
};

struct MoePolicyOverride {
  std::string dtype;
  int64_t n;
  int64_t k;
  MoePolicyConfig policy;
};

inline MoePolicyId parse_policy_name(const std::string& name) {
  if (name == "wg_256_128_32_sg_8_2_1") {
    return MoePolicyId::kWG_256_128_32_SG_8_2_1;
  }
  if (name == "wg_8_64_32_sg_1_4_1") {
    return MoePolicyId::kWG_8_64_32_SG_1_4_1;
  }
  if (name == "wg_16_64_32_sg_1_4_1") {
    return MoePolicyId::kWG_16_64_32_SG_1_4_1;
  }
  if (name == "wg_32_64_32_sg_1_4_1") {
    return MoePolicyId::kWG_32_64_32_SG_1_4_1;
  }
  if (name == "wg_128_256_32_sg_4_8_1") {
    return MoePolicyId::kWG_128_256_32_SG_4_8_1;
  }
  if (name == "wg_128_64_32_sg_4_2_1") {
    return MoePolicyId::kWG_128_64_32_SG_4_2_1;
  }
  if (name == "wg_128_128_32_sg_4_2_1") {
    return MoePolicyId::kWG_128_128_32_SG_4_2_1;
  }
  if (name == "wg_256_64_32_sg_8_2_1") {
    return MoePolicyId::kWG_256_64_32_SG_8_2_1;
  }
  if (name == "wg_256_256_32_sg_8_4_1") {
    return MoePolicyId::kWG_256_256_32_SG_8_4_1;
  }
  throw std::runtime_error("unsupported policy: " + name);
}

inline MoePolicyId
default_policy_for_avg_m(const std::string& dtype_key, int avg_m) {
  const bool is_w8a16 = (dtype_key == "fp8_w8a16");
  const bool is_w4a16 = (dtype_key == "mxfp4_w4a16");

  if (is_w8a16) {
    if (avg_m <= 8) {
      return MoePolicyId::kWG_8_64_32_SG_1_4_1;
    }
    if (avg_m <= 16) {
      return MoePolicyId::kWG_16_64_32_SG_1_4_1;
    }
    if (avg_m <= 32) {
      return MoePolicyId::kWG_32_64_32_SG_1_4_1;
    }
    return MoePolicyId::kWG_128_256_32_SG_4_8_1;
  }

  if (is_w4a16) {
    if (avg_m <= 8) {
      return MoePolicyId::kWG_8_64_32_SG_1_4_1;
    }
    if (avg_m <= 16) {
      return MoePolicyId::kWG_16_64_32_SG_1_4_1;
    }
    if (avg_m <= 32) {
      return MoePolicyId::kWG_32_64_32_SG_1_4_1;
    }
    return MoePolicyId::kWG_128_256_32_SG_4_8_1;
  }

  if (avg_m <= 8) {
    return MoePolicyId::kWG_8_64_32_SG_1_4_1;
  }
  if (avg_m <= 16) {
    return MoePolicyId::kWG_16_64_32_SG_1_4_1;
  }
  if (avg_m <= 32) {
    return MoePolicyId::kWG_32_64_32_SG_1_4_1;
  }
  return MoePolicyId::kWG_256_128_32_SG_8_2_1;
}

inline std::string normalize_dtype_key(std::string key) {
  std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return key;
}

class MoePolicyManager {
 public:
  static MoePolicyManager& get() {
    static MoePolicyManager instance;
    return instance;
  }

  MoePolicyId
  select_policy(const std::string& dtype, int64_t n, int64_t k, int avg_m) {
    std::lock_guard<std::mutex> lock(mutex_);
    maybe_reload_if_needed_locked();

    const std::string dtype_key = normalize_dtype_key(dtype);

    for (const auto& ov : policy_overrides_) {
      if (ov.dtype == dtype_key && ov.n == n && ov.k == k) {
        return select_from_config(ov.policy, avg_m);
      }
    }

    auto it = global_configs_.find(dtype_key);
    if (it == global_configs_.end()) {
      return default_policy_for_avg_m(dtype_key, avg_m);
    }
    return select_from_config(it->second, avg_m);
  }

 private:
  MoePolicyManager() { maybe_reload_if_needed_locked(); }

  static MoePolicyId select_from_config(const MoePolicyConfig& cfg, int avg_m) {
    for (const auto& rule : cfg.rules) {
      if (avg_m <= rule.max_avg_m) {
        return rule.policy;
      }
    }
    return cfg.default_policy;
  }

  static bool parse_policy_config_object(
      const nlohmann::json& block, MoePolicyConfig* out_config) {
    if (!block.is_object() || !block.contains("default_policy")) {
      return false;
    }

    out_config->default_policy =
        parse_policy_name(block.at("default_policy").get<std::string>());
    out_config->rules.clear();

    if (!block.contains("rules") || !block.at("rules").is_array()) {
      return false;
    }
    for (const auto& rule : block.at("rules")) {
      if (!rule.is_object() || !rule.contains("avg_m_max") ||
          !rule.contains("policy")) {
        continue;
      }
      out_config->rules.push_back(
          MoePolicyRule{
              rule.at("avg_m_max").get<int>(),
              parse_policy_name(rule.at("policy").get<std::string>()),
          });
    }

    std::sort(
        out_config->rules.begin(),
        out_config->rules.end(),
        [](const MoePolicyRule& lhs, const MoePolicyRule& rhs) {
          return lhs.max_avg_m < rhs.max_avg_m;
        });
    return true;
  }

  bool parse_v3_global_and_overrides(const nlohmann::json& root) {
    if (!root.is_object() || !root.contains("global")) {
      return false;
    }

    const auto& global = root.at("global");
    if (!global.is_object()) {
      return false;
    }

    bool has_any = false;
    for (const auto& global_item : global.items()) {
      MoePolicyConfig cfg;
      if (!parse_policy_config_object(global_item.value(), &cfg)) {
        continue;
      }
      global_configs_.emplace(
          normalize_dtype_key(global_item.key()), std::move(cfg));
      has_any = true;
    }

    if (root.contains("overrides") && root.at("overrides").is_array()) {
      for (const auto& ov : root.at("overrides")) {
        if (!ov.is_object() || !ov.contains("dtype") || !ov.contains("n") ||
            !ov.contains("k") || !ov.contains("policies")) {
          continue;
        }

        MoePolicyConfig cfg;
        if (!parse_policy_config_object(ov.at("policies"), &cfg)) {
          continue;
        }

        policy_overrides_.push_back(
            MoePolicyOverride{
                normalize_dtype_key(ov.at("dtype").get<std::string>()),
                ov.at("n").get<int64_t>(),
                ov.at("k").get<int64_t>(),
                std::move(cfg),
            });
      }
    }

    return has_any;
  }

  static std::string resolve_policy_path() {
    if (const char* env_path = std::getenv("VLLM_XPU_MOE_POLICY_PATH")) {
      if (env_path[0] != '\0') {
        return std::string(env_path);
      }
    }
    // Default path for in-repo development. Production environments can
    // override this with VLLM_XPU_MOE_POLICY_PATH.
    return "tools/kernel_tunner/default_moe_policy.json";
  }

  void maybe_reload_if_needed_locked() {
    const std::string path = resolve_policy_path();
    if (initialized_ && path == loaded_policy_path_) {
      return;
    }
    load_from_json_locked(path);
  }

  void load_from_json_locked(const std::string& path) {
    global_configs_.clear();
    policy_overrides_.clear();
    loaded_policy_path_ = path;
    initialized_ = true;

    std::ifstream f(path);
    if (!f.is_open()) {
      return;
    }

    nlohmann::json root;
    try {
      f >> root;
    } catch (...) {
      return;
    }

    try {
      bool parsed = parse_v3_global_and_overrides(root);
      if (!parsed) {
        global_configs_.clear();
        policy_overrides_.clear();
      }
    } catch (...) {
      // If parsing/mapping fails, keep empty config and use legacy fallback.
      global_configs_.clear();
      policy_overrides_.clear();
    }
  }

  std::mutex mutex_;
  std::string loaded_policy_path_;
  bool initialized_ = false;
  std::unordered_map<std::string, MoePolicyConfig> global_configs_;
  std::vector<MoePolicyOverride> policy_overrides_;
};

inline std::string
get_dtype_policy_key(at::ScalarType a_dtype, at::ScalarType b_dtype) {
  if (b_dtype == at::kFloat8_e4m3fn || b_dtype == at::kFloat8_e5m2) {
    return "fp8_w8a16";
  }

  // Quantized W4 paths store packed values in uint8.
  if (b_dtype == at::kByte) {
    return "mxfp4_w4a16";
  }

  return a_dtype == at::kBFloat16 ? "bf16" : "fp16";
}

}  // namespace MoE
