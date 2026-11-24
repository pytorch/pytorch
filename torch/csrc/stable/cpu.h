#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

HIDDEN_NAMESPACE_BEGIN(torch, stable, cpu)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

/**
 * @brief Looks up a key in the CPU capabilities reported by
 * at::cpu::get_cpu_capabilities().
 *
 * The runtime snapshots the ATen map on first lookup and exposes individual
 * values by NUL-terminated key, keeping the map and IValue types behind the
 * ABI.
 *
 * T must be int64_t, bool, or std::string. A missing key returns std::nullopt.
 * A key with a different value type raises an error.
 *
 * Minimum compatible version: PyTorch 2.16.
 * Build time minimum version: PyTorch 2.16.
 */
template <typename T>
inline std::optional<T> get_cpu_capability(const char* key) {
  bool ret_key_exists = false;
  if constexpr (std::is_same_v<T, int64_t>) {
    int64_t value = 0;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_get_cpu_capability_int(key, &value, &ret_key_exists));
    if (ret_key_exists) {
      return value;
    }
  } else if constexpr (std::is_same_v<T, bool>) {
    bool value = false;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_get_cpu_capability_bool(key, &value, &ret_key_exists));
    if (ret_key_exists) {
      return value;
    }
  } else {
    static_assert(
        std::is_same_v<T, std::string>,
        "CPU capability type must be int64_t, bool, or std::string");
    const char* value = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_get_cpu_capability_string(key, &value, &ret_key_exists));
    if (ret_key_exists) {
      return std::string(value);
    }
  }
  return std::nullopt;
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

HIDDEN_NAMESPACE_END(torch, stable, cpu)
