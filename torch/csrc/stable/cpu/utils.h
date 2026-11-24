#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cstdint>

HIDDEN_NAMESPACE_BEGIN(torch, stable, cpu)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

/**
 * @brief Queries the "l1d_cache_size" entry from
 * at::cpu::get_cpu_capabilities().
 *
 * @return The cache size in bytes, or 0 if unavailable.
 *
 * Minimum compatible version: PyTorch 2.16.
 * Build time minimum version: PyTorch 2.16.
 */
inline int64_t get_l1d_cache_size() {
  int64_t size = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_cpu_get_l1d_cache_size(&size));
  return size;
}

/**
 * @brief Queries the "l2_cache_size" entry from
 * at::cpu::get_cpu_capabilities().
 *
 * @return The cache size in bytes, or 0 if unavailable.
 *
 * Minimum compatible version: PyTorch 2.16.
 * Build time minimum version: PyTorch 2.16.
 */
inline int64_t get_l2_cache_size() {
  int64_t size = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_cpu_get_l2_cache_size(&size));
  return size;
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

HIDDEN_NAMESPACE_END(torch, stable, cpu)
