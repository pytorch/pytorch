#pragma once

#include <c10/macros/Export.h>

#include <optional>

namespace at {

// DEFAULT is an internal-only sentinel meaning "use legacy backend default
// unless a parent setting overrides it". NONE means "explicitly set to
// inherit/no-op". Append new values because Dynamo guards serialize this enum.
enum class TORCH_API Float32Precision {
  NONE,
  IEEE,
  TF32,
  BF16,
  DEFAULT,
  BF16X9
};

enum class TORCH_API CuDNNDepthwiseKernel { AUTO, CUDNN, NATIVE };

namespace impl {

// Unset fields use Context's process-wide defaults. Context setters continue to
// update those defaults, independently of these scoped overrides.
struct TORCH_API CuDNNFlags {
  std::optional<bool> enabled;
  std::optional<bool> benchmark;
  std::optional<int> benchmark_limit;
  std::optional<bool> deterministic;
  std::optional<bool> allow_tf32;
  std::optional<Float32Precision> fp32_precision;
  std::optional<Float32Precision> conv_fp32_precision;
  std::optional<Float32Precision> rnn_fp32_precision;
  std::optional<CuDNNDepthwiseKernel> depthwise_kernel;

  static const CuDNNFlags& get_tls_state();
  static void set_tls_state(const CuDNNFlags& state);
};

} // namespace impl
} // namespace at
