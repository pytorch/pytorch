#pragma once
#include <cstdint>
#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <cmath>
namespace sdp {

constexpr int32_t num_backends = 3;
enum class SDPBackend {
  error = -1,
  math = 0,
  flash_attention = 1,
  efficient_attention = 2
};

inline double calculate_scale(const at::Tensor& query, c10::optional<double> scale) {
    double softmax_scale =
      scale.has_value() ? scale.value() : std::pow(query.size(-1), -0.5);
    return softmax_scale;
}

} // namespace sdp