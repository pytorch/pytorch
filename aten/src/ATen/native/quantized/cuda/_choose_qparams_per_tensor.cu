#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

namespace at {
namespace native {

std::tuple<double, int64_t> _choose_qparams_per_tensor_cuda(const Tensor& self, bool reduce_range) {
  int32_t qmin = 0;
  int32_t qmax = reduce_range ? 127 : 255;
  float xmin = self.min().item<float>();
  xmin = xmin < 0. ? xmin : 0.;
  float xmax = self.max().item<float>();
  xmax = xmax > 0. ? xmax : 0.;

  double scale = (static_cast<double>(xmax) - xmin) / (qmax - qmin);
  if (float(scale) == 0.0f || std::isinf(1.0f / float(scale))) {
    scale = 0.1;
  }

  double zero_point_from_min = qmin - xmin / static_cast<double>(scale);
  double zero_point_from_max = qmax - xmax / static_cast<double>(scale);
  double zero_point_from_min_error =
      std::abs(qmin) - std::abs(xmin / static_cast<double>(scale));
  double zero_point_from_max_error =
      std::abs(qmax) - std::abs(xmax / static_cast<double>(scale));
  double initial_zero_point =
      zero_point_from_min_error < zero_point_from_max_error
      ? zero_point_from_min
      : zero_point_from_max;

  int32_t nudged_zero_point = 0;
  if (initial_zero_point < qmin) {
    nudged_zero_point = qmin;
  } else if (initial_zero_point > qmax) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = nearbyint(initial_zero_point);
  }

  return {scale, nudged_zero_point};
}

} // namespace native
} // namespace at
