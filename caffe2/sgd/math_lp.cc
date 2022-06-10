#include "caffe2/sgd/math_lp.h"

namespace caffe2 {

namespace internal {

template <>
void dot<float, float, float>(
    const int N,
    const float* x,
    const float* y,
    float* z,
    CPUContext* ctx) {
  math::Dot<float, CPUContext>(N, x, y, z, ctx);
}

template <>
void dot<float, at::Half, float>(
    const int N,
    const float* x,
    const at::Half* y,
    float* z,
    CPUContext* ctx) {
#ifdef _MSC_VER
  std::vector<float> tmp_y_vec(N);
  float* tmp_y = tmp_y_vec.data();
#else
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  float tmp_y[N];
#endif
  for (int i = 0; i < N; i++) {
#ifdef __F16C__
    tmp_y[i] = _cvtss_sh(y[i], 0); // TODO: vectorize
#else
    tmp_y[i] = y[i];
#endif
  }
  math::Dot<float, CPUContext>(N, x, tmp_y, z, ctx);
}

} // namespace internal
} // namespace caffe2
