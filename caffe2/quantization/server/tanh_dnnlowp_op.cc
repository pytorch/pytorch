#include "caffe2/quantization/server/elementwise_dnnlowp_op.h"
#include "caffe2/quantization/server/tanh.h"

namespace caffe2 {

using namespace dnnlowp;

template <typename T>
class TanhFunctor {
 public:
  explicit TanhFunctor() : tanh_() {};

  inline void
  operator()(const int n, const T *x, T *y) {
    for (int i = 0; i < n; ++i) {
      y[i] = tanh_.Compute(x[i]);
    }
  }

  TensorQuantizationParams GetOutputQuantizationParams() const {
    return tanh_.GetOutputQuantizationParams();
  }

 private:
  Tanh<T> tanh_;
};

REGISTER_CPU_OPERATOR_WITH_ENGINE(
  Tanh, DNNLOWP,
  UnaryElementwiseWithArgsDNNLowPOp<std::uint8_t, TanhFunctor<std::uint8_t>>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
  Int8Tanh, DNNLOWP,
  UnaryElementwiseWithArgsDNNLowPOp<std::uint8_t, TanhFunctor<std::uint8_t>>);

} // namespace caffe2
