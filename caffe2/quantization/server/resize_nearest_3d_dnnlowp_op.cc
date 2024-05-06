#include "caffe2/quantization/server/resize_nearest_3d_dnnlowp_op.h"

namespace caffe2 {

template <typename T>
bool ResizeNearest3DDNNLowPOp<T>::RunOnDevice() {
  using namespace dnnlowp;

  this->ParseDNNLowPOperatorArguments_();

  // Choose quantization params
  in_qparams_[0] = GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

  const auto& X = InputTensorCPU_(0);
  auto* Y = OutputTensorCPU_(0);

  CAFFE_ENFORCE_EQ(X.ndim(), 5);
  const int N = X.dim32(0);
  // input frames
  const int IF = X.dim32(1);
  const int IH = X.dim32(2);
  const int IW = X.dim32(3);
  const int C = X.dim32(4);
  const int OF = IF * temporal_scale_;
  const int OH = IH * height_scale_;
  const int OW = IW * width_scale_;

  vector<int> buffer_shape{N, OF, OH, OW, C};
  Y->Resize(buffer_shape);
  const T* X_data = X.template data<T>();
  T* Y_data = Y->template mutable_data<T>();

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int n = 0; n < N; ++n) {
    for (int t = 0; t < OF; ++t) {
      const int in_f = std::min((int)(t / temporal_scale_), (IF - 1));
      for (int y = 0; y < OH; ++y) {
        const int in_y = std::min((int)(y / height_scale_), (IH - 1));
        for (int x = 0; x < OW; ++x) {
          const int in_x = std::min((int)(x / width_scale_), (IW - 1));
          std::memcpy(
              &Y_data[((((n * OF) + t) * OH + y) * OW + x) * C],
              &X_data[((((n * IF) + in_f) * IH + in_y) * IW + in_x) * C],
              C * sizeof(T));
        }
      }
    }
  }
  // Even if there is a pre-chosen quantization parameters for the output,
  // it is ignored because resize nearest output quantization should be same
  // as the input.
  PropagateOutputTensorQuantizationParams(this, 0, in_qparams_[0]);

  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8ResizeNearest3D,
    DNNLOWP,
    ResizeNearest3DDNNLowPOp<uint8_t>);

} // namespace caffe2
