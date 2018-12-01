#include "caffe2/quantization/server/channel_shuffle_dnnlowp_op.h"

#include "caffe2/quantization/server/caffe2_dnnlowp_utils.h"
#include "caffe2/quantization/server/transpose.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <typename T>
ChannelShuffleDNNLowPOp<T>::ChannelShuffleDNNLowPOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : BaseType(operator_def, ws),
      order_(StringToStorageOrder(
          OperatorBase::GetSingleArgument<std::string>("order", "NCHW"))),
      OP_SINGLE_ARG(int, "group", group_, 1) {
  CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
}

template <typename T>
bool ChannelShuffleDNNLowPOp<T>::RunOnDevice() {
  return order_ == StorageOrder::NCHW ? RunOnDeviceWithOrderNCHW()
                                      : RunOnDeviceWithOrderNHWC();
}

template <typename T>
bool ChannelShuffleDNNLowPOp<T>::RunOnDeviceWithOrderNCHW() {
  using namespace dnnlowp;

  this->ParseDNNLowPOperatorArguments_();

  // Choose quantization params
  TensorQuantizationParams in_qparams =
      GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

  const auto& X = InputTensorCPU_(0);
  auto* Y = OutputTensorCPU_(0);
  Y->ResizeLike(X);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int G = group_;
  CAFFE_ENFORCE_EQ(C % G, 0);
  const int K = C / G;
  const int HxW = X.numel() / (N * C);
  const int stride = C * HxW;
  const T* X_data = X.template data<T>();
  T* Y_data = Y->template mutable_data<T>();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < N; ++i) {
    ConstEigenMatrixMap<T> X_mat(X_data, K * HxW, G);
    for (int j = 0; j < K; ++j) {
      EigenMatrixMap<T>(Y_data + j * G * HxW, HxW, G) =
          X_mat.block(j * HxW, 0, HxW, G);
    }
    X_data += stride;
    Y_data += stride;
  }

  // Even if there is a pre-chosen quantization parameters for the output,
  // it is ignored because channel shuffle output quantization should be same
  // as the input.
  PropagateOutputTensorQuantizationParams(this, 0, in_qparams);

  return true;
}

template <typename T>
bool ChannelShuffleDNNLowPOp<T>::RunOnDeviceWithOrderNHWC() {
  using namespace dnnlowp;

  this->ParseDNNLowPOperatorArguments_();

  // Choose quantization params
  TensorQuantizationParams in_qparams =
      GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

  const auto& X = InputTensorCPU_(0);
  auto* Y = OutputTensorCPU_(0);
  Y->ResizeLike(X);
  const auto C = X.dim32(X.ndim() - 1);
  const auto G = this->group_;
  CAFFE_ENFORCE(C % G == 0, "");
  const auto K = C / G;
  std::array<int, 2> dims = {G, K};
  std::array<int, 2> axes = {1, 0};
  const T* X_data = X.template data<T>();
  T* Y_data = Y->template mutable_data<T>();

  if (G == 4 && std::is_same<T, std::uint8_t>::value && GetCpuId().avx2()) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (auto i = 0; i < X.numel(); i += C) {
      // Transpose each C = GxK matrix
      fbgemm::transpose_4rows(
          K, (const std::uint8_t*)(X_data + i), (std::uint8_t*)(Y_data + i));
    }
  } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (auto i = 0; i < X.numel(); i += C) {
      // Transpose each C = GxK matrix
      math::Transpose(
          2, dims.data(), axes.data(), X_data + i, Y_data + i, &context_);
    }
  }

  // Even if there is a pre-chosen quantization parameters for the output,
  // it is ignored because channel shuffle output quantization should be same
  // as the input.
  PropagateOutputTensorQuantizationParams(this, 0, in_qparams);

  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    ChannelShuffle,
    DNNLOWP,
    ChannelShuffleDNNLowPOp<uint8_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8ChannelShuffle,
    DNNLOWP,
    ChannelShuffleDNNLowPOp<uint8_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    ChannelShuffle,
    DNNLOWP_16,
    ChannelShuffleDNNLowPOp<uint16_t>);

} // namespace caffe2
