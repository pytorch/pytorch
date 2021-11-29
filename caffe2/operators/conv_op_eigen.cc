#include "Eigen/Core"
#include "caffe2/utils/eigen_utils.h"

#if EIGEN_VERSION_AT_LEAST(3, 3, 0)

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"

#include "unsupported/Eigen/CXX11/Tensor"

namespace caffe2 {

template <typename T>
class EigenConvOp final : public ConvPoolOpBase<CPUContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CPUContext);
  explicit EigenConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(group_ == 1, "Group convolution not supported yet.");
  }
  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~EigenConvOp() override {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

// The NCHW implementation: we do explicit transposes before and after, which
// are not ideal but provides a compatible path instead of throwing the error.
template <typename T>
bool EigenConvOp<T>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto* Y = Output(0);
  const int N = X.dim32(0), C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
  CAFFE_ENFORCE(4 == filter.dim());
  const int M = filter.dim32(0);
  CAFFE_ENFORCE(filter.dim32(1) == C);
  CAFFE_ENFORCE(filter.dim32(2) == kernel_h());
  CAFFE_ENFORCE(filter.dim32(3) == kernel_w());
  ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
  Eigen::array<int64_t, 4> kernel_shuffles
      { {int64_t(2), int64_t(3), int64_t(1), int64_t(0)} };
  Eigen::array<int64_t, 4> input_shuffles
      { {int64_t(0), int64_t(2), int64_t(3), int64_t(1)} };

  Eigen::Tensor<T, 4, Eigen::RowMajor> filter_tensor =
      Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<T*>(filter.template data<T>()),
          M,
          C,
          kernel_h(),
          kernel_w())
          .shuffle(kernel_shuffles);
  Eigen::Tensor<T, 4, Eigen::RowMajor> X_tensor =
      Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<T*>(X.template data<T>()), N, C, H, W)
          .shuffle(input_shuffles);

  // For Eigen, the definition of row and col actually correspond to width
  // and height instead of the other way round, so notice how we pass the
  // stride, pad and dilation values.
  typedef typename Eigen::internal::traits<
      Eigen::Tensor<T, 4, Eigen::RowMajor>>::Index TensorIndex;
  Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

  Eigen::DSizes<TensorIndex, 2> pre_contract_dims;
  pre_contract_dims[1] = kernel_h() * kernel_w() * C;
  pre_contract_dims[0] = Y->numel() / M;

  Eigen::DSizes<TensorIndex, 2> kernel_dims;
  kernel_dims[0] = kernel_h() * kernel_w() * C;
  kernel_dims[1] = M;

  Eigen::array<TensorIndex, 4> bcast_dims;
  bcast_dims[0] = N;
  bcast_dims[1] = Y->dim32(1);
  bcast_dims[2] = Y->dim32(2);
  bcast_dims[3] = 1;

  Eigen::Tensor<T, 4, Eigen::RowMajor> Y_tensor(
      Y->dim32(0), Y->dim32(2), Y->dim32(3), Y->dim32(1));
  Y_tensor = X_tensor
                 .extract_image_patches(
                     kernel_w(),
                     kernel_h(),
                     stride_w(),
                     stride_h(),
                     dilation_w(),
                     dilation_h(),
                     1,
                     1,
                     pad_l(),
                     pad_r(),
                     pad_t(),
                     pad_b(),
                     0)
                 .reshape(pre_contract_dims)
                 .contract(filter_tensor.reshape(kernel_dims), contract_dims)
                 .reshape(Y_tensor.dimensions());
  if (InputSize() == 3) {
    auto& bias = Input(BIAS);
    CAFFE_ENFORCE(1 == bias.dim());
    CAFFE_ENFORCE(bias.dim32(0) == M);
    // It seems that the bias broadcast is still slower so let's do the
    // following for now.
    EigenArrayMap<T> Y_arr(
        Y_tensor.data(), static_cast<int64_t>(M), Y->numel() / M);
    ConstEigenVectorArrayMap<T> bias_arr(bias.template data<T>(), M);
    Y_arr = Y_arr.colwise() + bias_arr;
  }

  // Do a last transpose.
  Eigen::array<int64_t, 4> output_shuffles
      { {int64_t(0), int64_t(3), int64_t(1), int64_t(2) } };

  Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>(
      Y->template mutable_data<T>(), N, M, Y->dim32(2), Y->dim32(3)) =
      Y_tensor.shuffle(output_shuffles);
  return true;
}

template <typename T>
bool EigenConvOp<T>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto* Y = Output(0);
  const int N = X.dim32(0), H = X.dim32(1), W = X.dim32(2), C = X.dim32(3);
  CAFFE_ENFORCE(4 == filter.dim());
  const int M = filter.dim32(0);
  CAFFE_ENFORCE(filter.dim32(1) == kernel_h());
  CAFFE_ENFORCE(filter.dim32(2) == kernel_w());
  CAFFE_ENFORCE(filter.dim32(3) == C);
  ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
  // Eigen expects filter to be of shape (kernel_h, kernel_w, C, M) for
  // optimization purposes, so we will create a temp one.
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> temp_filter(
      M, kernel_h() * kernel_w() * C);
  temp_filter = ConstEigenArrayMap<T>(
                    filter.template data<T>(), kernel_h() * kernel_w() * C, M)
                    .transpose();

  // Create tensor maps, and call spatial convolution.
  // TODO(jiayq): right now we const cast away the const pointer, but we will
  // need to figure out how to properly do a const tensormap.
  Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> X_tensor(
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<T*>(X.template data<T>()), N, H, W, C);
  Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> Y_tensor(
      Y->template mutable_data<T>(), N, Y->dim32(1), Y->dim32(2), M);
  Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> filter_tensor(
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<T*>(temp_filter.data()), kernel_h(), kernel_w(), C, M);

  // For Eigen, the definition of row and col actually correspond to width
  // and height instead of the other way round, so notice how we pass the
  // stride, pad and dilation values.
  typedef typename Eigen::internal::traits<
      Eigen::Tensor<T, 4, Eigen::RowMajor>>::Index TensorIndex;
  Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

  Eigen::DSizes<TensorIndex, 2> pre_contract_dims;
  pre_contract_dims[1] = kernel_h() * kernel_w() * C;
  pre_contract_dims[0] = Y->numel() / M;

  Eigen::DSizes<TensorIndex, 2> kernel_dims;
  kernel_dims[0] = kernel_h() * kernel_w() * C;
  kernel_dims[1] = M;

  Eigen::array<TensorIndex, 4> bcast_dims;
  bcast_dims[0] = N;
  bcast_dims[1] = Y->dim32(1);
  bcast_dims[2] = Y->dim32(2);
  bcast_dims[3] = 1;

  Y_tensor = X_tensor
                 .extract_image_patches(
                     kernel_w(),
                     kernel_h(),
                     stride_w(),
                     stride_h(),
                     dilation_w(),
                     dilation_h(),
                     1,
                     1,
                     pad_l(),
                     pad_r(),
                     pad_t(),
                     pad_b(),
                     0)
                 .reshape(pre_contract_dims)
                 .contract(filter_tensor.reshape(kernel_dims), contract_dims)
                 .reshape(Y_tensor.dimensions());

  if (InputSize() == 3) {
    auto& bias = Input(BIAS);
    CAFFE_ENFORCE(1 == bias.dim());
    CAFFE_ENFORCE(bias.dim32(0) == M);
    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> bias_tensor(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<T*>(bias.template data<T>()), 1, 1, 1, M);
    // It seems that the bias broadcast is still slower so let's do the
    // following for now.
    EigenArrayMap<T> Y_arr(
        Y->template mutable_data<T>(), static_cast<int64_t>(M), Y->numel() / M);
    ConstEigenVectorArrayMap<T> bias_arr(bias.template data<T>(), M);
    Y_arr = Y_arr.colwise() + bias_arr;
  }
  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv, EIGEN, EigenConvOp<float>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv1D, EIGEN, EigenConvOp<float>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv2D, EIGEN, EigenConvOp<float>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv3D, EIGEN, EigenConvOp<float>);

} // namespace caffe2

#endif
