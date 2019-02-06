#include "caffe2/operators/elementwise_div_op.h"
#include "caffe2/utils/eigen_utils.h"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

namespace caffe2 {

namespace {

template <typename TGrad, typename TIn, typename TOut>
void ComputeDivGradient(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    const int* C_dims,
    const TGrad* dC,
    const TIn* B,
    const TOut* C,
    TGrad* dA,
    TGrad* dB,
    CPUContext* context) {
  const int A_size =
      std::accumulate(A_dims, A_dims + ndim, 1, std::multiplies<int>());
  const int B_size =
      std::accumulate(B_dims, B_dims + ndim, 1, std::multiplies<int>());
  const int C_size =
      std::accumulate(C_dims, C_dims + ndim, 1, std::multiplies<int>());
  if (dA != nullptr) {
    math::Set<TGrad, CPUContext>(A_size, TGrad(0), dA, context);
  }
  math::Set<TGrad, CPUContext>(B_size, TGrad(0), dB, context);
  std::vector<int> index(ndim, 0);
  for (int C_index = 0; C_index < C_size; ++C_index) {
    const int B_index =
        math::utils::GetIndexFromDims(ndim, B_dims, index.data());
    dB[B_index] += -dC[C_index] * C[C_index] / B[B_index];
    if (dA != nullptr) {
      const int A_index =
          math::utils::GetIndexFromDims(ndim, A_dims, index.data());
      dA[A_index] += dC[C_index] / B[B_index];
    }
    math::utils::IncreaseIndexInDims(ndim, C_dims, index.data());
  }
}

} // namespace

template <>
template <typename TGrad, typename TIn, typename TOut>
bool DivFunctor<CPUContext>::Backward(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims,
    const TGrad* dC,
    const TIn* /* A */,
    const TIn* B,
    const TOut* C,
    TGrad* dA,
    TGrad* dB,
    CPUContext* context) const {
  if (A_dims == B_dims) {
    const int size = std::accumulate(
        A_dims.cbegin(), A_dims.cend(), 1, std::multiplies<int>());
    EigenVectorMap<TGrad>(dB, size) =
        -ConstEigenVectorArrayMap<TGrad>(dC, size) *
        ConstEigenVectorArrayMap<TOut>(C, size) /
        ConstEigenVectorArrayMap<TIn>(B, size);
    math::Div(size, dC, B, dA, context);
    return true;
  }
  const int ndim = std::max(A_dims.size(), B_dims.size());
  std::vector<int> A_broadcast_dims(ndim);
  std::vector<int> B_broadcast_dims(ndim);
  std::vector<int> C_broadcast_dims(ndim);
  math::utils::ComputeBroadcastBinaryOpDims(
      A_dims.size(),
      A_dims.data(),
      B_dims.size(),
      B_dims.data(),
      A_broadcast_dims.data(),
      B_broadcast_dims.data(),
      C_broadcast_dims.data());
  if (dA == dC) {
    ComputeDivGradient<TGrad, TIn, TOut>(
        ndim,
        A_broadcast_dims.data(),
        B_broadcast_dims.data(),
        C_broadcast_dims.data(),
        dC,
        B,
        C,
        nullptr,
        dB,
        context);
    math::Div(
        A_dims.size(),
        A_dims.data(),
        B_dims.size(),
        B_dims.data(),
        dC,
        B,
        dA,
        context);
  } else {
    ComputeDivGradient<TGrad, TIn, TOut>(
        ndim,
        A_broadcast_dims.data(),
        B_broadcast_dims.data(),
        C_broadcast_dims.data(),
        dC,
        B,
        C,
        dA,
        dB,
        context);
  }
  return true;
}

template <>
class BinaryElementwiseWithArgsGradientOp<
    NumericTypes,
    CPUContext,
    BinaryFunctorWithDefaultCtor<DivFunctor<CPUContext>>,
    SameTypeAsInput,
    SameTypeAsInput>
    final : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);

  BinaryElementwiseWithArgsGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        OP_SINGLE_ARG(bool, "broadcast", legacy_broadcast_, false),
        OP_SINGLE_ARG(int, "axis", axis_, -1),
        OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
        OP_SINGLE_ARG(string, "order", order_, "NCHW"),
        functor_(*this) {
    if (legacy_broadcast_) {
      if (axis_ != -1) {
        // Get axis from an explicit axis argument.
        CAFFE_ENFORCE_EQ(
            axis_str_.size(),
            0,
            "Args axis and axis_str cannot be used simultaneously.");
      } else if (axis_str_.size()) {
        // Get the axis index semantically.
        CAFFE_ENFORCE_EQ(
            axis_str_.size(), 1, "Unsupported axis string", axis_str_);
        const size_t semantic_axis_ = order_.find(axis_str_);
        CAFFE_ENFORCE_NE(
            semantic_axis_,
            string::npos,
            "Unrecognizable axis string ",
            axis_str_,
            " from order string ",
            order_);
        axis_ = semantic_axis_;
      } else {
        CAFFE_ENFORCE(
            axis_ == -1 && axis_str_.empty(),
            "Do not specify axis or axis_str if broadcast is not enabled.");
      }
    }
  }

  bool RunOnDevice() override {
    return DispatchHelper<NumericTypes>::call(this, Input(1));
  }

  template <typename T>
  bool DoRunWithType() {
    const T* dC_data = nullptr;
    const T* A_data = nullptr;
    const T* B_data = nullptr;
    const T* C_data = nullptr;
    std::vector<int> A_dims;
    std::vector<int> B_dims;
    at::IntArrayRef dA_sizes;
    at::IntArrayRef dB_sizes;
    if (InputSize() == 3) {
      const auto& B = Input(0);
      const auto& C = Input(1);
      const auto& dC = Input(2);
      if (legacy_broadcast_) {
        if (B.numel() == 1) {
          A_dims = {static_cast<int>(C.numel())};
          B_dims = {1};
        } else {
          size_t pre, n, post;
          std::tie(pre, n, post) =
              elementwise_ops_utils::ComputeLegacyBroadcastSizes(C, B, axis_);
          A_dims = {static_cast<int>(pre),
                    static_cast<int>(n),
                    static_cast<int>(post)};
          B_dims = {static_cast<int>(n), 1};
        }
      } else {
        std::copy(
            C.sizes().cbegin(), C.sizes().cend(), std::back_inserter(A_dims));
        std::copy(
            B.sizes().cbegin(), B.sizes().cend(), std::back_inserter(B_dims));
      }
      B_data = B.template data<T>();
      C_data = C.template data<T>();
      dC_data = dC.template data<T>();
      dA_sizes = C.sizes();
      dB_sizes = B.sizes();
    } else {
      const auto& dC = Input(0);
      const auto& A = Input(1);
      const auto& B = Input(2);
      const auto& C = Input(3);
      if (legacy_broadcast_) {
        if (B.numel() == 1) {
          A_dims = {static_cast<int>(A.numel())};
          B_dims = {1};
        } else {
          size_t pre, n, post;
          std::tie(pre, n, post) =
              elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
          A_dims = {static_cast<int>(pre),
                    static_cast<int>(n),
                    static_cast<int>(post)};
          B_dims = {static_cast<int>(n), 1};
        }
      } else {
        std::copy(
            A.sizes().cbegin(), A.sizes().cend(), std::back_inserter(A_dims));
        std::copy(
            B.sizes().cbegin(), B.sizes().cend(), std::back_inserter(B_dims));
      }
      dC_data = dC.template data<T>();
      A_data = A.template data<T>();
      B_data = B.template data<T>();
      C_data = C.template data<T>();
      dA_sizes = A.sizes();
      dB_sizes = B.sizes();
    }
    auto* dA = Output(0, dA_sizes, at::dtype<T>());
    auto* dB = Output(1, dB_sizes, at::dtype<T>());
    auto* dA_data = dA->template mutable_data<T>();
    auto* dB_data = dB->template mutable_data<T>();
    return functor_.Backward(
        A_dims,
        B_dims,
        dC_data,
        A_data,
        B_data,
        C_data,
        dA_data,
        dB_data,
        &context_);
  }

 private:
  const bool legacy_broadcast_;
  int axis_;
  const std::string axis_str_;
  const std::string order_;

  BinaryFunctorWithDefaultCtor<DivFunctor<CPUContext>> functor_;
};

REGISTER_CPU_OPERATOR(
    DivGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        CPUContext,
        DivFunctor<CPUContext>>);

namespace {

class GetDivGradient final : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "DivGradient",
        "",
        std::vector<std::string>{GO(0), I(0), I(1), O(0)},
        std::vector<std::string>{GI(0), GI(1)});
  }
};

} // namespace

REGISTER_GRADIENT(Div, GetDivGradient);

} // namespace caffe2
