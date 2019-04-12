#include <ATen/core/op_registration/op_registration.h>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

using caffe2::BaseContext;
using caffe2::Tensor;

namespace caffe2 {
namespace {

template <class DataType, class Context>
class fc_op_cpu final : public c10::OperatorKernel {
 public:
  void operator()(
      const at::Tensor& X_,
      const at::Tensor& W_,
      const at::Tensor& b_,
      const at::Tensor& Y_,
      int64_t axis,
      int64_t axis_w) {
    Tensor X{C10Tensor(X_)};
    Tensor W{C10Tensor(W_)};
    Tensor b{C10Tensor(b_)};
    Tensor Y{C10Tensor(Y_)};
    CPUContext context;

    constexpr bool TransposeWeight = true;

    CAFFE_ENFORCE(b.dim() == 1, b.dim());
    // batch size
    const auto canonical_axis = X.canonical_axis_index(axis);
    const auto M = X.size_to_dim(canonical_axis);
    const auto K = X.size_from_dim(canonical_axis);
    const auto canonical_axis_w = W.canonical_axis_index(axis_w);
    const int N = TransposeWeight ? W.size_to_dim(canonical_axis_w)
                                  : W.size_from_dim(canonical_axis_w);

    auto dimErrorString = [&]() {
      return c10::str(
          "Dimension mismatch: ",
          "X: ",
          X.sizes(),
          ", W: ",
          W.sizes(),
          ", b: ",
          b.sizes(),
          ", axis: ",
          axis,
          ", M: ",
          M,
          ", N: ",
          N,
          ", K: ",
          K);
    };

    // Error checking
    CAFFE_ENFORCE(M == X.numel() / K, dimErrorString());
    CAFFE_ENFORCE(K == W.numel() / N, dimErrorString());
    CAFFE_ENFORCE(N == b.dim32(0), dimErrorString());
    CAFFE_ENFORCE(N == b.numel(), dimErrorString());

    Y_shape_cache_ = X.sizes().vec();
    // This is an invariant of canonical_axis, so we can DCHECK.
    DCHECK_LE(canonical_axis + 1, Y_shape_cache_.size());
    Y_shape_cache_.resize(canonical_axis + 1);
    Y_shape_cache_[canonical_axis] = N;
    Y.Resize(Y_shape_cache_);
    CAFFE_ENFORCE(M * N == Y.numel(), dimErrorString());

    if (X.numel() == 0) {
      // skip the rest of the computation if X is empty
      Y.template mutable_data<DataType>();
      return;
    }

    // default to FLOAT as math.h does.
    caffe2::TensorProto::DataType math_type =
        caffe2::TensorProto_DataType_FLOAT;
    if (caffe2::fp16_type<DataType>()) {
      math_type = caffe2::TensorProto_DataType_FLOAT16;
    }

    // W * x
    caffe2::math::Gemm<DataType, Context, caffe2::DefaultEngine>(
        CblasNoTrans,
        TransposeWeight ? CblasTrans : CblasNoTrans,
        M,
        N,
        K,
        1,
        X.template data<DataType>(),
        W.template data<DataType>(),
        0,
        Y.template mutable_data<DataType>(),
        static_cast<Context*>(&context),
        math_type);
    // Add bias term
    Tensor bias_multiplier(bias_multiplier_);
    ReinitializeTensor(
        &bias_multiplier, {M}, at::dtype<DataType>().device(CPU));
    caffe2::math::Set<DataType, Context>(
        M,
        caffe2::convert::To<float, DataType>(1),
        bias_multiplier.template mutable_data<DataType>(),
        static_cast<Context*>(&context));
    caffe2::math::Gemm<DataType, Context, caffe2::DefaultEngine>(
        CblasNoTrans,
        CblasNoTrans,
        M,
        N,
        1,
        1,
        bias_multiplier.template data<DataType>(),
        b.template data<DataType>(),
        1,
        Y.template mutable_data<DataType>(),
        static_cast<Context*>(&context),
        math_type);
  }

 private:
  vector<int64_t> Y_shape_cache_;
  at::Tensor bias_multiplier_ = at::Tensor(C10Tensor(Tensor()));
};

static auto registry = c10::RegisterOperators().op(
    FunctionSchema(
        "_c10_experimental::FullyConnected",
        "",
        (std::vector<c10::Argument>{c10::Argument("X"),
                                    c10::Argument("W"),
                                    c10::Argument("b"),
                                    c10::Argument("output"),
                                    c10::Argument("axis", IntType::get()),
                                    c10::Argument("axis_w", IntType::get())}),
        (std::vector<c10::Argument>{})),
    c10::kernel<fc_op_cpu<float, CPUContext>>(),
    c10::dispatchKey(CPUTensorId()));

} // namespace

REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(
    "_c10_experimental::FullyConnected",
    C10FC_DontUseThisOpYet)

} // namespace caffe2
