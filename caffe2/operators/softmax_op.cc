#include "caffe2/operators/softmax_op.h"
#include "caffe2/operators/softmax_shared.h"

namespace caffe2 {

// Implementation for the CPU context.
template <>
bool SoftmaxOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int N = X.size_to_dim(canonical_axis);
  const int D = X.size_from_dim(canonical_axis);
  Y->ResizeLike(X);
  float* Ydata = Y->mutable_data<float>();
  // First, get scales
  if (scale_.size() != N) {
    scale_.Resize(N);
  }
  if (rowmax_.size() != N) {
    rowmax_.Resize(N);
  }
  if (sum_multiplier_.size() != D) {
    sum_multiplier_.Resize(D);
    math::Set<float, CPUContext>(D, 1.f, sum_multiplier_.mutable_data<float>(),
                                 &context_);
  }

  SoftmaxCPU(
      context_,
      N,
      D,
      X.data<float>(),
      Ydata,
      scale_.mutable_data<float>(),
      sum_multiplier_.data<float>(),
      false,
      rowmax_.mutable_data<float>());
  return true;
}

// Implementation for the CPU context.
template <>
bool SoftmaxGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  const auto canonical_axis = Y.canonical_axis_index(axis_);
  const int N = Y.size_to_dim(canonical_axis);
  const int D = Y.size_from_dim(canonical_axis);
  // First, get scales
  if (scale_.size() != N) {
    scale_.Resize(N);
  }
  if (sum_multiplier_.size() != D) {
    sum_multiplier_.Resize(D);
    math::Set<float, CPUContext>(D, 1.f, sum_multiplier_.mutable_data<float>(),
                                 &context_);
  }
  dX->ResizeLike(Y);
  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  context_.Copy<float, CPUContext, CPUContext>(Y.size(), dYdata, dXdata);
  float* scaledata = scale_.mutable_data<float>();
  for (int i = 0; i < N; ++i) {
    math::Dot<float, CPUContext>(D, Ydata + i * D, dYdata + i * D,
                                 scaledata + i, &context_);
  }
  math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans, N, D, 1, -1,
                                scaledata, sum_multiplier_.data<float>(), 1,
                                dXdata, &context_);
  math::Mul<float, CPUContext>(Y.size(), dXdata, Ydata, dXdata,
                               &context_);
  return true;
}

REGISTER_CPU_OPERATOR(Softmax, SoftmaxOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(SoftmaxGradient, SoftmaxGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(Softmax)
  .NumInputs(1)
  .NumOutputs(1)
  .IdenticalTypeAndShape()
  .SetDoc(R"DOC(
The operator computes the softmax normalized values for each layer in the batch
 of the given input. The input is a 2-D tensor (Tensor<float>) of size
(batch_size x input_feature_dimensions). The output tensor has the same shape
and contains the softmax normalized values of the corresponding input.

X does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
X \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then X will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the X tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors.
)DOC")
  .Arg("axis",
       "(int) default to 1; describes the axis of the inputs when coerced "
       "to 2D; defaults to one because the 0th axis most likely describes "
       "the batch_size")
  .Input(0, "input",
         "The input tensor that's coerced into a 2D matrix of size (NxD) "
         "as described above.")
  .Output(0, "output", "The softmax normalized output values with the same "
          "shape as input tensor.")
  .InheritOnnxSchema("Softmax");

// Input: Y, dY. Output: dX
OPERATOR_SCHEMA(SoftmaxGradient).NumInputs(2).NumOutputs(1);

class GetSoftmaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient", "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Softmax, GetSoftmaxGradient);
REGISTER_GRADIENT(SoftmaxFp16, GetSoftmaxGradient);

}  // namespace caffe2
