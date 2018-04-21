#include "caffe2/operators/reduction_ops.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SumElements, SumElementsOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(SumElementsInt, SumElementsIntOp<int, CPUContext>);
REGISTER_CPU_OPERATOR(SumSqrElements, SumSqrElementsOp<CPUContext>);

REGISTER_CPU_OPERATOR(
    SumElementsGradient,
    SumElementsGradientOp<float, CPUContext>);

REGISTER_CPU_OPERATOR(RowwiseMax, MaxReductionOp<float, CPUContext, true>);
REGISTER_CPU_OPERATOR(
    RowwiseMaxGradient,
    MaxReductionGradientOp<float, CPUContext, true>);
REGISTER_CPU_OPERATOR(
    ColwiseMaxGradient,
    MaxReductionGradientOp<float, CPUContext, false>);
REGISTER_CPU_OPERATOR(ColwiseMax, MaxReductionOp<float, CPUContext, false>);

OPERATOR_SCHEMA(SumElements)
    .NumInputs(1)
    .NumOutputs(1)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc("Sums the elements of the input tensor.")
    .Arg("average", "whether to average or not")
    .Input(0, "X", "Tensor to sum up")
    .Output(0, "sum", "Scalar sum");

OPERATOR_SCHEMA(SumElementsInt)
    .NumInputs(1)
    .NumOutputs(1)
    .ScalarType(TensorProto::INT32)
    .SetDoc("Sums the integer elements of the input tensor.")
    .Input(0, "X", "Tensor to sum up")
    .Output(0, "sum", "Scalar sum");
SHOULD_NOT_DO_GRADIENT(SumElementsInt);

OPERATOR_SCHEMA(SumSqrElements)
    .NumInputs(1)
    .NumOutputs(1)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc("Sums the squares elements of the input tensor.")
    .Arg("average", "whether to average or not")
    .Input(0, "X", "Tensor to sum up")
    .Output(0, "sum", "Scalar sum of squares");

OPERATOR_SCHEMA(SumElementsGradient).NumInputs(2).NumOutputs(1);

class GetSumElementsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SumElementsGradient",
        "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(SumElements, GetSumElementsGradient);

OPERATOR_SCHEMA(RowwiseMax)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc("Compute row-wise max reduction of the input tensor.")
    .Input(
        0,
        "X",
        "A tenosr of dimensions batch_size x M x N to compute rowwise-max.")
    .Output(0, "Y", "batch_size x M rowwise-max results matrix.");

OPERATOR_SCHEMA(RowwiseMaxGradient).NumInputs(3).NumOutputs(1);
class GetRowwiseMaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "RowwiseMaxGradient",
        "",
        vector<string>{I(0), O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(RowwiseMax, GetRowwiseMaxGradient);

OPERATOR_SCHEMA(ColwiseMaxGradient);

OPERATOR_SCHEMA(ColwiseMax)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc("Compute column-wise max reduction of the input tensor.")
    .Input(
        0,
        "X",
        "A tenosr of dimensions batch_size x M x N to compute colwise-max.")
    .Output(0, "Y", "batch_size x N column-max results matrix.");

OPERATOR_SCHEMA(ColumnMaxGradient).NumInputs(3).NumOutputs(1);
class GetColwiseMaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ColwiseMaxGradient",
        "",
        vector<string>{I(0), O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(ColwiseMax, GetColwiseMaxGradient);

template <typename T, class Context>
bool SumElementsGradientOp<T, Context>::RunOnDevice()
// TODO: T21635077 fix float-divide-by-zero undefined behavior
#if defined(__has_feature)
#if __has_feature(__address_sanitizer__)
    __attribute__((__no_sanitize__("float-divide-by-zero")))
#endif
#endif
{
  auto& X = Input(0);
  TensorCPU sum_grad = TensorCPU(Input(1));
  auto* dX = Output(0);
  dX->ResizeLike(X);
  DCHECK_EQ(sum_grad.size(), 1);
  math::Set<T, Context>(
      dX->size(),
      static_cast<T>(sum_grad.data<T>()[0] * (average_ ? 1.0 / X.size() : 1)),
      dX->template mutable_data<T>(),
      &context_);
  return true;
}

template <typename T, class Context, bool ROWWISE>
bool MaxReductionGradientOp<T, Context, ROWWISE>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);

  auto* dX = Output(0);
  dX->ResizeLike(X);

  CAFFE_ENFORCE_EQ(X.ndim(), 3);

  const int batch_size = X.dim32(0);
  const int M = X.dim32(1);
  const int N = X.dim32(2);

  const T* Xdata = X.template data<T>();
  const T* Ydata = Y.template data<T>();
  const T* dYdata = dY.template data<T>();
  T* dXdata = dX->template mutable_data<T>();

  const int input_size = M * N;
  for (int i = 0; i < batch_size; ++i) {
    const T* Xdata_i = Xdata + i * input_size;
    T* dXdata_i = dXdata + i * input_size;
    if (ROWWISE) {
      const T* Ydata_i = Ydata + i * M;
      const T* dYdata_i = dYdata + i * M;
      for (int m = 0; m < M; ++m) {
        const T* Xdata_m = Xdata_i + m * N;
        T* dXdata_m = dXdata_i + m * N;
        for (int n = 0; n < N; ++n) {
          if (Xdata_m[n] == Ydata_i[m]) {
            dXdata_m[n] = dYdata_i[m];
          } else {
            dXdata_m[n] = static_cast<T>(0);
          }
        }
      }
    } else {
      const T* Ydata_i = Ydata + i * N;
      const T* dYdata_i = dYdata + i * N;
      for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
          const T* Xdata_m = Xdata_i + m * N;
          T* dXdata_m = dXdata_i + m * N;
          if (Xdata_m[n] == Ydata_i[n]) {
            dXdata_m[n] = dYdata_i[n];
          } else {
            dXdata_m[n] = static_cast<T>(0);
          }
        }
      }
    }
  }

  return true;
}

} // namespace caffe2
