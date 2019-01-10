#include "caffe2/operators/arg_ops.h"

#include <functional>

#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T, class Compare, class Context>
void ComputeArgImpl(
    const TIndex prev_size,
    const TIndex next_size,
    const TIndex n,
    const Compare& comp,
    const T* X,
    TIndex* Y,
    Context* context) {
  math::Set<TIndex, Context>(prev_size * next_size, TIndex(0), Y, context);
  for (TIndex i = 0; i < prev_size; ++i) {
    const T* cur_X = X + i * n * next_size + next_size;
    for (TIndex k = 1; k < n; ++k) {
      for (TIndex j = 0; j < next_size; ++j) {
        TIndex* cur_Y = Y + i * next_size + j;
        if (comp(*cur_X, X[i * n * next_size + *cur_Y * next_size + j])) {
          *cur_Y = k;
        }
        ++cur_X;
      }
    }
  }
}

} // namespace

template <>
template <typename T>
bool ArgMaxReducer<CPUContext>::operator()(
    const TIndex prev_size,
    const TIndex next_size,
    const TIndex n,
    const T* X,
    TIndex* Y,
    CPUContext* context) const {
  ComputeArgImpl(prev_size, next_size, n, std::greater<T>(), X, Y, context);
  return true;
}

template <>
template <typename T>
bool ArgMinReducer<CPUContext>::operator()(
    const TIndex prev_size,
    const TIndex next_size,
    const TIndex n,
    const T* X,
    TIndex* Y,
    CPUContext* context) const {
  ComputeArgImpl(prev_size, next_size, n, std::less<T>(), X, Y, context);
  return true;
}

REGISTER_CPU_OPERATOR(ArgMax, ArgOp<CPUContext, ArgMaxReducer<CPUContext>>);
REGISTER_CPU_OPERATOR(ArgMin, ArgOp<CPUContext, ArgMinReducer<CPUContext>>);

namespace {

std::vector<TensorShape> InferTensor(
    const OperatorDef& def,
    const std::vector<TensorShape>& in) {
  std::vector<TensorShape> out(1);
  ArgumentHelper helper(def);
  int axis = helper.GetSingleArgument("axis", -1);
  const bool keep_dims = helper.GetSingleArgument("keepdims", true);
  const auto& in_dims = in[0].dims();
  auto* out_dims = out[0].mutable_dims();
  if (axis == -1) {
    axis = in_dims.size() - 1;
  }
  for (int i = 0; i < axis; ++i) {
    out_dims->Add(in_dims.Get(i));
  }
  if (keep_dims) {
    out_dims->Add(1);
  }
  for (int i = axis + 1; i < in_dims.size(); ++i) {
    out_dims->Add(in_dims.Get(i));
  }
  out[0].set_data_type(TensorProto::INT64);
  return out;
}

} // namespace

OPERATOR_SCHEMA(ArgMax)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(InferTensor)
    .SetDoc(R"DOC(
Retrive the argmax of the axis dimension. Given an input tensor of shape
[a_0, a_1, ..., a_{n-1}] and two arguments axis as int and keepdims as bool,
returns one output:
- Index tensor which contains the indices of the largest element. It has the
  same dims as X.dims() with the dimension along axis equals 1 when
  keepdims == true otherwise removed.
    )DOC")
    .Input(0, "X", "Tenor of shape [a_0, a_1, ..., a_{n-1}].")
    .Output(0, "Indices", "Tensor of indices for the largest values.")
    .Arg("axis", "The axis to get argmax.")
    .Arg("keepdims", "Whether to keep the axis dim in the output.");

OPERATOR_SCHEMA(ArgMin)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(InferTensor)
    .SetDoc(R"DOC(
Retrive the argmin of the axis dimension. Given an input tensor of shape
[a_0, a_1, ..., a_{n-1}] and two arguments axis as int and keepdims as bool,
returns one output:
- Index tensor which contains the indices of the largest element. It has the
  same dims as X.dims() with the dimension along axis equals 1 when
  keepdims == true otherwise removed.
    )DOC")
    .Input(0, "X", "Tenor of shape [a_0, a_1, ..., a_{n-1}].")
    .Output(0, "Indices", "Tensor of indices for the largest values.")
    .Arg("axis", "The axis to get argmin.")
    .Arg("keepdims", "Whether to keep the axis dim in the output.");

NO_GRADIENT(ArgMax);
NO_GRADIENT(ArgMin);

} // namespace caffe2
