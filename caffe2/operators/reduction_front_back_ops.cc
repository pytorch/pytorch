#include "caffe2/operators/reduction_front_back_ops.h"
#include "caffe2/core/operator_gradient.h"

namespace caffe2 {

// ReduceFrontMax
template <>
void MaxReduceDimsOp<float, CPUContext, true>::Compute(
    int rows,
    int cols,
    const float* data,
    float* out_data) {
  for (int i = 0; i < cols; i++) {
    float mx = data[i];
    for (int j = 1; j < rows; j++) {
      mx = std::max(mx, data[j * cols + i]);
    }
    out_data[i] = mx;
  }
}

// ReduceBackMax
template <>
void MaxReduceDimsOp<float, CPUContext, false>::Compute(
    int rows,
    int cols,
    const float* data,
    float* out_data) {
  for (int i = 0; i < rows; i++) {
    float mx = data[i * cols];
    for (int j = 1; j < cols; j++) {
      mx = std::max(mx, data[i * cols + j]);
    }
    out_data[i] = mx;
  }
}

// ReduceFrontMaxGradient
template <>
void MaxReduceDimsGradientOp<float, CPUContext, true>::Compute(
    int rows,
    int cols,
    const float* dYdata,
    const float* Xdata,
    const float* Ydata,
    float* dXdata) {
  int len = cols * rows;
  for (int i = 0; i < len; i++) {
    int col = i % cols;
    dXdata[i] = Xdata[i] == Ydata[col] ? dYdata[col] : 0.0f;
  }
}

// ReduceBackMaxGradient
template <>
void MaxReduceDimsGradientOp<float, CPUContext, false>::Compute(
    int rows,
    int cols,
    const float* dYdata,
    const float* Xdata,
    const float* Ydata,
    float* dXdata) {
  int len = cols * rows;
  for (int i = 0; i < len; i++) {
    int row = i / cols;
    dXdata[i] = Xdata[i] == Ydata[row] ? dYdata[row] : 0.0f;
  }
}

REGISTER_CPU_OPERATOR(ReduceFrontMax, MaxReduceDimsOp<float, CPUContext, true>);
REGISTER_CPU_OPERATOR(
    ReduceFrontMaxGradient,
    MaxReduceDimsGradientOp<float, CPUContext, true>);

REGISTER_CPU_OPERATOR(ReduceBackMax, MaxReduceDimsOp<float, CPUContext, false>);
REGISTER_CPU_OPERATOR(
    ReduceBackMaxGradient,
    MaxReduceDimsGradientOp<float, CPUContext, false>);

class GetReduceFrontMaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ReduceFrontMaxGradient",
        "",
        vector<string>{GO(0), I(0), O(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ReduceFrontMax, GetReduceFrontMaxGradient);

class GetReduceBackMaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ReduceBackMaxGradient",
        "",
        vector<string>{GO(0), I(0), O(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ReduceBackMax, GetReduceBackMaxGradient);

OPERATOR_SCHEMA(ReduceFrontMax)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("num_reduce_dims", "Number of dimensions to reduce")
    .SetDoc(
        R"DOC("Reduces the input tensor along the first dimension of the input
                 tensor by applying 'Max')DOC")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      CAFFE_ENFORCE_EQ(1, in.size());
      ArgumentHelper helper(def);
      int num_reduce_dims = helper.GetSingleArgument<int>("num_reduce_dim", 1);
      int start_index = num_reduce_dims;
      int end_index = in[0].dims_size();

      vector<int> output_shape;
      for (int i = start_index; i < end_index; ++i) {
        output_shape.push_back(in[0].dims(i));
      }
      return vector<TensorShape>{
          CreateTensorShape(output_shape, in[0].data_type())};
    });
OPERATOR_SCHEMA(ReduceFrontMaxGradient).NumInputs(3).NumOutputs(1);

OPERATOR_SCHEMA(ReduceBackMax)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("num_reduce_dims", "Number of dimensions to reduce")
    .SetDoc(
        R"DOC("Reduces the input tensor along the last dimension of the
              input tensor by applying 'Max')DOC")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      CAFFE_ENFORCE_EQ(1, in.size());
      ArgumentHelper helper(def);
      int num_reduce_dims = helper.GetSingleArgument<int>("num_reduce_dim", 1);
      int end_index = in[0].dims_size() - num_reduce_dims;

      vector<int> output_shape;
      for (int i = 0; i < end_index; ++i) {
        output_shape.push_back(in[0].dims(i));
      }
      return vector<TensorShape>{
          CreateTensorShape(output_shape, in[0].data_type())};
    });
OPERATOR_SCHEMA(ReduceBackMaxGradient).NumInputs(3).NumOutputs(1);

} // namespace caffe2
