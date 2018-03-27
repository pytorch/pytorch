#include "caffe2/operators/reduction_front_back_ops.h"
#include "caffe2/core/operator_gradient.h"

namespace caffe2 {

/***
  Sum Ops
***/

// ReduceFrontSum: columnwise sum
template <>
template <typename T>
void SumReduceDimsOp<CPUContext, true, false>::Compute(
    int rows,
    int cols,
    const T* in_data,
    const int32_t* lengths_data,
    T* out_data) {
  for (int j = 0; j < cols; j++) {
    T sum = in_data[j];
    int length = lengths_data == nullptr ? rows : lengths_data[j];
    for (int i = 1; i < length; i++) {
      sum += in_data[i * cols + j];
    }
    out_data[j] = sum;
  }
}

// ReduceBackSum: rowwise sum
template <>
template <typename T>
void SumReduceDimsOp<CPUContext, false, false>::Compute(
    int rows,
    int cols,
    const T* in_data,
    const int32_t* lengths_data,
    T* out_data) {
  for (int i = 0; i < rows; i++) {
    int offset = i * cols;
    T sum = in_data[offset];
    int length = lengths_data == nullptr ? cols : lengths_data[i];
    for (int j = 1; j < length; j++) {
      sum += in_data[offset + j];
    }
    out_data[i] = sum;
  }
}

// ReduceFrontSumGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CPUContext, true, false>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    const int* lengths_data,
    T* dXdata) {
  for (int i = 0; i < rows * cols; i++) {
    int row = i / cols;
    int col = i % cols;
    if (lengths_data == nullptr || row < lengths_data[col]) {
      dXdata[i] = dYdata[col];
    } else {
      dXdata[i] = 0;
    }
  }
}

// ReduceBackSumGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CPUContext, false, false>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    const int* lengths_data,
    T* dXdata) {
  for (int i = 0; i < rows * cols; i++) {
    int row = i / cols;
    int col = i % cols;
    if (lengths_data == nullptr || col < lengths_data[row]) {
      dXdata[i] = dYdata[row];
    } else {
      dXdata[i] = 0;
    }
  }
}

REGISTER_CPU_OPERATOR(ReduceFrontSum, SumReduceDimsOp<CPUContext, true, false>);
REGISTER_CPU_OPERATOR(
    ReduceFrontSumGradient,
    SumReduceDimsGradientOp<CPUContext, true, false>);

class GetReduceFrontSumGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> grad_in = {GO(0), I(0)};
    if (def_.input_size() == 2) {
      grad_in.push_back(I(1));
    }
    return SingleGradientDef(
        "ReduceFrontSumGradient", "", grad_in, vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ReduceFrontSum, GetReduceFrontSumGradient);

REGISTER_CPU_OPERATOR(ReduceBackSum, SumReduceDimsOp<CPUContext, false, false>);
REGISTER_CPU_OPERATOR(
    ReduceBackSumGradient,
    SumReduceDimsGradientOp<CPUContext, false, false>);

class GetReduceBackSumGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> grad_in = {GO(0), I(0)};
    if (def_.input_size() == 2) {
      grad_in.push_back(I(1));
    }
    return SingleGradientDef(
        "ReduceBackSumGradient", "", grad_in, vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ReduceBackSum, GetReduceBackSumGradient);

#define REDUCTION_OP_SHAPE_INFERENCE(is_front_reducer)                      \
  CAFFE_ENFORCE_LE(1, in.size());                                           \
  CAFFE_ENFORCE_GE(2, in.size());                                           \
  ArgumentHelper helper(def);                                               \
  int num_reduce_dims = helper.GetSingleArgument<int>("num_reduce_dim", 1); \
  int start_index = is_front_reducer ? num_reduce_dims : 0;                 \
  int end_index = is_front_reducer ? in[0].dims_size()                      \
                                   : in[0].dims_size() - num_reduce_dims;   \
  vector<int> output_shape;                                                 \
  for (int i = start_index; i < end_index; ++i) {                           \
    output_shape.push_back(in[0].dims(i));                                  \
  }                                                                         \
  return vector<TensorShape>{                                               \
      CreateTensorShape(output_shape, in[0].data_type())};

OPERATOR_SCHEMA(ReduceFrontSum)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .Arg("num_reduce_dims", "Number of dimensions to reduce.")
    .SetDoc(R"DOC(
Reduces the input tensor along the first dimension of the input
tensor by applying 'Sum'.  When lengths is given, sum is only computed
with subsets of elements correspondingly.
)DOC")
    .Input(0, "data_in", "(T<D1..., Dn>) Input data.")
    .Input(
        1,
        "lengths",
        "Num of elements in each sample, should have size D2 x D3 x ... x Dn.")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      REDUCTION_OP_SHAPE_INFERENCE(true)
    });
OPERATOR_SCHEMA(ReduceFrontSumGradient).NumInputs(2, 3).NumOutputs(1);

OPERATOR_SCHEMA(ReduceBackSum)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .Arg("num_reduce_dims", "Number of dimensions to reduce.")
    .SetDoc(R"DOC(
Reduces the input tensor along the last dimension of the
input tensor by applying 'Sum'.  When lengths is given, sum is only computed
with subsets of elements correspondingly.
)DOC")
    .Input(0, "data_in", "(T<D1..., Dn>) Input data.")
    .Input(
        1,
        "lengths",
        "Num of elements in each sample, should have size D1 x D2 x ... x D(n-1).")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      REDUCTION_OP_SHAPE_INFERENCE(false)
    });
OPERATOR_SCHEMA(ReduceBackSumGradient).NumInputs(2, 3).NumOutputs(1);

/***
  Mean Ops
***/

// ReduceFrontMean: columnwise mean
template <>
template <typename T>
void SumReduceDimsOp<CPUContext, true, true>::Compute(
    int rows,
    int cols,
    const T* in_data,
    const int32_t* lengths_data,
    T* out_data) {
  for (int j = 0; j < cols; j++) {
    T sum = in_data[j];
    int length = lengths_data == nullptr ? rows : lengths_data[j];
    for (int i = 1; i < length; i++) {
      sum += in_data[i * cols + j];
    }
    out_data[j] = sum / length;
  }
}

// ReduceBackMean: rowwise mean
template <>
template <typename T>
void SumReduceDimsOp<CPUContext, false, true>::Compute(
    int rows,
    int cols,
    const T* in_data,
    const int32_t* lengths_data,
    T* out_data) {
  for (int i = 0; i < rows; i++) {
    int offset = i * cols;
    T sum = in_data[offset];
    int length = lengths_data == nullptr ? cols : lengths_data[i];
    for (int j = 1; j < length; j++) {
      sum += in_data[offset + j];
    }
    out_data[i] = sum / length;
  }
}

// ReduceFrontMeanGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CPUContext, true, true>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    const int* lengths_data,
    T* dXdata) {
  for (int i = 0; i < rows * cols; i++) {
    int row = i / cols;
    int col = i % cols;
    if (lengths_data == nullptr) {
      dXdata[i] = dYdata[col] / rows;
    } else if (row < lengths_data[col]) {
      dXdata[i] = dYdata[col] / lengths_data[col];
    } else {
      dXdata[i] = 0;
    }
  }
}

// ReduceBackMeanGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CPUContext, false, true>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    const int* lengths_data,
    T* dXdata) {
  for (int i = 0; i < rows * cols; i++) {
    int row = i / cols;
    int col = i % cols;
    if (lengths_data == nullptr) {
      dXdata[i] = dYdata[row] / cols;
    } else if (col < lengths_data[row]) {
      dXdata[i] = dYdata[row] / lengths_data[row];
    } else {
      dXdata[i] = 0;
    }
  }
}

REGISTER_CPU_OPERATOR(ReduceFrontMean, SumReduceDimsOp<CPUContext, true, true>);
REGISTER_CPU_OPERATOR(
    ReduceFrontMeanGradient,
    SumReduceDimsGradientOp<CPUContext, true, true>);

class GetReduceFrontMeanGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> grad_in = {GO(0), I(0)};
    if (def_.input_size() == 2) {
      grad_in.push_back(I(1));
    }
    return SingleGradientDef(
        "ReduceFrontMeanGradient", "", grad_in, vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ReduceFrontMean, GetReduceFrontMeanGradient);

OPERATOR_SCHEMA(ReduceFrontMean)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .Arg("num_reduce_dims", "Number of dimensions to reduce.")
    .SetDoc(R"DOC(
Reduces the input tensor along the first dimension of the input
tensor by applying 'Mean'. When lengths is given, mean is only computed
with subsets of elements correspondingly.
)DOC")
    .Input(0, "data_in", "(T<D1..., Dn>) Input data.")
    .Input(
        1,
        "lengths",
        "Num of elements in each sample, should have size D2 x D3 x ... x Dn.")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      REDUCTION_OP_SHAPE_INFERENCE(true)
    });
OPERATOR_SCHEMA(ReduceFrontMeanGradient).NumInputs(2, 3).NumOutputs(1);

REGISTER_CPU_OPERATOR(ReduceBackMean, SumReduceDimsOp<CPUContext, false, true>);
REGISTER_CPU_OPERATOR(
    ReduceBackMeanGradient,
    SumReduceDimsGradientOp<CPUContext, false, true>);

class GetReduceBackMeanGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> grad_in = {GO(0), I(0)};
    if (def_.input_size() == 2) {
      grad_in.push_back(I(1));
    }
    return SingleGradientDef(
        "ReduceBackMeanGradient", "", grad_in, vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ReduceBackMean, GetReduceBackMeanGradient);

OPERATOR_SCHEMA(ReduceBackMean)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .Arg("num_reduce_dims", "Number of dimensions to reduce.")
    .SetDoc(R"DOC(
Reduces the input tensor along the last dimension of the input
tensor by applying 'Mean'. When lengths is given, mean is only computed
with subsets of elements correspondingly.
)DOC")
    .Input(0, "data_in", "(T<D1..., Dn>) Input data.")
    .Input(
        1,
        "lengths",
        "Num of elements in each sample, should have size D1 x D2 x ... x D(n-1).")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      REDUCTION_OP_SHAPE_INFERENCE(false)
    });
OPERATOR_SCHEMA(ReduceBackMeanGradient).NumInputs(2, 3).NumOutputs(1);

/***
  Max Ops
***/

// ReduceFrontMax
template <>
void MaxReduceDimsOp<float, CPUContext, true>::Compute(
    int rows,
    int cols,
    const float* data,
    const int32_t* lengths_data,
    float* out_data) {
  for (int i = 0; i < cols; i++) {
    float mx = data[i];
    int length = lengths_data == nullptr ? rows : lengths_data[i];
    for (int j = 1; j < length; j++) {
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
    const int32_t* lengths_data,
    float* out_data) {
  for (int i = 0; i < rows; i++) {
    float mx = data[i * cols];
    int length = lengths_data == nullptr ? cols : lengths_data[i];
    for (int j = 1; j < length; j++) {
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
    const int32_t* lengths_data,
    float* dXdata) {
  int len = cols * rows;
  for (int i = 0; i < len; i++) {
    int col = i % cols;
    int row = i / cols;
    if (lengths_data != nullptr && row >= lengths_data[col]) {
      dXdata[i] = 0.0f;
    } else {
      dXdata[i] = Xdata[i] == Ydata[col] ? dYdata[col] : 0.0f;
    }
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
    const int32_t* lengths_data,
    float* dXdata) {
  int len = cols * rows;
  for (int i = 0; i < len; i++) {
    int row = i / cols;
    int col = i % cols;
    if (lengths_data == nullptr || col < lengths_data[row]) {
      dXdata[i] = Xdata[i] == Ydata[row] ? dYdata[row] : 0.0f;
    } else {
      dXdata[i] = 0.0f;
    }
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
    vector<string> grad_in = {GO(0), I(0), O(0)};
    if (def_.input_size() == 2) {
      grad_in.push_back(I(1));
    }
    return SingleGradientDef(
        "ReduceFrontMaxGradient", "", grad_in, vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ReduceFrontMax, GetReduceFrontMaxGradient);

class GetReduceBackMaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> grad_in = {GO(0), I(0), O(0)};
    if (def_.input_size() == 2) {
      grad_in.push_back(I(1));
    }
    return SingleGradientDef(
        "ReduceBackMaxGradient", "", grad_in, vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ReduceBackMax, GetReduceBackMaxGradient);

OPERATOR_SCHEMA(ReduceFrontMax)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .Arg("num_reduce_dims", "Number of dimensions to reduce")
    .SetDoc(R"DOC(
Reduces the input tensor along the first dimension of the input
tensor by applying 'Max'. When lengths is given, max is only computed
with subsets of elements correspondingly.
)DOC")
    .Input(0, "data_in", "(T<D1..., Dn>) Input data.")
    .Input(
        1,
        "lengths",
        "Num of elements in each sample, should have size D2 x D3 ... x Dn.")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      REDUCTION_OP_SHAPE_INFERENCE(true)
    });
OPERATOR_SCHEMA(ReduceFrontMaxGradient).NumInputs(3, 4).NumOutputs(1);

OPERATOR_SCHEMA(ReduceBackMax)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .Arg("num_reduce_dims", "Number of dimensions to reduce")
    .SetDoc(R"DOC(
Reduces the input tensor along the last dimension of the
input tensor by applying 'Max'. When lengths is given, max is only computed
with subsets of elements correspondingly.
)DOC")
    .Input(0, "data_in", "(T<D1..., Dn>) Input data.")
    .Input(
        1,
        "lengths",
        "Num of elements in each sample, should have size D1 x D2 x ... x D(n-1).")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      REDUCTION_OP_SHAPE_INFERENCE(false)
    });
OPERATOR_SCHEMA(ReduceBackMaxGradient).NumInputs(3, 4).NumOutputs(1);

#undef REDUCTION_OP_SHAPE_INFERENCE

} // namespace caffe2
