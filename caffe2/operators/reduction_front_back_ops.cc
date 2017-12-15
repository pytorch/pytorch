/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
    T* out_data) {
  for (int j = 0; j < cols; j++) {
    T sum = in_data[j];
    for (int i = 1; i < rows; i++) {
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
    T* out_data) {
  for (int i = 0; i < rows; i++) {
    int offset = i * cols;
    T sum = in_data[offset];
    for (int j = 1; j < cols; j++) {
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
    T* dXdata) {
  for (int i = 0; i < rows * cols; i++) {
    dXdata[i] = dYdata[i % cols];
  }
}

// ReduceBackSumGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CPUContext, false, false>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    T* dXdata) {
  for (int i = 0; i < rows * cols; i++) {
    dXdata[i] = dYdata[i / cols];
  }
}

REGISTER_CPU_OPERATOR(ReduceFrontSum, SumReduceDimsOp<CPUContext, true, false>);
REGISTER_CPU_OPERATOR(
    ReduceFrontSumGradient,
    SumReduceDimsGradientOp<CPUContext, true, false>);

class GetReduceFrontSumGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ReduceFrontSumGradient",
        "",
        vector<string>{GO(0), I(0)},
        vector<string>{GI(0)});
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
    return SingleGradientDef(
        "ReduceBackSumGradient",
        "",
        vector<string>{GO(0), I(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ReduceBackSum, GetReduceBackSumGradient);

#define REDUCTION_OP_SHAPE_INFERENCE(is_front_reducer)                      \
  CAFFE_ENFORCE_EQ(1, in.size());                                           \
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
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("num_reduce_dims", "Number of dimensions to reduce")
    .SetDoc(
        R"DOC("Reduces the input tensor along the first dimension of the input
                 tensor by applying 'Sum')DOC")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      REDUCTION_OP_SHAPE_INFERENCE(true)
    });
OPERATOR_SCHEMA(ReduceFrontSumGradient).NumInputs(2).NumOutputs(1);

OPERATOR_SCHEMA(ReduceBackSum)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("num_reduce_dims", "Number of dimensions to reduce")
    .SetDoc(
        R"DOC("Reduces the input tensor along the last dimension of the
              input tensor by applying 'Sum')DOC")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      REDUCTION_OP_SHAPE_INFERENCE(false)
    });
OPERATOR_SCHEMA(ReduceBackSumGradient).NumInputs(2).NumOutputs(1);

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
    T* out_data) {
  for (int j = 0; j < cols; j++) {
    T sum = in_data[j];
    for (int i = 1; i < rows; i++) {
      sum += in_data[i * cols + j];
    }
    out_data[j] = sum / rows;
  }
}

// ReduceBackMean: rowwise mean
template <>
template <typename T>
void SumReduceDimsOp<CPUContext, false, true>::Compute(
    int rows,
    int cols,
    const T* in_data,
    T* out_data) {
  for (int i = 0; i < rows; i++) {
    int offset = i * cols;
    T sum = in_data[offset];
    for (int j = 1; j < cols; j++) {
      sum += in_data[offset + j];
    }
    out_data[i] = sum / cols;
  }
}

// ReduceFrontMeanGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CPUContext, true, true>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    T* dXdata) {
  for (int i = 0; i < rows * cols; i++) {
    dXdata[i] = dYdata[i % cols] / rows;
  }
}

// ReduceBackMeanGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CPUContext, false, true>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    T* dXdata) {
  for (int i = 0; i < rows * cols; i++) {
    dXdata[i] = dYdata[i / cols] / cols;
  }
}

REGISTER_CPU_OPERATOR(ReduceFrontMean, SumReduceDimsOp<CPUContext, true, true>);
REGISTER_CPU_OPERATOR(
    ReduceFrontMeanGradient,
    SumReduceDimsGradientOp<CPUContext, true, true>);

class GetReduceFrontMeanGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ReduceFrontMeanGradient",
        "",
        vector<string>{GO(0), I(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ReduceFrontMean, GetReduceFrontMeanGradient);

OPERATOR_SCHEMA(ReduceFrontMean)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("num_reduce_dims", "Number of dimensions to reduce")
    .SetDoc(
        R"DOC("Reduces the input tensor along the first dimension of the input
                 tensor by applying 'Mean')DOC")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      REDUCTION_OP_SHAPE_INFERENCE(true)
    });
OPERATOR_SCHEMA(ReduceFrontMeanGradient).NumInputs(2).NumOutputs(1);

REGISTER_CPU_OPERATOR(ReduceBackMean, SumReduceDimsOp<CPUContext, false, true>);
REGISTER_CPU_OPERATOR(
    ReduceBackMeanGradient,
    SumReduceDimsGradientOp<CPUContext, false, true>);

class GetReduceBackMeanGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ReduceBackMeanGradient",
        "",
        vector<string>{GO(0), I(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ReduceBackMean, GetReduceBackMeanGradient);

OPERATOR_SCHEMA(ReduceBackMean)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("num_reduce_dims", "Number of dimensions to reduce")
    .SetDoc(
        R"DOC("Reduces the input tensor along the last dimension of the
              input tensor by applying 'Mean')DOC")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      REDUCTION_OP_SHAPE_INFERENCE(false)
    });
OPERATOR_SCHEMA(ReduceBackMeanGradient).NumInputs(2).NumOutputs(1);

/***
  Max Ops
***/

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
      REDUCTION_OP_SHAPE_INFERENCE(true)
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
      REDUCTION_OP_SHAPE_INFERENCE(false)
    });
OPERATOR_SCHEMA(ReduceBackMaxGradient).NumInputs(3).NumOutputs(1);

#undef REDUCTION_OP_SHAPE_INFERENCE

} // namespace caffe2
