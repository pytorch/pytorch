// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef CAFFE2_OPERATORS_BATCH_SPARSE_TO_DENSE_OP_H_
#define CAFFE2_OPERATORS_BATCH_SPARSE_TO_DENSE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class BatchSparseToDenseOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit BatchSparseToDenseOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int64_t, "dense_last_dim", dense_last_dim_, -1),
        OP_SINGLE_ARG(T, "default_value", default_value_, static_cast<T>(0)) {}
  bool RunOnDevice() {
    auto& lengths = Input(LENGTHS);
    auto& indices = Input(INDICES);
    auto& values = Input(VALUES);

    CAFFE_ENFORCE_EQ(indices.numel(), values.numel());
    CAFFE_ENFORCE_EQ(lengths.dim(), 1);
    CAFFE_ENFORCE_EQ(indices.dim(), 1);

    const int64_t* lengths_data = lengths.template data<int64_t>();
    const int64_t* indices_data = indices.template data<int64_t>();
    const T* values_data = values.template data<T>();
    int64_t batch_size = lengths.numel();

    vector<int64_t> output_shape = {batch_size};
    if (InputSize() == 4) {
      auto& shaper = Input(3);
      CAFFE_ENFORCE_EQ(shaper.dim(), 2);
      if (dense_last_dim_ == -1) {
        dense_last_dim_ = shaper.size(1);
      } else {
        CAFFE_ENFORCE(
            dense_last_dim_ == shaper.size(1),
            "The last dim argument is not aligned with the shape input last dim");
      }
    } else {
      CAFFE_ENFORCE(dense_last_dim_ >= 1, "The last dim of dense must be >= 1");
    }
    output_shape.push_back(dense_last_dim_);
    auto* output = Output(0, output_shape, at::dtype<T>());
    T* output_data = output->template mutable_data<T>();
    math::Set(
        output->numel(),
        static_cast<T>(default_value_),
        output_data,
        &context_);

    FillInDenseValues(
        batch_size,
        indices.numel(),
        lengths_data,
        indices_data,
        values_data,
        output_data,
        &context_);

    return true;
  }

 private:
  void FillInDenseValues(
      const int64_t batch_size,
      const int64_t indice_lengths,
      const int64_t* lengths_data,
      const int64_t* indices_data,
      const T* values_data,
      T* output_data,
      Context* context);

  int64_t dense_last_dim_;
  T default_value_;
  INPUT_TAGS(LENGTHS, INDICES, VALUES);

  // len_prefix_sum_ and len_prefix_tmp_ are buffers on the GPU. It is not used
  // in the CPUContext implementation.
  Tensor len_prefix_sum_{Context::GetDeviceType()};
  Tensor len_prefix_tmp_{Context::GetDeviceType()};
};

template <typename T, class Context>
class BatchDenseToSparseOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit BatchDenseToSparseOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  bool RunOnDevice() {
    auto& lengths = Input(LENGTHS);
    auto& indices = Input(INDICES);
    auto& dense = Input(DENSE);

    CAFFE_ENFORCE_EQ(lengths.dim(), 1);
    CAFFE_ENFORCE_EQ(indices.dim(), 1);
    CAFFE_ENFORCE_EQ(dense.dim(), 2);
    const int64_t* lengths_data = lengths.template data<int64_t>();
    const int64_t* indices_data = indices.template data<int64_t>();
    const T* dense_data = dense.template data<T>();

    int64_t batch_size = lengths.numel();

    CAFFE_ENFORCE_EQ(batch_size, dense.size(0));
    dense_last_dim_ = dense.size(1);
    vector<int64_t> output_shape = indices.sizes().vec();
    auto* output = Output(0, output_shape, at::dtype<T>());
    T* output_data = output->template mutable_data<T>();

    FillInSparseValues(
        batch_size,
        indices.numel(),
        lengths_data,
        indices_data,
        dense_data,
        output_data,
        &context_);

    return true;
  }

 private:
  void FillInSparseValues(
      const int64_t batch_size,
      const int64_t indice_lengths,
      const int64_t* lengths_data,
      const int64_t* indices_data,
      const T* dense_data,
      T* output_data,
      Context* context);

  int64_t dense_last_dim_;
  INPUT_TAGS(LENGTHS, INDICES, DENSE);

  // len_prefix_sum_ and len_prefix_tmp_ are buffers on the GPU. It is not used
  // in the CPUContext implementation.
  Tensor len_prefix_sum_{Context::GetDeviceType()};
  Tensor len_prefix_tmp_{Context::GetDeviceType()};
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_BATCH_SPARSE_TO_DENSE_OP_H_
