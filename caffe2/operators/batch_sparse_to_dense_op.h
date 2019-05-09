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
  bool RunOnDevice() override;

 private:
  int64_t dense_last_dim_;
  T default_value_;
  INPUT_TAGS(LENGTHS, INDICES, VALUES);
};

template <typename T, class Context>
class BatchDenseToSparseOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit BatchDenseToSparseOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  bool RunOnDevice() override;

 private:
  int64_t dense_last_dim_;
  INPUT_TAGS(LENGTHS, INDICES, DENSE);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_BATCH_SPARSE_TO_DENSE_OP_H_
