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

#ifndef CAFFE2_OPERATORS_MATMUL_OP_H_
#define CAFFE2_OPERATORS_MATMUL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context, class Engine = DefaultEngine>
class BatchMatMulOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BatchMatMulOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        trans_a_(OperatorBase::GetSingleArgument<int>("trans_a", 0)),
        trans_b_(OperatorBase::GetSingleArgument<int>("trans_b", 0)),
        use_scratch_(OperatorBase::GetSingleArgument<int>("use_scratch", 0)) {
    if (use_scratch_)
      scratch_ = std::make_shared<Tensor<Context> >();
  }
  ~BatchMatMulOp() {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& A = Input(0);
    const auto& B = Input(1);
    auto* Y = Output(0);

    CAFFE_ENFORCE_EQ(A.ndim(), B.ndim());
    auto ndim = A.ndim();
    CAFFE_ENFORCE_GE(ndim, 2);
    for (int axis = 0; axis < (ndim - 2); ++axis) {
      CAFFE_ENFORCE_EQ(
          A.dim32(axis),
          B.dim32(axis),
          "Every axis of A and B should match except for the last two. Axis No",
          axis);
    }

    int a_dim0, a_dim1, b_dim0, b_dim1;

    if (trans_a_) {
      a_dim0 = A.dim32(ndim - 1);
      a_dim1 = A.dim32(ndim - 2);
    } else {
      a_dim0 = A.dim32(ndim - 2);
      a_dim1 = A.dim32(ndim - 1);
    }

    if (trans_b_) {
      b_dim0 = B.dim32(ndim - 1);
      b_dim1 = B.dim32(ndim - 2);
    } else {
      b_dim0 = B.dim32(ndim - 2);
      b_dim1 = B.dim32(ndim - 1);
    }

    // Error checking
    CAFFE_ENFORCE(
        a_dim1 == b_dim0,
        "Dimension mismatch: ",
        trans_a_ ? "trans(A): " : "A: ",
        a_dim0,
        " ",
        a_dim1,
        trans_b_ ? ", trans(B): " : ", B: ",
        b_dim0,
        " ",
        b_dim1);

    auto y_dims = A.dims();
    y_dims[ndim - 2] = a_dim0;
    y_dims[ndim - 1] = b_dim1;
    Y->Resize(y_dims);

    const auto batches = A.size_to_dim(ndim - 2);
    if (!batches) {
      Y->template mutable_data<T>(); // create output tensor
      return true;
    }

    // Y = A * B
    math::GemmBatched<T, Context, Engine>(
        trans_a_ ? CblasTrans : CblasNoTrans,
        trans_b_ ? CblasTrans : CblasNoTrans,
        A.size(),
        batches,
        B.size(),
        batches,
        a_dim0, // M
        b_dim1, // N
        a_dim1, // K
        1,
        A.template data<T>(),
        B.template data<T>(),
        0,
        Y->template mutable_data<T>(),
        &context_,
        use_scratch_ ? scratch_.get() : nullptr);
    return true;
  }

 protected:
  bool trans_a_;
  bool trans_b_;

  bool use_scratch_;
  std::shared_ptr<Tensor<Context> > scratch_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MATMUL_OP_H_
