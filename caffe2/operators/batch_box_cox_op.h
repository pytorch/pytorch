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

#ifndef CAFFE_OPERATORS_BATCH_BOX_COX_OPS_H_
#define CAFFE_OPERATORS_BATCH_BOX_COX_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class BatchBoxCoxOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BatchBoxCoxOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        min_block_size_(
            OperatorBase::GetSingleArgument<int>("min_block_size", 256)) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(DATA));
  }

  template <typename T>
  bool DoRunWithType();

 protected:
  template <typename T>
  void BoxCoxNaive(
      TIndex N,
      TIndex D,
      const T* data_ptr,
      const T* lambda1_ptr,
      const T* lambda2_ptr,
      T k_eps,
      T* output_ptr);

#ifdef CAFFE2_USE_MKL
  template <typename T>
  void BoxCoxNonzeroLambda(
      TIndex D,
      const T* data_ptr,
      const T* lambda1,
      const T* lambda2,
      T k_eps,
      T* output_ptr);

  template <typename T>
  void BoxCoxZeroLambda(
      TIndex D,
      const T* data_ptr,
      const T* lambda2,
      T k_eps,
      T* output_ptr);

  template <typename T>
  void BoxCoxMixedLambda(
      const T* data_ptr,
      const vector<int>& nonzeros,
      const vector<int>& zeros,
      const T* lambda1,
      const T* lambda2,
      const T* lambda2_z,
      T k_eps,
      T* buffer,
      T* output_ptr);

  vector<int> nonzeros_, zeros_;

  // Buffers used by the MKL version are cached across calls.
  struct CachedBuffers {
    virtual ~CachedBuffers() {}
    int type_;
  };
  template <typename T>
  struct TypedCachedBuffers : public CachedBuffers {
    vector<T> lambda1_, lambda2_, lambda2_z_;
    vector<T> accumulator_;
  };
  template <typename T>
  TypedCachedBuffers<T>& GetBuffers();
  unique_ptr<CachedBuffers> buffers_;

#endif // CAFFE2_USE_MKL

  int min_block_size_;

  INPUT_TAGS(DATA, LAMBDA1, LAMBDA2);
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_BATCH_BOX_COX_OPS_H_
