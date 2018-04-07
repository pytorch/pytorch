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

#ifndef CAFFE_OPERATORS_UNIQUE_OPS_H_
#define CAFFE_OPERATORS_UNIQUE_OPS_H_

#include <cmath>
#include <google/dense_hash_map>

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

/**
 * Deduplicates input indices vector and optionally produces reverse remapping.
 * Current implementation produces a sorted list but it's not guaranteed in
 * general.
 */
template <class Context>
class UniqueOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(UniqueOp)

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType();

 private:
  vector<int> order_;
  Tensor<Context> thrust_unique_buffer_;
  Tensor<Context> cuda_order_buffer_;
  Tensor<Context> second_order_buffer_;

 public:
  OUTPUT_TAGS(UNIQUE, REMAPPING);
};

// Implementation of Unique using the hash-map from the sparsehash library
template <class Context>
class SparseHashUniqueOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SparseHashUniqueOp)

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType();

 protected:
  OUTPUT_TAGS(UNIQUE, REMAPPING);

 private:
  Tensor<CPUContext> input_buffer_;
  Tensor<CPUContext> unique_buffer_;
  Tensor<CPUContext> remapping_buffer_;

  template <typename T>
  bool RunOnCPU(
      const Tensor<CPUContext>& inputTensor,
      Tensor<CPUContext>* uniqueTensor,
      Tensor<CPUContext>* remappingTensor) {
    int N = inputTensor.dim32(0);
    CAFFE_ENFORCE_EQ(inputTensor.ndim(), 1, "Input should be a vector");

    int* remapping = nullptr;
    if (remappingTensor) {
      remappingTensor->ResizeLike(inputTensor);
      remapping = remappingTensor->template mutable_data<int>();
    }

    auto* input = inputTensor.template data<T>();
    google::dense_hash_map<T, int> hashMap(N);

    // We assume input tensor only contains positive values
    hashMap.set_empty_key(-1);
    int K = 0;
    for (int i = 0; i < N; ++i) {
      CAFFE_ENFORCE_GE(
          input[i], 0, "SparseHashUnique assumes inputs are positive numbers");
      auto p = hashMap.insert(std::make_pair(input[i], K));
      if (p.second) {
        ++K;
      }
      if (remapping) {
        remapping[i] = p.first->second;
      }
    }
    uniqueTensor->Resize(K);
    auto* unique = uniqueTensor->template mutable_data<T>();
    for (auto p : hashMap) {
      unique[p.second] = p.first;
    }

    return true;
  }
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_UNIQUE_OPS_H_
