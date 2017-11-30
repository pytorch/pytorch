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

#ifndef CAFFE2_OPERATORS_PACK_SEGMENTS_H_
#define CAFFE2_OPERATORS_PACK_SEGMENTS_H_

#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class PackSegmentsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  // USE_SIMPLE_CTOR_DTOR(PackSegmentsOp)
  USE_DISPATCH_HELPER;

  PackSegmentsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        pad_minf_(OperatorBase::GetSingleArgument<bool>("pad_minf", false)),
        return_presence_mask_(OperatorBase::GetSingleArgument<bool>(
            "return_presence_mask",
            false)) {
    if (pad_minf_) {
      padding_ = -1.0 * std::numeric_limits<float>::infinity();
    } else {
      padding_ = 0;
    }
  }

  bool RunOnDevice() {
    return DispatchHelper<TensorTypes<int, long>>::call(this, Input(LENGTHS));
  }

  template <typename T>
  bool DoRunWithType();

  template <typename T, typename Data_T>
  bool DoRunWithType2();

  INPUT_TAGS(LENGTHS, DATA);

 private:
  bool pad_minf_;
  float padding_;
  bool return_presence_mask_;

  // Scratch space required by the CUDA version
  Tensor<Context> lengths_prefix_sum_buffer_;
  Tensor<Context> lengths_prefix_sum_;
  Tensor<Context> dev_max_length_buffer_;
  Tensor<Context> dev_max_length_;
  Tensor<CPUContext> host_max_length_;
};

template <class Context>
class UnpackSegmentsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(UnpackSegmentsOp)
  USE_DISPATCH_HELPER;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int, long>>::call(this, Input(LENGTHS));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& data = Input(DATA);
    const auto& lengths = Input(LENGTHS);
    auto* output = Output(0);

    CAFFE_ENFORCE(data.ndim() >= 2, "DATA should be at least 2-D");
    CAFFE_ENFORCE(lengths.ndim() == 1, "LENGTH should be 1-D");

    const T* l = lengths.template data<T>();

    T max_length = 0;
    for (T i = 0; i < lengths.dim(0); ++i) {
      max_length = std::max(max_length, l[i]);
    }
    T total_l = std::accumulate(l, l + lengths.dim(0), 0);

    auto shape = data.dims();
    CAFFE_ENFORCE(
        shape[0] == lengths.dim(0), "LENGTH should match DATA in dimension 0");
    shape.erase(shape.begin());
    shape[0] = total_l;
    output->Resize(shape);
    // create output tensor
    auto* out = static_cast<char*>(output->raw_mutable_data(data.meta()));
    if (!(data.dim(0) * data.dim(1))) {
      return true;
    }
    int block_size = data.size() / (data.dim(0) * data.dim(1));
    int block_bytesize = data.nbytes() / (data.dim(0) * data.dim(1));
    const auto* d = static_cast<const char*>(data.raw_data());
    int start = 0;
    for (int i = 0; i < lengths.dim(0); ++i) {
      context_.template CopyItems<Context, Context>(
          data.meta(),
          l[i] * block_size,
          d + block_bytesize * data.dim(1) * i,
          out + block_bytesize * start);
      start += l[i];
    }
    return true;
  }

  INPUT_TAGS(LENGTHS, DATA);
};

} // namespace caffe2
#endif // CAFFE2_OPERATORS_PACK_SEGMENTS_H_
