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

#ifndef CAFFE2_OPERATORS_INDEX_HASH_OPS_H_
#define CAFFE2_OPERATORS_INDEX_HASH_OPS_H_

#include "caffe2/core/asan.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class IndexHashOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  IndexHashOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        seed_(OperatorBase::GetSingleArgument<int64_t>("seed", 0)),
        modulo_(OperatorBase::GetSingleArgument<int64_t>("modulo", 0)) {
    CAFFE_ENFORCE_GT(modulo_, 0, "MODULO should be > 0");
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& indices = Input(INDICES);
    auto* hashed_indices = Output(HASHED_INDICES);
    hashed_indices->ResizeLike(indices);

    CAFFE_ENFORCE_GE(
        static_cast<int64_t>(std::numeric_limits<T>::max()),
        modulo_,
        "MODULO shouldn't be larger than the numeric limit of the indices");

    auto N = indices.size();
    auto* indices_data = indices.template data<T>();
    auto* hashed_indices_data = hashed_indices->template mutable_data<T>();

    for (auto i = 0; i < N; i++) {
      hashed_indices_data[i] = hash(indices_data[i]);
    }

    return true;
  }

 protected:
  template <typename T>
  T hash(T id) CAFFE2_NO_SANITIZE("signed-integer-overflow") {
    int8_t* bytes = (int8_t*)&id;
    T hashed = seed_ * 0xDEADBEEF;
    for (int i = 0; i < sizeof(T) / sizeof(int8_t); i++) {
      hashed = hashed * 65537 + bytes[i];
    }
    hashed = static_cast<T>((modulo_ + hashed % modulo_) % modulo_);
    return hashed;
  }

 private:
  INPUT_TAGS(INDICES);
  OUTPUT_TAGS(HASHED_INDICES);

  int64_t seed_;
  int64_t modulo_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INDEX_HASH_OPS_H_
