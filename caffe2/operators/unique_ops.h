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
  Tensor thrust_unique_buffer_;
  Tensor cuda_order_buffer_{Context::GetDeviceType()};
  Tensor second_order_buffer_{Context::GetDeviceType()};

 public:
  OUTPUT_TAGS(UNIQUE, REMAPPING);
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_UNIQUE_OPS_H_
