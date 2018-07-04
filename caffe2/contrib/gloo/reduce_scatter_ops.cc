/**
 * Copyright (c) 2018-present, Facebook, Inc.
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

#include "reduce_scatter_ops.h"

#include <gloo/reduce_scatter.h>
#include <gloo/types.h>

namespace caffe2 {
namespace gloo {

template <class Context>
void ReduceScatterOp<Context>::initializeHalvingDoubling() {
  if (init_.template IsType<float>()) {
    algorithm_.reset(new ::gloo::ReduceScatterHalvingDoubling<float>(
        init_.context,
        init_.template getOutputs<float>(),
        init_.size,
        recvCounts_));
  } else if (init_.template IsType<::caffe2::float16>()) {
    algorithm_.reset(new ::gloo::ReduceScatterHalvingDoubling<::gloo::float16>(
        init_.context,
        init_.template getOutputs<::gloo::float16>(),
        init_.size,
        recvCounts_));
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

namespace {

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    ReduceScatter,
    GLOO,
    ReduceScatterOp<CPUContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
