/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

#include "allgather_ops.h"

#include <gloo/allgather_ring.h>

namespace caffe2 {
namespace gloo {

template <class Context>
void AllgatherOp<Context>::initializeAlgorithm() {
  if (init_.template IsType<float>()) {
    algorithm_.reset(new ::gloo::AllgatherRing<float>(
        init_.context,
        init_.template getInputs<float>(),
        init_.template getOutput<float>(),
        init_.size));
  } else if (init_.template IsType<long>()) {
    algorithm_.reset(new ::gloo::AllgatherRing<long>(
        init_.context,
        init_.template getInputs<long>(),
        init_.template getOutput<long>(),
        init_.size));
  } else if (init_.template IsType<int>()) {
    algorithm_.reset(new ::gloo::AllgatherRing<int>(
        init_.context,
        init_.template getInputs<int>(),
        init_.template getOutput<int>(),
        init_.size));
  } else if (init_.template IsType<float16>()) {
    algorithm_.reset(new ::gloo::AllgatherRing<::gloo::float16>(
        init_.context,
        init_.template getInputs<::gloo::float16>(),
        init_.template getOutput<::gloo::float16>(),
        init_.size));
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

namespace {

REGISTER_CPU_OPERATOR_WITH_ENGINE(Allgather, GLOO, AllgatherOp<CPUContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
