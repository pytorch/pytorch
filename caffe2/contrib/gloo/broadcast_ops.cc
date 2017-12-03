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

#include "broadcast_ops.h"

#include <gloo/broadcast_one_to_all.h>

namespace caffe2 {
namespace gloo {

template <class Context>
void BroadcastOp<Context>::initializeAlgorithm() {
  if (init_.template IsType<float>()) {
    algorithm_.reset(new ::gloo::BroadcastOneToAll<float>(
        init_.context, init_.template getOutputs<float>(), init_.size, root_));
  } else if (init_.template IsType<long>()) {
    algorithm_.reset(new ::gloo::BroadcastOneToAll<long>(
        init_.context, init_.template getOutputs<long>(), init_.size, root_));
  } else if (init_.template IsType<int>()) {
    algorithm_.reset(new ::gloo::BroadcastOneToAll<int>(
        init_.context, init_.template getOutputs<int>(), init_.size, root_));
  } else if (init_.template IsType<float16>()) {
    algorithm_.reset(new ::gloo::BroadcastOneToAll<::gloo::float16>(
        init_.context,
        init_.template getOutputs<::gloo::float16>(),
        init_.size,
        root_));
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

namespace {

REGISTER_CPU_OPERATOR_WITH_ENGINE(Broadcast, GLOO, BroadcastOp<CPUContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
