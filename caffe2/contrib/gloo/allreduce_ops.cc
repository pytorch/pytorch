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

#include "allreduce_ops.h"

#include <gloo/allreduce_halving_doubling.h>
#include <gloo/allreduce_ring.h>
#include <gloo/allreduce_ring_chunked.h>
#include <gloo/types.h>

namespace caffe2 {
namespace gloo {

template <class Context>
void AllreduceOp<Context>::initializeHalvingDoubling() {
  if (init_.template IsType<float>()) {
    algorithm_.reset(new ::gloo::AllreduceHalvingDoubling<float>(
        init_.context, init_.template getOutputs<float>(), init_.size));
  } else if (init_.template IsType<::caffe2::float16>()) {
    algorithm_.reset(new ::gloo::AllreduceHalvingDoubling<::gloo::float16>(
        init_.context,
        init_.template getOutputs<::gloo::float16>(),
        init_.size));
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

template <class Context>
void AllreduceOp<Context>::initializeRingFull() {
  if (init_.template IsType<float>()) {
    algorithm_.reset(new ::gloo::AllreduceRing<float>(
        init_.context, init_.template getOutputs<float>(), init_.size));
  } else if (init_.template IsType<::caffe2::float16>()) {
    algorithm_.reset(new ::gloo::AllreduceRing<::gloo::float16>(
        init_.context,
        init_.template getOutputs<::gloo::float16>(),
        init_.size));
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

template <class Context>
void AllreduceOp<Context>::initializeRingChunked() {
  if (init_.template IsType<float>()) {
    algorithm_.reset(new ::gloo::AllreduceRingChunked<float>(
        init_.context, init_.template getOutputs<float>(), init_.size));
  } else if (init_.template IsType<::caffe2::float16>()) {
    algorithm_.reset(new ::gloo::AllreduceRingChunked<::gloo::float16>(
        init_.context,
        init_.template getOutputs<::gloo::float16>(),
        init_.size));
  } else {
    CAFFE_ENFORCE(false, "Unhandled type: ", init_.meta.name());
  }
}

namespace {

REGISTER_CPU_OPERATOR_WITH_ENGINE(Allreduce, GLOO, AllreduceOp<CPUContext>);

} // namespace
} // namespace gloo
} // namespace caffe2
