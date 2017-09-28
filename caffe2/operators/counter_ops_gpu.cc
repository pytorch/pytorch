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

#include "caffe2/core/context_gpu.h"
#include "counter_ops.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(CreateCounter, CreateCounterOp<int64_t, CUDAContext>);
REGISTER_CUDA_OPERATOR(ResetCounter, ResetCounterOp<int64_t, CUDAContext>);
REGISTER_CUDA_OPERATOR(CountDown, CountDownOp<int64_t, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    CheckCounterDone,
    CheckCounterDoneOp<int64_t, CUDAContext>);
REGISTER_CUDA_OPERATOR(CountUp, CountUpOp<int64_t, CUDAContext>);
REGISTER_CUDA_OPERATOR(RetrieveCount, RetrieveCountOp<int64_t, CUDAContext>);
} // namespace caffe2
