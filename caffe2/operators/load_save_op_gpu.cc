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
#include "caffe2/operators/load_save_op.h"

namespace caffe2 {

template <>
void LoadOp<CUDAContext>::SetCurrentDevice(BlobProto* proto) {
  if (proto->has_tensor()) {
    auto* device_detail = proto->mutable_tensor()->mutable_device_detail();
    device_detail->set_device_type(CUDA);
    device_detail->set_cuda_gpu_id(CaffeCudaGetDevice());
  }
}

REGISTER_CUDA_OPERATOR(Load, LoadOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Save, SaveOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Checkpoint, CheckpointOp<CUDAContext>);
}  // namespace caffe2
