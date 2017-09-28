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

#pragma once

#include <cstddef>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/logging.h"

#include <nccl.h>
#include <unordered_map>

#define NCCL_VERSION_MIN(major, minor, patch) \
  ((NCCL_MAJOR > major) || \
    ((NCCL_MAJOR == major) && ((NCCL_MINOR > minor) || \
      ((NCCL_MINOR == minor) && (NCCL_PATCH >= patch)) )))

namespace caffe2 {

namespace nccl {

#define CAFFE_NCCL_CHECK(condition)    \
  do {                                 \
    ncclResult_t status = (condition); \
    CAFFE_ENFORCE_EQ(                  \
        status,                        \
        ncclSuccess,                   \
        " ",                           \
        "Error at: ",                  \
        __FILE__,                      \
        __LINE__,                      \
        ": ",                          \
        ncclGetErrorString(status));   \
  } while (0)

struct NCCLElement {
  const TensorCUDA* src{nullptr};
  TensorCUDA* dst{nullptr};
  int device{0};
};

struct NCCLExecution {
  int stream_gpu_id{0};
  cudaStream_t stream{nullptr};
  std::vector<NCCLElement> elements;
  size_t root{0};
};

template <typename T>
class NCCL {
 public:
  static void AllReduce(const NCCLExecution& ex);
  static void Broadcast(const NCCLExecution& ex);
  static void Reduce(const NCCLExecution& ex);
  static void AllGather(const NCCLExecution& ex);
  static void ReduceScatter(const NCCLExecution& ex);
};
}
}
