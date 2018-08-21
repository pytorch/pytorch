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
