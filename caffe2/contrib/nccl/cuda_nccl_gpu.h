#pragma once

#include <cstddef>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/logging.h"

#include <unordered_map>
#include <nccl.h>

namespace caffe2 {

namespace nccl {


#define CAFFE_NCCL_CHECK(condition)                                     \
  do {                                                                     \
    ncclResult_t status = (condition);                                     \
    CAFFE_CHECK_EQ(status, ncclSuccess) << " "                             \
                                        << "Error at: " << __FILE__ << ":" \
                                        << __LINE__ << ": "                \
                                        << ncclGetErrorString(status);     \
  } while (0)


class NCCLContext {
 public:
  explicit NCCLContext(const std::vector<int>& devices) : devices_(devices) {
    comms_.resize(devices_.size());
    CAFFE_NCCL_CHECK(
        ncclCommInitAll(comms_.data(), devices_.size(), devices_.data()));

    streams_.resize(devices_.size());
    for (auto i = 0; i < devices_.size(); ++i) {
      DeviceGuard g(devices_[i]);
      CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }
  }

  ~NCCLContext() {
    for (auto i = 0; i < devices_.size(); ++i) {
      DeviceGuard g(devices_[i]);
      CUDA_CHECK(cudaStreamDestroy(streams_[i]));
    }
    for (auto& comm: comms_) {
      ncclCommDestroy(comm);
    }
  }

  std::vector<ncclComm_t>& comms() {
    return comms_;
  }

  std::vector<cudaStream_t>& streams() {
    return streams_;
  }

 private:
  std::vector<int> devices_;
  std::vector<ncclComm_t> comms_;
  std::vector<cudaStream_t> streams_;
};

struct NCCLElement {
  const TensorCUDA* src{nullptr};
  TensorCUDA* dst{nullptr};
  int device{0};
};

template<typename T>
class NCCL {
 public:
  static void AllReduce(const std::vector<NCCLElement>& ctxs);
  static void Broadcast(const std::vector<NCCLElement>& ctxs, int root);
  static void Reduce(const std::vector<NCCLElement>& ctxs, int root);
  static void AllGather(const std::vector<NCCLElement>& ctxs);
};

}
}
