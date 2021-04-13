#pragma once

#ifdef USE_TENSORPIPE

#include <torch/csrc/distributed/rpc/macros.h>
#include <torch/csrc/distributed/rpc/utils.h>

#ifdef USE_CUDA_NOT_ROCM
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace tensorpipe {
class Message;
} // namespace tensorpipe

namespace torch {
namespace distributed {
namespace rpc {

#ifdef USE_CUDA_NOT_ROCM
using at::cuda::CUDAStream;
#endif

// A general device context class for both CPU and CUDA. If CUDA is not
// available, all CUDA-related methods will be no-ops.
struct TORCH_API LazyStreamContext {
  LazyStreamContext(const LazyStreamContext& other) = delete;
  LazyStreamContext(LazyStreamContext&& other) = delete;
  LazyStreamContext& operator=(const LazyStreamContext& rhs) = delete;
  LazyStreamContext& operator=(LazyStreamContext&& rhs) & = delete;

  LazyStreamContext() = default;
  virtual ~LazyStreamContext() = default;
  virtual void waitForCurrentStreams(const std::vector<torch::Tensor>& = {}) {}
  virtual std::set<c10::DeviceIndex> devices() const {
    return {};
  }

#ifdef USE_CUDA_NOT_ROCM
  virtual std::vector<CUDAStream> getReservedStreams() const {
    throw std::runtime_error(
        "Attempting to access CUDA streams, but torch is not built with CUDA");
  }

  virtual CUDAStream getStream(c10::DeviceIndex index) {
    throw std::runtime_error(c10::str(
        "Attempting to access CUDA stream of device ",
        index,
        ", but torch is not built with CUDA"));
  }
#endif
};

#ifndef USE_CUDA_NOT_ROCM

// CUDA is not available, use CPU device context.
inline std::shared_ptr<LazyStreamContext> createLazyStreamContext() {
  return std::make_shared<LazyStreamContext>();
}

#else

// CUDA is available. Implement CUDA-related operations.
struct TORCH_CUDA_CPP_API CudaLazyStreamContext : public LazyStreamContext {
  using LazyStreamContext::LazyStreamContext;

  // let streams in this context wiat for current streams.
  void waitForCurrentStreams(
      const std::vector<torch::Tensor>& tensors = {}) override {
    for (const auto& tensor : tensors) {
      if (tensor.is_cuda()) {
        getStream(tensor.device().index());
      }
    }

    for (const auto& entry : streams_) {
      at::cuda::CUDAEvent event;
      event.record(at::cuda::getCurrentCUDAStream(entry.first));
      event.block(entry.second);
    }
  }

  // get all streams used in this context
  std::vector<CUDAStream> getReservedStreams() const override {
    std::vector<CUDAStream> reservedStreams;
    reservedStreams.reserve(streams_.size());
    for (const auto& entry : streams_) {
      reservedStreams.push_back(entry.second);
    }
    return reservedStreams;
  }

  // get a stream for the given device. If it is the first time using that
  // device, allocate a new stream and store it in the map.
  CUDAStream getStream(c10::DeviceIndex index) override {
    auto iter = streams_.find(index);
    if (iter == streams_.end()) {
      auto cudaStream = at::cuda::getStreamFromPool(
          /* isHighPriority */ false, /* device */ index);
      streams_.emplace(index, cudaStream);
      return cudaStream;
    } else {
      return iter->second;
    }
  }

  std::set<c10::DeviceIndex> devices() const override {
    std::set<c10::DeviceIndex> devices;
    for (const auto& entry : streams_) {
      devices.insert(entry.first);
    }
    return devices;
  }

 private:
  std::unordered_map<c10::DeviceIndex, CUDAStream> streams_;
};

inline std::shared_ptr<LazyStreamContext> createLazyStreamContext() {
  return std::make_shared<CudaLazyStreamContext>();
}

#endif

// A struct that holds pointers that keep alive all the memory that will be
// accessed by TensorPipe during a write operation.
struct TensorpipeWriteBuffers {
  // Allocate on heap so pointers stay valid as we move the holder.
  std::unique_ptr<MessageType> type;
  std::unique_ptr<int64_t> id;
  std::vector<char> payload;
  std::vector<char> pickle;
  // This contains the original tensors and the clones of the sparse tensors.
  std::vector<torch::Tensor> tensors;
  // This contains the copies of the data of the tensors that didn't own their
  // memory, e.g., the ones created from torch::from_blob() with no deleter.
  std::vector<std::vector<char>> copiedTensors;
};

// A struct that holds pointers that keep alive all the memory that will be
// accessed by TensorPipe during a read operation.
struct TensorpipeReadBuffers {
  // Allocate on heap so pointers stay valid as we move the holder.
  std::unique_ptr<MessageType> type;
  std::unique_ptr<int64_t> id;
  std::vector<char> payload;
  std::vector<char> pickle;
  std::vector<c10::DataPtr> tensors;
};

// Convert an RPC message into a TensorPipe message, plus a holder to all the
// data that must be kept alive while the write is performed asynchronously.
TORCH_API std::tuple<tensorpipe::Message, TensorpipeWriteBuffers>
tensorpipeSerialize(
    Message&& rpcMessage,
    std::vector<c10::DeviceIndex> devices = {},
    const std::shared_ptr<LazyStreamContext>& =
        std::make_shared<LazyStreamContext>());

// Allocate the buffers that will hold the incoming data. They will be managed
// by the returned holder, which must be kept alive until the asynchronous read
// has finished. Pointers to these buffers will be stored in-place in the
// TensorPipe message.
TORCH_API TensorpipeReadBuffers tensorpipeAllocate(
    tensorpipe::Message& tpMessage,
    const std::shared_ptr<LazyStreamContext>& ctx =
        std::make_shared<LazyStreamContext>());

// Convert a TensorPipe message back into an RPC message. This requires the data
// to be available and can thus only be performed once the asynchronous read has
// completed. The holder can be destroyed once this function returns.
TORCH_API Message tensorpipeDeserialize(
    tensorpipe::Message&& tpMessage,
    TensorpipeReadBuffers&& holder);

} // namespace rpc
} // namespace distributed
} // namespace torch

#endif // USE_TENSORPIPE
