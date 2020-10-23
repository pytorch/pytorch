#pragma once

#ifdef USE_TENSORPIPE

#include <torch/csrc/distributed/rpc/utils.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace tensorpipe {
class Message;
} // namespace tensorpipe

namespace torch {
namespace distributed {
namespace rpc {

#ifdef USE_CUDA
using at::cuda::CUDAStream;
#endif

struct DevicesContext {

  DevicesContext(const DevicesContext& other) = default;
  DevicesContext(DevicesContext&& other) = default;

  DevicesContext& operator=(const DevicesContext& rhs) = default;
  DevicesContext& operator=(DevicesContext&& rhs) & = default;

  void synchronize() {}

#ifndef USE_CUDA
  explicit DevicesContext(bool noCuda=true) : noCuda_(noCuda) {}
#else
  // Use the noCuda arg to disable streams management when deviceMaps are not
  // set.
  explicit DevicesContext(bool noCuda=true) : noCuda_(noCuda) {
    if (!noCuda_) {
      auto deviceNum = at::cuda::device_count();
      streams_.reserve(deviceNum);
      for (c10::DeviceIndex idx = 0; idx < deviceNum; ++idx) {
        streams_.emplace_back(at::cuda::getStreamFromPool(
          /* isHighPriority */ false, /* device */ idx));
      }
    }
  }

  inline const std::vector<CUDAStream>& streams() const {
    return streams_;
  }

 private:
  std::vector<CUDAStream> streams_;
#endif

 private:
  const bool noCuda_;
};


struct DevicesStateGuard {

#ifdef USE_CUDA
  DevicesStateGuard(const DevicesContext& ctx) {
    const auto& streams = ctx.streams();
    std::vector<CUDAStream> prevStreams_;
    prevStreams_.reserve(streams.size());
    for (const auto& stream: streams) {
      prevStreams_.emplace_back(
          at::cuda::getCurrentCUDAStream(stream.device_index()));
      at::cuda::setCurrentCUDAStream(stream);
    }
  }

  ~DevicesStateGuard() noexcept {
    for (auto& stream : prevStreams_) {
      at::cuda::setCurrentCUDAStream(std::move(stream));
    }
  }
#else
  DevicesStateGuard(DevicesContext /* unused */) {};
#endif

  DevicesStateGuard(const DevicesStateGuard& other) = delete;
  DevicesStateGuard(DevicesStateGuard&& other) = delete;
  DevicesStateGuard& operator=(const DevicesStateGuard& rhs) = delete;
  DevicesStateGuard& operator=(DevicesStateGuard&& rhs) = delete;

 private:
#ifdef USE_CUDA
  std::vector<CUDAStream> prevStreams_;
#endif
};


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
    const DevicesContext& = DevicesContext(/* noCuda */ true));

// Allocate the buffers that will hold the incoming data. They will be managed
// by the returned holder, which must be kept alive until the asynchronous read
// has finished. Pointers to these buffers will be stored in-place in the
// TensorPipe message.
TORCH_API TensorpipeReadBuffers
tensorpipeAllocate(
    tensorpipe::Message& tpMessage,
    const DevicesContext& ctx = DevicesContext(/* noCuda */ true));

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
