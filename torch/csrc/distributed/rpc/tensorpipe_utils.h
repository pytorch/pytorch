#pragma once

#ifdef USE_TENSORPIPE

#include <torch/csrc/distributed/rpc/macros.h>
#include <torch/csrc/distributed/rpc/utils.h>

#ifdef USE_CUDA_NOT_ROCM
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
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


struct FullDeviceContext {

  FullDeviceContext(const FullDeviceContext& other) = delete;
  FullDeviceContext(FullDeviceContext&& other) = delete;

  FullDeviceContext& operator=(const FullDeviceContext& rhs) = delete;
  FullDeviceContext& operator=(FullDeviceContext&& rhs) & = delete;

  explicit FullDeviceContext(bool /* unused */) {}
  virtual void recordDataPtrs(
      const std::vector<c10::DataPtr>& dataPtrs) const {}
  virtual void recordTensors(const std::vector<torch::Tensor>& tensors) const {}
  virtual void blockCurrentStreams() const {}
  virtual void waitForCurrentStreams() const {}
  virtual void synchronize() const {}

#ifdef USE_CUDA_NOT_ROCM
  virtual const std::vector<CUDAStream>& streams() const {
    throw std::runtime_error(
        "Attempting to access CUDA streams, but torch is not built with CUDA");
  }
#endif

};

#ifndef USE_CUDA_NOT_ROCM

inline std::shared_ptr<FullDeviceContext> createFullDeviceContext(bool noCuda) {
  return std::make_shared<FullDeviceContext>(noCuda);
}

#else

struct CudaFullDeviceContext : public FullDeviceContext {

  // Use the noCuda arg to disable streams management when deviceMaps are not
  // set.
  explicit CudaFullDeviceContext(bool noCuda) : FullDeviceContext(noCuda) {
    if (!noCuda) {
      auto deviceNum = at::cuda::device_count();
      streams_.reserve(deviceNum);
      for (c10::DeviceIndex idx = 0; idx < deviceNum; ++idx) {
        streams_.emplace_back(at::cuda::getStreamFromPool(
          /* isHighPriority */ false, /* device */ idx));
      }
    }
  }

  void recordDataPtrs(const std::vector<c10::DataPtr>& dataPtrs) const override;
  void recordTensors(const std::vector<torch::Tensor>& tensors) const override;
  void blockCurrentStreams() const override;
  void waitForCurrentStreams() const override;
  void synchronize() const override;
  const std::vector<CUDAStream>& streams() const override;

 private:
  std::vector<CUDAStream> streams_;
};

inline std::shared_ptr<FullDeviceContext> createFullDeviceContext(bool noCuda) {
  return std::make_shared<CudaFullDeviceContext>(noCuda);
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
    const std::shared_ptr<FullDeviceContext>& =
        std::make_shared<FullDeviceContext>(/* noCuda */ true));

// Allocate the buffers that will hold the incoming data. They will be managed
// by the returned holder, which must be kept alive until the asynchronous read
// has finished. Pointers to these buffers will be stored in-place in the
// TensorPipe message.
TORCH_API TensorpipeReadBuffers
tensorpipeAllocate(
    tensorpipe::Message& tpMessage,
    const std::shared_ptr<FullDeviceContext>& ctx =
        std::make_shared<FullDeviceContext>(/* noCuda */ true));

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
