#include <torch/csrc/distributed/rpc/tensorpipe_agent.h>
#include <torch/csrc/distributed/rpc/tensorpipe_utils.h>

#if defined(USE_TENSORPIPE) && !defined(USE_ROCM)

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wextra-semi")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated")
#include <tensorpipe/tensorpipe.h>
#include <tensorpipe/tensorpipe_cuda.h>
C10_DIAGNOSTIC_POP()
C10_DIAGNOSTIC_POP()

namespace torch::distributed::rpc {
namespace {

#if TENSORPIPE_HAS_CUDA_IPC_CHANNEL

std::unique_ptr<ChannelRegistration> makeCudaIpcChannel() {
  auto context = tensorpipe::channel::cuda_ipc::create();
  return std::make_unique<ChannelRegistration>(
      ChannelRegistration{std::move(context), kCudaIpcChannelPriority});
}

// The cuda_ipc channels use cudaMemcpy to transmit CUDA tensor across processes
C10_REGISTER_CREATOR(TensorPipeChannelRegistry, cuda_ipc, makeCudaIpcChannel)

#endif

#if TENSORPIPE_HAS_CUDA_GDR_CHANNEL

std::unique_ptr<ChannelRegistration> makeCudaGdrChannel() {
  auto context = tensorpipe::channel::cuda_gdr::create();
  return std::make_unique<ChannelRegistration>(
      ChannelRegistration{std::move(context), kCudaGdrChannelPriority});
}

// The cuda_gdr channel sends CUDA memory over InfiniBand using GPUDirect RDMA.
// It directly registers the user-provided tensor with libibverbs, an operation
// which is expensive the first time, but it then caches the registration in
// order to amortize the cost and get low latency for subsequent transfers. A
// ready-to-send/ready-to-receive handshake is still needed before the transfer
// in order to ensure readiness and to agree on the device indices and thus the
// queue pair to use. It automatically pairs each GPU to the "closest" NIC if
// there are multiple of them (closest = longest prefix match in PCI tree).
C10_REGISTER_CREATOR(TensorPipeChannelRegistry, cuda_gdr, makeCudaGdrChannel)

#endif

std::unique_ptr<ChannelRegistration> makeCudaXthChannel() {
  auto context = tensorpipe::channel::cuda_xth::create();
  return std::make_unique<ChannelRegistration>(
      ChannelRegistration{std::move(context), kCudaXthChannelPriority});
}

// The cuda_xth channel supports same-process GPU-to-GPU comm
C10_REGISTER_CREATOR(TensorPipeChannelRegistry, cuda_xth, makeCudaXthChannel)

std::unique_ptr<ChannelRegistration> makeCudaBasicChannel() {
  auto context = tensorpipe::channel::cuda_basic::create(
      tensorpipe::channel::basic::create());
  return std::make_unique<ChannelRegistration>(
      ChannelRegistration{std::move(context), kCudaBasicChannelPriority});
}

// The cuda_basic is the fallback channel for GPU-to-GPU comm
C10_REGISTER_CREATOR(
    TensorPipeChannelRegistry,
    cuda_basic,
    makeCudaBasicChannel)

class TensorpipeCudaConverter : public TensorpipeDeviceTypeConverter {
 public:
  std::optional<std::vector<char>> prepareTensorForSending(
      const c10::Storage& storage,
      const std::vector<c10::Stream>& streams,
      tensorpipe::Message& message) const override {
    auto stream =
        at::cuda::CUDAStream(getStreamForDevice(streams, storage.device()));
    // record tensor data ptrs on TensorPipe streams, so that the tensors
    // won't be destructed before TensorPipe finishing sending them.
    c10::cuda::CUDACachingAllocator::recordStream(storage.data_ptr(), stream);

    tensorpipe::CudaBuffer buffer;
    buffer.ptr = static_cast<char*>(storage.mutable_data());
    buffer.stream = stream.stream();

    tensorpipe::Message::Tensor tensor;
    tensor.buffer = buffer;
    tensor.length = storage.nbytes();

    message.tensors.push_back(std::move(tensor));

    return std::nullopt;
  }

  at::DataPtr allocateTensorForReceiving(
      c10::DeviceIndex deviceIndex,
      size_t length,
      const std::vector<c10::Stream>& streams,
      tensorpipe::Allocation& allocation) const override {
    c10::Device device(c10::kCUDA, deviceIndex);
    at::cuda::CUDAStream stream(getStreamForDevice(streams, device));
    // CUDACachingAllocator will call recordStream accordingly on the current
    // stream.
    at::cuda::CUDAStreamGuard guard(stream);
    at::DataPtr dataPtr =
        c10::cuda::CUDACachingAllocator::get()->allocate(length);

    tensorpipe::CudaBuffer buffer;
    buffer.ptr = dataPtr.get();
    buffer.stream = stream.stream();

    tensorpipe::Allocation::Tensor tensor;
    tensor.buffer = buffer;

    allocation.tensors.push_back(tensor);

    return dataPtr;
  }
};

C10_REGISTER_TENSORPIPE_DEVICE_TYPE_CONVERTER(CUDA, TensorpipeCudaConverter)

} // namespace
} // namespace torch::distributed::rpc

#endif
