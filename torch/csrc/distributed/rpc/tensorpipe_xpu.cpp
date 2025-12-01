#include <torch/csrc/distributed/rpc/tensorpipe_agent.h>
#include <torch/csrc/distributed/rpc/tensorpipe_utils.h>

#if defined(USE_TENSORPIPE)

#include <c10/core/StreamGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated")
#include <tensorpipe/tensorpipe.h>
#include <tensorpipe/tensorpipe_xpu.h>
C10_DIAGNOSTIC_POP()

namespace torch::distributed::rpc {
namespace {

std::unique_ptr<ChannelRegistration> makeXpuBasicChannel() {
  auto context = tensorpipe::channel::xpu_basic::create(
      tensorpipe::channel::basic::create());
  return std::make_unique<ChannelRegistration>(
      ChannelRegistration{std::move(context), kXpuBasicChannelPriority});
}

// The xpu_basic is the fallback channel for GPU-to-GPU comm
C10_REGISTER_CREATOR(TensorPipeChannelRegistry, xpu_basic, makeXpuBasicChannel)

class TensorpipeXpuConverter : public TensorpipeDeviceTypeConverter {
 public:
  std::optional<std::vector<char>> prepareTensorForSending(
      const c10::Storage& storage,
      const std::vector<c10::Stream>& streams,
      tensorpipe::Message& message) const override {
    auto stream =
        at::xpu::XPUStream(getStreamForDevice(streams, storage.device()));
    // record tensor data ptrs on TensorPipe streams, so that the tensors
    // won't be destructed before TensorPipe finishing sending them.
    c10::xpu::XPUCachingAllocator::recordStream(storage.data_ptr(), stream);

    tensorpipe::XpuBuffer buffer;
    buffer.ptr = static_cast<char*>(storage.mutable_data());
    buffer.queue = &stream.queue();

    tensorpipe::Message::Tensor tensor;
    tensor.buffer = buffer;
    tensor.length = storage.nbytes();
    int index = static_cast<int>(storage.device().index());
    std::string name = c10::DeviceTypeName(storage.device().type(), true);
    tensor.targetDevice = tensorpipe::Device(name, index);

    message.tensors.push_back(std::move(tensor));

    return std::nullopt;
  }

  at::DataPtr allocateTensorForReceiving(
      c10::DeviceIndex deviceIndex,
      size_t length,
      const std::vector<c10::Stream>& streams,
      tensorpipe::Allocation& allocation) const override {
    c10::Device device(c10::kXPU, deviceIndex);
    at::xpu::XPUStream stream(getStreamForDevice(streams, device));
    // XPUCachingAllocator will call recordStream accordingly on the current
    // stream.
    c10::StreamGuard guard(stream);
    at::DataPtr dataPtr =
        c10::xpu::XPUCachingAllocator::get()->allocate(length);

    tensorpipe::XpuBuffer buffer;
    buffer.ptr = dataPtr.get();
    buffer.queue = &stream.queue();

    tensorpipe::Allocation::Tensor tensor;
    tensor.buffer = buffer;

    allocation.tensors.push_back(tensor);

    return dataPtr;
  }
};

C10_REGISTER_TENSORPIPE_DEVICE_TYPE_CONVERTER(XPU, TensorpipeXpuConverter)

} // namespace
} // namespace torch::distributed::rpc

#endif
