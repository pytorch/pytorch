#include <torch/csrc/distributed/rpc/macros.h>
#include <torch/csrc/distributed/rpc/tensorpipe_utils.h>

#ifdef USE_TENSORPIPE

#ifdef USE_CUDA_NOT_ROCM
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include <tensorpipe/tensorpipe.h>

namespace torch {
namespace distributed {
namespace rpc {
namespace {

// The TensorPipe agent splits the RPC message's information across multiple
// payloads. This allows the agent to provide the data to TensorPipe without
// performing a copy into a single contiguous buffer, and without storing it as
// metadata, which is less efficient.

// First come the rpc::Message::type() and ::id().
constexpr int kTpMessageTypeIdx = 0;
constexpr int kTpMessageIdIdx = 1;
// Then comes the rpc::Message::payload();
constexpr int kTpMessagePayloadIdx = 2;
// Last comes the pickle of rpc::Message::tensors() (with the tensors themselves
// stored as, well, tensors in the tensorpipe::Message).
constexpr int kTpMessagePickleIdx = 3;

inline c10::Device indexToDevice(c10::DeviceIndex index) {
  if (index == -1) {
    return c10::Device(at::kCPU);
  } else {
    return c10::Device(at::kCUDA, index);
  }
}

} // namespace

std::tuple<tensorpipe::Message, TensorpipeWriteBuffers> tensorpipeSerialize(
    Message&& rpcMessage,
    std::vector<c10::Device> devices,
    const std::shared_ptr<LazyStreamContext>& ctx) {
  tensorpipe::Message tpMessage;
  TensorpipeWriteBuffers buffers;

  // Metadata
  buffers.type = std::make_unique<MessageType>(rpcMessage.type());
  buffers.id = std::make_unique<int64_t>(rpcMessage.id());
  // kTpMessageTypeIdx = 0
  tpMessage.payloads.push_back(
      tensorpipe::Message::Payload{buffers.type.get(), sizeof(MessageType)});
  // kTpMessageIdIdx = 1
  tpMessage.payloads.push_back(
      tensorpipe::Message::Payload{buffers.id.get(), sizeof(int64_t)});

  // Payload
  buffers.payload = std::move(rpcMessage.payload());
  // TensorPipe uses the same Message class for both reading and writing, thus
  // it uses non-const pointers even though it doesn't modify them when writing.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  char* payloadPtr = const_cast<char*>(buffers.payload.data());
  // kTpMessagePayloadIdx = 2
  tpMessage.payloads.push_back(
      tensorpipe::Message::Payload{payloadPtr, buffers.payload.size()});

  {
    // The function below might allocate new tensors if there are Tensor views.
    // Apply stream guard here to include those Tensor allocation operations to
    // the streams.
    c10::MultiStreamGuard guard(
        ctx ? ctx->getReservedStreams() : ArrayRef<Stream>({}));
    // Tensors
    buffers.tensors = cloneSparseTensors(rpcMessage.tensors()).vec();
  }

  torch::jit::Pickler pickler([&](const void* buf, size_t sz) -> size_t {
    buffers.pickle.insert(
        buffers.pickle.end(),
        static_cast<const char*>(buf),
        static_cast<const char*>(buf) + sz);
    return sz;
  });
  pickler.protocol();
  pickler.pushIValue(buffers.tensors);
  pickler.stop();
  // kTpMessagePickleIdx = 3
  tpMessage.payloads.push_back(tensorpipe::Message::Payload{
      buffers.pickle.data(), buffers.pickle.size()});
  const auto& tensorDataVec = pickler.tensorData();
  for (size_t i = 0; i < tensorDataVec.size(); ++i) {
    // This is different from jit::getWriteableTensorData as it avoids copying
    // tensor to CPU.
    const auto& tensorData =
        jit::getWriteableTensorData(tensorDataVec[i], /* toCpu */ false);
    tensorpipe::Device targetDevice = devices.empty() || devices[i].is_cpu()
        ? tensorpipe::Device{tensorpipe::kCpuDeviceType, 0}
        : tensorpipe::Device{tensorpipe::kCudaDeviceType, devices[i].index()};

    // Enforce memory copy if tensor is created from torch::from_blob, means
    // that the tensor doesn't own the memory.
    if (!tensorData.storageHasDeleter()) {
      std::vector<char> storageData(
          tensorData.data(), tensorData.data() + tensorData.sizeInBytes());
      tensorpipe::CpuBuffer buffer;
      buffer.ptr = storageData.data();

      tensorpipe::Message::Tensor tensor;
      tensor.buffer = buffer;
      tensor.length = storageData.size();
      tensor.targetDevice = std::move(targetDevice);

      tpMessage.tensors.push_back(std::move(tensor));
      buffers.copiedTensors.push_back(std::move(storageData));
    } else {
      // TensorPipe uses the same Message class for both reading and writing, so
      // it uses non-const ptrs even though it doesn't modify them when writing.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      char* tensorPtr = const_cast<char*>(tensorData.data());
      if (tensorDataVec[i].device().is_cpu()) {
        tensorpipe::CpuBuffer buffer;
        buffer.ptr = tensorPtr;

        tensorpipe::Message::Tensor tensor;
        tensor.buffer = buffer;
        tensor.length = tensorData.sizeInBytes();
        tensor.targetDevice = std::move(targetDevice);

        tpMessage.tensors.push_back(std::move(tensor));
#ifdef USE_CUDA_NOT_ROCM
      } else if (tensorDataVec[i].device().is_cuda()) {
        auto stream =
            at::cuda::CUDAStream(ctx->getStream(tensorDataVec[i].device()));
        tensorpipe::CudaBuffer buffer;
        buffer.ptr = tensorPtr;
        buffer.stream = stream.stream();

        tensorpipe::Message::Tensor tensor;
        tensor.buffer = buffer;
        tensor.length = tensorData.sizeInBytes();
        tensor.targetDevice = std::move(targetDevice);

        tpMessage.tensors.push_back(std::move(tensor));
        // record tensor data ptrs on TensorPipe streams, so that the tensors
        // won't be destructed before TensorPipe finishing sending them.
        c10::cuda::CUDACachingAllocator::recordStream(
            tensorDataVec[i].storage().data_ptr(), stream);
#endif
      } else {
        TORCH_CHECK(
            false,
            "Attempting to send a Tensor with unexpected device type ",
            tensorDataVec[i].device());
      }
    }
  }

  return std::make_tuple(std::move(tpMessage), std::move(buffers));
}

std::pair<tensorpipe::Allocation, TensorpipeReadBuffers> tensorpipeAllocate(
    const tensorpipe::Descriptor& tpDescriptor,
    const std::shared_ptr<LazyStreamContext>& ctx) {
  tensorpipe::Allocation tpAllocation;
  TensorpipeReadBuffers buffers;

  TORCH_INTERNAL_ASSERT(
      tpDescriptor.payloads.size() == 4,
      "message expected to contain 4 payloads, whereas it contained ",
      tpDescriptor.payloads.size(),
      " payloads");
  tpAllocation.payloads.resize(tpDescriptor.payloads.size());

  TORCH_INTERNAL_ASSERT(
      tpDescriptor.payloads[kTpMessageTypeIdx].length == sizeof(MessageType),
      "first payload expected to contain ",
      sizeof(MessageType),
      " bytes, whereas it contained ",
      tpDescriptor.payloads[kTpMessageTypeIdx].length,
      " bytes");
  buffers.type = std::make_unique<MessageType>();
  tpAllocation.payloads[kTpMessageTypeIdx].data = buffers.type.get();

  TORCH_INTERNAL_ASSERT(
      tpDescriptor.payloads[kTpMessageIdIdx].length == sizeof(int64_t),
      "second payload expected to contain ",
      sizeof(int64_t),
      " bytes, whereas it contained ",
      tpDescriptor.payloads[kTpMessageIdIdx].length,
      " bytes");
  buffers.id = std::make_unique<int64_t>();
  tpAllocation.payloads[kTpMessageIdIdx].data = buffers.id.get();

  // FIXME The two resizes below zero out the vectors, which is not needed.
  buffers.payload.resize(tpDescriptor.payloads[kTpMessagePayloadIdx].length);
  tpAllocation.payloads[kTpMessagePayloadIdx].data = buffers.payload.data();

  buffers.pickle.resize(tpDescriptor.payloads[kTpMessagePickleIdx].length);
  tpAllocation.payloads[kTpMessagePickleIdx].data = buffers.pickle.data();

  size_t numTensors = tpDescriptor.tensors.size();
  tpAllocation.tensors.resize(numTensors);
  for (size_t tensorIdx = 0; tensorIdx < numTensors; ++tensorIdx) {
    const tensorpipe::Descriptor::Tensor& tensor =
        tpDescriptor.tensors[tensorIdx];
    TORCH_INTERNAL_ASSERT(tensor.targetDevice.has_value());
    if (tensor.targetDevice->type == tensorpipe::kCpuDeviceType) {
      buffers.tensors.emplace_back(
          at::getCPUAllocator()->allocate(tensor.length));
      tensorpipe::CpuBuffer buffer;
      buffer.ptr = buffers.tensors.back().get();
      tpAllocation.tensors[tensorIdx].buffer = buffer;
#ifdef USE_CUDA_NOT_ROCM
    } else if (tensor.targetDevice->type == tensorpipe::kCudaDeviceType) {
      c10::Device device(c10::kCUDA, tensor.targetDevice->index);
      auto stream = at::cuda::CUDAStream(ctx->getStream(device));
      // CUDACachingAllocator will call recordStream accordingly on the current
      // stream.
      at::cuda::CUDAStreamGuard guard(stream);
      buffers.tensors.emplace_back(
          c10::cuda::CUDACachingAllocator::get()->allocate(tensor.length));
      tensorpipe::CudaBuffer buffer;
      buffer.ptr = buffers.tensors.back().get();
      buffer.stream = stream.stream();
      tpAllocation.tensors[tensorIdx].buffer = buffer;
#endif
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unrecognized TensorPipe buffer type.");
    }
  }

  return {std::move(tpAllocation), std::move(buffers)};
}

Message tensorpipeDeserialize(
    tensorpipe::Descriptor&& tpDescriptor,
    TensorpipeReadBuffers&& buffers) {
  // Tensors
  std::vector<at::Tensor> tensors;
  const char* pickleData = buffers.pickle.data();
  size_t pickleLen = buffers.pickle.size();
  size_t picklePos = 0;
  auto pickleReadFunc = [&](char* buf, size_t n) -> size_t {
    if (picklePos >= pickleLen || n == 0) {
      return 0;
    }
    size_t toCopy = std::min(picklePos + n, pickleLen) - picklePos;
    memcpy(buf, pickleData + picklePos, toCopy);
    picklePos += toCopy;
    return toCopy;
  };
  auto tensorReadFunc = [&](const std::string& ename) -> at::DataPtr {
    unsigned long index = std::stoul(ename);
    return std::move(buffers.tensors.at(index));
  };

  // No need to pass typeResolver here, as it always processes string and
  // tensors only
  torch::jit::Unpickler unpickler(
      pickleReadFunc,
      nullptr,
      nullptr,
      tensorReadFunc,
      {},
      /* use_storage_device*/ true);

  auto ival = unpickler.parse_ivalue();
  for (auto&& t : ival.toTensorList()) {
    tensors.emplace_back(std::move(t));
  }

  for (size_t i = 0; i < tpDescriptor.tensors.size(); ++i) {
    auto& tensor = tpDescriptor.tensors[i];
    if (tensor.targetDevice.has_value() &&
        tensor.targetDevice->type == tensorpipe::kCudaDeviceType) {
      TORCH_INTERNAL_ASSERT(
          tensors[i].device() == indexToDevice(tensor.targetDevice->index),
          "Tensor ",
          i,
          " in message ",
          *buffers.id,
          " was expected to be received on device ",
          tensor.targetDevice->index,
          ", but got it on ",
          tensors[i].device());
    }
  }

  return Message(
      std::move(buffers.payload),
      std::move(tensors),
      *buffers.type,
      *buffers.id);
}
} // namespace rpc
} // namespace distributed
} // namespace torch

#endif // USE_TENSORPIPE
