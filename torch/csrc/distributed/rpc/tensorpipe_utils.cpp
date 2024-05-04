#include <torch/csrc/distributed/rpc/tensorpipe_utils.h>

#ifdef USE_TENSORPIPE

#include <c10/util/irange.h>
#include <limits>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated")
#include <tensorpipe/tensorpipe.h>
C10_DIAGNOSTIC_POP()

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

class TensorpipeCpuConverter : public TensorpipeDeviceTypeConverter {
 public:
  c10::optional<std::vector<char>> prepareTensorForSending(
      const c10::Storage& storage,
      const std::vector<c10::Stream>& /* streams */,
      tensorpipe::Message& message) const override {
    // Enforce memory copy if tensor is created from torch::from_blob, means
    // that the tensor doesn't own the memory.
    bool storageHasDeleter = storage.data_ptr().get_context() != nullptr;
    if (!storageHasDeleter) {
      std::vector<char> storageData(
          static_cast<const char*>(storage.data()),
          static_cast<const char*>(storage.data()) + storage.nbytes());

      tensorpipe::CpuBuffer buffer;
      buffer.ptr = storageData.data();

      tensorpipe::Message::Tensor tensor;
      tensor.buffer = buffer;
      tensor.length = storageData.size();

      message.tensors.push_back(std::move(tensor));

      return c10::make_optional(std::move(storageData));
    } else {
      tensorpipe::CpuBuffer buffer;
      buffer.ptr = static_cast<char*>(storage.mutable_data());

      tensorpipe::Message::Tensor tensor;
      tensor.buffer = buffer;
      tensor.length = storage.nbytes();

      message.tensors.push_back(std::move(tensor));

      return c10::nullopt;
    }
  }

  at::DataPtr allocateTensorForReceiving(
      c10::DeviceIndex /* deviceIndex */,
      size_t length,
      const std::vector<c10::Stream>& /* streams */,
      tensorpipe::Allocation& allocation) const override {
    at::DataPtr dataPtr = at::getCPUAllocator()->allocate(length);

    tensorpipe::CpuBuffer buffer;
    buffer.ptr = dataPtr.get();

    tensorpipe::Allocation::Tensor tensor;
    tensor.buffer = buffer;

    allocation.tensors.push_back(std::move(tensor));

    return dataPtr;
  }
};

C10_REGISTER_TENSORPIPE_DEVICE_TYPE_CONVERTER(CPU, TensorpipeCpuConverter);

c10::DeviceType convertDeviceType(const std::string& tpDeviceType) {
  if (tpDeviceType == tensorpipe::kCpuDeviceType) {
    return c10::kCPU;
  } else if (tpDeviceType == tensorpipe::kCudaDeviceType) {
    return c10::kCUDA;
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unrecognized TensorPipe buffer type.");
  }
}

} // namespace

// As the vector of streams will typically be very small (1-8 items) we expect
// a linear search to be as fast (or faster?) than if we used a hashmap.
const c10::Stream& getStreamForDevice(
    const std::vector<c10::Stream>& streams,
    const c10::Device& device) {
  for (const c10::Stream& stream : streams) {
    if (stream.device() == device) {
      return stream;
    }
  }
  TORCH_INTERNAL_ASSERT(false, "No stream found for device ", device);
}

std::array<
    std::atomic<const TensorpipeDeviceTypeConverter*>,
    static_cast<size_t>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)>
    device_type_converter_registry;

TensorpipeDeviceTypeConverterRegistrar::TensorpipeDeviceTypeConverterRegistrar(
    DeviceType type,
    const TensorpipeDeviceTypeConverter* impl) {
  device_type_converter_registry[static_cast<size_t>(type)].store(impl);
}

std::tuple<tensorpipe::Message, TensorpipeWriteBuffers> tensorpipeSerialize(
    c10::intrusive_ptr<Message> rpcMessage,
    std::vector<c10::Device> devices,
    const std::vector<c10::Stream>& streams) {
  tensorpipe::Message tpMessage;
  TensorpipeWriteBuffers buffers;

  // Metadata
  buffers.type = std::make_unique<MessageType>(rpcMessage->type());
  buffers.id = std::make_unique<int64_t>(rpcMessage->id());
  // kTpMessageTypeIdx = 0
  tpMessage.payloads.push_back(
      tensorpipe::Message::Payload{buffers.type.get(), sizeof(MessageType)});
  // kTpMessageIdIdx = 1
  tpMessage.payloads.push_back(
      tensorpipe::Message::Payload{buffers.id.get(), sizeof(int64_t)});

  // Payload
  buffers.payload = std::move(rpcMessage->payload());
  // TensorPipe uses the same Message class for both reading and writing, thus
  // it uses non-const pointers even though it doesn't modify them when writing.
  char* payloadPtr = buffers.payload.data();
  // kTpMessagePayloadIdx = 2
  tpMessage.payloads.push_back(
      tensorpipe::Message::Payload{payloadPtr, buffers.payload.size()});

  {
    // The function below might allocate new tensors if there are Tensor views.
    // Apply stream guard here to include those Tensor allocation operations to
    // the streams.
    c10::MultiStreamGuard guard(streams);
    // Tensors
    buffers.tensors = cloneSparseTensors(rpcMessage->tensors()).vec();
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
  const std::vector<torch::Tensor>& tensorDataVec = pickler.tensorData();
  tpMessage.tensors.reserve(tensorDataVec.size());
  for (const auto i : c10::irange(tensorDataVec.size())) {
    const torch::Tensor& tensor = tensorDataVec[i];

    const TensorpipeDeviceTypeConverter* converter =
        getDeviceTypeConverter(tensor.device().type());
    TORCH_CHECK(
        converter != nullptr,
        "Attempting to send a Tensor with unexpected device type ",
        tensor.device());

    TORCH_INTERNAL_ASSERT(tpMessage.tensors.size() == i);
    c10::optional<std::vector<char>> maybeCopiedTensor =
        converter->prepareTensorForSending(
            tensor.storage(), streams, tpMessage);
    TORCH_INTERNAL_ASSERT(tpMessage.tensors.size() == i + 1);

    tensorpipe::Device targetDevice = devices.empty() || devices[i].is_cpu()
        ? tensorpipe::Device{tensorpipe::kCpuDeviceType, 0}
        : tensorpipe::Device{tensorpipe::kCudaDeviceType, devices[i].index()};
    tpMessage.tensors.back().targetDevice = std::move(targetDevice);

    if (maybeCopiedTensor.has_value()) {
      buffers.copiedTensors.push_back(std::move(maybeCopiedTensor).value());
    }
  }

  return std::make_tuple(std::move(tpMessage), std::move(buffers));
}

std::pair<tensorpipe::Allocation, TensorpipeReadBuffers> tensorpipeAllocate(
    const tensorpipe::Descriptor& tpDescriptor,
    const std::vector<c10::Stream>& streams) {
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
  tpAllocation.tensors.reserve(numTensors);
  for (const auto tensorIdx : c10::irange(numTensors)) {
    const tensorpipe::Descriptor::Tensor& tensor =
        tpDescriptor.tensors[tensorIdx];
    TORCH_INTERNAL_ASSERT(tensor.targetDevice.has_value());
    c10::DeviceType targetDeviceType =
        convertDeviceType(tensor.targetDevice->type);

    const TensorpipeDeviceTypeConverter* converter =
        getDeviceTypeConverter(targetDeviceType);
    TORCH_INTERNAL_ASSERT(
        converter != nullptr,
        "Attempting to receive a Tensor with unexpected device type ",
        targetDeviceType);

    TORCH_INTERNAL_ASSERT(tpAllocation.tensors.size() == tensorIdx);
    TORCH_INTERNAL_ASSERT(
        tensor.targetDevice->index <=
        std::numeric_limits<c10::DeviceIndex>::max());
    at::DataPtr dataPtr = converter->allocateTensorForReceiving(
        static_cast<c10::DeviceIndex>(tensor.targetDevice->index),
        tensor.length,
        streams,
        tpAllocation);
    TORCH_INTERNAL_ASSERT(tpAllocation.tensors.size() == tensorIdx + 1);

    buffers.tensors.push_back(std::move(dataPtr));
  }

  return {std::move(tpAllocation), std::move(buffers)};
}

c10::intrusive_ptr<Message> tensorpipeDeserialize(
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

  for (const auto i : c10::irange(tpDescriptor.tensors.size())) {
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

  return c10::make_intrusive<Message>(
      std::move(buffers.payload),
      std::move(tensors),
      *buffers.type,
      *buffers.id);
}
} // namespace rpc
} // namespace distributed
} // namespace torch

#endif // USE_TENSORPIPE
