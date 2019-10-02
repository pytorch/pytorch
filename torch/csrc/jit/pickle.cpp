#include <ATen/core/ivalue.h>
#include <ATen/ATen.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/pickle.h>
#include <torch/csrc/jit/pickler.h>

#include <torch/cuda.h>

namespace torch {
namespace jit {

// These are both defined in `torch/serialization.py`
const char* torch_save_magic_number =
    "\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19";
uint16_t protocol_version = 1001;

void pickle(
    std::function<void(const char* data_start, size_t data_len)> writer,
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table) {
  Pickler pickler(std::move(writer), tensor_table);
  pickler.protocol();
  pickler.pushIValue(ivalue);
  pickler.stop();
}

std::vector<char> pickle(
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table) {
  std::vector<char> data;

  pickle(
      [&](const char* bytes, size_t len) {
        data.insert(data.end(), bytes, bytes + len);
      },
      ivalue,
      tensor_table);

  return data;
}

// This has to live here instead of the C++ API to mirror torch.save since the
// mobile build excludes the C++ API
std::vector<char> pickle_save(const at::IValue& ivalue) {
  std::vector<char> data;

  auto writer = [&](const char* bytes, size_t len) {
    data.insert(data.end(), bytes, bytes + len);
  };

  jit::Pickler pickler(writer, /*tensor_table=*/nullptr);
  // Output data to match torch.save, see torch/serialization.py for details
  // Magic number (0x1950a86a20f9469cfc6c)
  pickler.protocol();
  pickler.pushLong(torch_save_magic_number);
  pickler.stop();

  // Protocol Version
  pickler.protocol();
  pickler.pushInt(protocol_version);
  pickler.stop();

  // sys_info, this isn't actually used in de-serialization so we can leave this
  // one empty
  pickler.protocol();
  pickler.pushEmptyDict();
  pickler.stop();

  jit::Pickler data_pickler(writer, /*tensor_table=*/nullptr);
  data_pickler.protocol();
  data_pickler.pushIValue(ivalue);
  data_pickler.stop();

  auto writeable_tensors = data_pickler.tensorData();

  c10::List<std::string> keys;
  keys.reserve(writeable_tensors.size());
  for (size_t i = 0; i < writeable_tensors.size(); i++) {
    keys.emplace_back(std::to_string(i));
  }
  jit::pickle(writer, keys);

  for (const auto& tensor_data : writeable_tensors) {
    const char* addr = tensor_data.data();
    size_t numel = tensor_data.numel();
    writer(reinterpret_cast<const char*>(&numel), sizeof(numel));
    writer(addr, tensor_data.sizeInBytes());
  }

  return data;
}

IValue pickle_load(const std::function<bool(char*, size_t)>& reader) {
  // This follows the loading of `serialization.py`. The pickle binary is
  // composed of 5 pickle archives (START ... STOP) catted together, followed
  // by the binary storage of each tensor storage.

  // Read the magic number
  std::string magic_number =
      Unpickler(reader, nullptr, nullptr).parse_ivalue().to<std::string>();
  TORCH_CHECK(
      magic_number == torch_save_magic_number,
      "The magic number read from the file was not correct");

  // Read the version number
  uint64_t version =
      Unpickler(reader, nullptr, nullptr).parse_ivalue().to<int64_t>();
  TORCH_CHECK(
      version == protocol_version,
      "Protocol version was incorrect, expected ",
      protocol_version,
      " but got ",
      version);

  // Read the system metadata number
  Unpickler(reader, nullptr, nullptr).parse_ivalue();

  // Read the main pickle data (the actual values, tensors will only have a key
  // to their storage)
  Unpickler data_pickle(reader, nullptr, nullptr);
  auto data = data_pickle.parse_ivalue();

  // Read a vector of storage keys that specifies the order their binaries are
  // saved
  auto storage_keys = Unpickler(reader, nullptr, nullptr)
                          .parse_ivalue();

  // Get all the storages encountered by the unpickler
  const auto& uninitialized_storages = data_pickle.uninitializedStorages();

  for (const auto& key : storage_keys.toGenericListRef()) {
    // Read each storage in order and set it from the file
    // Each storage is saved as [8 byte numel, binary data]
    auto item = uninitialized_storages.find(key.toStringRef());
    TORCH_INTERNAL_ASSERT(item != uninitialized_storages.end());
    const auto* storage_ptr = item->second.second;
    auto type = item->second.first;

    int64_t numel;
    reader(reinterpret_cast<char*>(&numel), sizeof(numel));
    // TORCH_CHECK(
    //     storage_ptr->numel() == numel,
    //     "Expected ",
    //     storage_ptr->numel(),
    //     " elements but found ",
    //     numel);

    size_t size = storage_ptr->numel() * storage_ptr->elementSize();
    char* dest = reinterpret_cast<char*>(storage_ptr->data());

    if (storage_ptr->device_type() == at::DeviceType::CUDA) {
      // std::cout << "Cuda bad!\n";
      std::vector<char> data;
      data.resize(size);
      // reader(data.data(), size);

      auto dest_data_ptr = storage_ptr->allocator()->allocate(storage_ptr->numel());

      // reader(reinterpret_cast<char*>(dest_data_ptr.get()), size);


      // auto type = storage_ptr->device();
      auto options = at::CPU(type).options();
      auto storage = at::Storage(
          at::CPU(type).typeMeta(),
          storage_ptr->numel(),
          /*allocator=*/at::getCPUAllocator(),
          /*resizable=*/false);
      reader(reinterpret_cast<char*>(storage.data()), size);
      at::Tensor tensor = at::empty({0}, options).set_(storage);
      tensor = tensor.to(storage_ptr->device(), tensor.scalar_type());

      // reader(dest, size);
      // at::DataPtr data_ptr(data.data(), storage_ptr->device());
      // storage_ptr->set_data_ptr(tensor.storage().data_ptr());
      std::cout << "I loaded\n";
      std::cout << tensor << "\n";
      // *storage_ptr->unsafeGetStorageImpl() = tensor.storage().unsafeReleaseStorageImpl();
      // storage_ptr->set_data_ptr(std::move(dest_data_ptr));

      // cudaMemcpy(dest, data.data(), size, cudaMemcpyHostToDevice);
    } else {
      reader(dest, size);
    }
    // cudaMemcpy(storage_ptr->data(), data, size, );
    // if (THCudaCheck(cudaMemcpy(THWStorage_(data)(LIBRARY_STATE storage), data, size * sizeof(scalar_t), cudaMemcpyHostToDevice));)
  }

  return data;
}

IValue unpickle(
    std::function<bool(char*, size_t)> reader,
    ClassResolver class_resolver,
    const std::vector<at::Tensor>* tensor_table) {
  Unpickler unpickler(
      std::move(reader), std::move(class_resolver), tensor_table);
  return unpickler.parse_ivalue();
}

IValue unpickle(
    const char* data,
    size_t size,
    ClassResolver class_resolver,
    const std::vector<at::Tensor>* tensor_table) {
  size_t bytes_read = 0;
  return unpickle(
      [&](char* buffer, size_t len) {
        if (bytes_read + len > size) {
          return false;
        }
        // Copy len bytes into buffer
        const char* start = data + bytes_read;
        std::memcpy(buffer, start, len);
        bytes_read += len;
        return true;
      },
      std::move(class_resolver),
      tensor_table);
}

} // namespace jit
} // namespace torch
