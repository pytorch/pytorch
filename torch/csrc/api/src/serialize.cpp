#include <torch/csrc/jit/pickle.h>
#include <torch/csrc/jit/pickler.h>
#include <torch/serialize.h>

#include <vector>

namespace torch {

// These are both defined in `torch/serialization.py`
const char* torch_save_magic_number =
    "\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19";
uint16_t protocol_version = 1001;

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
  IValue dict_ivalue =
      c10::impl::GenericDict(c10::impl::deprecatedUntypedDict());
  pickler.pushDict(dict_ivalue);
  pickler.stop();

  std::vector<at::Tensor> tensors;
  jit::pickle(writer, ivalue, &tensors);

  std::vector<at::IValue> keys;
  keys.reserve(tensors.size());
  std::vector<TypePtr> types(tensors.size(), StringType::get());

  // Each unique storage should only be saved 1 time
  std::unordered_set<const void*> memoized_storages;

  for (size_t i = 0; i < tensors.size(); i++) {
    void* addr = tensors.at(i).storage().unsafeGetStorageImpl();
    if (memoized_storages.count(addr) > 0) {
      continue;
    }
    keys.emplace_back(std::to_string(i));
    memoized_storages.insert(addr);
  }
  memoized_storages.clear();

  auto keys_tuple = at::ivalue::Tuple::create(keys, TupleType::create(types));
  jit::pickle(writer, keys_tuple, &tensors);

  for (const auto& tensor : tensors) {
    void* addr = tensor.storage().unsafeGetStorageImpl();
    if (memoized_storages.count(addr) > 0) {
      continue;
    }
    auto data = jit::getWriteableTensorData(tensor);
    size_t numel = data.numel();
    writer(reinterpret_cast<const char*>(&numel), sizeof(numel));
    writer(data.data(), data.sizeInBytes());
    memoized_storages.insert(addr);
  }

  return data;
}

} // namespace torch
