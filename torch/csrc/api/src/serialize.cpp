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

  jit::Pickler data_pickler(writer, /*tensor_table=*/nullptr);
  data_pickler.protocol();
  data_pickler.pushIValue(ivalue);
  data_pickler.stop();

  auto writeable_tensors = data_pickler.tensorData();

  std::vector<at::IValue> keys;
  keys.reserve(writeable_tensors.size());
  std::vector<TypePtr> types(writeable_tensors.size(), StringType::get());

  for (size_t i = 0; i < writeable_tensors.size(); i++) {
    keys.emplace_back(std::to_string(i));
  }

  auto keys_tuple = at::ivalue::Tuple::create(keys, TupleType::create(types));
  jit::pickle(writer, keys_tuple);

  for (const auto& tensor_data : writeable_tensors) {
    const char* addr = tensor_data.data();
    size_t numel = tensor_data.numel();
    writer(reinterpret_cast<const char*>(&numel), sizeof(numel));
    writer(addr, tensor_data.sizeInBytes());
  }

  return data;
}

} // namespace torch
