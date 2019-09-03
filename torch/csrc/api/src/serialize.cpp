#include <torch/csrc/jit/pickle.h>
#include <torch/csrc/jit/pickler.h>
#include <torch/serialize.h>

#include <vector>

namespace torch {

// These are both defined in `torch/serialization.py`
const char* torch_save_magic_number =
    "\x0a\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19";
const char* protocol_version = "\xe9\x03";

std::vector<char> save(const at::IValue& ivalue) {
  std::vector<char> data;

  auto writer = [&](const char* bytes, size_t len) {
    data.insert(data.end(), bytes, bytes + len);
  };


  jit::unsafe_pickle(
      writer, jit::PickleOpCode::LONG1, torch_save_magic_number);
  jit::unsafe_pickle(
      writer, jit::PickleOpCode::BININT2, protocol_version);
  jit::unsafe_pickle(writer, jit::PickleOpCode::EMPTY_DICT, "");

  std::vector<at::Tensor> tensors;
  jit::pickle<jit::LiteralPickler>(writer, ivalue, &tensors);

  std::vector<at::IValue> keys;
  keys.reserve(tensors.size());
  std::vector<TypePtr> types(tensors.size(), StringType::get());

  for (size_t i = 0; i < tensors.size(); i++) {
    keys.push_back(std::to_string(i));
  }

  auto keys_tuple = at::ivalue::Tuple::create(keys, TupleType::create(types));
  jit::pickle(writer, keys_tuple, &tensors);

  for (auto tensor : tensors) {
    auto data = jit::getWriteableTensorData(tensor);
    size_t numel = data.numel();
    writer(reinterpret_cast<const char*>(&numel), sizeof(numel));
    writer(data.data(), data.sizeInBytes());
  }

  return data;
}

} // namespace torch
