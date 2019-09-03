#include <ATen/core/ivalue.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/pickle.h>
#include <torch/csrc/jit/pickler.h>

namespace torch {
namespace jit {

std::vector<WriteableStorageData> pickle(
    std::function<void(const char* data_start, size_t data_len)> writer,
    const IValue& ivalue) {
  Pickler pickler(std::move(writer));
  pickler.protocol();
  pickler.pushIValue(ivalue);
  pickler.stop();
  return pickler.tensorData();
}

std::pair<std::vector<char>, std::vector<WriteableStorageData>> pickle(
    const IValue& ivalue) {
  std::vector<char> data;

  auto storages = pickle(
      [&](const char* bytes, size_t len) {
        data.insert(data.end(), bytes, bytes + len);
      },
      ivalue);

  return std::make_pair(data, storages);
}

void unsafe_pickle(
    std::function<void(const char*, size_t)> writer,
    jit::PickleOpCode op,
    std::string data) {
  Pickler pickler(std::move(writer));
  pickler.protocol();
  pickler.pushOp(op, data);
  pickler.stop();
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
