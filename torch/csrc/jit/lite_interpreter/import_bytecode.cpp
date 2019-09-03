//#include <torch/csrc/autograd/variable.h>
#include "import_bytecode.h"
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/script/compilation_unit.h>
#include <torch/csrc/jit/pickler.h>
#include <caffe2/serialize/inline_container.h>
//#include <caffe2/serialize/istream_adapter.h>


#include <fstream>
#include <string>
#include <vector>

namespace torch {
namespace jit {
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::ReadAdapterInterface;

namespace {
// The deserializer class which loads the bytecode package from bc files.
//
class BytecodeDeserializer final {
 public:
  explicit BytecodeDeserializer(std::unique_ptr<PyTorchStreamReader> reader);
  mobile::Bytecode deserialize(c10::optional<at::Device> device);

 private:
  c10::IValue readArchive(const std::string& archive_name);
  std::shared_ptr<script::CompilationUnit> compilation_unit_;
  std::unordered_set<std::string> imported_libs_;
  std::unique_ptr<PyTorchStreamReader> reader_;
  c10::optional<at::Device> device_;
};


BytecodeDeserializer::BytecodeDeserializer(std::unique_ptr<PyTorchStreamReader> reader)
    : compilation_unit_(std::make_shared<script::CompilationUnit>()), reader_(std::move(reader)) {}

mobile::Bytecode BytecodeDeserializer::deserialize(c10::optional<at::Device> device) {
  mobile::Bytecode bc;
  device_ = device;
  auto methods = readArchive("bytecode").toTuple();
  auto data = readArchive("data").toObject();
  return bc;
}

c10::IValue BytecodeDeserializer::readArchive(const std::string& archive_name) {
  std::stringstream picklename;
  picklename << archive_name << ".pkl";
  at::DataPtr pickle_ptr;
  size_t pickle_size;
  std::tie(pickle_ptr, pickle_size) = reader_->getRecord(picklename.str());

  size_t bytes_read = 0;
  auto data = reinterpret_cast<const char*>(pickle_ptr.get());
  auto reader = [&](char* buffer, size_t len) {
    if (bytes_read + len > pickle_size) {
      return false;
    }
    // Copy len bytes into buffer
    const char* start = data + bytes_read;
    std::memcpy(buffer, start, len);
    bytes_read += len;
    return true;
  };

  auto class_resolver = [&](const c10::QualifiedName& qn) {
    if (compilation_unit_->get_class(qn) == nullptr) {
      auto typeptr = ClassType::create(qn, compilation_unit_, true);
      compilation_unit_->register_type(typeptr);
    }
    return c10::StrongTypePtr(
        compilation_unit_, compilation_unit_->get_class(qn));
  };
  auto read_record = [&](const std::string& name) {
    std::stringstream ss;
    ss << archive_name << "/" << name;
    return std::get<0>(reader_->getRecord(ss.str()));
  };
  Unpickler unpickler(
      reader, std::move(class_resolver), std::move(read_record), device_);
  return unpickler.parse_ivalue();
}

} // namespace

mobile::Bytecode load_bytecode(
    std::istream& in,
    c10::optional<at::Device> device) {
  std::unique_ptr<IStreamAdapter> rai =
      caffe2::make_unique<IStreamAdapter>(&in);
  auto bc = load_bytecode(std::move(rai), device);
  return bc;
}

mobile::Bytecode load_bytecode(
    const std::string& filename,
    c10::optional<at::Device> device) {
  std::unique_ptr<FileAdapter> rai = caffe2::make_unique<FileAdapter>(filename);
  auto module = load_bytecode(std::move(rai), device);
  return module;
}

mobile::Bytecode load_bytecode(
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<c10::Device> device) {
  auto reader = torch::make_unique<PyTorchStreamReader>(std::move(rai));
  BytecodeDeserializer deserializer(std::move(reader));
  return deserializer.deserialize(device);
}

} // namespace jit
} // namespace torch
