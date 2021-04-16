#include <torch/csrc/jit/mobile/export_data.h>

#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>

#include <caffe2/serialize/inline_container.h>

#include <ATen/core/jit_type.h>
#include <string>
#include <vector>

namespace torch {
namespace jit {
namespace mobile {

char const* toString(OpCode op);

namespace {

class ScriptModuleSerializer {
 public:
  explicit ScriptModuleSerializer(const std::string& filename)
      : writer_(filename) {}

  explicit ScriptModuleSerializer(
      const std::function<size_t(const void*, size_t)>& writer_func)
      : writer_(writer_func) {}

  void serialize(const IValue& object) {
    // Serialize just the data
    writeArchive("data", object);
  }

  void writeArchive(const std::string& archive_name, const IValue& value) {
    std::vector<char> data;
    // Vector to capture the run-time class types during pickling the IValues
    std::vector<c10::ClassTypePtr> memoizedClassTypes;
    Pickler data_pickle(
        [&](const char* buf, size_t size) {
          data.insert(data.end(), buf, buf + size);
        },
        nullptr,
        [&](const c10::ClassTypePtr& t) {
          return type_name_uniquer_.getUniqueName(t);
        },
        &memoizedClassTypes);
    data_pickle.protocol();
    data_pickle.pushIValue(value);
    data_pickle.stop();
    size_t i = 0;
    std::string prefix = archive_name + "/";
    for (const auto& td : data_pickle.tensorData()) {
      WriteableTensorData writable_td = getWriteableTensorData(td);
      std::string fname = prefix + c10::to_string(i++);
      writer_.writeRecord(fname, writable_td.data(), writable_td.sizeInBytes());
    }
    std::string fname = archive_name + ".pkl";
    writer_.writeRecord(fname, data.data(), data.size());
  }

  caffe2::serialize::PyTorchStreamWriter writer_;
  TypeNameUniquer type_name_uniquer_;
};

} // namespace

void _save_data(const Module& module, std::ostream& out) {
  ScriptModuleSerializer serializer(
      [&](const void* buf, size_t nbytes) -> size_t {
        out.write(static_cast<const char*>(buf), nbytes);
        return !out ? 0 : nbytes;
      });
  serializer.serialize(module._ivalue());
}

void _save_data(const Module& module, const std::string& filename) {
  ScriptModuleSerializer serializer(filename);
  serializer.serialize(module._ivalue());
}

} // namespace mobile

void _save_parameters(
    const std::map<std::string, at::Tensor>& map,
    std::ostream& out) {
  mobile::ScriptModuleSerializer serializer(
      [&](const void* buf, size_t nbytes) -> size_t {
        out.write(static_cast<const char*>(buf), nbytes);
        return !out ? 0 : nbytes;
      });
  c10::Dict<std::string, at::Tensor> dict;
  for (const auto& e : map) {
    dict.insert(e.first, e.second);
  }
  serializer.serialize(dict);
}

void _save_parameters(
    const std::map<std::string, at::Tensor>& map,
    const std::string& filename) {
  mobile::ScriptModuleSerializer serializer(filename);
  c10::Dict<std::string, at::Tensor> dict;
  for (const auto& e : map) {
    dict.insert(e.first, e.second);
  }
  serializer.serialize(dict);
}

} // namespace jit
} // namespace torch
