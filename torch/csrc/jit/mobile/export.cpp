#include <torch/csrc/jit/mobile/module.h>

#include <torch/csrc/api/include/torch/ordered_dict.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>
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

static IValue Tup(std::vector<IValue> ivalues) {
  return c10::ivalue::Tuple::create(std::move(ivalues));
}

static IValue Table(
    const std::vector<std::pair<std::string, IValue>>& entries) {
  std::vector<IValue> ivalue_entries;
  for (const auto& e : entries) {
    ivalue_entries.push_back(Tup({e.first, e.second}));
  }
  return Tup(std::move(ivalue_entries));
}

} // namespace

class ScriptModuleSerializer {
 public:
  explicit ScriptModuleSerializer(const std::string& filename)
      : writer_(filename) {}

  explicit ScriptModuleSerializer(
      const std::function<size_t(const void*, size_t)>& writer_func)
      : writer_(writer_func) {}

  void serialize(const IValue object) {
    // Serialize the model object
    writeArchive("data", object);

    // Acquires and sets minimum (dynamic) version
    for (auto& item : file_streams_) {
      writer_.setMinVersion(item.value().minVersion());
    }
  }

  void writeArchive(const std::string& archive_name, const IValue& value) {
    std::vector<char> data;
    // Vector to capture the run-time class types during pickling the IValues
    std::vector<c10::ClassTypePtr> memorizedClassTypes;
    Pickler data_pickle(
        [&](const char* buf, size_t size) {
          data.insert(data.end(), buf, buf + size);
        },
        nullptr,
        [&](const c10::ClassTypePtr& t) {
          return type_name_uniquer_.getUniqueName(t);
        },
        &memorizedClassTypes);
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

    // serialize all the captured run-time class types
    for (const c10::ClassTypePtr& wroteType : memorizedClassTypes) {
      convertNamedType(wroteType);
    }
  }

  void convertNamedType(const c10::NamedTypePtr& class_type) {
    if (converted_types_.count(class_type)) {
      return;
    }
    converted_types_.insert(class_type);
    auto qualname = type_name_uniquer_.getUniqueName(class_type);
    const std::string qualifier = qualname.prefix();
    PythonPrint* pp = file_streams_.find(qualifier);

    auto type_printer =
        [&](const c10::ConstTypePtr& t) -> c10::optional<std::string> {
      auto namedType = t->cast<c10::NamedType>();
      if (namedType && namedType->name()) {
        return type_name_uniquer_.getUniqueName(namedType).qualifiedName();
      }
      return c10::nullopt;
    };
    if (!pp) {
      pp = &file_streams_.insert(
          qualifier,
          PythonPrint(
              constant_table_,
              class_deps_,
              type_printer,
              /*enforce_importable=*/true));
    }
    pp->printNamedType(class_type);
  }

  caffe2::serialize::PyTorchStreamWriter writer_;
  std::vector<at::IValue> constant_table_;
  std::unordered_set<c10::NamedTypePtr> converted_types_;
  std::vector<c10::NamedTypePtr> class_deps_;
  TypeNameUniquer type_name_uniquer_;

  // qualifier, e.g. '__torch__.Bar' -> PythonPrint for the file that will be
  // created
  OrderedDict<std::string, PythonPrint> file_streams_;
};

void Module::save_data(std::ostream& out) const {
  ScriptModuleSerializer serializer(
      [&](const void* buf, size_t nbytes) -> size_t {
        out.write(static_cast<const char*>(buf), nbytes);
        return !out ? 0 : nbytes;
      });
  serializer.serialize(this->object_);
}

void Module::save_data(const std::string& filename) const {
  ScriptModuleSerializer serializer(filename);
  serializer.serialize(this->object_);
}

} // namespace mobile
} // namespace jit
} // namespace torch