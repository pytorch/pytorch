
#include <ATen/core/ivalue.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/mobile/backport.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/observer.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>
#include <torch/csrc/jit/serialization/unpickler.h>
#include <torch/custom_class.h>

#include <exception>
#include <fstream>
#include <string>
#include <vector>

namespace c10 {
// std::string serializeType(const Type &t);
TypePtr parseType(const std::string& pythonStr);
} // namespace c10

namespace torch {
namespace jit {

namespace {
static constexpr const char* kArchiveNameConstants = "constants";
static constexpr const char* kArchiveNameBytecode = "bytecode";

TypePtr resolveTypeName(
    std::shared_ptr<CompilationUnit>& compilation_unit,
    const c10::QualifiedName& qn) {
  // HACK: first we check whether the name starts with special prefix to
  // tell if it's a supported pytorch class type. There are two special
  // prefixes. "__torch__" for nn module, and "torch.jit" from to_backend.
  // This is a reliable
  // check today, but there is no guarantee that this is the case. The
  // real solution is to merge type parsers so we can share class
  // resolution logic.
  static const c10::QualifiedName torchPrefix = "__torch__";
  static const c10::QualifiedName jitPrefix = "torch.jit";
  if (torchPrefix.isPrefixOf(qn) || jitPrefix.isPrefixOf(qn)) {
    if (compilation_unit->get_class(qn) == nullptr) {
      auto typeptr = ClassType::create(qn, compilation_unit, true);
      compilation_unit->register_type(typeptr);
    }
    return compilation_unit->get_class(qn);
  } else {
    return c10::parseType(qn.qualifiedName());
  }
}

c10::IValue readArchive(
    const std::string& archive_name,
    std::shared_ptr<mobile::CompilationUnit> compilation_unit,
    std::unique_ptr<caffe2::serialize::PyTorchStreamReader>& stream_reader) {
  std::stringstream picklename;
  picklename << archive_name << ".pkl";
  at::DataPtr pickle_ptr;
  size_t pickle_size;
  std::tie(pickle_ptr, pickle_size) =
      stream_reader->getRecord(picklename.str());

  size_t bytes_read = 0;
  auto data = reinterpret_cast<const char*>(pickle_ptr.get());
  auto reader = [&](char* buffer, size_t len) -> size_t {
    if (bytes_read >= pickle_size) {
      return 0;
    }
    len = std::min(pickle_size - bytes_read, len);
    // Copy len bytes into buffer
    const char* start = data + bytes_read;
    std::memcpy(buffer, start, len);
    bytes_read += len;
    return len;
  };
  std::shared_ptr<CompilationUnit> jit_compilation_unit =
      std::make_shared<CompilationUnit>();
  auto type_resolver = [&](const c10::QualifiedName& qn) {
    return c10::StrongTypePtr(
        jit_compilation_unit, resolveTypeName(jit_compilation_unit, qn));
  };

  auto obj_loader = [&](at::StrongTypePtr type, IValue input) {
    auto cls = type.type_->expect<at::ClassType>();
    auto qn = cls->name();
    c10::QualifiedName method_name(qn.value(), "__setstate__");
    auto setstate = compilation_unit->find_function(method_name);
    auto find_custom_class_with_setstate = [&qn]() -> c10::ClassTypePtr {
      auto custom_class_type = torch::jit::getCustomClass(qn->qualifiedName());
      if (custom_class_type && custom_class_type->findMethod("__setstate__")) {
        return custom_class_type;
      }
      return nullptr;
    };
    if (setstate) {
      auto obj = c10::ivalue::Object::create(type, 0);
      Stack stack({obj, input});
      setstate->run(stack);
      return obj;
    } else if (auto custom_class_type = find_custom_class_with_setstate()) {
      auto obj = c10::ivalue::Object::create(
          c10::StrongTypePtr(nullptr, custom_class_type), 1);
      Stack stack({obj, input});
      custom_class_type->getMethod("__setstate__").run(stack);
      return obj;
    } else {
      auto dict = std::move(input).toGenericDict();
      size_t ndict = dict.size();
      auto obj = c10::ivalue::Object::create(type, ndict);
      auto it = dict.begin();
      for (size_t i = 0; i < ndict; ++i) {
        std::stringstream name;
        name << it->key();
        cls->addOrCheckAttribute(name.str(), it->key().type());
        obj->setSlot(i, it->value());
        ++it;
      }
      return obj;
    }
  };

  static const std::string slash = "/";
  auto read_record = [&](const std::string& name) {
    std::size_t found = name.find(slash);
    std::stringstream ss;
    // In version 4, the tensor root_key doesn't include the parent path
    // To support backward compatibility, when the name doesn't include slash
    // assume it's version 4 and attach the archive_name_plus_slash
    // The example tensor format is:
    // torch._utils._rebuild_tensor_v2(
    //     pers.obj(('storage', torch.FloatStorage, '17', 'cpu', 22736),),
    //     0,
    //     (1, 464, 7, 7),
    //     (22736, 49, 7, 1),
    //     False,
    //     collections.OrderedDict())
    if (found == std::string::npos) {
      ss << archive_name << slash << name;
      return std::get<0>(stream_reader->getRecord(ss.str()));
    }

    // In version 4+, the tensor root_key in bytecode will include the parent
    // path. The example tensor format is: torch._utils._rebuild_tensor_v2(
    //     pers.obj(('storage', torch.FloatStorage, 'constants/17', 'cpu',
    //     22736),), 0, (1, 464, 7, 7), (22736, 49, 7, 1), False,
    //     collections.OrderedDict())
    ss << name;
    return std::get<0>(stream_reader->getRecord(ss.str()));
  };
  c10::optional<at::Device> device;

  Unpickler unpickler(
      reader,
      std::move(type_resolver),
      std::move(obj_loader),
      std::move(read_record),
      device);
  return unpickler.parse_ivalue();
}

void writeArchive(
    caffe2::serialize::PyTorchStreamWriter& writer,
    const std::string& archive_name,
    const IValue& value,
    TensorIndexMap& tensors_archive_table,
    bool use_tensors_archive_table = false) {
  std::vector<char> data;
  TypeNameUniquer type_name_uniquer;
  // Vector to capture the run-time class types during pickling the IValues
  std::vector<c10::ClassTypePtr> memoizedClassTypes;
  Pickler data_pickle(
      [&](const char* buf, size_t size) {
        data.insert(data.end(), buf, buf + size);
      },
      nullptr,
      [&](const c10::ClassTypePtr& t) {
        return type_name_uniquer.getUniqueName(t);
      },
      &memoizedClassTypes);
  if (use_tensors_archive_table && !tensors_archive_table.empty()) {
    data_pickle.updateTensorsArchiveTable(tensors_archive_table);
  }
  data_pickle.protocol();
  data_pickle.pushIValue(value);
  data_pickle.stop();
  size_t i = 0;
  std::string prefix = archive_name + "/";

  // TODO: currently there exists logic only for archive constant and
  // bytecode, to avoid exporting duplicate tensors. The logic can be more
  // generic such that it can be used by other tensors from other archive, to
  // avoid deduplicating tensors among different archives.

  // Store all tensors from archives `constants` to tensors_archive_table
  if (archive_name == kArchiveNameConstants) {
    const auto tensor_candidates = data_pickle.tensorData();
    for (size_t tensor_index = 0; tensor_index < tensor_candidates.size();
         tensor_index++) {
      tensors_archive_table[tensor_candidates[tensor_index]] =
          std::make_pair(kArchiveNameConstants, tensor_index);
    }
  }

  // Export deduplicate tensors only if use_tensors_archive_table is set to
  // true and archive name is `bytecode`
  bool can_use_tensors_archive_table =
      (use_tensors_archive_table && archive_name == kArchiveNameBytecode);

  for (const auto& td : data_pickle.tensorData()) {
    WriteableTensorData writable_td = getWriteableTensorData(td);
    std::string fname = prefix + c10::to_string(i++);
    if (can_use_tensors_archive_table) {
      //      std::cout << "using tensor archive table" << std::endl;
      const auto found = tensors_archive_table.find(td);
      if (found == tensors_archive_table.end()) {
        writer.writeRecord(
            fname, writable_td.data(), writable_td.sizeInBytes());
      }
    } else {
      writer.writeRecord(fname, writable_td.data(), writable_td.sizeInBytes());
    }
  }
  std::string fname = archive_name + ".pkl";
  writer.writeRecord(fname, data.data(), data.size());

  // serialize all the captured run-time class types
  //  for (const c10::ClassTypePtr& wroteType : memoizedClassTypes) {
  //    convertNamedType(wroteType);
  //  }
}

} // namespace

using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::ReadAdapterInterface;

// Forward declare so that _backport_for_mobile() overloads can
// call this method directly.
void _backport_for_mobile_impl(std::unique_ptr<ReadAdapterInterface> rai);

void _backport_for_mobile(std::istream& in) {
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  _backport_for_mobile(std::move(rai));
}

void _backport_for_mobile(const std::string& filename) {
  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
  _backport_for_mobile_impl(std::move(rai));
}

void _backport_for_mobile(std::unique_ptr<ReadAdapterInterface> rai) {
  _backport_for_mobile_impl(std::move(rai));
}

void _backport_for_mobile_impl(std::unique_ptr<ReadAdapterInterface> rai) {
  std::cout << "backporting..." << std::endl;

  // Verify that we're loading a zip archive and not a torch.save pickle archive
  // (marked by the 0x80 0x02 bytes at the start)
  uint8_t first_short[2];
  rai->read(
      /*pos=*/0,
      /*buf=*/&first_short,
      /*n=*/2,
      /*what=*/"checking archive");
  if (first_short[0] == 0x80 && first_short[1] == 0x02) {
    // NB: zip files by spec can start with any data, so technically they might
    // start with 0x80 0x02, but in practice zip files start with a file entry
    // which begins with 0x04034b50. Furthermore, PyTorch will never produce zip
    // files that do not start with the file entry, so it is relatively safe to
    // perform this check.
    TORCH_CHECK(false, "file issue");
  }
  auto reader = torch::make_unique<caffe2::serialize::PyTorchStreamReader>(
      std::move(rai));
  c10::optional<std::vector<IValue>> bvals;
  auto mcu = std::make_shared<mobile::CompilationUnit>();
  auto compilation_unit = std::make_shared<CompilationUnit>();
  bvals = readArchive("bytecode", mcu, reader).toTuple()->elements();

  auto it = reader->getAllRecords();
  auto it_2 = reader->version();

  std::cout << "pass bytecode " << std::endl;

  c10::optional<std::vector<IValue>> ts_vals;
  ts_vals = readArchive("constants", mcu, reader).toTuple()->elements();

  std::cout << "pass constatns " << std::endl;
  const auto ivalue_constants = ts_vals.value();
  //  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
  //    out.write(static_cast<const char*>(buf), nbytes);
  //    return !out ? 0 : nbytes;
  //  }
  //  caffe2::serialize::PyTorchStreamWriter writer(writer_func);
  TensorIndexMap tensors_archive_table;
  auto file_name = "output_test.pkl";

  caffe2::serialize::PyTorchStreamWriter writer(file_name);

  auto constants_data = c10::ivalue::Tuple::create(std::move(ivalue_constants));
  auto bytecode_data = c10::ivalue::Tuple::create(std::move(bvals.value()));

  writeArchive(writer, "constants", constants_data, tensors_archive_table);
  writeArchive(writer, "bytecode", bytecode_data, tensors_archive_table, true);

  //  std::string archive_name = "bytecode";
  //  std::stringstream picklename;
  //  picklename << archive_name << ".pkl";
  //  at::DataPtr pickle_ptr;
  //  size_t pickle_size;
  //  std::tie(pickle_ptr, pickle_size) =
  //      stream_reader->getRecord(picklename.str());
  //
  //  size_t bytes_read = 0;
  //  auto data = reinterpret_cast<const char*>(pickle_ptr.get());
  //  auto reader = [&](char* buffer, size_t len) -> size_t {
  //    if (bytes_read >= pickle_size) {
  //      return 0;
  //    }
  //    len = std::min(pickle_size - bytes_read, len);
  //    // Copy len bytes into buffer
  //    const char* start = data + bytes_read;
  //    std::memcpy(buffer, start, len);
  //    bytes_read += len;
  //    return len;
  //  };
}

} // namespace jit
} // namespace torch
