#include <torch/csrc/jit/serialization/import.h>
#include <ATen/core/functional.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/serialization/import_export_helpers.h>
#ifndef C10_MOBILE
#include <torch/csrc/jit/serialization/import_legacy.h>
#endif
#include <torch/csrc/jit/frontend/script_type_parser.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/serialization/import_source.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>
#include <torch/csrc/jit/serialization/unpickler.h>

#include "caffe2/serialize/file_adapter.h"
#include "caffe2/serialize/inline_container.h"
#include "caffe2/serialize/istream_adapter.h"

#include <ATen/ATen.h>

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {

using caffe2::serialize::FileAdapter;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::ReadAdapterInterface;

void postSetStateValidate(const IValue& v) {
  auto obj = v.toObject();
  const auto& objType = obj->type();
  for (size_t i = 0; i < objType->numAttributes(); i++) {
    const auto& attrType = objType->getAttribute(i);
    const auto& attrName = objType->getAttributeName(i);
    const auto& slot = obj->getSlot(i);
    // const auto attrType = objType->getAttribute(i);
    // Verify that all the non-optional attributes have been initialized
    // TODO: Issue #20497
    if (attrType->kind() != TypeKind::OptionalType) {
      TORCH_CHECK(
          !slot.isNone(),
          "The field '",
          attrName,
          "' was left unitialized after __setstate__, but expected a ",
          "value of type '",
          attrType->python_str(),
          "'");
    }
  }
}

IValue readArchiveAndTensors(
    const std::string& archive_name,
    c10::optional<TypeResolver> type_resolver,
    c10::optional<ObjLoader> obj_loader,
    c10::optional<at::Device> device,
    PyTorchStreamReader& stream_reader) {
  std::string picklename = archive_name + ".pkl";
  at::DataPtr pickle_ptr;
  size_t pickle_size;
  std::tie(pickle_ptr, pickle_size) = stream_reader.getRecord(picklename);

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

  std::string archive_name_plus_slash = archive_name + "/";
  auto read_record = [&](const std::string& name) {
    std::string ss = archive_name_plus_slash + name;
    return std::get<0>(stream_reader.getRecord(ss));
  };

  Unpickler unpickler(
      reader,
      type_resolver ? std::move(*type_resolver) : nullptr,
      obj_loader ? std::move(*obj_loader) : nullptr,
      std::move(read_record),
      device);
  unpickler.set_version(stream_reader.version());
  return unpickler.parse_ivalue();
}

namespace {

// This is a deserializer class which loads script modules from pt files.
// Content of the file is written using PyTorchStreamWriter, for details please
// check caffe2/serialize/inline_container.h.
// The module is saved in pickle. readArchive() is called to parse and construct
// the constant table and the script module.
class ScriptModuleDeserializer final {
 public:
  ScriptModuleDeserializer(
      std::shared_ptr<CompilationUnit> cu,
      std::unique_ptr<PyTorchStreamReader> reader)
      : compilation_unit_(cu),
        reader_(std::move(reader)),
        source_importer_(
            compilation_unit_,
            &constants_table_,
            [this](const std::string& qualifier) {
              return findSourceInArchiveFromQualifier(
                  *reader_, export_prefix_, qualifier);
            },
            reader_->version()) {}

  Module deserialize(
      c10::optional<at::Device> device,
      ExtraFilesMap& extra_files);

 private:
  IValue readArchive(const std::string& archive_name);

  std::shared_ptr<CompilationUnit> compilation_unit_;
  std::unique_ptr<PyTorchStreamReader> reader_;
  c10::optional<at::Device> device_;
  std::vector<at::Tensor> constants_table_;
  SourceImporter source_importer_;
  std::string export_prefix_ = "code/";
};

IValue ScriptModuleDeserializer::readArchive(const std::string& archive_name) {
  auto type_resolver = [&](const c10::QualifiedName& qn) {
    auto cls = source_importer_.loadNamedType(qn)->expect<ClassType>();
    return c10::StrongTypePtr(compilation_unit_, std::move(cls));
  };

  // Decouple how to get obj from type. In this file it's dependent on
  // Method.run() and graph executor, etc.
  // For bytecode import we need to decouple these dependencies.
  auto obj_loader = [&](at::StrongTypePtr type, IValue input) {
    auto cls = type.type_->expect<at::ClassType>();
    auto qn = cls->name();
    size_t n = cls->numAttributes();
    if (checkHasValidSetGetState(cls)) {
      auto obj = c10::ivalue::Object::create(type, n);
      // XXX: Do not optimize __setstate__, so that we don't try to
      // specialize the class before it is initialized.
      setGraphExecutorOptimize(false);
      Function* set_state = cls->getMethod("__setstate__");
      // since we are in the middle of unpickling we might still have lists and
      // dicts that do not have accurate tags (e.g. they report they are
      // List[Any]). But we need to run __setstate__ which will check the input
      // type and may access the tags. Since setstate has a known input type, we
      // can correctly restore the tags now by apply the input type of set_state
      // to the state object being passed.
      // TODO: Remove once [serialization type tags] is landed
      restoreAccurateTypeTags(
          input, set_state->getSchema().arguments().at(1).type());
      (*set_state)({obj, input});
      setGraphExecutorOptimize(true);
      postSetStateValidate(obj);
      return obj;
    } else {
      auto dict = std::move(input).toGenericDict();
      auto obj = c10::ivalue::Object::create(type, n);
      for (size_t i = 0; i < n; ++i) {
        obj->setSlot(i, dict.at(cls->getAttributeName(i)));
      }
      return obj;
    }
  };

  return readArchiveAndTensors(
      archive_name, type_resolver, obj_loader, device_, *reader_.get());
}

void  rewriteQuantizedConvForBC(const Module& module) {
  const std::string& old_quantized_conv2d = R"(
graph(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point):
         %r = quantized::conv2d(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point)
         return (%r) )";
  const std::string& old_quantized_conv2d_relu = R"(
graph(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point):
         %r = quantized::conv2d_relu(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point)
         return (%r) )";

  const std::string& old_quantized_conv3d = R"(
graph(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point):
         %r = quantized::conv3d(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point)
         return (%r) )";

  const std::string& old_quantized_conv3d_relu = R"(
graph(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point):
         %r = quantized::conv3d_relu(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point)
         return (%r) )";

  const std::string& new_quantized_conv2d = R"(
graph(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point):
         %r = quantized::conv2d(%x, %packed_params, %r_scale, %r_zero_point)
         return (%r) )";

  const std::string& new_quantized_conv2d_relu = R"(
graph(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point):
         %r = quantized::conv2d_relu(%x, %packed_params, %r_scale, %r_zero_point)
         return (%r) )";

  const std::string& new_quantized_conv3d = R"(
graph(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point):
         %r = quantized::conv3d(%x, %packed_params, %r_scale, %r_zero_point)
         return (%r) )";

  const std::string& new_quantized_conv3d_relu = R"(
graph(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point):
         %r = quantized::conv3d_relu(%x, %packed_params, %r_scale, %r_zero_point)
         return (%r) )";

  SubgraphRewriter rewriter;
  std::vector<std::vector<std::string>> patterns_and_replacements = {
    {old_quantized_conv2d, new_quantized_conv2d},
    {old_quantized_conv2d_relu, new_quantized_conv2d_relu},
    {old_quantized_conv3d, new_quantized_conv3d},
    {old_quantized_conv3d_relu, new_quantized_conv3d_relu},
  };
  rewriter.runOnModule(module);
  for (const Module& child : module.children()) {
    rewriteQuantizedConvForBC(child);
  }
}

Module ScriptModuleDeserializer::deserialize(
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  C10_LOG_API_USAGE_ONCE("torch.script.load");
  device_ = device;
  // Load extra files.
  for (const auto& kv : extra_files) {
    const std::string& key = "extra/" + kv.first;
    if (reader_->hasRecord(key)) {
      at::DataPtr meta_ptr;
      size_t meta_size;
      std::tie(meta_ptr, meta_size) = reader_->getRecord(key);
      extra_files[kv.first] =
          std::string(static_cast<char*>(meta_ptr.get()), meta_size);
    }
  }
  if (reader_->hasRecord("model.json")) {
#ifndef C10_MOBILE
    return torch::jit::LEGACY_deserialize(
        compilation_unit_, std::move(reader_), device_);
#else
    AT_ERROR("Legacy model format is not supported on mobile.");
#endif
  }
  auto tuple = readArchive("constants").toTuple();
  for (auto constant : tuple->elements()) {
    constants_table_.push_back(constant.toTensor());
  }
  auto m = Module(readArchive("data").toObject());
  rewriteQuantizedConvForBC(m);
  return m;
}

} // namespace

Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::istream& in,
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  auto reader = torch::make_unique<PyTorchStreamReader>(&in);
  ScriptModuleDeserializer deserializer(std::move(cu), std::move(reader));
  return deserializer.deserialize(device, extra_files);
}

Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& filename,
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  auto reader = torch::make_unique<PyTorchStreamReader>(filename);
  ScriptModuleDeserializer deserializer(std::move(cu), std::move(reader));
  return deserializer.deserialize(device, extra_files);
}

Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  auto reader = torch::make_unique<PyTorchStreamReader>(std::move(rai));
  ScriptModuleDeserializer deserializer(std::move(cu), std::move(reader));
  return deserializer.deserialize(device, extra_files);
}

Module load(
    std::istream& in,
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  auto module = load(std::move(rai), device, extra_files);
  return module;
}

Module load(
    const std::string& filename,
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
  auto module = load(std::move(rai), device, extra_files);
  return module;
}

Module load(
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<c10::Device> device,
    ExtraFilesMap& extra_files) {
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
    TORCH_CHECK(
        false,
        "`torch::jit::load()` received a file from `torch.save()`, "
        "but `torch::jit::load()` can only load files"
        " produced by `torch.jit.save()`");
  }

  auto reader = torch::make_unique<PyTorchStreamReader>(std::move(rai));
  auto cu = std::make_shared<CompilationUnit>();

  ScriptModuleDeserializer deserializer(std::move(cu), std::move(reader));
  return deserializer.deserialize(device, extra_files);
}

} // namespace jit
} // namespace torch
