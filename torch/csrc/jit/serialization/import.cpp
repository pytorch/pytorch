#include <ATen/core/interned_strings.h>
#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/in_memory_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <caffe2/serialize/istream_adapter.h>
#include <caffe2/serialize/read_adapter_interface.h>

#include <torch/csrc/jit/api/compilation_unit.h>

#include <ATen/core/functional.h>
#include <ATen/core/ivalue_inl.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/script_type_parser.h>
#include <torch/csrc/jit/ir/graph_utils.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/mobile/file_format.h>
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/operator_upgraders/upgraders_entry.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/import_export_helpers.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <torch/csrc/jit/serialization/import_source.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>
#include <torch/csrc/jit/serialization/unpickler.h>

#include <ATen/ATen.h>
#include <fmt/format.h>

#include <string>
#include <utility>
#include <vector>

namespace torch::jit {

using caffe2::serialize::MemoryReadAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::ReadAdapterInterface;

static void postSetStateValidate(const IValue& v) {
  auto obj = v.toObject();
  const auto& objType = obj->type();
  for (const auto i : c10::irange(objType->numAttributes())) {
    const auto& attrType = objType->getAttribute(i);
    const auto& attrName = objType->getAttributeName(i);
    const auto& slot = obj->getSlot(i);
    // const auto attrType = objType->getAttribute(i);
    // Verify that all the non-optional attributes have been initialized
    // TODO: Issue #20497
    if (attrType->kind() != TypeKind::UnionType &&
        attrType->kind() != TypeKind::OptionalType &&
        attrType->kind() != TypeKind::NoneType) {
      TORCH_CHECK(
          !slot.isNone(),
          fmt::format(
              "The field '{}' was left uninitialized after '__setstate__', "
              "but expected a value of type '{}'",
              attrName,
              attrType->repr_str()));
    }
  }
}

// Decouple how to get obj from type. In this file it's dependent on
// Method.run() and graph executor, etc.
// For bytecode import we need to decouple these dependencies.
c10::intrusive_ptr<c10::ivalue::Object> ObjLoaderFunc(
    const at::StrongTypePtr& type,
    IValue input) {
  auto cls = type.type_->expect<at::ClassType>();
  auto qn = cls->name();
  size_t n = cls->numAttributes();
  if (checkHasValidSetGetState(cls)) {
    auto obj = c10::ivalue::Object::create(type, n);
    // XXX: Do not optimize __setstate__, so that we don't try to
    // specialize the class before it is initialized.
    GraphOptimizerEnabledGuard guard(false);
    Function& set_state = cls->getMethod("__setstate__");
    // since we are in the middle of unpickling we might still have lists and
    // dicts that do not have accurate tags (e.g. they report they are
    // List[Any]). But we need to run __setstate__ which will check the input
    // type and may access the tags. Since setstate has a known input type, we
    // can correctly restore the tags now by apply the input type of set_state
    // to the state object being passed.
    // TODO: Remove once [serialization type tags] is landed
    restoreAccurateTypeTags(
        input, set_state.getSchema().arguments().at(1).type());
    set_state({obj, input});
    postSetStateValidate(obj);
    return obj;
  } else {
    auto dict = std::move(input).toGenericDict();
    auto obj = c10::ivalue::Object::create(type, n);
    for (const auto i : c10::irange(n)) {
      obj->setSlot(i, dict.at(cls->getAttributeName(i)));
    }
    return obj;
  }
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
      std::shared_ptr<PyTorchStreamReader> reader)
      : compilation_unit_(std::move(cu)),
        reader_(std::move(reader)),
        code_prefix_("code/"),
        pickle_dir_prefix_(""),
        tensor_dir_prefix_(""),
        source_importer_(
            compilation_unit_,
            &constants_table_,
            [this](const std::string& qualifier) {
              return findSourceInArchiveFromQualifier(
                  *reader_, code_prefix_, qualifier);
            },
            reader_->version()) {}

  ScriptModuleDeserializer(
      std::shared_ptr<CompilationUnit> cu,
      std::shared_ptr<PyTorchStreamReader> reader,
      std::string pickle_dir_prefix,
      std::string tensor_dir_prefix,
      std::shared_ptr<DeserializationStorageContext> storage_context)
      : compilation_unit_(std::move(cu)),
        reader_(std::move(reader)),
        storage_context_(std::move(storage_context)),
        code_prefix_(".data/ts_code/code/"),
        pickle_dir_prefix_(std::move(pickle_dir_prefix)),
        tensor_dir_prefix_(std::move(tensor_dir_prefix)),
        source_importer_(
            compilation_unit_,
            &constants_table_,
            [this](const std::string& qualifier) {
              return findSourceInArchiveFromQualifier(
                  *reader_, code_prefix_, qualifier);
            },
            reader_->version()) {}

  Module deserialize(
      std::optional<at::Device> device,
      ExtraFilesMap& extra_files,
      bool restore_shapes = false);

 private:
  IValue readArchive(const std::string& archive_name);

  std::shared_ptr<CompilationUnit> compilation_unit_;
  std::shared_ptr<PyTorchStreamReader> reader_;
  std::shared_ptr<DeserializationStorageContext> storage_context_;
  std::optional<at::Device> device_;
  std::vector<at::IValue> constants_table_;
  std::string code_prefix_;
  std::string pickle_dir_prefix_;
  std::string tensor_dir_prefix_;
  SourceImporter source_importer_;
};

IValue ScriptModuleDeserializer::readArchive(const std::string& archive_name) {
  auto type_resolver = [&](const c10::QualifiedName& qn) {
    auto cls = source_importer_.loadType(qn);
    return c10::StrongTypePtr(compilation_unit_, std::move(cls));
  };

  return readArchiveAndTensors(
      /*archive_name=*/archive_name,
      /*pickle_prefix=*/pickle_dir_prefix_,
      /*tensor_prefix=*/tensor_dir_prefix_,
      type_resolver,
      ObjLoaderFunc,
      device_,
      *reader_,
      nullptr,
      storage_context_);
}

void rewriteQuantizedConvForBC(const Module& module) {
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
  static const std::vector<std::pair<std::string, std::string>>
      patterns_and_replacements = {
          {old_quantized_conv2d, new_quantized_conv2d},
          {old_quantized_conv2d_relu, new_quantized_conv2d_relu},
          {old_quantized_conv3d, new_quantized_conv3d},
          {old_quantized_conv3d_relu, new_quantized_conv3d_relu},
      };
  for (const auto& item : patterns_and_replacements) {
    rewriter.RegisterRewritePattern(item.first, item.second);
  }
  rewriter.runOnModule(module);

  for (const Module& child : module.children()) {
    rewriteQuantizedConvForBC(child);
  }
}

Module ScriptModuleDeserializer::deserialize(
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    bool restore_shapes) {
  // we populate the upgraders map before any load starts
  populate_upgraders_graph_map();

  C10_LOG_API_USAGE_ONCE("torch.jit.load");
  device_ = device;
  // Load extra files.
  for (const auto& kv : extra_files) {
    const std::string& key = "extra/" + kv.first;
    if (reader_->hasRecord(key)) {
      auto [meta_ptr, meta_size] = reader_->getRecord(key);
      extra_files[kv.first] =
          std::string(static_cast<char*>(meta_ptr.get()), meta_size);
    }
  }
  if (reader_->hasRecord("model.json") && code_prefix_ == "code/") {
    AT_ERROR("Legacy model format is not supported on mobile.");
  }
  auto tuple = readArchive("constants").toTuple();
  for (auto constant : tuple->elements()) {
    constants_table_.push_back(constant.toIValue());
  }
  auto m_ivalue = readArchive("data");
  auto m = Module(m_ivalue.toObject());
  rewriteQuantizedConvForBC(m);
  // Checking for and loading saved traced inputs
  if (restore_shapes && reader_->hasRecord("traced_inputs.pkl")) {
    auto dict = readArchive("traced_inputs").toGenericDict();
    for (const auto& entry : dict) {
      auto inputs = entry.value().toList().vec();
      auto g =
          toGraphFunction(m.get_method(entry.key().toStringRef()).function())
              .graph();
      Stack stack(inputs.begin(), inputs.end());
      // Added the module as the first input if we are missing
      // an input as traced modules refer to self as an additional input
      if (g->inputs().size() == stack.size() + 1) {
        stack.insert(stack.begin(), m_ivalue);
      }
      setInputTensorTypes(*g, stack, /*complete=*/true);
      PropagateInputShapes(g);
    }
  } else {
    if (restore_shapes) {
      TORCH_WARN("Cannot restore shapes as no traced inputs were stored");
    }
  }
  c10::LogAPIUsageMetadata(
      "torch.script.load.metadata",
      {{"serialization_id", reader_->serializationId()}});
  return m;
}
} // namespace

Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::istream& in,
    std::optional<at::Device> device,
    bool load_debug_files) {
  ExtraFilesMap extra_files;
  return import_ir_module(
      std::move(cu), in, device, extra_files, load_debug_files);
}

static Module _load_jit_module_from_bytes(
    const std::shared_ptr<char>& data,
    size_t size,
    std::shared_ptr<CompilationUnit> cu,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool restore_shapes);

Module parse_and_initialize_jit_module(
    const std::shared_ptr<char>& data,
    size_t size,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device) {
  populate_upgraders_graph_map();
  ExtraFilesMap jit_files;
  std::vector<IValue> jit_constants;
  mobile::Module mobilem = parse_and_initialize_mobile_module_for_jit(
      data.get(), size, jit_files, jit_constants, device, &extra_files);

  Module m = jitModuleFromSourceAndConstants(
      mobilem._ivalue(),
      jit_files,
      jit_constants,
      static_cast<int32_t>(mobilem.bytecode_version()));
  m.set_delete_memory(data);
  return m;
}

Module load_jit_module_from_file(
    const std::string& filename,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device) {
  auto data = get_file_content(filename.c_str());
  return parse_and_initialize_jit_module(
      std::get<0>(data), std::get<1>(data), extra_files, device);
}

Module load_jit_module_from_stream(
    std::istream& in,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device) {
  auto data = get_stream_content(in);
  return parse_and_initialize_jit_module(
      std::get<0>(data), std::get<1>(data), extra_files, device);
}

Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::istream& in,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files,
    bool restore_shapes) {
  in.seekg(0, in.beg);
  // NOTE: Zipformat can be large files. So using stream version directly
  // instead of reading the file all at once.
  if (getFileFormat(in) != FileFormat::FlatbufferFileFormat) {
    auto reader = std::make_unique<PyTorchStreamReader>(&in);
    reader->setShouldLoadDebugSymbol(load_debug_files);
    ScriptModuleDeserializer deserializer(std::move(cu), std::move(reader));
    return deserializer.deserialize(device, extra_files, restore_shapes);
  }
  auto [data, size] = get_stream_content(in);
  return _load_jit_module_from_bytes(
      data, size, cu, device, extra_files, restore_shapes);
}

// For reading unified serialization format from torch.Package.
Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::shared_ptr<PyTorchStreamReader> reader,
    std::shared_ptr<DeserializationStorageContext> storage_context,
    std::optional<at::Device> device,
    const std::string& ts_id) {
  ScriptModuleDeserializer deserializer(
      std::move(cu),
      std::move(reader),
      /* pickle_dir_prefix = */ ".data/ts_code/" + ts_id + "/",
      /* tensor_dir_prefix = */ ".data/",
      std::move(storage_context));
  ExtraFilesMap extra_files;
  return deserializer.deserialize(device, extra_files);
}

Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& filename,
    std::optional<at::Device> device,
    bool load_debug_files) {
  ExtraFilesMap extra_files;
  return import_ir_module(
      std::move(cu), filename, device, extra_files, load_debug_files);
}

Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& filename,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files,
    bool restore_shapes) {
  // NOTE: Zipformat can be large files. So using stream version directly
  // instead of reading the file all at once.
  if (getFileFormat(filename) != FileFormat::FlatbufferFileFormat) {
    auto reader = std::make_unique<PyTorchStreamReader>(filename);
    reader->setShouldLoadDebugSymbol(load_debug_files);
    ScriptModuleDeserializer deserializer(std::move(cu), std::move(reader));
    return deserializer.deserialize(device, extra_files, restore_shapes);
  }
  auto [data, size] = get_file_content(filename.c_str());
  return _load_jit_module_from_bytes(
      data, size, cu, device, extra_files, restore_shapes);
}

Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::unique_ptr<ReadAdapterInterface> rai,
    std::optional<at::Device> device,
    bool load_debug_files) {
  ExtraFilesMap extra_files;
  return import_ir_module(
      std::move(cu), std::move(rai), device, extra_files, load_debug_files);
}

Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::unique_ptr<ReadAdapterInterface> rai,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files) {
  std::shared_ptr<ReadAdapterInterface> rai_shared = std::move(rai);
  return import_ir_module(
      std::move(cu), rai_shared, device, extra_files, load_debug_files);
}

Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::shared_ptr<ReadAdapterInterface> rai,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files) {
  auto reader = std::make_shared<PyTorchStreamReader>(std::move(rai));
  reader->setShouldLoadDebugSymbol(load_debug_files);
  ScriptModuleDeserializer deserializer(std::move(cu), std::move(reader));
  return deserializer.deserialize(device, extra_files);
}

Module load(
    std::istream& in,
    std::optional<at::Device> device,
    bool load_debug_files) {
  auto cu = std::make_shared<CompilationUnit>();
  return import_ir_module(std::move(cu), in, device, load_debug_files);
}

Module load(
    std::istream& in,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files) {
  auto cu = std::make_shared<CompilationUnit>();
  return import_ir_module(
      std::move(cu), in, device, extra_files, load_debug_files);
}

Module load(
    const std::string& filename,
    std::optional<at::Device> device,
    bool load_debug_files) {
  auto cu = std::make_shared<CompilationUnit>();
  return import_ir_module(std::move(cu), filename, device, load_debug_files);
}

Module load(
    const std::string& filename,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files) {
  auto cu = std::make_shared<CompilationUnit>();
  return import_ir_module(
      std::move(cu), filename, device, extra_files, load_debug_files);
}

Module load(
    std::shared_ptr<ReadAdapterInterface> rai,
    std::optional<c10::Device> device,
    bool load_debug_files) {
  auto cu = std::make_shared<CompilationUnit>();
  ExtraFilesMap extra_files;
  return import_ir_module(
      std::move(cu), std::move(rai), device, extra_files, load_debug_files);
}

Module load(
    std::shared_ptr<ReadAdapterInterface> rai,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files) {
  auto cu = std::make_shared<CompilationUnit>();
  return import_ir_module(
      std::move(cu), std::move(rai), device, extra_files, load_debug_files);
}

Module _load_jit_module_from_bytes(
    const std::shared_ptr<char>& data,
    size_t size,
    std::shared_ptr<CompilationUnit> cu,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool restore_shapes) {
  TORCH_CHECK(size >= kFileFormatHeaderSize, "Unrecognized data format");
  auto format = getFileFormat(data.get());
  switch (format) {
    case FileFormat::FlatbufferFileFormat: {
      return parse_and_initialize_jit_module(data, size, extra_files, device);
    }
    case FileFormat::ZipFileFormat: {
      auto rai = std::make_unique<MemoryReadAdapter>(data.get(), size);
      auto reader = std::make_unique<PyTorchStreamReader>(std::move(rai));
      ScriptModuleDeserializer deserializer(std::move(cu), std::move(reader));
      return deserializer.deserialize(device, extra_files, restore_shapes);
    }

    default:
      TORCH_CHECK(false, "Unrecognized data format");
  }
}

// Replace object with a newly created but equivalent object.
// The goal is to replace object's methods. However, since object's
// methods are attached to type; we need to replace it's type.
// Non-objects are unchanged; however, nested structures such as list, dict
// are also reconstructed because they might contain an object.
static IValue recreateObject(IValue ivalue, const TypeResolver& resolver) {
  if (ivalue.isObject()) {
    auto obj = ivalue.toObject();
    auto classtype_old = obj->type();
    auto newtype = resolver(*classtype_old->name());
    size_t n = classtype_old->numAttributes();
    auto newobj = c10::ivalue::Object::create(newtype, n);
    for (const auto i : c10::irange(n)) {
      newobj->setSlot(i, recreateObject(obj->getSlot(i), resolver));
    }
    return newobj;
  } else if (ivalue.isList()) {
    auto res = c10::impl::GenericList(ivalue.type()->containedType(0));
    for (const auto& ival : ivalue.toList()) {
      res.emplace_back(recreateObject(ival, resolver));
    }
    return res;
  } else if (ivalue.isGenericDict()) {
    auto result = c10::impl::GenericDict(
        ivalue.type()->containedType(0), ivalue.type()->containedType(1));
    for (const auto& kv : ivalue.toGenericDict()) {
      result.insert_or_assign(
          recreateObject(kv.key(), resolver),
          recreateObject(kv.value(), resolver));
    }
    return result;
  } else if (ivalue.isTuple()) {
    std::vector<IValue> res;
    for (const auto& ival : ivalue.toTuple()->elements()) {
      res.push_back(recreateObject(ival, resolver));
    }
    return c10::ivalue::Tuple::create(res);
  }
  // Leaf types are returned verbatim.
  return ivalue;
}

Module jitModuleFromSourceAndConstants(
    const IValue& ivalue,
    const ExtraFilesMap& source,
    const std::vector<IValue>& constants,
    int32_t version) {
  auto compilation_unit = std::make_shared<CompilationUnit>();
  SourceImporter importer(
      compilation_unit,
      &constants,
      [&source](const std::string& qualifier) -> std::shared_ptr<Source> {
        auto source_iter = source.find(qualifier);
        if (source_iter == source.end()) {
          return nullptr;
        }
        return std::make_shared<Source>(
            source_iter->second, qualifier, 1, nullptr, Source::COPIES_STRING);
      },
      version);
  auto type_resolver = [&](const c10::QualifiedName& qn) {
    auto cls = importer.loadType(qn);
    return c10::StrongTypePtr(compilation_unit, std::move(cls));
  };
  auto newIvalue = recreateObject(ivalue, type_resolver).toObject();
  Module m(newIvalue);
  rewriteQuantizedConvForBC(m);
  return m;
}

} // namespace torch::jit
