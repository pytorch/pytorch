#ifdef FLATBUFFERS_VERSION_MAJOR
#error "flatbuffer_loader.h must not include any flatbuffers headers"
#endif // FLATBUFFERS_VERSION_MAJOR

#include <array>
#include <istream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/dynamic_type.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/qualified_name.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/util/Exception.h>
#include <c10/util/ScopeExit.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/mobile/file_format.h>
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/observer.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <torch/custom_class.h>
#include <optional>

#ifndef DISABLE_UPGRADER
#include <torch/csrc/jit/mobile/parse_bytecode.h>
#include <torch/csrc/jit/mobile/upgrader_mobile.h>
#endif

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

#if defined(FB_XPLAT_BUILD) || defined(FBCODE_CAFFE2)
#include <torch/csrc/jit/serialization/mobile_bytecode_generated_fbsource.h> // NOLINT
namespace flatbuffers = flatbuffers_fbsource;
#define FLATBUFFERS_MAX_ALIGNMENT FLATBUFFERS_FBSOURCE_MAX_ALIGNMENT
#else
#include <torch/csrc/jit/serialization/mobile_bytecode_generated.h> // NOLINT
#endif

namespace torch::jit {

// Our own alignment requirement does not need to be exactly the same as what
// flatbuffers supports, but what flatbuffers supports needs to satisfy our
// requirement.
static_assert(
    kFlatbufferDataAlignmentBytes <= FLATBUFFERS_MAX_ALIGNMENT,
    "Sizes must be compatible");
static_assert(
    (kFlatbufferDataAlignmentBytes & ~(kFlatbufferDataAlignmentBytes - 1)) ==
        kFlatbufferDataAlignmentBytes,
    "Must be a power of 2");

namespace {

static constexpr std::string_view kCustomClassPrefix =
    "__torch__.torch.classes";
static constexpr std::string_view kTorchPrefix = "__torch__";
static constexpr std::string_view kJitPrefix = "torch.jit";

class FlatbufferLoader final {
 public:
  FlatbufferLoader();

  typedef IValue (
      *IValueParser)(FlatbufferLoader&, const mobile::serialization::IValue&);
  void registerIValueParser(
      mobile::serialization::IValueUnion ivalue_type,
      IValueParser parser);
  mobile::Module parseModule(mobile::serialization::Module* module, char* end);

  void extractJitSourceAndConstants(
      ExtraFilesMap* jit_sources,
      std::vector<IValue>* constants);

  using TypeResolver = TypePtr (*)(
      const std::string& type_str,
      const std::shared_ptr<CompilationUnit>& cu);

  void internal_registerTypeResolver(TypeResolver type_resolver);

  IValue& getIValue(uint32_t pos) {
    TORCH_CHECK(pos < all_ivalues_.size());
    return all_ivalues_[pos];
  }

  mobile::Function* getFunction(uint32_t pos) {
    return all_functions_[pos];
  }

  ClassTypePtr getType(uint32_t pos) {
    TORCH_CHECK(pos < all_types_.size());
    return all_types_[pos];
  }

  c10::Storage getStorage(uint32_t index);
  TypePtr getOrCreateTypeAnnotations(const flatbuffers::String* offset);
  ClassTypePtr getOrCreateClassTypeForObject(
      const mobile::serialization::Object* object);

  const mobile::serialization::Module* getCurrentFlatbufferInput() {
    return module_;
  }

  void setShouldCopyTensorMemory(bool should_copy_tensor_memory) {
    should_copy_tensor_memory_ = should_copy_tensor_memory;
  }

  std::shared_ptr<mobile::CompilationUnit> mcu_;
  std::shared_ptr<CompilationUnit> cu_;

 private:
  IValue parseIValue(const mobile::serialization::IValue* ivalue);
  std::unique_ptr<mobile::Function> parseFunction(
      const mobile::serialization::Function* method);
  void parseAndPopulate(
      uint32_t i,
      const mobile::serialization::IValue* ivalue);

  std::unordered_map<uint32_t, mobile::Function*> all_functions_;
  std::vector<ClassTypePtr> all_types_;
  std::unordered_set<uint32_t> initialized_types_;
  std::unordered_map<const flatbuffers::String*, TypePtr> type_annotations_;
  std::vector<bool> storage_loaded_;
  std::vector<c10::Storage> storages_;
  std::vector<IValue> all_ivalues_;
  std::array<
      IValueParser,
      static_cast<uint8_t>(mobile::serialization::IValueUnion::MAX) + 1>
      ivalue_parsers_;
  TypeResolver type_resolver_ = nullptr;
  mobile::serialization::Module* module_ = nullptr;
  bool module_parsed_ = false;
  bool should_copy_tensor_memory_ = false;
  // 0 -> mobile_ivalue_size_ elements are from the mobile module.
  uint32_t mobile_ivalue_size_ = 0;
};

IValue parseList(
    FlatbufferLoader& /*loader*/,
    const mobile::serialization::IValue& ivalue);
IValue parseTensor(
    FlatbufferLoader& /*loader*/,
    const mobile::serialization::IValue& ivalue);
IValue parseTuple(
    FlatbufferLoader& /*loader*/,
    const mobile::serialization::IValue& ivalue);
IValue parseDict(
    FlatbufferLoader& /*loader*/,
    const mobile::serialization::IValue& ivalue);
IValue parseObject(
    FlatbufferLoader& /*loader*/,
    const mobile::serialization::IValue& ivalue);
IValue parseIntList(
    FlatbufferLoader& /*unused*/,
    const mobile::serialization::IValue& ivalue);
IValue parseDoubleList(
    FlatbufferLoader& /*unused*/,
    const mobile::serialization::IValue& ivalue);
IValue parseBoolList(
    FlatbufferLoader& /*unused*/,
    const mobile::serialization::IValue& ivalue);
IValue parseBasic(
    FlatbufferLoader& /*unused*/,
    const mobile::serialization::IValue& ivalue);
IValue parseEnum(
    FlatbufferLoader& /*loader*/,
    const mobile::serialization::IValue& ivalue);

TypePtr resolveType(
    const std::string& type_string,
    const std::shared_ptr<CompilationUnit>& cu) {
  TypePtr type;
  std::string_view type_str(type_string);
  if (c10::starts_with(type_str, kCustomClassPrefix)) {
    type = getCustomClass(type_string);
    TORCH_CHECK(
        type, "The implementation of class ", type_string, " cannot be found.");
  } else if (
      c10::starts_with(type_str, kTorchPrefix) ||
      c10::starts_with(type_str, kJitPrefix)) {
    c10::QualifiedName qn(type_string);
    if (cu->get_class(qn) == nullptr) {
      auto classtype = ClassType::create(qn, cu, true);
      cu->register_type(classtype);
      type = classtype;
    } else {
      type = cu->get_class(qn);
    }
  } else {
    type = c10::parseType(type_string);
  }
  return type;
}

FlatbufferLoader::FlatbufferLoader()
    : mcu_(std::make_shared<mobile::CompilationUnit>()),
      cu_(std::make_shared<CompilationUnit>()),
      ivalue_parsers_{nullptr} {
  registerIValueParser(mobile::serialization::IValueUnion::NONE, &parseBasic);
  registerIValueParser(mobile::serialization::IValueUnion::Int, &parseBasic);
  registerIValueParser(mobile::serialization::IValueUnion::Bool, &parseBasic);
  registerIValueParser(mobile::serialization::IValueUnion::Double, &parseBasic);
  registerIValueParser(
      mobile::serialization::IValueUnion::ComplexDouble, &parseBasic);
  registerIValueParser(
      mobile::serialization::IValueUnion::TensorMetadata, &parseTensor);
  registerIValueParser(mobile::serialization::IValueUnion::String, &parseBasic);
  registerIValueParser(mobile::serialization::IValueUnion::List, &parseList);
  registerIValueParser(
      mobile::serialization::IValueUnion::IntList, &parseIntList);
  registerIValueParser(
      mobile::serialization::IValueUnion::DoubleList, &parseDoubleList);
  registerIValueParser(
      mobile::serialization::IValueUnion::BoolList, &parseBoolList);
  registerIValueParser(mobile::serialization::IValueUnion::Tuple, &parseTuple);
  registerIValueParser(mobile::serialization::IValueUnion::Dict, &parseDict);
  registerIValueParser(
      mobile::serialization::IValueUnion::Object, &parseObject);
  registerIValueParser(mobile::serialization::IValueUnion::Device, &parseBasic);
  registerIValueParser(
      mobile::serialization::IValueUnion::EnumValue, &parseEnum);
  internal_registerTypeResolver(&resolveType);
}

void FlatbufferLoader::registerIValueParser(
    mobile::serialization::IValueUnion ivalue_type,
    IValueParser parser) {
  ivalue_parsers_[static_cast<uint8_t>(ivalue_type)] = parser;
}

void FlatbufferLoader::internal_registerTypeResolver(
    TypeResolver type_resolver) {
  type_resolver_ = type_resolver;
}

void parseExtraFilesFromVector(
    const flatbuffers::Vector<flatbuffers::Offset<
        torch::jit::mobile::serialization::ExtraFile>>* files,
    ExtraFilesMap* extra_files) {
  for (uint32_t i = 0; i < files->size(); ++i) {
    const auto* extra_file = files->Get(i);
    (*extra_files)[extra_file->name()->str()] = extra_file->content()->str();
  }
}

void parseExtraFiles(
    mobile::serialization::Module* module,
    ExtraFilesMap& extra_files) {
  auto extra_files_offsets = module->extra_files();
  parseExtraFilesFromVector(extra_files_offsets, &extra_files);
}

void FlatbufferLoader::parseAndPopulate(
    uint32_t i,
    const mobile::serialization::IValue* ivalue) {
  if (const auto* func = ivalue->val_as_Function()) {
    auto func_ptr = parseFunction(func);
    all_functions_[i] = func_ptr.get();
    mcu_->register_function(std::move(func_ptr));
  } else {
    all_ivalues_[i] = parseIValue(ivalue);
  }
}

mobile::Module FlatbufferLoader::parseModule(
    mobile::serialization::Module* module,
    char* end) {
  module_ = module;
  all_ivalues_.clear();
  all_types_.clear();
  storages_.clear();
  storage_loaded_.clear();
  module_parsed_ = false;

  const auto* ivalues = module->ivalues();
  TORCH_CHECK(
      ivalues && module->object_types(),
      "Parsing flatbuffer module: Corrupted ivalues/object_types field");
  TORCH_CHECK(
      reinterpret_cast<const char*>(ivalues) < end, "Corrupted ivalues field");
  TORCH_CHECK(
      module->storage_data_size() >= 0,
      "Parsing flatbuffer module: illegal storage_data_size: ",
      module->storage_data_size(),
      ", expected to be non negative");
  all_ivalues_.resize(ivalues->size());
  all_types_.resize(module->object_types()->size());
  storages_.resize(module->storage_data_size());
  storage_loaded_.resize(module->storage_data_size(), false);

  mobile_ivalue_size_ = module_->mobile_ivalue_size();
  if (mobile_ivalue_size_ == 0 || mobile_ivalue_size_ > ivalues->size()) {
    mobile_ivalue_size_ = ivalues->size();
  }

  for (uint32_t i = 0; i < mobile_ivalue_size_; i++) {
    const auto* ival = ivalues->Get(i);
    TORCH_CHECK(
        reinterpret_cast<const char*>(ival) < end, "Corrupted ivalue item")
    parseAndPopulate(i, ival);
  }
  IValue& module_ivalue = getIValue(module->state_obj());

  // register functions
  for (const auto& f : all_functions_) {
    uint32_t class_index =
        ivalues->Get(f.first)->val_as_Function()->class_type();
    ClassTypePtr class_type = all_types_[class_index];
    class_type->addMethod(f.second);
  }

  module_parsed_ = true;
  auto m = mobile::Module(module_ivalue.toObject(), mcu_);
  m.set_min_operator_version(module->operator_version());
  m.set_bytecode_version(module->bytecode_version());
  return m;
}

void appendUpgraderFunctions(mobile::Function* function) {
#ifndef DISABLE_UPGRADER
  for (auto& byteCodeFunctionWithOperator : getUpgraderBytecodeList()) {
    function->append_function(byteCodeFunctionWithOperator.function);
  }
#endif
}

std::unique_ptr<mobile::Function> FlatbufferLoader::parseFunction(
    const mobile::serialization::Function* method) {
  auto function = std::make_unique<mobile::Function>(
      c10::QualifiedName(method->qn()->str()));
  // TODO(qihan) add debug handle
  // const auto* debug_handle = method->debug_info()->debug_handle();
  for (const auto* inst : *method->instructions()) {
    function->append_instruction(
        static_cast<OpCode>(inst->op()), inst->x(), inst->n());
  }

  for (uint32_t i : *method->constants()) {
    function->append_constant(getIValue(i));
  }

  appendUpgraderFunctions(function.get());
  // 2. Decides if upgrader is needed
  const uint32_t operator_version = module_->operator_version();
  bool use_upgrader =
      (operator_version < caffe2::serialize::kProducedFileFormatVersion);

  for (const auto* op : *method->operators()) {
    std::optional<int> num_args = std::nullopt;
    if (op->num_args_serialized() > -1) {
      num_args = op->num_args_serialized();
    }

    function->append_operator(
        op->name()->str(), op->overload_name()->str(), num_args);
  }

  function->initialize_operators(true);

  for (const auto i : *method->type_annotations()) {
    function->append_type(getOrCreateTypeAnnotations(i));
  }

  // 3. If upgrader is needed, change change the OP instruction to CALL
  // instruction (In next PR, use_upgrader will be parsed to parseInstruction
  // function and do the actual change)
  if (use_upgrader) {
#ifndef DISABLE_UPGRADER
    applyUpgrader(function.get(), operator_version);
#endif
  }

  function->set_register_size(method->register_size());
  if (method->schema()) {
    try {
      auto parseArgList = [this](const auto* args_fb) {
        std::vector<c10::Argument> args;
        for (const auto* arg_tb : *args_fb) {
          IValue default_value = getIValue(arg_tb->default_value());
          TypePtr type_ptr = getOrCreateTypeAnnotations(arg_tb->type());
          auto arg = c10::Argument(
              arg_tb->name()->str(),
              std::move(type_ptr),
              std::nullopt /*N*/,
              std::move(default_value));
          args.emplace_back(std::move(arg));
        }
        return args;
      };
      c10::FunctionSchema schema(
          method->qn()->str(),
          "" /*overload_name*/,
          parseArgList(method->schema()->arguments()),
          parseArgList(method->schema()->returns()),
          false /*is_varargs*/,
          false /*is_varret*/);

      function->setSchema(std::move(schema));
    } catch (const c10::Error& e) {
    }
  }
  return function;
}

IValue parseEnum(
    FlatbufferLoader& loader,
    const mobile::serialization::IValue& ivalue) {
  const auto* enum_val = ivalue.val_as_EnumValue();
  auto enum_type = loader.getOrCreateTypeAnnotations(enum_val->type_name())
                       ->cast<c10::EnumType>();
  AT_ASSERT(
      enum_type,
      "Enum with type: " + enum_val->type_name()->str() + " not found.");
  IValue val = loader.getIValue(enum_val->value());
  for (const auto& p : enum_type->enumNamesValues()) {
    if (p.second == val) {
      auto enum_holder = c10::make_intrusive<at::ivalue::EnumHolder>(
          enum_type, p.first, p.second);
      return IValue(std::move(enum_holder));
    }
  }
  AT_ASSERT(
      false, "Enum with type: " + enum_val->type_name()->str() + " not found.");
}

IValue parseBasic(
    FlatbufferLoader& /*unused*/,
    const mobile::serialization::IValue& ivalue) {
  switch (ivalue.val_type()) {
    case mobile::serialization::IValueUnion::NONE:
      return {};
    case mobile::serialization::IValueUnion::Int:
      return ivalue.val_as_Int()->int_val();
    case mobile::serialization::IValueUnion::Bool:
      return ivalue.val_as_Bool()->bool_val();
    case mobile::serialization::IValueUnion::Double:
      return ivalue.val_as_Double()->double_val();
    case mobile::serialization::IValueUnion::ComplexDouble: {
      const auto* comp = ivalue.val_as_ComplexDouble();
      return c10::complex<double>(comp->real(), comp->imag());
    }
    case mobile::serialization::IValueUnion::String:
      return ivalue.val_as_String()->data()->str();
    case mobile::serialization::IValueUnion::Device: {
      return c10::Device(ivalue.val_as_Device()->str()->str());
    }
    default:
      return {};
  }
}

at::Tensor parseTensorFromMetadata(
    FlatbufferLoader* loader,
    const mobile::serialization::TensorMetadata* tensor_md) {
  auto type = static_cast<at::ScalarType>(tensor_md->scalar_type());
  auto options = at::device(at::kCPU).dtype(type);
  at::Tensor tensor;
  if (tensor_md->quantized_schema() != nullptr) {
    // is quantized
    const auto* schema = tensor_md->quantized_schema();
    auto qscheme_type = static_cast<at::QScheme>(schema->qscheme());
    switch (qscheme_type) {
      case at::kPerTensorAffine: {
        tensor = at::_empty_affine_quantized(
            {0}, options, schema->scale(), schema->zero_point());
      } break;
      case at::kPerChannelAffineFloatQParams:
      case at::kPerChannelAffine: {
        at::Tensor scales = parseTensorFromMetadata(loader, schema->scales());
        at::Tensor zero_points =
            parseTensorFromMetadata(loader, schema->zero_points());
        tensor = at::_empty_per_channel_affine_quantized(
            {0}, scales, zero_points, schema->axis(), options);
      } break;
      default:
        TORCH_CHECK(
            false,
            "Unsupported tensor quantization type in serialization ",
            toString(qscheme_type));
        break;
    }
  } else {
    tensor = at::empty({0}, options);
  }
  at::TensorImpl* impl = tensor.unsafeGetTensorImpl();

  c10::Storage storage;
  storage = loader->getStorage(tensor_md->storage_location_index());
  impl->set_storage_keep_dtype(storage);
  impl->set_storage_offset(tensor_md->storage_offset());

  std::vector<int64_t> size{
      tensor_md->sizes()->begin(), tensor_md->sizes()->end()};
  std::vector<int64_t> stride{
      tensor_md->strides()->begin(), tensor_md->strides()->end()};
  impl->set_sizes_and_strides(size, stride);
#ifndef MIN_EDGE_RUNTIME
  tensor = autograd::make_variable(tensor, tensor_md->requires_grad());
#endif
  return tensor;
}

IValue parseTensor(
    FlatbufferLoader& loader,
    const mobile::serialization::IValue& ivalue) {
  const mobile::serialization::TensorMetadata* tensor_md =
      ivalue.val_as_TensorMetadata();
  return parseTensorFromMetadata(&loader, tensor_md);
}

IValue parseList(
    FlatbufferLoader& loader,
    const mobile::serialization::IValue& ivalue) {
  const mobile::serialization::List* list = ivalue.val_as_List();
  auto res = c10::impl::GenericList(AnyType::get());
  for (auto i : *list->items()) {
    res.emplace_back(loader.getIValue(i));
  }
  auto type = loader.getOrCreateTypeAnnotations(list->annotation_str());
  res.unsafeSetElementType(type->containedType(0));
  return res;
}

template <typename T, typename U>
std::vector<T> parseListNative(const U* list) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(list != nullptr);
  return {list->items()->begin(), list->items()->end()};
}

IValue parseIntList(
    FlatbufferLoader& /*unused*/,
    const mobile::serialization::IValue& ivalue) {
  const auto& list = ivalue.val_as_IntList();
  return parseListNative<int64_t>(list);
}

IValue parseDoubleList(
    FlatbufferLoader& /*unused*/,
    const mobile::serialization::IValue& ivalue) {
  const auto& list = ivalue.val_as_DoubleList();
  return parseListNative<double>(list);
}

IValue parseBoolList(
    FlatbufferLoader& /*unused*/,
    const mobile::serialization::IValue& ivalue) {
  const auto& list = ivalue.val_as_BoolList();
  std::vector<uint8_t> res = parseListNative<uint8_t>(list);
  c10::List<bool> boollist;
  for (auto x : res) {
    boollist.push_back(x);
  }
  return boollist;
}

IValue parseTuple(
    FlatbufferLoader& loader,
    const mobile::serialization::IValue& ivalue) {
  const auto& tuple = ivalue.val_as_Tuple();
  const auto items = tuple->items();
  std::vector<IValue> res;
  res.reserve(items->size());
  for (auto i : *items) {
    res.emplace_back(loader.getIValue(i));
  }
  return c10::ivalue::Tuple::create(std::move(res));
}

IValue parseDict(
    FlatbufferLoader& loader,
    const mobile::serialization::IValue& ivalue) {
  const auto* dict = ivalue.val_as_Dict();
  auto result = c10::impl::GenericDict(AnyType::get(), AnyType::get());
  const auto* keys = dict->keys();
  const auto* values = dict->values();
  for (size_t i = 0; i < keys->size(); ++i) {
    uint32_t key = keys->Get(i);
    uint32_t val = values->Get(i);
    result.insert_or_assign(loader.getIValue(key), loader.getIValue(val));
  }
  auto type = loader.getOrCreateTypeAnnotations(dict->annotation_str());
  result.unsafeSetKeyType(type->containedType(0));
  result.unsafeSetValueType(type->containedType(1));
  return result;
}

ClassTypePtr FlatbufferLoader::getOrCreateClassTypeForObject(
    const mobile::serialization::Object* object) {
  auto cls = getType(object->type_index());
  const mobile::serialization::ObjectType* obj_type =
      module_->object_types()->Get(object->type_index());
  if (cls == nullptr) {
    std::string_view qn_str(
        obj_type->type_name()->c_str(), obj_type->type_name()->size());
    if (c10::starts_with(qn_str, kTorchPrefix) ||
        c10::starts_with(qn_str, kJitPrefix)) {
      c10::QualifiedName qn(obj_type->type_name()->str());
      cls = cu_->get_class(qn);
      if (cls == nullptr) {
        cls = ClassType::create(qn, cu_, true);
        cu_->register_type(cls);
      }
    } else {
      cls = c10::parseType(std::string(qn_str))->cast<ClassType>();
    }
    TORCH_CHECK(object->type_index() < all_ivalues_.size());
    all_types_[object->type_index()] = cls;

    if (obj_type->type() == mobile::serialization::TypeType::CLASS_WITH_FIELD) {
      for (uint32_t i = 0; i < object->attrs()->size(); i++) {
        IValue val = getIValue(object->attrs()->Get(i));
        // Need to use concrete object's field's type to set type of field.
        cls->addAttribute(
            obj_type->attr_names()->Get(i)->str(),
            val.type<c10::DynamicType>());
      }
    }
    initialized_types_.insert(object->type_index());
  }
  return cls;
}

IValue parseObject(
    FlatbufferLoader& loader,
    const mobile::serialization::IValue& ivalue) {
  const mobile::serialization::Object* object = ivalue.val_as_Object();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(object != nullptr);
  const auto* cur_input = loader.getCurrentFlatbufferInput();
  const mobile::serialization::ObjectType* obj_type =
      cur_input->object_types()->Get(object->type_index());
  auto cls = loader.getOrCreateClassTypeForObject(object);
  Stack stack;
  switch (obj_type->type()) {
    case mobile::serialization::TypeType::CLASS_WITH_FIELD: {
      auto obj = c10::ivalue::Object::create(
          at::StrongTypePtr(loader.cu_, cls), object->attrs()->size());
      for (uint32_t i = 0; i < object->attrs()->size(); i++) {
        IValue val = loader.getIValue(object->attrs()->Get(i));
        obj->setSlot(i, std::move(val));
      }
      return obj;
    }
    case mobile::serialization::TypeType::CLASS_WITH_SETSTATE: {
      IValue input = loader.getIValue(object->state());
      mobile::Function* setstate = loader.getFunction(object->setstate_func());
      auto obj =
          c10::ivalue::Object::create(at::StrongTypePtr(loader.cu_, cls), 0);
      stack.emplace_back(obj);
      stack.emplace_back(std::move(input));
      setstate->run(stack);
      return obj;
    }
    case mobile::serialization::TypeType::CUSTOM_CLASS: {
      auto custom_class_type =
          torch::jit::getCustomClass(cls->name()->qualifiedName());
      IValue input = loader.getIValue(object->state());
      auto obj = c10::ivalue::Object::create(
          c10::StrongTypePtr(nullptr, custom_class_type), 1);
      stack.emplace_back(obj);
      stack.emplace_back(std::move(input));
      custom_class_type->getMethod("__setstate__").run(stack);
      return obj;
    }
    default:
      AT_ASSERT(false, "need to be object");
  }
}

IValue FlatbufferLoader::parseIValue(
    const mobile::serialization::IValue* ivalue) {
  return ivalue_parsers_[static_cast<uint32_t>(ivalue->val_type())](
      *this, *ivalue);
}

void deleteNothing2(void* /*unused*/);
void deleteNothing2(void* /*unused*/) {}

c10::Storage FlatbufferLoader::getStorage(uint32_t index) {
  TORCH_CHECK(index < storage_loaded_.size());
  TORCH_CHECK(index < storages_.size());
  if (!storage_loaded_[index]) {
    auto* storage = module_->storage_data()->GetMutableObject(index);
    size_t size = storage->data()->size();

    at::DataPtr data;
    if (should_copy_tensor_memory_) {
      auto* allocator = at::GetCPUAllocator();
      data = allocator->allocate(size);
      memcpy(data.get(), storage->data()->data(), size);
    } else {
      void* ptr = static_cast<void*>(storage->mutable_data()->data());
      data = at::DataPtr(ptr, ptr, deleteNothing2, DeviceType::CPU);
    }
    storages_[index] =
        c10::Storage(c10::Storage::use_byte_size_t(), size, std::move(data));
    storage_loaded_[index] = true;
  }
  return storages_[index];
}

TypePtr FlatbufferLoader::getOrCreateTypeAnnotations(
    const flatbuffers::String* offset) {
  auto iter = type_annotations_.find(offset);
  if (iter != type_annotations_.end()) {
    return iter->second;
  }
  TypePtr type = type_resolver_(offset->str(), cu_);
  type_annotations_[offset] = type;
  return type;
}

void FlatbufferLoader::extractJitSourceAndConstants(
    ExtraFilesMap* jit_sources,
    std::vector<IValue>* constants) {
  AT_ASSERT(
      module_parsed_,
      "Need to first parse a flatbuffer file before extracting jit_sources");

  const auto* ivalues = module_->ivalues();
  for (uint32_t i = mobile_ivalue_size_; i < ivalues->size(); i++) {
    const auto* ival = ivalues->Get(i);
    parseAndPopulate(i, ival);
  }
  // register functions
  for (const auto& f : all_functions_) {
    if (f.first >= mobile_ivalue_size_) {
      uint32_t class_index =
          ivalues->Get(f.first)->val_as_Function()->class_type();
      ClassTypePtr class_type = all_types_[class_index];
      class_type->addMethod(f.second);
    }
  }
  const auto* jit_constants = module_->jit_constants();
  for (const auto i : c10::irange(jit_constants->size())) {
    constants->emplace_back(getIValue(jit_constants->Get(i)));
  }
  parseExtraFilesFromVector(module_->jit_sources(), jit_sources);
}

} // namespace

mobile::Module parse_and_initialize_mobile_module(
    void* data,
    size_t size,
    std::optional<at::Device> /*unused*/,
    ExtraFilesMap* extra_files,
    bool should_copy_tensor_memory) {
  // TODO(T128189662): If not copying, enforce that data is aligned to
  // kFlatbufferDataAlignmentBytes, and add unit tests.

  // Validate Flatbuffer module before parsing.
  flatbuffers::Verifier verifier(reinterpret_cast<uint8_t*>(data), size);
  TORCH_CHECK(
      mobile::serialization::VerifyModuleBuffer(verifier),
      "Malformed Flatbuffer module");

  FlatbufferLoader loader;
  loader.setShouldCopyTensorMemory(should_copy_tensor_memory);

  // Flatbuffer doesn't seem to have a way to provide the buffer size when
  // interacting with the buffer.
  auto* flatbuffer_module = mobile::serialization::GetMutableModule(data);
  auto* end = static_cast<char*>(data) + size;
  mobile::Module m = loader.parseModule(flatbuffer_module, end);
  if (extra_files != nullptr) {
    parseExtraFiles(flatbuffer_module, *extra_files);
  }
  return m;
}

mobile::Module parse_and_initialize_mobile_module(
    std::shared_ptr<char> data,
    size_t size,
    std::optional<at::Device> device,
    ExtraFilesMap* extra_files) {
  mobile::Module m = parse_and_initialize_mobile_module(
      data.get(),
      size,
      device,
      extra_files,
      /*should_copy_tensor_memory=*/false);
  m.set_delete_memory(std::move(data));
  return m;
}

mobile::Module parse_and_initialize_mobile_module_for_jit(
    void* data,
    size_t size,
    ExtraFilesMap& jit_sources,
    std::vector<IValue>& jit_constants,
    std::optional<at::Device> /*unused*/,
    ExtraFilesMap* extra_files) {
  TORCH_CHECK(
      mobile::serialization::ModuleBufferHasIdentifier(data), "Format error");
  // TODO(T128189662): Enforce that data is aligned to
  // kFlatbufferDataAlignmentBytes, and add unit tests.

  // Validate Flatbuffer module before parsing.
  flatbuffers::Verifier verifier(reinterpret_cast<uint8_t*>(data), size);
  TORCH_CHECK(
      mobile::serialization::VerifyModuleBuffer(verifier),
      "Malformed Flatbuffer module");

  FlatbufferLoader loader;
  auto* flatbuffer_module = mobile::serialization::GetMutableModule(data);
  auto* end = static_cast<char*>(data) + size;
  mobile::Module m = loader.parseModule(flatbuffer_module, end);
  if (extra_files != nullptr) {
    parseExtraFiles(flatbuffer_module, *extra_files);
  }

  loader.extractJitSourceAndConstants(&jit_sources, &jit_constants);
  return m;
}

mobile::Module load_mobile_module_from_file(
    const std::string& filename,
    std::optional<c10::Device> device,
    ExtraFilesMap* extra_files) {
  auto [data, size] = get_file_content(filename.c_str());
  return parse_and_initialize_mobile_module(
      std::move(data), size, device, extra_files);
}

uint64_t get_bytecode_version(std::istream& in) {
  auto [data, size] = get_stream_content(in);
  return get_bytecode_version_from_bytes(data.get());
}

uint64_t get_bytecode_version(const std::string& filename) {
  auto [data, size] = get_file_content(filename.c_str());
  return get_bytecode_version_from_bytes(data.get());
}

uint64_t get_bytecode_version_from_bytes(char* flatbuffer_content) {
  TORCH_CHECK(
      mobile::serialization::ModuleBufferHasIdentifier(flatbuffer_content),
      "Format error");
  auto* flatbuffer_module =
      mobile::serialization::GetMutableModule(flatbuffer_content);
  return flatbuffer_module->bytecode_version();
}

mobile::ModuleInfo get_module_info_from_flatbuffer(char* flatbuffer_content) {
  auto* ff_module = mobile::serialization::GetMutableModule(flatbuffer_content);
  mobile::ModuleInfo minfo;
  minfo.operator_version = ff_module->operator_version();
  minfo.bytecode_version = ff_module->bytecode_version();

  uint32_t mobile_ivalue_size = ff_module->mobile_ivalue_size();
  if (mobile_ivalue_size == 0) {
    mobile_ivalue_size = ff_module->ivalues()->size();
  }

  std::vector<std::string> type_name_list;
  for (uint32_t i = 0; i < mobile_ivalue_size; i++) {
    const auto* ival = ff_module->ivalues()->Get(i);
    if (const auto* func = ival->val_as_Function()) {
      minfo.function_names.insert(func->qn()->str());
      for (const auto* op : *func->operators()) {
        at::OperatorName opname(op->name()->str(), op->overload_name()->str());
        minfo.opname_to_num_args[mobile::operator_str(opname)] =
            op->num_args_serialized();
      }
      for (const auto* type_ann : *func->type_annotations()) {
        type_name_list.push_back(type_ann->str());
      }
    }
  }
  c10::TypeParser parser(type_name_list);
  parser.parseList();
  minfo.type_names = parser.getContainedTypes();
  return minfo;
}

mobile::Module load_mobile_module_from_stream_with_copy(
    std::istream& in,
    std::optional<at::Device> device,
    ExtraFilesMap* extra_files) {
  auto [data, size] = get_stream_content(in);
  return parse_and_initialize_mobile_module(
      std::move(data), size, device, extra_files);
}

mobile::Module parse_flatbuffer_no_object(
    std::shared_ptr<char> data,
    size_t size,
    std::optional<at::Device> device) {
  (void)device;
  (void)size;

  // Validate Flatbuffer module before parsing.
  flatbuffers::Verifier verifier(reinterpret_cast<uint8_t*>(data.get()), size);
  TORCH_CHECK(
      mobile::serialization::VerifyModuleBuffer(verifier),
      "Malformed Flatbuffer module");

  auto* flatbuffer_module = mobile::serialization::GetMutableModule(data.get());
  FlatbufferLoader loader;
  // replace parserObject with to handle only class with field case
  // function.
  loader.registerIValueParser(
      mobile::serialization::IValueUnion::Object,
      +[](FlatbufferLoader& loader,
          const mobile::serialization::IValue& ivalue) {
        const mobile::serialization::Object* object = ivalue.val_as_Object();
        auto cls = loader.getOrCreateClassTypeForObject(object);
        auto obj = c10::ivalue::Object::create(
            at::StrongTypePtr(loader.cu_, cls), object->attrs()->size());
        for (uint32_t i = 0; i < object->attrs()->size(); i++) {
          IValue val = loader.getIValue(object->attrs()->Get(i));
          obj->setSlot(i, std::move(val));
        }
        return static_cast<c10::IValue>(obj);
      });

  auto* end = data.get() + size;
  mobile::Module m = loader.parseModule(flatbuffer_module, end);
  m.set_delete_memory(std::move(data));
  return m;
}

bool register_flatbuffer_loader() {
  return true;
}

} // namespace torch::jit
