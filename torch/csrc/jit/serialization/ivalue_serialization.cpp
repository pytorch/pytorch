#include <torch/csrc/jit/serialization/ivalue_serialization.h>
#include "torch/csrc/jit/serialization/mobile_bytecode_generated.h"

#include <ATen/ATen.h>
#include <ATen/core/Dict.h>
#include <aten/src/ATen/quantized/Quantizer.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/serialization/mobile_bytecode_generated.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <string>


namespace torch {
namespace jit {

using mobile::serialization::CreateTupleDirect;
using mobile::serialization::CreateList;
using mobile::serialization::CreateDict;
using mobile::serialization::CreateTensorMetadataDirect;
using mobile::serialization::CreateArg;
using mobile::serialization::CreateOperator;
using mobile::serialization::CreateObject;
using mobile::serialization::CreateFunctionDirect;
using mobile::serialization::CreateDebugInfo;
using mobile::serialization::CreateModule;
using flatbuffers::FlatBufferBuilder;

char const* toString(OpCode op);
std::tuple<
    std::vector<c10::OperatorName>,
    std::vector<std::string>,
    std::vector<int64_t>>
// TODO get rid of thsi
convertInstructionsForMobile2(
  const MobileCode* code,
  std::vector<Instruction>* instructions,
  BackendDebugInfoRecorder& debug_info_recorder
) {
  std::vector<c10::OperatorName> opnames;
  std::vector<std::string> method_names;
  std::vector<int64_t> op_debug_handles;
  for (size_t i = 0; i < instructions->size(); ++i) {
    Instruction ins = instructions->at(i);
    if (ins.op == OP || ins.op == OPN) {
      auto node = code->instructions_source()[i];
      opnames.emplace_back(node->schema().operator_name());
    }
    // CALL nodes at this point represent built-in (i.e. non-Graph)
    // functions that were not inlined. Here we convert the CALL
    // instructions for these functions into INTERFACE_CALL instructions
    // s.t. at runtime, we will look up the Function* on the Type of the
    // 0th argument in the stack and call that directly.
    if (ins.op == CALL) {
      auto node = code->instructions_source()[i];
      if (node->kind() == prim::CallMethod) {
        // NB: replacing instruction
        auto method_name_idx =
            code->constant_table().size() + method_names.size();
        method_names.emplace_back(node->s(attr::name));
        Instruction new_instr{
            INTERFACE_CALL,
            static_cast<int32_t>(method_name_idx),
            static_cast<uint16_t>(node->inputs().size())};
        instructions->at(i) = new_instr;
      } else {
        TORCH_INTERNAL_ASSERT(
            false, "Unsupported node kind on CALL opcode for mobile");
      }
    } else if (ins.op == RET) {
      auto node = code->instructions_source()[i];
      for (const auto& input : node->inputs()) {
        const auto& input_type = input->type();
        if (input_type->kind() == TypeKind::TupleType) {
          if (const auto& name_typed_input =
                  input_type->cast<at::NamedType>()) {
            TORCH_CHECK(
                !name_typed_input->name(),
                "A named tuple type is not supported in mobile module. ",
                "Workaround: instead of using a named tuple type's fields, ",
                "use a dictionary type's key-value pair itmes or ",
                "a pytorch class (class Foo(torch.nn.Module))'s attributes.'");
          }
        } else if (
            input_type->kind() == TypeKind::ListType ||
            input_type->kind() == TypeKind::DictType) {
          for (const TypePtr& element_type : input_type->containedTypes()) {
            TORCH_CHECK(
                element_type->kind() != TypeKind::ClassType,
                "Returining a list or dictionary with pytorch class type ",
                "is not supported in mobile module "
                "(List[Foo] or Dict[int, Foo] for class Foo(torch.nn.Module)). "
                "Workaround: instead of using pytorch class as their element type, ",
                "use a combination of list, dictionary, and single types.");
          }
        }
      }
    } else {
      TORCH_CHECK(
          isOpSupportedInMobile(ins.op),
          toString(ins.op),
          " is not supported in mobile module.");
    }
    auto node = code->instructions_source()[i];
    int64_t debug_handle = debug_info_recorder.getNextDebugHandle(node);
    // Note 1-to-1 correspondence between instructions and debug handles
    op_debug_handles.emplace_back(debug_handle);
  }
  return std::make_tuple(opnames, method_names, op_debug_handles);
}

bool ValidSetGetState(const c10::ClassType* cls) {
  // Check that the schemas for __getstate__ and __setstate__ are correct
  auto getstate = cls->findMethod("__getstate__");
  if (getstate == nullptr) {
    return false;
  }
  auto get_schema = getstate->getSchema();

  // Check __getstate__
  //   __getstate__ is expected to be (self) -> T
  TORCH_CHECK(
      get_schema.arguments().size() == 1,
      "'__getstate__' must have 'self' as its only argument, but found ",
      get_schema.arguments().size(),
      " arguments");
  TORCH_CHECK(
      get_schema.returns().size() == 1,
      "'__getstate__' must return 1 value, but found ",
      get_schema.returns().size());

  // Check __setstate__ if the method exists
  //   __setstate__ is expected to be (self, T) -> None
  auto setstate = cls->findMethod("__setstate__");
  if (!setstate) {
    return false;
  }
  auto set_schema = setstate->getSchema();

  TORCH_CHECK(
      set_schema.arguments().size() == 2,
      "'__setstate__' must have 'self' and the state as its "
      "only arguments, but found ",
      set_schema.arguments().size(),
      " arguments");
  TORCH_CHECK(
      set_schema.returns().size() == 1,
      "'__setstate__' must return None, but found ",
      set_schema.returns().size(),
      " return values");
  TORCH_CHECK(
      set_schema.returns().at(0).type()->isSubtypeOf(NoneType::get()),
      "'__setstate__' must return None, but found value of type",
      set_schema.returns().at(0).type()->annotation_str());

  // Check that the return type of __getstate__ matches the input to
  // __setstate__
  auto get_type = get_schema.returns().at(0).type();
  auto set_type = set_schema.arguments().at(1).type();

  TORCH_CHECK(
      get_type->isSubtypeOf(set_type),
      "'__getstate__'s return type (",
      get_type->annotation_str(),
      ") does not match '__setstate__'s argument type (",
      set_type->annotation_str(),
      ")");

  return true;
}

flatbuffers::Offset<jit::mobile::serialization::Schema> IValueFlatbufferSerializer::CreateFBSchema(
    flatbuffers::FlatBufferBuilder& fbb,
    const std::vector<Argument>& args,
    const std::vector<Argument>& returns,
    c10::TypePrinter type_printer) {
  std::vector<flatbuffers::Offset<jit::mobile::serialization::Arg>> arg_vec;
  arg_vec.reserve(args.size());
  std::vector<flatbuffers::Offset<jit::mobile::serialization::Arg>> return_vec;
  return_vec.reserve(returns.size());
  for (const auto& arg : args) {
    int index = storeIValueAndGetIndex(fbb, arg.default_value());
    arg_vec.emplace_back(CreateArg(
        fbb,
        fbb.CreateSharedString(arg.name()),
        fbb.CreateSharedString(arg.type()->annotation_str(type_printer)),
        index));
  }

  for (const auto& ret : returns) {
    int index = storeIValueAndGetIndex(fbb, ret.default_value());
    return_vec.emplace_back(CreateArg(
        fbb,
        fbb.CreateSharedString(ret.name()),
        fbb.CreateSharedString(ret.type()->annotation_str(type_printer)),
        index));
  }
  return CreateSchema(fbb, fbb.CreateVector(arg_vec), fbb.CreateVector(return_vec));
}

flatbuffers::Offset<mobile::serialization::Function>
IValueFlatbufferSerializer::functionToFB(
    FlatBufferBuilder& fbb,
    const std::string& qn,
    const Function& func
) {
  // const auto qn = func.qualname().qualifiedName();
  auto graph = func.graph()->copy();
  Inline(*graph);

  std::shared_ptr<MobileCode> code;
  code = std::make_shared<MobileCode>(
      graph,
      func.name(),
      emit_default_input_instructions_);
  auto instructions_copy = code->instructions();

  std::vector<c10::OperatorName> opnames;
  std::vector<std::string> method_names;
  std::vector<int64_t> op_debug_handles;
  std::tie(opnames, method_names, op_debug_handles) = convertInstructionsForMobile2(
    code.get(), &instructions_copy, debug_info_recorder_);

  // instructions
  std::vector<mobile::serialization::Instruction> instruction_vector;
  for (const auto& inst: instructions_copy) {
    instruction_vector.emplace_back(inst.op, inst.N, inst.X);
  }

  // operators
  std::vector<flatbuffers::Offset<mobile::serialization::Operator>> operator_vector;
  auto op_to_specified_args = code->op_to_num_specified_args();
  operator_vector.reserve(opnames.size());
  for (const auto& opname : opnames) {
    auto unique_name = c10::toString(opname);
    // For operator with vararg, adding default arguments would be confusing and
    // is not allowed. For an operator with num_args = -1, it means the number
    // of arguments is not available for this operator, we don't do any backward
    // compatibility adaptation at runtime.
    int num_args = -1;
    auto it = op_to_specified_args.find(unique_name);
    if (it != op_to_specified_args.end()) {
      num_args = it->second;
    }
    operator_vector.push_back(
      CreateOperator(fbb, fbb.CreateSharedString(opname.name),
                     fbb.CreateSharedString(opname.overload_name),
                     num_args));
  }

  const auto& constants = code->constant_table();

  std::vector<uint32_t> constant_indexes;
  for (const auto& constant : constants) {
    constant_indexes.push_back(storeIValueAndGetIndex(fbb, constant));
  }

  // types
  static const std::string torch_prefix("__torch__");
  static const std::string class_prefix("__torch__.torch.classes");
  std::vector<flatbuffers::Offset<flatbuffers::String>> type_offsets;

  for (const TypePtr& t : code->type_table()) {
    auto type_str = t->annotation_str();
    if (type_str.find(torch_prefix) == 0) {
      std::cerr << "type is " << type_str << std::endl;
      TORCH_CHECK(
          type_str.find(class_prefix) == 0,
          "__torch__ types other than torchbind (__torch__.torch.classes)"
          "are not supported in lite interpreter. ",
          "Workaround: instead of using arbitrary class type (class Foo()), ",
          "define a pytorch class (class Foo(torch.nn.Module)).");
    }

    type_offsets.push_back(fbb.CreateSharedString(type_str));
  }

  // since the register location is embedded into the bytecode, pass the
  // register size
  auto register_size = static_cast<int>(code->register_size());

  // schema
  const auto& schema = func.getSchema();
  auto type_printer =
      [&](const c10::ConstTypePtr& t) -> c10::optional<std::string> {
    auto namedType = t->cast<c10::NamedType>();
    if (namedType && namedType->name()) {
      return type_name_uniquer_.getUniqueName(namedType).qualifiedName();
    }
    return c10::nullopt;
  };
  TORCH_CHECK(
      schema.overload_name().empty(), // @TODO: is this check correct?
      "Overloads are not supported in mobile modules.");
  TORCH_CHECK(
      !schema.is_vararg(), "Python *args are not supported in mobile modules.");
  TORCH_CHECK(
      !schema.is_varret(),
      "A variable number of return values is not supported in mobile modules.");

  auto schema_offset = CreateFBSchema(
    fbb, schema.arguments(), schema.returns(), type_printer);
  auto debug_info_offset =  CreateDebugInfo(fbb, fbb.CreateVector(op_debug_handles));

  auto function_offset = CreateFunctionDirect(
    fbb,
    qn.c_str(),
    &instruction_vector,
    &operator_vector,
    &constant_indexes,
    &type_offsets,
    register_size,
    schema_offset,
    debug_info_offset);
  return function_offset;
}


flatbuffers::DetachedBuffer
IValueFlatbufferSerializer::serializeModule(
  const Module& module, bool include_tensor_data_in_flatbuffer) {

  FlatBufferBuilder fbb;
  auto methods = module.get_methods();

  std::vector<uint32_t> functions_index;
  functions_index.reserve(methods.size());
  for (const auto& method : methods) {
    auto func_offset = storeFunctionAndGetIndex(
      fbb, method.function().qualname().qualifiedName(),
      method.function());
    functions_index.push_back(func_offset);
  }

  auto functions_offset = fbb.CreateVector(functions_index);
  uint32_t ivalue_index = storeIValueAndGetIndex(fbb, module._ivalue());

  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<mobile::serialization::StorageData>>> storage_data_offset = 0;
  if (include_tensor_data_in_flatbuffer) {
    std::vector<flatbuffers::Offset<mobile::serialization::StorageData>> storage_data;
    for (const auto& td : tensor_data_) {
      WriteableTensorData writable_td = getWriteableTensorData(td);
      auto storage_offset = mobile::serialization::CreateStorageData(
        fbb, fbb.CreateVector(reinterpret_cast<const uint8_t*>(writable_td.data()), writable_td.sizeInBytes()));
      storage_data.push_back(storage_offset);
    }
    storage_data_offset = fbb.CreateVector(storage_data);
  }

  auto mod = CreateModule(fbb, functions_offset, ivalue_index,
      fbb.CreateVector(ivalue_types_), fbb.CreateVector(ivalue_offsets_),
      tensor_data_.size(), storage_data_offset,
      fbb.CreateVector(obj_types_offset_));
  fbb.Finish(mod);
  return fbb.Release();
}


flatbuffers::Offset<mobile::serialization::Tuple>
IValueFlatbufferSerializer::tupleToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& tuple) {
    const auto& elements = tuple.toTuple()->elements();
    std::vector<uint32_t> items = storeIValuesAndGetIndexes(fbb, elements.begin(), elements.end());
    return CreateTupleDirect(fbb, &items);
}

flatbuffers::Offset<mobile::serialization::List>
IValueFlatbufferSerializer::listToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& list) {
    const auto& elements = list.toList();
    std::vector<uint32_t> items = storeIValuesAndGetIndexes(fbb, elements.begin(), elements.end());
    return CreateList(fbb, fbb.CreateVector(items), fbb.CreateSharedString(list.type()->annotation_str()));
}

flatbuffers::Offset<mobile::serialization::Dict>
IValueFlatbufferSerializer::dictToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& ivalue) {
  const auto& dict = ivalue.toGenericDict();
  std::vector<uint32_t> keys;
  std::vector<uint32_t> values;
  keys.reserve(dict.size());
  values.reserve(dict.size());
  for (const auto& entry: dict) {
    int key_index = storeIValueAndGetIndex(fbb, entry.key());
    keys.push_back(key_index);
    int value_index = storeIValueAndGetIndex(fbb, entry.value());
    values.push_back(value_index);
  }
  return CreateDict(fbb,
    fbb.CreateVector(keys),
    fbb.CreateVector(values),
    fbb.CreateSharedString(ivalue.type()->annotation_str()));
}

flatbuffers::Offset<mobile::serialization::ObjectType>
IValueFlatbufferSerializer::classTypeToFB(
  FlatBufferBuilder& fbb, ClassTypePtr class_ptr
) {

  mobile::serialization::TypeType typetype = mobile::serialization::TypeType_UNSET;

  uint32_t state_index = 0;
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> names_offset = 0;
  auto setstate = class_ptr->findMethod("__setstate__");
  if (setstate == nullptr) {
    size_t num_attr = class_ptr->numAttributes();
    std::vector<flatbuffers::Offset<flatbuffers::String>> names;
    std::vector<uint32_t> type_index;
    for (size_t i = 0; i < num_attr; ++i) {
      names.push_back(fbb.CreateSharedString(class_ptr->getAttributeName(i)));
    }
    names_offset = fbb.CreateVector(names);
    typetype = mobile::serialization::TypeType_CLASS_WITH_FIELD;
  } else {
    if (setstate->isGraphFunction()) {
      typetype = mobile::serialization::TypeType_CLASS_WITH_SETSTATE;
    } else {
      typetype = mobile::serialization::TypeType_CUSTOM_CLASS;
    }
  }
  std::cerr << " Calling CreateOjbect type: " << mobile::serialization::EnumNameTypeType(typetype) << std::endl;

  /*

  inline flatbuffers::Offset<ObjectType> CreateObjectType(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> type_name = 0,
    torch::jit::mobile::serialization::TypeType type = torch::jit::mobile::serialization::TypeType_UNSET,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> attr_names = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint32_t>> attr_sample_indexes = 0,
    uint32_t setstate = 0) {
  */
  auto name_offset = fbb.CreateString(type_name_uniquer_.getUniqueName(class_ptr).qualifiedName());
  std::cerr << "    classoffset is " << (uint64_t) name_offset.o << std::endl;
  return CreateObjectType(fbb, name_offset, typetype, names_offset);
}

uint32_t IValueFlatbufferSerializer::storeFunctionAndGetIndex(
  flatbuffers::FlatBufferBuilder& fbb, const std::string& qn, const Function& function) {

  auto iter = qn_to_serialized_values_.find(qn);
  if (iter != qn_to_serialized_values_.end()) {
    return iter->second;
  }

  uint32_t index = insertIValue(mobile::serialization::IValue_Function,
             functionToFB(fbb, qn, function).Union());
  qn_to_serialized_values_[qn] = index;
  return index;
}

uint32_t IValueFlatbufferSerializer::storeClassTypeAndGetIndex(FlatBufferBuilder& fbb, ClassTypePtr class_ptr) {

  const auto& type_str = class_ptr->name()->qualifiedName();
  auto iter = qn_to_serialized_values_.find(type_str);
  if (iter != qn_to_serialized_values_.end()) {
    return iter->second;
  }

  auto offset = classTypeToFB(fbb, class_ptr);
  uint32_t res = obj_types_offset_.size();
  obj_types_offset_.push_back(offset);
  std::cerr << "      index is " << res << std::endl;
  qn_to_serialized_values_[type_str] = res;
  return res;
}

flatbuffers::Offset<mobile::serialization::Object>
IValueFlatbufferSerializer::objectToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& ivalue) {
  auto obj = ivalue.toObject();
  auto type = obj->type();
  // rename type?
  // check getstate

  bool setstate = ValidSetGetState(type.get());

  // save state as ivalue
  flatbuffers::Offset<flatbuffers::Vector<uint32_t>> attrs = 0;
  uint32_t state_index = 0;
  uint32_t setstate_func_index = 0;
  if (setstate) {
    Function& getstate = type->getMethod("__getstate__");
    auto state = getstate({obj});
    state_index = storeIValueAndGetIndex(fbb, state);
    Function& setstate = type->getMethod("__setstate__");
    if (setstate.isGraphFunction()) {
      const auto qn =
          type_name_uniquer_.getUniqueName(type).qualifiedName() + "." +
          setstate.name();
      setstate_func_index = storeFunctionAndGetIndex(fbb, qn, setstate);
    }
  } else {
    size_t num_attr = type->numAttributes();
    std::vector<uint32_t>tuple_index;
    for (size_t i = 0; i < num_attr; ++i) {
      tuple_index.push_back(storeIValueAndGetIndex(fbb, obj->getSlot(i)));
    }
    attrs = fbb.CreateVector(tuple_index);
    std::cerr << " num of attrs is " << num_attr << std::endl;
  }

  uint32_t type_index = storeClassTypeAndGetIndex(fbb, type);
  std::cerr << "Serializing obj of type " << type->name()->qualifiedName() << " index is " << type_index << std::endl;
  return CreateObject(fbb, type_index, state_index, attrs, setstate_func_index);
}

flatbuffers::Offset<mobile::serialization::TensorMetadata>
IValueFlatbufferSerializer::IValueFlatbufferSerializer::tensorToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& ivalue) {
  auto& tensor = ivalue.toTensor();
  bool quantized = tensor.is_quantized();
  const at::Storage& storage = tensor.storage();

  flatbuffers::Offset<mobile::serialization::QuantizedSchema> qschema_offset = 0;
  if (quantized) {
    double scale = 0;
    int32_t zero_point = 0;
    flatbuffers::Offset<mobile::serialization::TensorMetadata> scales = 0;
    flatbuffers::Offset<mobile::serialization::TensorMetadata> zero_points = 0;
    int32_t axis = 0;

    switch (tensor.qscheme()) {
      case at::kPerTensorAffine:
        scale = tensor.q_scale();
        zero_point = tensor.q_zero_point();
        break;
      case at::kPerChannelAffineFloatQParams:
      case at::kPerChannelAffine: {
        scales = tensorToFB(fbb, tensor.q_per_channel_scales());
        zero_points = tensorToFB(fbb, tensor.q_per_channel_zero_points());
        axis = tensor.q_per_channel_axis();
      } break;
      default:
        TORCH_CHECK(
            false,
            "Unsupported tensor quantization type in serialization ",
            toString(tensor.qscheme()));
        break;
    }

    qschema_offset = mobile::serialization::CreateQuantizedSchema(
      fbb,
      static_cast<int8_t>(tensor.qscheme()),
      scale,
      zero_point,
      scales,
      zero_points,
      axis);
  }

  void* addr = storage.unsafeGetStorageImpl();
  uint32_t storage_index;
  auto it = memoized_storage_map_.find(addr);
  if (it != memoized_storage_map_.end()) {
    storage_index = it->second;
  } else {
    storage_index = tensor_data_.size();
    memoized_storage_map_[addr] = storage_index;
    tensor_data_.push_back(tensor);
  }

  std::vector<int> sizes{tensor.sizes().begin(), tensor.sizes().end()};
  std::vector<int> strides{tensor.strides().begin(), tensor.strides().end()};

  return CreateTensorMetadataDirect(
    fbb,
    /* storage_location_index */ storage_index,
    /* scalar_type */ static_cast<int8_t>(tensor.scalar_type()),
    /* int32_t storage_offset */ tensor.storage_offset(),
    /* sizes */ &sizes,
    /* strides */ &strides,
    /* bool requires_grad */ tensor.requires_grad(),
    /* qschema */ qschema_offset);

}


uint32_t IValueFlatbufferSerializer::storeIValueAndGetIndex(flatbuffers::FlatBufferBuilder& fbb, const IValue& ivalue) {
  if (ivalue.isNone()) {
    return 0;
  }
  if (ivalue.isBool()) {
    return ivalue.toBool() ? 2 : 1;
  }
  bool is_hashable = IValue::hashable(ivalue);
  if (is_hashable) {
    auto iter = cached_ivalues_.find(ivalue);
    if (iter != cached_ivalues_.end()) {
      return iter->second;
    }
  }
  uint8_t type;
  flatbuffers::Offset<void> offset;
  std::tie(type, offset) = iValueToFB(fbb, ivalue);
  uint32_t index = insertIValue(type, offset);
  if (is_hashable) {
    cached_ivalues_[ivalue] = index;
  }
  return index;
}

std::tuple<
    mobile::serialization::IValue,
    flatbuffers::Offset<void>>
IValueFlatbufferSerializer::iValueToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& ivalue) {
  mobile::serialization::IValue ivalue_type;
  flatbuffers::Offset<void> offset;

  if (ivalue.isTensor()) {
    ivalue_type = mobile::serialization::IValue_TensorMetadata;
    offset = tensorToFB(fbb, ivalue).Union();
  } else if (ivalue.isTuple()) {
    ivalue_type = mobile::serialization::IValue_Tuple;
    offset = tupleToFB(fbb, ivalue).Union();
  } else if (ivalue.isDouble()) {
    ivalue_type = mobile::serialization::IValue_Double;
    offset = fbb.CreateStruct(mobile::serialization::Double(ivalue.toDouble())).Union();
  } else if (ivalue.isInt()) {
    ivalue_type = mobile::serialization::IValue_Int;
    offset = fbb.CreateStruct(mobile::serialization::Int(ivalue.toInt())).Union();
  } else if (ivalue.isBool()) {
    ivalue_type = mobile::serialization::IValue_Bool;
    offset = fbb.CreateStruct(mobile::serialization::Bool(ivalue.toBool())).Union();
  } else if (ivalue.isString()) {
    ivalue_type = mobile::serialization::IValue_String;
    offset = mobile::serialization::CreateString(fbb,
      fbb.CreateSharedString(ivalue.toString()->string())).Union();
  } else if (ivalue.isGenericDict()) {
    ivalue_type = mobile::serialization::IValue_Dict;
    offset = dictToFB(fbb, ivalue).Union();
  } else if (ivalue.isNone()) {
    ivalue_type = mobile::serialization::IValue_NONE;
    offset = 0;
  } else if (ivalue.isIntList()) {
    ivalue_type = mobile::serialization::IValue_IntList;
    offset = mobile::serialization::CreateIntList(fbb, fbb.CreateVector(ivalue.toIntVector())).Union();
  } else if (ivalue.isDoubleList()) {
    ivalue_type = mobile::serialization::IValue_DoubleList;
    offset = mobile::serialization::CreateDoubleList(fbb, fbb.CreateVector(ivalue.toDoubleVector())).Union();
  } else if (ivalue.isBoolList()) {
    ivalue_type = mobile::serialization::IValue_BoolList;
    auto boollist = ivalue.toBoolList();
    std::vector<uint8_t> bool_vec(boollist.begin(), boollist.end());
    offset = mobile::serialization::CreateBoolListDirect(fbb, &bool_vec).Union();
  } else if (ivalue.isList()) {
    ivalue_type = mobile::serialization::IValue_List;
    offset = listToFB(fbb, ivalue).Union();
  } else if (ivalue.isObject()) {
    ivalue_type = mobile::serialization::IValue_Object;
    offset = objectToFB(fbb, ivalue).Union();
  } else if (ivalue.isDevice()) {
    AT_ERROR("Cannot serialize device");
  } else if (ivalue.isCapsule()) {
    std::cerr << "Cannot serialize capsule" << std::endl;
    ivalue_type = mobile::serialization::IValue_NONE;
    offset = 0;
    // AT_ERROR("Cannot serialize capsule");
  } else if (ivalue.isRRef()) {
    AT_ERROR("Cannot serialize rref");
  } else if (ivalue.isEnum()) {
    std::cerr << "Cannot serialize enum" << std::endl;
  } else {
    AT_ERROR("Unknown IValue type for pickling: ", ivalue.tagKind());
  }
  return {ivalue_type, offset};
}

} // namespace jit
} // namespace torch
