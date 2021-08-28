#include <torch/csrc/jit/serialization/ivalue_serialization.h>

#include <ATen/ATen.h>
#include <ATen/core/Dict.h>
#include <aten/src/ATen/quantized/Quantizer.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/serialization/mobile_bytecode_generated.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <string>

using flatbuffers::FlatBufferBuilder;

namespace torch {
namespace jit {

using mobile::serialization::CreateTupleDirect;
using mobile::serialization::CreateListDirect;
using mobile::serialization::CreateDictDirect;
using mobile::serialization::CreateTensorMetadataDirect;

bool ValidSetGetState(const std::shared_ptr<c10::ClassType>& cls) {
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





flatbuffers::Offset<mobile::serialization::Tuple>
IValueFlatbufferSerializer::tupleToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& tuple) {
    std::vector<uint8_t> types;
    std::vector<flatbuffers::Offset<void>> offsets;
    const auto& elements = tuple.toTuple()->elements();
    std::tie(types, offsets) = iValueIteratorToFB(fbb, elements.begin(), elements.end());
    return CreateTupleDirect(fbb, &types, &offsets);
}

flatbuffers::Offset<mobile::serialization::List>
IValueFlatbufferSerializer::listToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& list) {
    std::vector<uint8_t> types;
    std::vector<flatbuffers::Offset<void>> offsets;
    auto elements = list.toList();
    std::tie(types, offsets) = iValueIteratorToFB(fbb, elements.begin(), elements.end());
    return CreateListDirect(fbb, &types, &offsets);
}

flatbuffers::Offset<mobile::serialization::Dict>
IValueFlatbufferSerializer::dictToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& ivalue) {
  std::vector<uint8_t> key_types;
  std::vector<flatbuffers::Offset<void>> key_offsets;
  std::vector<uint8_t> value_types;
  std::vector<flatbuffers::Offset<void>> value_offsets;
  uint8_t type;
  flatbuffers::Offset<void> offset;
  auto dict = ivalue.toGenericDict();
  for (const auto& entry: dict) {
    std::tie(type, offset) = iValueToFB(fbb, entry.key());
    key_types.push_back(type);
    key_offsets.push_back(offset);
    std::tie(type, offset) = iValueToFB(fbb, entry.value());
    value_types.push_back(type);
    value_offsets.push_back(offset);
  }
  return CreateDictDirect(fbb, &key_types, &key_offsets, &value_types, &value_offsets);
}

flatbuffers::Offset<mobile::serialization::Object>
IValueFlatbufferSerializer::objectToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& ivalue) {
  auto obj = ivalue.toObject();
  auto type = obj->type();
  // rename type?
  // check getstate

  int type_index;
  auto iter = memoized_class_map_.find(type);
  if (iter != memoized_class_map_.end()) {
    type_index = iter->second;
  } else {
    type_index = memoized_class_types_.size();
    memoized_class_types_.push_back(type);
    memoized_class_map_[type] = type_index;
  }

  mobile::serialization::IValue state_type;  
  flatbuffers::Offset<void> state_offset;  
  bool setstate = false;

  if (ValidSetGetState(type)) {
    Function& getstate = type->getMethod("__getstate__");
    auto state = getstate({obj});
    std::tie(state_type, state_offset) = iValueToFB(fbb, state);
    setstate = true;
  } else {
    size_t num_attr = type->numAttributes();
    std::vector<IValue> ivalues;
    for (size_t i = 0; i < num_attr; ++i) {
      ivalues.push_back(obj->getSlot(i));
    }
    IValue state = c10::ivalue::Tuple::create(std::move(ivalues));
    std::tie(state_type, state_offset) = iValueToFB(fbb, state);
  }
  return CreateObject(
    fbb, type_index, setstate, state_type, state_offset);
}

flatbuffers::Offset<mobile::serialization::TensorMetadata>
IValueFlatbufferSerializer::IValueFlatbufferSerializer::tensorToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& ivalue) {
  auto& tensor = ivalue.toTensor();
  bool quantized = tensor.is_quantized();
  const at::Storage& storage = tensor.storage();
  AT_ASSERT(!quantized);

  void* addr = storage.unsafeGetStorageImpl();
  uint32_t storage_index;
  auto it = memoized_storage_map_.find(addr);
  if (it != memoized_storage_map_.end()) {
    storage_index = it->second;
  } else {
    storage_index = memoized_storage_map_.size();
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
    /* int64_t nbytes */ storage.nbytes(),
    /* int32_t element_size */ tensor.element_size(),
    /* sizes */ &sizes,
    /* strides */ &strides,
    /* bool requires_grad */ false);

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
      fbb.CreateString(ivalue.toString()->string())).Union();
  } else if (ivalue.isGenericDict()) {
    ivalue_type = mobile::serialization::IValue_Dict;
    offset = dictToFB(fbb, ivalue).Union();
  } else if (ivalue.isNone()) {
    ivalue_type = mobile::serialization::IValue_NONE;
    offset = 0;
  /*}  else if (ivalue.isIntList()) {
  } else if (ivalue.isTensorList()) {
  } else if (ivalue.isDoubleList()) {
  } else if (ivalue.isBoolList()) { */
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

IValue IValueDeserializer::parseTensor(const mobile::serialization::TensorMetadata* tensor_md) {

  at::ScalarType type = static_cast<at::ScalarType>(tensor_md->scalar_type());
  auto options = at::CPU(type).options();
  at::Tensor tensor = at::empty({0}, options);
  // requires_grad?
  at::TensorImpl* impl = tensor.unsafeGetTensorImpl();
  impl->set_storage_keep_dtype(tensor_data_->at(tensor_md->storage_location_index()));
  impl->set_storage_offset(tensor_md->storage_offset());

  std::vector<int64_t> size{tensor_md->sizes()->begin(), tensor_md->sizes()->end()};
  std::vector<int64_t> stride{tensor_md->strides()->begin(), tensor_md->strides()->end()};
  impl->set_sizes_and_strides(size, stride);

  return tensor;
}
IValue IValueDeserializer::parseList(const mobile::serialization::List* list) {
  auto res = c10::impl::GenericList(AnyType::get());
  const auto* items_type = list->items_type();
  const auto* items = list->items();
  for (size_t i = 0; i < items_type->size(); ++i) {
    res.emplace_back(parseIValue(items_type->GetEnum<mobile::serialization::IValue>(i), items->GetAs<void>(i)));
  }
  return res;
}
IValue IValueDeserializer::parseTuple(const mobile::serialization::Tuple* tuple) {
  std::vector<IValue> res;
  const auto* items_type = tuple->items_type();
  const auto* items = tuple->items();
  for (size_t i = 0; i < items_type->size(); ++i) {
    res.emplace_back(parseIValue(items_type->GetEnum<mobile::serialization::IValue>(i), items->GetAs<void>(i)));
  }
  return c10::ivalue::Tuple::create(res);
}
IValue IValueDeserializer::parseDict(const mobile::serialization::Dict* dict) {
  auto result = c10::impl::GenericDict(AnyType::get(), AnyType::get());
  const auto* keys_type = dict->keys_type();
  const auto* keys = dict->keys();
  const auto* values_type = dict->values_type();
  const auto* values = dict->values();
  for (size_t i = 0; i < keys_type->size(); ++i) {
    IValue key = parseIValue(keys_type->GetEnum<mobile::serialization::IValue>(i), keys->GetAs<void>(i));
    IValue value = parseIValue(values_type->GetEnum<mobile::serialization::IValue>(i), values->GetAs<void>(i));
    result.insert_or_assign(std::move(key), std::move(value));
  }
  return result;
}

IValue IValueDeserializer::parseObject(const mobile::serialization::Object* object) {
  IValue state = parseIValue(object->state_type(), object->state());
  auto type_ptr = types_->at(object->type_index());
  if (object->use_setstate()) {
    std::cerr << "TODO setstate\n";
    return IValue();
  } else {
    const auto& elements = state.toTuple()->elements(); 
    size_t ndict = elements.size();
    auto obj = c10::ivalue::Object::create(type_ptr, ndict);
    //auto obj = c10::ivalue::Object::create(type, elements.size());
    size_t i = 0;
    for (const auto& ival : elements) {
      obj->setSlot(i, ival);
      ++i;
    }
    return obj;
  }
  return IValue();
}

IValue IValueDeserializer::parseIValue(const mobile::serialization::IValue ivalue_type, const void* ivalue_data) {
  switch (ivalue_type) {
    case mobile::serialization::IValue_NONE:
      return {};
    case mobile::serialization::IValue_Int:
      return static_cast<const mobile::serialization::Int*>(ivalue_data)->int_val();
    case mobile::serialization::IValue_Bool:
      return static_cast<const mobile::serialization::Bool*>(ivalue_data)->bool_val();
    case mobile::serialization::IValue_Double:
      return static_cast<const mobile::serialization::Double*>(ivalue_data)->double_val();
    case mobile::serialization::IValue_TensorMetadata:
      return parseTensor(static_cast<const mobile::serialization::TensorMetadata*>(ivalue_data));
    case mobile::serialization::IValue_String:
      return static_cast<const mobile::serialization::String*>(ivalue_data)->data()->str();
    case mobile::serialization::IValue_List:
      return parseList(static_cast<const mobile::serialization::List*>(ivalue_data));
    case mobile::serialization::IValue_Tuple:
      return parseTuple(static_cast<const mobile::serialization::Tuple*>(ivalue_data));
    case mobile::serialization::IValue_Dict:
      return parseDict(static_cast<const mobile::serialization::Dict*>(ivalue_data));
    case mobile::serialization::IValue_Object:
      return parseObject(static_cast<const mobile::serialization::Object*>(ivalue_data));
    default:
      return {};
  }
}


} // namespace jit
} // namespace torch
