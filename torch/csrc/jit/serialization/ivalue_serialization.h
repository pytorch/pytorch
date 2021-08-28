#pragma once


#include <ATen/core/qualified_name.h>
#include <string>
#include <utility>
#include <vector>
#include <flatbuffers/flatbuffers.h>

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/utils/disallow_copy.h>
#include <torch/csrc/jit/serialization/mobile_bytecode_generated.h>

namespace torch {
namespace jit {

using ::c10::IValue;

class IValueFlatbufferSerializer{

 public:
  template <typename It>
  std::tuple<
    std::vector<uint8_t>,
    std::vector<flatbuffers::Offset<void>>>
  iValueIteratorToFB(flatbuffers::FlatBufferBuilder& fbb, It begin, It end) {
    std::vector<uint8_t> types;
    std::vector<flatbuffers::Offset<void>> offsets;
    uint8_t type;
    flatbuffers::Offset<void> offset;
    for (; begin != end; ++begin) {
        std::tie(type, offset) = iValueToFB(fbb, *begin);
        types.push_back(type);
        offsets.push_back(offset);
    }
    return {types, offsets};
  }

  flatbuffers::Offset<mobile::serialization::Tuple>
  tupleToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& tuple);

  flatbuffers::Offset<mobile::serialization::List>
  listToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& list);

  flatbuffers::Offset<mobile::serialization::Dict>
  dictToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& list);

  flatbuffers::Offset<mobile::serialization::Object>
  objectToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& ivalue);

  flatbuffers::Offset<mobile::serialization::TensorMetadata>
  tensorToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& ivalue);

  std::tuple<
      mobile::serialization::IValue, 
      flatbuffers::Offset<void>>
  iValueToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& ivalue);

  std::vector<at::Tensor> tensor_data_;
  std::unordered_map<const void*, uint32_t> memoized_storage_map_;
  std::vector<c10::ClassTypePtr> memoized_class_types_;
  std::unordered_map<c10::ClassTypePtr, int> memoized_class_map_;
};

class IValueDeserializer {
 public:
  IValueDeserializer(
    const std::vector<c10::Storage>& tensor_data,
    const std::vector<c10::StrongTypePtr>& types) :
    tensor_data_(&tensor_data), types_(&types) {}

  IValue parseIValue(const mobile::serialization::IValue ivalue_type, const void* ivalue_data);
  IValue parseList(const mobile::serialization::List* list);
  IValue parseTensor(const mobile::serialization::TensorMetadata* tensor);
  IValue parseTuple(const mobile::serialization::Tuple* tuple);
  IValue parseDict(const mobile::serialization::Dict* dict);
  IValue parseObject(const mobile::serialization::Object* object);


  const std::vector<c10::Storage>* tensor_data_;
  const std::vector<c10::StrongTypePtr>* types_;
};



} // namespace jit
} // namespace torch
