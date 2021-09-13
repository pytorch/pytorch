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
#include <c10/util/Exception.h>
#include <torch/csrc/jit/backends/backend_debug_handler.h>
#include <torch/csrc/jit/backends/backend_debug_info.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/type_hashing.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/method.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/callstack_debug_info_serialization.h>
#include <torch/csrc/jit/serialization/ivalue_serialization.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/csrc/jit/serialization/import_export_helpers.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>
#include <torch/csrc/jit/serialization/mobile_bytecode_generated.h>
#include <flatbuffers/flatbuffers.h>

namespace torch {
namespace jit {

using ::c10::IValue;

class IValueFlatbufferSerializer{

 public:
  IValueFlatbufferSerializer(
    BackendDebugInfoRecorder& debug_info_recorder, TypeNameUniquer& type_name_uniquer,
    bool emit_default_input_instructions )
    : debug_info_recorder_(debug_info_recorder), type_name_uniquer_(type_name_uniquer), emit_default_input_instructions_(emit_default_input_instructions)  {
      insertIValue(mobile::serialization::IValue_NONE, 0);
      insertIValue(mobile::serialization::IValue_Bool, 0);
      insertIValue(mobile::serialization::IValue_Bool, 0);
    }

  flatbuffers::DetachedBuffer
  serializeModule(const Module& module, bool include_tensor_data_in_flatbuffer);

  // TODO
  std::vector<at::Tensor> tensor_data_;

 private:
  template <typename It>
  std::vector<uint32_t>
  storeIValuesAndGetIndexes(flatbuffers::FlatBufferBuilder& fbb, It begin, It end) {
    std::vector<uint32_t> indexes;
    for (; begin != end; ++begin) {
      indexes.push_back(storeIValueAndGetIndex(fbb, *begin));
    }
    return indexes;
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

  flatbuffers::Offset<mobile::serialization::Function>
  functionToFB(flatbuffers::FlatBufferBuilder& fbb,
    const std::string& qn,
    const Function& func);


  std::tuple<
      mobile::serialization::IValue,
      flatbuffers::Offset<void>>
  iValueToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& ivalue);

  flatbuffers::Offset<jit::mobile::serialization::Schema> CreateFBSchema(
    flatbuffers::FlatBufferBuilder& fbb,
    const std::vector<Argument>& args,
    const std::vector<Argument>& returns,
    c10::TypePrinter type_printer);

flatbuffers::Offset<mobile::serialization::ObjectType>
classTypeToFB(
  flatbuffers::FlatBufferBuilder& fbb, ClassTypePtr class_ptr);

  uint32_t storeIValueAndGetIndex(flatbuffers::FlatBufferBuilder& fbb, const IValue& ivalue);
  uint32_t storeFunctionAndGetIndex(flatbuffers::FlatBufferBuilder& fbb,
    const std::string& qn,
    const Function& function);

  uint32_t storeClassTypeAndGetIndex(
    flatbuffers::FlatBufferBuilder& fbb, ClassTypePtr class_type);

  // cached stuff
  uint32_t insertIValue(uint8_t type, flatbuffers::Offset<void> offset)  {
    uint32_t size = ivalue_types_.size();
    ivalue_types_.push_back(type);
    ivalue_offsets_.push_back(offset);
    return size;
  }

  BackendDebugInfoRecorder& debug_info_recorder_;
  TypeNameUniquer& type_name_uniquer_;
  bool emit_default_input_instructions_;

  std::unordered_map<const void*, uint32_t> memoized_storage_map_;

  std::vector<uint8_t> ivalue_types_;
  std::vector<flatbuffers::Offset<void>> ivalue_offsets_;

  std::vector<flatbuffers::Offset<mobile::serialization::ObjectType>> obj_types_offset_;

  // qualified name to serialized class, type or function
  std::unordered_map<std::string, uint32_t> qn_to_serialized_values_;

  // cache of some ivalues
  struct IValueHash {
    size_t operator()(const IValue& val) const { return IValue::hash(val) ;}
  };

  std::unordered_map<IValue, uint32_t, IValueHash> cached_ivalues_;
};


} // namespace jit
} // namespace torch
