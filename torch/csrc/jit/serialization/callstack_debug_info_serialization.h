#pragma once

#include <c10/core/Allocator.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/scope.h>

#include <ATen/core/ivalue.h>

#include <vector>

#include <c10/util/flat_hash_map.h>

namespace c10 {
struct IValue;
}

namespace torch::jit {

class Pickler;
class InlinedCallStackSerializer {
 public:
  // Serialize InlinedCallStack as
  // SerializedInlinedCallStack =
  // [module_info, source range tag, SerializedInlinedCallStack]
  // module_info = [ClassType.qualifiedName, instance_name]
  // source_range_tag = unique source range id
  c10::IValue serialize(
      const InlinedCallStackPtr& cs_ptr,
      const SourceRangeTagMap& source_range_tags);

 private:
  // module_info = [ClassType.qualifiedName, instance_name]
  c10::IValue serialize_module_instance_info(
      const std::optional<ModuleInstanceInfo>& m);

  // This caches serialized inlined callstack ptr, since many
  // InlinedCallStackPtr can refer to the same one.
  ska::flat_hash_map<InlinedCallStackPtr, c10::IValue>
      serialized_inlined_callstack_;
  // This caches serialized module instance info.
  // There might be many nodes that are part of the same
  // parent, grandparent etc. module.
  ska::flat_hash_map<std::string, c10::IValue> serialized_module_instance_info_;
};

class TORCH_API CallStackDebugInfoPickler {
 public:
  CallStackDebugInfoPickler() = default;

  std::vector<char> pickle(
      const std::unordered_map<int64_t, DebugInfoTuple>& callstack_ptrs,
      const SourceRangeTagMap& source_range_tags);

 private:
  InlinedCallStackSerializer css_;
};

class InlinedCallStackDeserializer {
 public:
  InlinedCallStackPtr deserialize(
      const c10::IValue& iv,
      const ska::flat_hash_map<int64_t, SourceRange>& source_range_map,
      const std::shared_ptr<CompilationUnit>& cu);

 private:
  std::optional<ModuleInstanceInfo> deserialize_module_instance_info(
      const c10::IValue& iv,
      const std::shared_ptr<CompilationUnit>& cu);

  ska::
      flat_hash_map<c10::intrusive_ptr<c10::ivalue::Tuple>, InlinedCallStackPtr>
          cached_inlined_callstacks_;
  ska::flat_hash_map<c10::intrusive_ptr<c10::ivalue::Tuple>, ModuleInstanceInfo>
      cached_module_instance_info_;
};

class TORCH_API CallStackDebugInfoUnpickler {
 public:
  ska::flat_hash_map<int64_t, DebugInfoTuple> unpickle(
      const at::DataPtr& data,
      size_t size,
      const ska::flat_hash_map<int64_t, SourceRange>& source_range_map,
      const std::shared_ptr<CompilationUnit>& cu);

 private:
  InlinedCallStackDeserializer csds_;
};

} // namespace torch::jit
