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

namespace torch {
namespace jit {

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
      const c10::optional<ModuleInstanceInfo>& m);

  // This caches serialized inlined callstack ptr, since many
  // InlinedCallStackPtr can refer to the same one.
  ska::flat_hash_map<InlinedCallStackPtr, c10::IValue>
      serialized_inlined_callstack_;
};

class TORCH_API InlinedCallStackPickler {
 public:
  InlinedCallStackPickler() = default;

  std::vector<char> pickle(
      const std::unordered_map<int64_t, DelegateDebugInfoType>& callstack_ptrs,
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
  c10::optional<ModuleInstanceInfo> deserialize_module_instance_info(
      const c10::IValue& iv,
      const std::shared_ptr<CompilationUnit>& cu);

  ska::
      flat_hash_map<c10::intrusive_ptr<c10::ivalue::Tuple>, InlinedCallStackPtr>
          cached_inlined_callstacks_;
};

class TORCH_API InlinedCallStackUnpickler {
 public:
  ska::flat_hash_map<int64_t, DelegateDebugInfoType> unpickle(
      at::DataPtr&& data,
      size_t size,
      const ska::flat_hash_map<int64_t, SourceRange>& source_range_map,
      const std::shared_ptr<CompilationUnit>& cu);

 private:
  InlinedCallStackDeserializer csds_;
};

} // namespace jit
} // namespace torch
