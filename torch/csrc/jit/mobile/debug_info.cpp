#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/mobile/debug_info.h>
#include <torch/csrc/jit/serialization/callstack_debug_info_serialization.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/serialization/pickle.h>

#include <c10/util/string_view.h>

namespace torch {
namespace jit {

namespace {

// This function construct stacktrace with module hierarchy
// Module hierarchy will contain information about where in the
// module hierarchy this source is. For example if conv2d op
// exist in hierarcy A->B->C->Conv2d with type annotations of
// A -> TopM, B->MyModule, C->SomeModule, then module hierarchy
// will be TopM(A).MyModule(B).SomeModule(C).Conv2d(conv)
// Source level stack information will be from model source code.
std::pair<std::string, std::string> getStackTraceWithModuleHierarchy(
    const DebugInfoPair& source_callstack,
    const std::string& root_scope_string,
    const std::string& top_module_type_name) {
  constexpr size_t kSourceRange = 1;
  constexpr size_t kModuleInstanceInfo = 2;
  std::vector<StackEntry> entries;

  const SourceRange& range = source_callstack.first;
  InlinedCallStackPtr callstack_ptr = source_callstack.second;
  std::string module_info =
      root_scope_string + "(" + top_module_type_name + ")";
  std::ostringstream ss;
  if (!callstack_ptr) {
    // If not cs then top level node
    entries.emplace_back(StackEntry{"FunctionName_UNKNOWN", range});
  } else {
    for (const auto& element : callstack_ptr->vec()) {
      const auto& opt_module_instance_info =
          std::get<kModuleInstanceInfo>(element);
      if (opt_module_instance_info.has_value()) {
        const auto& module_instance_info = opt_module_instance_info.value();
        if (module_instance_info.class_type()) {
          const auto& class_type = module_instance_info.class_type();
          const auto& instance_name = module_instance_info.instance_name();
          auto type_name = class_type->name()->qualifiedName();
          type_name = type_name.substr(type_name.find_last_of('.') + 1);
          module_info.append(".")
              .append(instance_name)
              .append("(")
              .append(type_name)
              .append(")");
        } else {
          module_info += ".(UNKNOWN_INSTANCE(UNKNOWN_TYPE)";
        }
      } else {
        module_info += ".(UNKNOWN_INSTANCE(UNKNOWN_TYPE)";
      }
      // Now add source range info to stack
      // When we serialize function names, those can be added here.
      // TODO: Add function name separately
      entries.emplace_back(
          StackEntry{"FunctionName_UNKNOWN", std::get<kSourceRange>(element)});
    }
    entries.emplace_back(StackEntry{"FunctionName_UNKNOWN", range});
  }
  ss << "Module hierarchy:" << module_info << "\n";
  format_stack_trace(ss, entries);
  return {ss.str(), std::move(module_info)};
}

} // namespace

MobileDebugTable::MobileDebugTable(
    std::unique_ptr<caffe2::serialize::PyTorchStreamReader>& reader,
    const std::shared_ptr<CompilationUnit>& cu) {
  ska::flat_hash_map<int64_t, SourceRange> source_range_map;
  const std::vector<std::string>& record_names = reader->getAllRecords();
  const c10::string_view suffix(".debug_pkl");
  for (const auto& record_name : record_names) {
    if (c10::string_view(record_name).ends_with(suffix)) {
      at::DataPtr debug_data;
      size_t debug_size{0};
      std::tie(debug_data, debug_size) = reader->getRecord(record_name);
      auto ivalues =
          jit::unpickle(
              reinterpret_cast<const char*>(debug_data.get()), debug_size)
              .toTuple()
              ->elements();
      SourceRangeDeserializer deserializer;
      for (auto& val : ivalues) {
        auto tup_elems = val.toTuple()->elements();
        // For BC we decode only tuples with 3 elements
        // assuming it contains
        // byte_offset, debug_handle (=source range tag), source range
        if (tup_elems.size() == 3) {
          int64_t debug_handle = tup_elems[kSourceRangeTagIndex].toInt();
          auto source_range =
              deserializer.deserialize(tup_elems[kSourceRangeIndex]);
          source_range_map.emplace(debug_handle, std::move(source_range));
        }
      }
    }
  }
  const std::string callstack_debug_file("callstack_debug_map.pkl");
  if (reader->hasRecord("callstack_debug_map.pkl")) {
    at::DataPtr callstack_data;
    size_t callstack_data_size{0};
    std::tie(callstack_data, callstack_data_size) =
        reader->getRecord(callstack_debug_file);
    CallStackDebugInfoUnpickler unpickler;
    callstack_ptr_map_ = unpickler.unpickle(
        std::move(callstack_data), callstack_data_size, source_range_map, cu);
  }
}

std::string MobileDebugTable::getModuleHierarchyInfo(
    const int64_t debug_handle,
    const std::string& top_module_type_name) const {
  const auto it = callstack_ptr_map_.find(debug_handle);
  if (it == callstack_ptr_map_.end()) {
    return "debug_handle:" + std::to_string(debug_handle);
  }
  return (getStackTraceWithModuleHierarchy(
              it->second, "top", top_module_type_name))
      .second;
}

std::string MobileDebugTable::getSourceDebugString(
    const int64_t debug_handle,
    const std::string& top_module_type_name) const {
  const auto it = callstack_ptr_map_.find(debug_handle);
  if (it == callstack_ptr_map_.end()) {
    return "debug_handle:" + std::to_string(debug_handle);
  }
  return (getStackTraceWithModuleHierarchy(
              it->second, "top", top_module_type_name))
      .first;
}

} // namespace jit
} // namespace torch
