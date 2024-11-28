#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/mobile/debug_info.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/serialization/callstack_debug_info_serialization.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/serialization/pickle.h>

#include <c10/util/string_view.h>

namespace torch::jit {

namespace {

C10_ALWAYS_INLINE std::string debugHandlesNotFoundMessage(
    const std::string& debug_handles_string) {
  return "Debug info for handle(s): " + debug_handles_string +
      ", was not found.";
}

std::pair<std::vector<StackEntry>, std::string> getStackTraceWithModuleHierarchy(
    const DebugInfoTuple& source_callstack,
    const std::string& caller_name) {
  std::vector<StackEntry> entries;

  const SourceRange& range =
      std::get<kDebugInfoTupleSourceRangeIndex>(source_callstack);
  InlinedCallStackPtr callstack_ptr =
      std::get<kDebugInfoTupleInlinedCSIndex>(source_callstack);
  std::string prev_function_name = caller_name;
  std::string module_info;
  if (!callstack_ptr) {
    // If not cs then top level node
    entries.emplace_back(StackEntry{prev_function_name, range});
    return {std::move(entries), std::move(module_info)};
  } else {
    while (callstack_ptr) {
      const auto& opt_module_instance_info = callstack_ptr->module_instance();
      if (opt_module_instance_info.has_value()) {
        const auto& module_instance_info = opt_module_instance_info.value();
        // Sometimes (e.g., in lowered backends) we augment instance name with
        // type name instead of losing type name. In those cases instance_name
        // includes both instance name and type name. See
        // callstack_debug_info_serialization.cpp
        if (module_instance_info.class_type()) {
          module_info.append(".").append(
              utils::get_module_info(module_instance_info));
        } else {
          module_info.append(".").append(module_instance_info.instance_name());
        }
      } else {
        module_info.append(".UNKNOWN_INSTANCE(UNKNOWN_TYPE)");
      }
      // Now add source range info to stack
      entries.emplace_back(
          StackEntry{prev_function_name, callstack_ptr->source_range()});
      prev_function_name = callstack_ptr->function_name();
      // Function name appended here
      // It is renamed to prev_function_name because for StackEntry
      // it will be appended in the next iteration. This is the format
      // in which format_stack_trace expects function names.
      module_info.append("::").append(prev_function_name);

      if (callstack_ptr->callee()) {
        callstack_ptr = callstack_ptr->callee().value();
      } else {
        callstack_ptr = c10::intrusive_ptr<InlinedCallStack>();
      }
    }
    entries.emplace_back(StackEntry{prev_function_name, range});
    return {std::move(entries), std::move(module_info)};
  }
}

// This function construct stacktrace with module hierarchy
// Module hierarchy will contain information about where in the
// module hierarchy this source is. For example if conv2d op
// exist in hierarcy A->B->C->Conv2d with type annotations of
// A -> TopM, B->MyModule, C->SomeModule, then module hierarchy
// will be TopM(A).MyModule(B).SomeModule(C).Conv2d(conv)
// Source level stack information will be from model source code.
std::pair<std::string, std::string> getStackTraceWithModuleHierarchy(
    const std::vector<DebugInfoTuple>& source_callstacks,
    const std::string& root_scope_string,
    const std::string& top_module_type_name) {
  std::vector<StackEntry> stack_entries;
  std::string module_info =
      root_scope_string + "(" + top_module_type_name + ")";
  std::string caller_fn_name = "<unknown>";
  module_info.append("::").append(caller_fn_name);
  for (const auto& debug_info : source_callstacks) {
    auto debug_info_pair =
        getStackTraceWithModuleHierarchy(debug_info, caller_fn_name);
    auto entries = std::move(debug_info_pair.first);
    stack_entries.insert(stack_entries.end(), entries.begin(), entries.end());
    module_info.append(debug_info_pair.second);
  }
  // Only last entry in the callstack will have a node name of interest.
  // Rest are likely CallMethod/CallFunction nodes
  auto last_entry = source_callstacks.back();
  const std::string& node_name =
      std::get<kDebugInfoTupleNodeNameIndex>(last_entry);
  module_info.append(".").append(node_name);
  std::ostringstream ss;
  ss << "Module hierarchy:" << module_info << "\n";
  format_stack_trace(ss, stack_entries);
  return {ss.str(), std::move(module_info)};
}

} // namespace

MobileDebugTable::MobileDebugTable(
    std::unique_ptr<caffe2::serialize::PyTorchStreamReader>& reader,
    const std::shared_ptr<CompilationUnit>& cu) {
  ska::flat_hash_map<int64_t, SourceRange> source_range_map;
  const std::vector<std::string>& record_names = reader->getAllRecords();
  constexpr std::string_view suffix(".debug_pkl");
  for (const auto& record_name : record_names) {
    if (c10::string_view_ends_with(std::string_view(record_name), suffix)) {
      auto [debug_data, debug_size] = reader->getRecord(record_name);
      auto ivalueTuple = jit::unpickle(
          reinterpret_cast<const char*>(debug_data.get()),
          debug_size,
          nullptr,
          {},
          c10::parseType);
      const auto& ivalues = ivalueTuple.toTuple()->elements();
      IValue lines;
      std::unique_ptr<SourceRangeDeserializer> deserializer;
      if (ivalues.size() == 3 && ivalues[0].isString() &&
          kFormatWithStringTable == ivalues[0].toStringRef()) {
        // new format
        deserializer = std::make_unique<SourceRangeDeserializer>(ivalues[1]);
        lines = ivalues[2];
      } else {
        deserializer = std::make_unique<SourceRangeDeserializer>();
        lines = ivalueTuple;
      }

      for (auto& val : lines.toTuple()->elements()) {
        auto tup_elems = std::move(*val.toTuple()).elements();
        // For BC we decode only tuples with 3 elements
        // assuming it contains
        // byte_offset, debug_handle (=source range tag), source range
        if (tup_elems.size() == 3) {
          int64_t debug_handle = tup_elems[kSourceRangeTagIndex].toInt();
          auto source_range =
              deserializer->deserialize(tup_elems[kSourceRangeIndex]);
          source_range_map.emplace(debug_handle, std::move(source_range));
        }
      }
    }
  }
  const std::string callstack_debug_file("callstack_debug_map.pkl");
  if (reader->hasRecord("callstack_debug_map.pkl")) {
    auto [callstack_data, callstack_data_size] =
        reader->getRecord(callstack_debug_file);
    CallStackDebugInfoUnpickler unpickler;
    callstack_ptr_map_ = unpickler.unpickle(
        callstack_data, callstack_data_size, source_range_map, cu);
  }
}

std::string MobileDebugTable::getModuleHierarchyInfo(
    const int64_t debug_handle,
    const std::string& top_module_type_name) const {
  const auto it = callstack_ptr_map_.find(debug_handle);
  if (it == callstack_ptr_map_.end()) {
    return debugHandlesNotFoundMessage(std::to_string(debug_handle));
  }
  return (getStackTraceWithModuleHierarchy(
              {it->second}, "top", top_module_type_name))
      .second;
}

std::string MobileDebugTable::getModuleHierarchyInfo(
    const std::vector<int64_t>& debug_handles,
    const std::string& top_module_type_name) const {
  return getSourceDebugModuleHierarchyInfo(debug_handles, top_module_type_name)
      .second;
}

std::string MobileDebugTable::getSourceDebugString(
    const int64_t debug_handle,
    const std::string& top_module_type_name) const {
  const auto it = callstack_ptr_map_.find(debug_handle);
  if (it == callstack_ptr_map_.end()) {
    return debugHandlesNotFoundMessage(std::to_string(debug_handle));
  }
  return (getStackTraceWithModuleHierarchy(
              {it->second}, "top", top_module_type_name))
      .first;
}

std::string MobileDebugTable::getSourceDebugString(
    const std::vector<int64_t>& debug_handles,
    const std::string& top_module_type_name) const {
  return getSourceDebugModuleHierarchyInfo(debug_handles, top_module_type_name)
      .first;
}

std::pair<std::string, std::string> MobileDebugTable::
    getSourceDebugModuleHierarchyInfo(
        const std::vector<int64_t>& debug_handles,
        const std::string& top_module_type_name) const {
  std::vector<DebugInfoTuple> debug_infos;
  bool debug_handle_not_found{false};
  for (auto it = debug_handles.rbegin(); it != debug_handles.rend(); ++it) {
    auto debug_handle = *it;
    const auto cs_it = callstack_ptr_map_.find(debug_handle);
    if (cs_it == callstack_ptr_map_.end()) {
      debug_handle_not_found = true;
      break;
    }
    debug_infos.emplace_back(cs_it->second);
  }
  if (debug_handle_not_found) {
    std::string debug_handles_string = "debug_handles:{";
    for (const auto debug_handle : debug_handles) {
      debug_handles_string += std::to_string(debug_handle);
    }
    debug_handles_string += "}";
    debug_handles_string = debugHandlesNotFoundMessage(debug_handles_string);
    return {debug_handles_string, debug_handles_string};
  }
  return (getStackTraceWithModuleHierarchy(
      debug_infos, "top", top_module_type_name));
}

} // namespace torch::jit
