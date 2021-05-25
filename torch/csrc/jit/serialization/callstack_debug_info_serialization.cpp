#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/serialization/callstack_debug_info_serialization.h>
#include <torch/csrc/jit/serialization/pickle.h>

namespace torch {
namespace jit {

namespace {
const int64_t kInvalidSourceRangeTag = -1;
} // namespace

c10::IValue InlinedCallStackSerializer::serialize(
    const InlinedCallStackPtr& cs_ptr,
    const SourceRangeTagMap& source_range_tags) {
  if (!cs_ptr) {
    return c10::IValue();
  }
  auto cs_it = serialized_inlined_callstack_.find(cs_ptr);
  if (cs_it != serialized_inlined_callstack_.end()) {
    return cs_it->second;
  }
  // Inlined callstack pointer is serialized as tuple of 3 elements
  // {IValue(module_instance_info), source_range_tag, IValue(InlinedCallStack)}
  std::vector<c10::IValue> elements;
  elements.reserve(3);
  elements.emplace_back(
      serialize_module_instance_info(cs_ptr->module_instance()));
  int64_t source_range_tag{kInvalidSourceRangeTag};
  const SourceRange& sr = cs_ptr->source_range().findSourceRangeThatGenerated()
      ? cs_ptr->source_range().findSourceRangeThatGenerated().value()
      : cs_ptr->source_range();
  auto sr_it = source_range_tags.find(sr);
  if (sr_it != source_range_tags.end()) {
    source_range_tag = sr_it->second;
  }
  elements.emplace_back(source_range_tag);
  if (cs_ptr->callee()) {
    elements.emplace_back(
        serialize(cs_ptr->callee().value(), source_range_tags));
  } else {
    elements.emplace_back(c10::IValue());
  }
  c10::IValue serialized_cs = c10::ivalue::Tuple::create(elements);
  serialized_inlined_callstack_[cs_ptr] = serialized_cs;
  return serialized_cs;
}

c10::IValue InlinedCallStackSerializer::serialize_module_instance_info(
    const c10::optional<ModuleInstanceInfo>& m) {
  if (!m) {
    return c10::IValue();
  }
  const auto& m_val = m.value();
  std::string module_type_name = m_val.class_type()->name()->qualifiedName();
  auto module_instance_name = m_val.instance_name();
  if (m_val.class_type()) {
    module_type_name = m_val.class_type()->name()->qualifiedName();
  }
  auto key_val = module_type_name + module_instance_name;
  auto m_inst_it = serialized_module_instance_info_.find(key_val);
  if (m_inst_it != serialized_module_instance_info_.end()) {
    return m_inst_it->second;
  }
  std::vector<c10::IValue> elements;
  // Module instance info is serialized as
  // {type name, instance name}
  elements = {module_type_name, module_instance_name};
  serialized_module_instance_info_[key_val] =
      c10::ivalue::Tuple::create(std::move(elements));
  return serialized_module_instance_info_[key_val];
}

std::vector<char> CallStackDebugInfoPickler::pickle(
    const std::unordered_map<int64_t, DebugInfoTuple>& callstack_ptrs,
    const SourceRangeTagMap& source_range_tags) {
  std::vector<c10::IValue> ivalues;
  for (const auto& it : callstack_ptrs) {
    int64_t debug_handle = it.first;
    std::vector<c10::IValue> elements;
    /*
     * Debug handles and debug info (source range + inlinded callstack)
     * are serialized as a tuple of 3 elements
     * {debug_handle, source_range_tag, serialized_callstack}
     */
    elements.reserve(3);
    elements.emplace_back(debug_handle);
    int64_t source_range_tag{kInvalidSourceRangeTag};
    const auto source_range =
        std::get<kDebugInfoTupleSourceRangeIndex>(it.second);
    const SourceRange& sr = source_range.findSourceRangeThatGenerated()
        ? source_range.findSourceRangeThatGenerated().value()
        : source_range;
    auto sr_it = source_range_tags.find(sr);
    if (sr_it != source_range_tags.end()) {
      source_range_tag = sr_it->second;
    }
    elements.emplace_back(source_range_tag);
    elements.emplace_back(std::get<kDebugInfoTupleNodeNameIndex>(it.second));
    const auto inlined_cs_ptr =
        std::get<kDebugInfoTupleInlinedCSIndex>(it.second);
    elements.emplace_back(css_.serialize(inlined_cs_ptr, source_range_tags));
    c10::IValue tuple = c10::ivalue::Tuple::create(elements);
    ivalues.emplace_back(tuple);
  }
  std::vector<at::Tensor> table;
  c10::IValue ivalue = c10::ivalue::Tuple::create(std::move(ivalues));
  auto result = jit::pickle(ivalue, &table);
  TORCH_CHECK(table.size() == 0, "Expected 0 tensors to be written");
  return result;
}

InlinedCallStackPtr InlinedCallStackDeserializer::deserialize(
    const c10::IValue& iv,
    const ska::flat_hash_map<int64_t, SourceRange>& source_range_map,
    const std::shared_ptr<CompilationUnit>& cu) {
  if (iv.isNone()) {
    return c10::intrusive_ptr<InlinedCallStack>();
  }
  auto tup = iv.toTuple();
  auto it = cached_inlined_callstacks_.find(tup);
  if (it != cached_inlined_callstacks_.end()) {
    return it->second;
  }

  auto tup_elems = tup->elements();
  TORCH_INTERNAL_ASSERT(tup_elems.size() == 3);
  // {IValue(module_instance_info), source_range_tag, IValue(InlinedCallStack)}
  auto module_instance_info =
      deserialize_module_instance_info(tup_elems[0], cu);
  int64_t source_range_tag = tup_elems[1].toInt();
  auto source_range_it = source_range_map.find(source_range_tag);
  TORCH_CHECK(
      source_range_tag == kInvalidSourceRangeTag ||
          source_range_it != source_range_map.end(),
      "Source range tag must exist in deserialized source range map."
      " Not found source range tag:",
      source_range_tag);
  SourceRange source_range;
  if (source_range_tag != kInvalidSourceRangeTag) {
    source_range = source_range_it->second;
  }
  auto callee = deserialize(tup_elems[2], source_range_map, cu);
  InlinedCallStackPtr cs_ptr;
  if (callee) {
    cs_ptr = c10::make_intrusive<InlinedCallStack>(
        callee, nullptr, source_range, module_instance_info);
  } else {
    cs_ptr = c10::make_intrusive<InlinedCallStack>(
        nullptr, source_range, module_instance_info);
  }
  cached_inlined_callstacks_[tup] = cs_ptr;
  // Invoking move constructor
  // It is not clear if copy-ellision can happen since
  // cs_ptr is copied into map above.
  // This is to help avoid ref count update
  return cs_ptr;
}

c10::optional<ModuleInstanceInfo> InlinedCallStackDeserializer::
    deserialize_module_instance_info(
        const c10::IValue& iv,
        const std::shared_ptr<CompilationUnit>& cu) {
  if (iv.isNone()) {
    return c10::nullopt;
  }
  auto tup = iv.toTuple();
  auto it = cached_module_instance_info_.find(tup);
  if (it != cached_module_instance_info_.end()) {
    return it->second;
  }
  auto tup_elems = iv.toTuple()->elements();
  TORCH_CHECK(tup_elems.size() == 2);
  std::string type_name = tup_elems[0].toString()->string();
  std::string instance_name = tup_elems[1].toString()->string();
  // type_name might be empty string ""
  // In that case type_ptr should be just nullptr
  auto type_ptr = cu->get_class(type_name);
  if (!type_ptr) {
    // We may have lost type information. For example in lowered backends
    // original class type has no relevance.
    // However, to correlate ops to their original modules
    // we saved both type name and instance name.
    // In such cases, when module is absorbed by lowered backend
    // we augment instance name with type name instead of losing it.
    auto last_dot_position = type_name.find_last_of('.');
    size_t substring_pos{0};
    if (last_dot_position != std::string::npos) {
      substring_pos = last_dot_position + 1;
    }
    type_name = type_name.substr(substring_pos);
    instance_name = instance_name + "(" + type_name + ")";
  }
  cached_module_instance_info_[tup] =
      ModuleInstanceInfo(type_ptr, instance_name);
  return cached_module_instance_info_[tup];
}

ska::flat_hash_map<int64_t, DebugInfoTuple> CallStackDebugInfoUnpickler::
    unpickle(
        at::DataPtr&& data,
        size_t size,
        const ska::flat_hash_map<int64_t, SourceRange>& source_range_map,
        const std::shared_ptr<CompilationUnit>& cu) {
  auto ival = jit::unpickle(reinterpret_cast<const char*>(data.get()), size);
  ska::flat_hash_map<int64_t, DebugInfoTuple> callstack_ptrs;
  auto& ivalues = ival.toTuple()->elements();
  for (auto& val : ivalues) {
    const auto tup_elems = val.toTuple()->elements();
    TORCH_CHECK(
        tup_elems.size() == 4,
        "Pickled map must have four elements: "
        "debug_handle, source_range_tag, op name, IValue(inlined_call_stack)");
    int64_t debug_handle = tup_elems[0].toInt();
    int64_t source_range_tag = tup_elems[1].toInt();
    const std::string& node_name = tup_elems[2].toStringRef();
    auto source_range_it = source_range_map.find(source_range_tag);
    TORCH_CHECK(
        source_range_it != source_range_map.end(),
        "Source range tag must exist in deserialized source range map.");
    auto source_range = source_range_it->second;
    TORCH_CHECK(
        callstack_ptrs.count(debug_handle) == 0,
        "Debug handles should be unique.");
    callstack_ptrs[debug_handle] = std::make_tuple(
        source_range,
        node_name,
        csds_.deserialize(tup_elems[3], source_range_map, cu));
  }
  return callstack_ptrs;
}

} // namespace jit
} // namespace torch
