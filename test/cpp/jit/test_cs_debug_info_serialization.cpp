#include <test/cpp/jit/test_utils.h>

#include <gtest/gtest.h>

#include <c10/core/TensorOptions.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/backends/backend_debug_handler.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/serialization/callstack_debug_info_serialization.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/custom_class.h>
#include <torch/torch.h>

#include <stack>
#include <unordered_set>

// Tests go in torch::jit
namespace torch {
namespace jit {

namespace {
bool validate_debug_info(
    const DebugInfoPair& pre_serialize,
    const DebugInfoPair& post_serialize) {
  auto sr1 = pre_serialize.first;
  auto sr2 = post_serialize.first;
  if (sr1 != sr2) {
    return false;
  }
  if (!pre_serialize.second.defined()) {
    return !post_serialize.second.defined();
  }
  if (!post_serialize.second.defined()) {
    return false;
  }
  auto vec1 = pre_serialize.second->vec();
  auto vec2 = post_serialize.second->vec();
  if (vec1.size() != vec2.size()) {
    return false;
  }
  for (size_t i = 0; i < vec1.size(); i++) {
    auto rhs_sr = std::get<1>(vec1[i]);
    auto lhs_sr = std::get<1>(vec2[i]);
    auto rhs_module = std::get<2>(vec1[i]);
    auto lhs_module = std::get<2>(vec2[i]);
    if (!((rhs_module.has_value() == lhs_module.has_value()) &&
          (rhs_module.has_value() &&
           (rhs_module.value().class_type()->name().value() ==
            lhs_module.value().class_type()->name().value()) &&
           (rhs_module.value().instance_name() ==
            lhs_module.value().instance_name())) &&
          (rhs_sr == lhs_sr))) {
      return false;
    }
  }
  return true;
}

TEST(CSDebugInfoSerializaitionTest, TwoSubmodules) {
  std::shared_ptr<CompilationUnit> cu = std::make_shared<CompilationUnit>();
  Module a("A", cu);
  a.define(R"JIT(
    def forward(self, x):
      return x + 1
  )JIT");
  Module b("B", cu);
  b.define(R"JIT(
    def forward(self, x):
      return x + 2
  )JIT");
  Module c("C", cu);
  c.register_module("A0", a);
  c.register_module("B0", b);
  c.define(R"JIT(
    def forward(self, x):
      return self.A0.forward(x) + self.B0.forward(x)
  )JIT");

  BackendDebugHandleManager debug_handle_manager;
  auto graph = c.get_method("forward").graph();
  Inline(*graph);
  std::stack<Block*> blocks_to_visit;

  // maps from source range to debug handle
  SourceRangeTagMap source_range_tags;
  // Maps from debug handle to source range
  ska::flat_hash_map<int64_t, SourceRange> source_range_map;
  int64_t source_range_tag{0};

  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      source_range_tags[n->sourceRange()] = source_range_tag;
      source_range_map[source_range_tag] = n->sourceRange();
      source_range_tag++;
      debug_handle_manager.getNextDebugHandleForInlinedCallStackPtr(n);
      if (n->callstack().has_value()) {
        for (const auto& e : n->callstack().value()->vec()) {
          auto sr = std::get<1>(e);
          source_range_tags[sr] = source_range_tag;
          source_range_map[source_range_tag] = sr;
          source_range_tag++;
        }
      }
    }
  }
  auto debug_handle_cs_ptr_map = debug_handle_manager.getCallStackPtrMap();
  CallStackDebugInfoPickler cs_debug_info_pickler;
  auto cs_data =
      cs_debug_info_pickler.pickle(debug_handle_cs_ptr_map, source_range_tags);
  at::DataPtr data_ptr(cs_data.data(), DeviceType::CPU);
  CallStackDebugInfoUnpickler unpickler;
  auto deserialized_cs_map = unpickler.unpickle(
      std::move(data_ptr), cs_data.size(), source_range_map, cu);
  for (const auto& it : debug_handle_cs_ptr_map) {
    auto handle = it.first;
    auto debug_info_one = it.second;
    TORCH_CHECK(
        deserialized_cs_map.count(handle),
        "Serialized debug handle must be in deserialized map.");
    auto debug_info_two = deserialized_cs_map[handle];
    ASSERT_TRUE(validate_debug_info(debug_info_one, debug_info_two));
  }
}

} // namespace

} // namespace jit
} // namespace torch
