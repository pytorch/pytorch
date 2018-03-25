#include "torch/csrc/jit/script/module.h"

namespace torch { namespace jit { namespace script {

static std::vector<Value*> inlineCallTo(Graph& g, Graph& callee, ArrayRef<Value*> inputs) {
  std::unordered_map<Value*, Value*> value_map;
  auto value_map_func = [&](Value* v) { return value_map.at(v); };
  JIT_ASSERT(callee.inputs().size() == inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    value_map[callee.inputs()[i]] = inputs[i];
  }
  for (auto* node : callee.nodes()) {
    auto* new_node =
        g.insertNode(g.createClone(node, value_map_func));
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      value_map[node->outputs()[i]] = new_node->outputs()[i];
    }
  }

  std::vector<Value*> outputs;
  for (auto* output : callee.outputs()) {
    outputs.push_back(value_map_func(output));
  }
  return outputs;
}

std::vector<Value*> Method::emit_call_to(Method & callee, ArrayRef<Value*> inputs) {
  JIT_ASSERT(!executor);
  auto fn = callee.graph();
  JIT_ASSERT(inputs.size() == callee.num_inputs());
  std::vector<Value*> all_inputs = inputs;
  // parameters to callee method (which become parameters to _this_ method
  // if they were not already)
  for(at::Tensor* member : callee.member_inputs) {
    all_inputs.push_back(get_or_add_parameter(member));
  }
  return inlineCallTo(*graph(), *callee.graph(), all_inputs);
}

}}}
