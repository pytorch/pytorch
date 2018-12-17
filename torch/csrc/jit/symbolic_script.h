#pragma once
// This file is temporary until native_functions.yaml and derivatives.yaml are merged.
// Ideally this should all go into native_functions.yaml

#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/script/module.h"

namespace torch { namespace jit {
  const std::unordered_map<const char*, std::string> symbolic_scripts({
    {"aten::mul(Tensor self, Tensor other) -> Tensor",
R"(
def forward(self, other):
    return self * other, (self, other)
def backward(ctx, grad_output):
    # type: (Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tensor]
    self, other = ctx
    return (grad_output * other).sum_to_size(self.size()), (grad_output * self).sum_to_size(other.size())
)"},
    });

  inline bool check_symbolic_scripts(Node* n) {
    for (auto & it : symbolic_scripts) {
      if (n->matches(it.first)) {
        return true;
      }
    }
    return false;
  }

  // This map is a workaround to cache compiled graphs. Ideally this graph
  // should be compiled only once and saved in Operator structure.
  // This should be done along with merging into native_functions.yaml.
  std::unordered_map<const char*, std::vector<std::shared_ptr<Graph>>> cached_symbolic_graphs;

  inline std::vector<std::shared_ptr<Graph>> get_cached_symbolic_graphs(const char* schema) {
    auto it = cached_symbolic_graphs.find(schema);
    if (it == cached_symbolic_graphs.end()) {
      // Compile the python code to a script module
      auto cu = std::make_shared<script::Module>();
      script::defineMethodsInModule(cu, symbolic_scripts.at(schema), script::nativeResolver, nullptr);
      auto fw_graph = cu->find_method("forward")->graph();
      auto bw_graph = cu->find_method("backward")->graph();
      std::vector<std::shared_ptr<Graph>> compiled_graphs{fw_graph, bw_graph};
      cached_symbolic_graphs.emplace_hint(it, schema, compiled_graphs);
      return compiled_graphs;
    } else {
      return it->second;
    }
  }
}}
