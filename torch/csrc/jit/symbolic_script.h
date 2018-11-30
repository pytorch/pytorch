#pragma once
// This file is temporary until native_functions.yaml and derivatives.yaml are merged.
// Ideally this should all go into native_functions.yaml

namespace torch { namespace jit {
  const std::unordered_map<std::string, std::string> symbolic_scripts({
    {"aten::mul(Tensor self, Tensor other) -> Tensor",
R"(
def forward(self, other):
    return self * other, (self, other)
def backward(ctx, grad_output):
    # type: (Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tensor]
    self, other = ctx
    return grad_output * other, grad_output * self
)"},
    });

  inline bool check_symbolic_scripts(Node* n) {
    for (auto & it : symbolic_scripts) {
      if (n->matches(it.first.c_str())) {
        return true;
      }
    }
    return false;
  }
}}
