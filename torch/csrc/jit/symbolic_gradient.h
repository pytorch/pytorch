#pragma once

namespace torch { namespace jit {
  const std::unordered_map<std::string, std::string> symbolic_grads({
    {"aten::mul(Tensor self, Tensor other) -> Tensor",
R"(
def forward(x, y):
    return x * y, (x, y)
def backward(ctx, doutput):
    # type: (Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tensor]
    x, y = ctx
    return doutput * y, doutput * x
)"},
    });
}}
