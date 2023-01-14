#include <torch/csrc/autograd/function_hook.h>

namespace torch::autograd {

FunctionPreHook::~FunctionPreHook() = default;
FunctionPostHook::~FunctionPostHook() = default;

} // namespace torch::autograd
