#include <torch/csrc/autograd/function_hook.h>

namespace torch {
namespace autograd {

FunctionPreHook::~FunctionPreHook() = default;
FunctionPostHook::~FunctionPostHook() = default;

} // namespace autograd
} // namespace torch
