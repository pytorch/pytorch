

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/tensorexpr/mem_arena.h>

namespace torch {
namespace jit {

void TORCH_API foxPass();
tensorexpr::KernelArena* TORCH_API enterNewKernelScope();
void TORCH_API exitKernelScope(tensorexpr::KernelArena *orig);

}} // namespace torch::jit
