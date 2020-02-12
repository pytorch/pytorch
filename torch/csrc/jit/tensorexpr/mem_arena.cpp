#include <stdexcept>
#include "torch/csrc/jit/tensorexpr/mem_arena.h"

namespace torch {
namespace jit {
namespace tensorexpr {

KernelArena::~KernelArena() {
  for (KernelScopedObject* p : kernel_objects_) {
    delete p;
  }
}

KernelScopedObject::KernelScopedObject() {
  KernelArena& kernel = KernelArena::GetCurrentKernelArena();
  kernel.kernel_objects_.push_back(this);
}

static std::vector<KernelArena*>& GetKernelArenaStack() {
  thread_local std::vector<KernelArena*> kernel_arena_stack;
  return kernel_arena_stack;
}

KernelArena& KernelArena::GetCurrentKernelArena() {
  std::vector<KernelArena*>& kernel_arena_stack = GetKernelArenaStack();
  if (kernel_arena_stack.empty()) {
    throw std::runtime_error(
        "A KernelScope must be bound before creating KernelScopedObject");
  }
  return *kernel_arena_stack.back();
}

KernelScope::KernelScope() : owning_kernel_arena_(true) {
  kernel_arena_ = new KernelArena;
  GetKernelArenaStack().push_back(kernel_arena_);
}

KernelScope::KernelScope(KernelArena& kernel_arena)
    : owning_kernel_arena_(false) {
  kernel_arena_ = &kernel_arena;
  GetKernelArenaStack().push_back(&kernel_arena);
}

KernelScope::~KernelScope() noexcept(false) {
  std::vector<KernelArena*>& kernel_arena_stack = GetKernelArenaStack();
  if (kernel_arena_ != kernel_arena_stack.back()) {
    throw std::runtime_error("Mismatch KernelScope and kernel");
  }
  if (owning_kernel_arena_) {
    delete kernel_arena_;
  }
  kernel_arena_stack.pop_back();
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
