#include <stdexcept>
#include "torch/csrc/jit/tensorexpr/mem_arena.h"

namespace torch {
namespace jit {
namespace tensorexpr {

thread_local KernelArena* current_arena = nullptr;

KernelArena::~KernelArena() {
  for (KernelScopedObject* p : kernel_objects_) {
    delete p;
  }
}

KernelScopedObject::KernelScopedObject() {
  KernelArena* kernel = KernelArena::GetCurrentKernelArena();
  kernel->kernel_objects_.push_back(this);
}

static std::vector<KernelArena*>& GetKernelArenaStack() {
  thread_local std::vector<KernelArena*> kernel_arena_stack;
  return kernel_arena_stack;
}

void KernelArena::SetCurrentKernelArena(KernelArena *new_kernel_arena) {
  current_arena = new_kernel_arena;
}

KernelArena* KernelArena::GetCurrentKernelArena() {
  return current_arena;
}

KernelScope::KernelScope() {
  old_kernel_arena_ = KernelArena::GetCurrentKernelArena();
  KernelArena::SetCurrentKernelArena(new KernelArena);
}

KernelScope::~KernelScope() {
  delete KernelArena::GetCurrentKernelArena();
  KernelArena::SetCurrentKernelArena(old_kernel_arena_);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
