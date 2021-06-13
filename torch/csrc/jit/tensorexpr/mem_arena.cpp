#include <c10/util/Exception.h>
#include <torch/csrc/jit/tensorexpr/mem_arena.h>
#include <stdexcept>

namespace torch {
namespace jit {
namespace tensorexpr {

namespace {
// Define in an anonymous namespace to hide this symbol from other compilation
// units
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local KernelArena* current_arena = nullptr;
} // namespace

KernelArena::~KernelArena() {
  for (KernelScopedObject* p : kernel_objects_) {
    delete p;
  }
}

KernelScopedObject::KernelScopedObject() {
  KernelArena* kernel = KernelArena::GetCurrentKernelArena();
  if (kernel == nullptr) {
    throw std::runtime_error(
        "KernelScope() must be constructed before calling this");
  }
  kernel->kernel_objects_.push_back(this);
}

void KernelArena::SetCurrentKernelArena(KernelArena* new_kernel_arena) {
  current_arena = new_kernel_arena;
}

KernelArena* KernelArena::GetCurrentKernelArena() {
  return current_arena;
}

KernelScope::KernelScope()
    : kernel_arena_(new KernelArena()),
      old_kernel_arena_(KernelArena::GetCurrentKernelArena()),
      owning_(true) {
  KernelArena::SetCurrentKernelArena(kernel_arena_);
}

KernelScope::KernelScope(KernelArena* arena_)
    : kernel_arena_(arena_),
      old_kernel_arena_(KernelArena::GetCurrentKernelArena()),
      owning_(false) {
  KernelArena::SetCurrentKernelArena(kernel_arena_);
}

KernelScope::~KernelScope() {
  if (KernelArena::GetCurrentKernelArena() != kernel_arena_) {
    // This should be an error, but it gets triggered in
    // caffe2/benchmarks/static_runtime:static_runtime_cpptest
    TORCH_WARN("KernelScope() destructed out of order, leaking memory");
    return;
  }
  KernelArena::SetCurrentKernelArena(old_kernel_arena_);
  if (owning_) {
    delete kernel_arena_;
  }
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
