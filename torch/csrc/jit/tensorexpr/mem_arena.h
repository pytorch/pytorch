#pragma once
#include <vector>
#include "torch/csrc/WindowsTorchApiMacro.h"

namespace torch {
namespace jit {
namespace tensorexpr {

class KernelScopedObject;

// An arena that manages all the underlying kernel-scoped objects.
class KernelArena {
 public:
  static KernelArena& GetCurrentKernelArena();
  TORCH_API KernelArena() {}
  TORCH_API ~KernelArena();

 private:
  KernelArena(const KernelArena&) = delete;
  KernelArena& operator=(const KernelArena&) = delete;
  friend class KernelScopedObject;
  std::vector<KernelScopedObject*> kernel_objects_; // owned
};

// A RAII convenience wrapper on top of a kernel.
// It either creates a Kernel, or take another existing Kernel, and sets it as
// the current Kernel, as long as this KernelScope object is alive.
class KernelScope {
 public:
  TORCH_API KernelScope();
  TORCH_API explicit KernelScope(KernelArena& kernel_arena);
  TORCH_API ~KernelScope() noexcept(false);

 private:
  KernelScope(const KernelScope&) = delete;
  KernelScope& operator=(const KernelScope&) = delete;
  bool owning_kernel_arena_ = false;
  KernelArena* kernel_arena_ =
      nullptr; // possibly owned, if owning_kernel_arena_ == true
};

// The base object managed by the Kernel.
// The object must be created through "new", and when the Kernel is destroyed,
// All its registered objects are destroyed through "delete".
class TORCH_API KernelScopedObject {
 public:
  KernelScopedObject();
  virtual ~KernelScopedObject() = default;

 private:
  KernelScopedObject(const KernelScopedObject&) = delete;
  KernelScopedObject& operator=(const KernelScopedObject&) = delete;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch

