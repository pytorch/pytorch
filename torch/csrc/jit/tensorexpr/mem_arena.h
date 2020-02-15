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
  static KernelArena* GetCurrentKernelArena();
  static void SetCurrentKernelArena(KernelArena* new_arena);
  TORCH_API KernelArena() {}
  TORCH_API ~KernelArena();

 private:
  KernelArena(const KernelArena&) = delete;
  KernelArena& operator=(const KernelArena&) = delete;
  friend class KernelScopedObject;
  std::vector<KernelScopedObject*> kernel_objects_; // owned
};

// A RAII convenience wrapper on top of a kernel.
// It creates a Kernel and sets it as the current Kernel, as long as this
// KernelScope object is alive. After that, the previous Kernel is set as
// current.
class KernelScope {
 public:
  TORCH_API KernelScope();
  TORCH_API ~KernelScope();

 private:
  KernelScope(const KernelScope&) = delete;
  KernelScope& operator=(const KernelScope&) = delete;
  KernelArena* kernel_arena_ = nullptr;     // owned, created in constructor
  KernelArena* old_kernel_arena_ =
      nullptr; // previous arena, will be restored in destructor
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

