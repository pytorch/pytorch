#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <vector>

namespace torch {
namespace jit {
namespace tensorexpr {

class KernelScopedObject;

// An arena that manages all the underlying kernel-scoped objects.
class KernelArena {
 public:
  static KernelArena* GetCurrentKernelArena();
  static void SetCurrentKernelArena(KernelArena* new_arena);
  TORCH_API KernelArena() = default;
  TORCH_API ~KernelArena();
  KernelArena(const KernelArena&) = delete;
  KernelArena& operator=(const KernelArena&) = delete;

 private:
  friend class KernelScopedObject;
  std::vector<KernelScopedObject*> kernel_objects_; // owned
};

// A RAII convenience wrapper on top of a kernel.
// It either creates or takes an existing Kernel and sets it as the current
// Kernel. When this object is destroyed, the previous Kernel is set as current,
// and the created kernel is freed. If the kernel was passed, it stays alive.
class KernelScope {
 public:
  TORCH_API KernelScope();
  TORCH_API explicit KernelScope(KernelArena* arena_);
  TORCH_API ~KernelScope();
  KernelScope(const KernelScope&) = delete;
  KernelScope& operator=(const KernelScope&) = delete;

 private:
  KernelArena* old_kernel_arena_ =
      nullptr; // previous arena, will be restored in destructor
  bool owning_ = false; // determines whether the arena will be freed along with
                        // the scope object
};

// The base object managed by the Kernel.
// The object must be created through "new", and when the Kernel is destroyed,
// All its registered objects are destroyed through "delete".
class TORCH_API KernelScopedObject {
 public:
  KernelScopedObject();
  virtual ~KernelScopedObject() = default;

  KernelScopedObject(const KernelScopedObject&) = delete;
  KernelScopedObject& operator=(const KernelScopedObject&) = delete;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
