#pragma once

#include <c10/util/Exception.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Types of debug print-outs
//!
//! These can be set through the `PYTORCH_NVFUSER_DUMP` environment variable
//!
enum class DebugDumpOption {
  FusionIr, //!< Dump the Fusion IR before lowering
  FusionIrMath, //!< Dump just the compute (math) part of the Fusion IR
  KernelIr, //!< Dump the compiler Kernel IR
  CudaKernel, //!< Dump the generated CUDA C++ kernel code
  CudaFull, //!< Dump the complete CUDA C++ code
  LaunchParam, //!< Dump the Launch parameters of kernel
  FusionSegments, //!< Dump Segmented Fusion Graph
  FusionSegmentsDrawing //!< Dump Segmented Fusion Graph
};

bool isDebugDumpEnabled(DebugDumpOption option);

bool useFallback();

//! Ceil integer division
constexpr int64_t ceilDiv(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

//! Simple mixin for suppressing copy & move operations, ex:
//!
//!  class Foo : public NonCopyable {
//!   ...
//!  };
//!
class NonCopyable {
 public:
  NonCopyable() = default;

  // No copy/move semantics
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
};

//! A generic root for a hierarchy of polymorphic classes:
//! - It ensures virtual destructors
//! - Provides the base->as<Derived>() and node->isA<T>() notation
class PolymorphicBase {
 public:
  virtual ~PolymorphicBase() = default;

  // Replacement for static_cast<T*>(ptr): ptr->as<T>()
  // (checked in DEBUG builds)
  template <class T>
  T* as() {
#ifdef NDEBUG
    auto downcast_ptr = static_cast<T*>(this);
#else
    auto downcast_ptr = dynamic_cast<T*>(this);
    TORCH_INTERNAL_ASSERT(downcast_ptr != nullptr);
#endif
    return downcast_ptr;
  }

  template <class T>
  const T* as() const {
#ifdef NDEBUG
    auto downcast_ptr = static_cast<const T*>(this);
#else
    auto downcast_ptr = dynamic_cast<const T*>(this);
    TORCH_INTERNAL_ASSERT(downcast_ptr != nullptr);
#endif
    return downcast_ptr;
  }

  //! Check if the runtime time is T (or derived from T)
  //!
  //! \note Don't use this for conditional casts. Instead, use:
  //!
  //!  if (auto t = dynamic_cast<T>(p)) { ... }
  //!
  //! instead of:
  //!
  //!  if (p->isA<T>()) { auto t = p->as<T>(); ... }
  //!
  template <class T>
  bool isA() const {
    return dynamic_cast<const T*>(this) != nullptr;
  }
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
