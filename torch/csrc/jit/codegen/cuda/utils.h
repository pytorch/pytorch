#pragma once

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/codegen/cuda/type.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

void debugPrint(const c10::TensorTypePtr& type);

bool is_zero_dim_tensor(const std::shared_ptr<c10::TensorType>& tensor_type);
bool is_zero_sized_tensor(const std::shared_ptr<c10::TensorType>& tensor_type);

bool is_cpu_scalar(const at::Tensor& tensor);
bool is_cpu_scalar(const c10::TensorType& tensor_type);

// TODO: merge these two
// check if input is compatible with 32b index mode
int getCommonDeviceCUDA(const at::ArrayRef<IValue>& inputs);
KernelIndexMode collectIndexMode(const at::ArrayRef<at::IValue>& inputs);

//! Types of debug print-outs
//!
//! These can be set through the `PYTORCH_NVFUSER_DUMP` environment variable
//!
enum class DebugDumpOption {
  FusionIr, //!< Dump the Fusion IR before lowering
  FusionIrMath, //!< Dump just the compute (math) part of the Fusion IR
  FusionIrPresched, //!< Dump the Fusion IR before it is scheduled.
  KernelIr, //!< Dump the compiler Kernel IR
  ComputeAtMap, //!< Dump the computeAt map
  CudaKernel, //!< Dump the generated CUDA C++ kernel code
  CudaFull, //!< Dump the complete CUDA C++ code
  CudaToFile, //!< Dump CUDA Strings to File
  DebugInfo, //!< Embed line info and debug info to compiled kernel, and dump
             //!< the full CUDA C++ code
  LaunchParam, //!< Dump the Launch parameters of kernel
  FusionSegments, //!< Dump Segmented Fusion Graph
  FusionSegmenterLog, //!< Dump Detailed Segmenter Logging
  FusionArgs, //!< Print the runtime fusion arguments
  KernelArgs, //!< Print the runtime kernel arguments when launching kernels
  EffectiveBandwidth, //! Measure kernel performance and print effective
                      //! bandwidth
  FusionSegmentsDrawing, //!< Dump Segmented Fusion Graph
  PrintPtxasLog, //!< Print the ptxas verbose log including register usage
  BufferReuseInfo, //!< Dump the analysis details of local/shared buffer re-use
  SchedulerDebug, //! Dump scheduler heuristic parameters
  ParallelDimensions, //!< Dump known parallel dimensions
  Halo, //! Halo information of tensors
  PerfDebugVerbose, //! When running kernels, print verbose information
                    //! associated with what's running
  PythonDefinition, //! Python Frontend Fusion Definition.
  PythonFrontendDebug, //! Python Frontend debug information.
  TransformPropagator, //! When running TransformPropagator, print propagation
                       //! path and replay result
  InlinePropagator, //! When running InlinePropagator, print propagation
                    //! path and inlining result
  Cubin, //! Dump compiled CUBIN
  Ptx //! Dump compiled PTX
};

TORCH_CUDA_CU_API bool isDebugDumpEnabled(DebugDumpOption option);

//! Types of features to disable
//!
//! These can be set through the `PYTORCH_NVFUSER_DISABLE` environment variable
//!
enum class DisableOption {
  ArchCheck, //! Disable hardware-specific checks to enable cross arch debug
  Fallback, //! Disable fallback
  Fma, //! Disable FMA instructions
  IndexHoist, //! Disable index hoisting
  Nvtx, //! Disable NVTX instrumentation
  PredicateElimination //! Disable predicate elimination
};

TORCH_CUDA_CU_API bool isOptionDisabled(DisableOption option);

//! Types of features to enable
//!
//! These can be set through the `PYTORCH_NVFUSER_ENABLE` environment variable
//!
enum class EnableOption {
  Complex, //! Enable complex support on python
  KernelProfile, //! Enable intra-kernel performance profiling
  LinearDecomposition, //! Enable linear-bias decomposition
  ConvDecomposition, //! Enable conv-bias decomposition
  TransposeScheduler //! Enable the experimental transpose scheduler
};

TORCH_CUDA_CU_API bool isOptionEnabled(EnableOption option);

// Check if fallback path should be used which will dispatch to eagermode if any
// errors are encountered. Helpful for debugging.
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

template <class T, std::enable_if_t<std::is_enum<T>::value, bool> = true>
constexpr unsigned int switch_pair(T t1, T t2) {
  constexpr unsigned int _WORD_SHIFT = 16;
  return ((unsigned int)t1 << _WORD_SHIFT) + (unsigned int)t2;
}

std::vector<int64_t> getTensorSizes(TensorTypePtr const& tensor_type);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
