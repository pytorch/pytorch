#pragma once

#include <cstring>
#include <functional>
#include <map>

#include "Evalue.h"
#include "kernel_runtime_context.h"

#include <c10/util/ArrayRef.h>

namespace torch {
namespace executor {

using KernelFunction = std::function<void(KernelRuntimeContext&, EValue**)>;

template<typename T>
using ArrayRef = at::ArrayRef<T>;

#define EXECUTORCH_SCOPE_PROF(x)

struct Kernel {
  const char* name_;
  KernelFunction kernel_;

  Kernel() = default;

  /**
   * We are doing a copy of the string pointer instead of duplicating the string
   * itself, we require the lifetime of the kernel name to be at least as long
   * as the kernel registry.
   */
  explicit Kernel(const char* name, KernelFunction func)
      : name_(name), kernel_(func) {}
};

/**
 * See KernelRegistry::hasKernelFn()
 */
bool hasKernelFn(const char* name);

/**
 * See KernelRegistry::getKernelFn()
 */
KernelFunction& getKernelFn(const char* name);


[[nodiscard]] bool register_kernels(const ArrayRef<Kernel>&);

struct KernelRegistry {
 public:
  KernelRegistry() : kernelRegSize_(0) {}

  bool register_kernels(const ArrayRef<Kernel>&);

  /**
   * Checks whether an kernel with a given name is registered
   */
  bool hasKernelFn(const char* name);

  /**
   * Checks whether an kernel with a given name is registered
   */
  KernelFunction& getKernelFn(const char* name);

 private:
  std::map<const char*, KernelFunction> kernels_map_;
  uint32_t kernelRegSize_;
};

} // namespace executor
} // namespace torch
