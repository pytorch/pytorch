#pragma once

#include <ATen/core/function_schema.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/either.h>
#include <c10/core/DispatchKey.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/DispatchKeyExtractor.h>

#include <array>
#include <atomic>
#include <iostream>
#include <mutex>
#include <type_traits>
#include <sstream>
#include <unordered_map>
#include <functional>

namespace c10 {

namespace impl {
/**
 * A KernelFunctionTable is a map from DispatchKey to a KernelFunction.
 * It can store zero or one KernelFunctions for each DispatchKey.
 */
class KernelFunctionTable final {
public:
  explicit KernelFunctionTable()
  : kernels_()
  , kernelCount_(0) {}

  enum class SetKernelResult : uint8_t {ADDED_NEW_KERNEL, OVERWROTE_EXISTING_KERNEL};
  C10_NODISCARD SetKernelResult setKernel(DispatchKey dispatchKey, KernelFunction kernel) {
    TORCH_INTERNAL_ASSERT(dispatchKey != DispatchKey::Undefined);
    auto& slot = kernels_[static_cast<uint8_t>(dispatchKey)];
    SetKernelResult result;;
    if (slot.isValid()) {
      result = SetKernelResult::OVERWROTE_EXISTING_KERNEL;
    } else {
      result = SetKernelResult::ADDED_NEW_KERNEL;
      ++kernelCount_;
    }
    slot = std::move(kernel);
    return result;
  }

  enum class RemoveKernelIfExistsResult : uint8_t {REMOVED_KERNEL, KERNEL_DIDNT_EXIST};
  RemoveKernelIfExistsResult removeKernelIfExists(DispatchKey dispatchKey) {
    auto& slot = kernels_[static_cast<uint8_t>(dispatchKey)];
    if (slot.isValid()) {
      --kernelCount_;
      slot = {};
      return RemoveKernelIfExistsResult::REMOVED_KERNEL;
    } else {
      return RemoveKernelIfExistsResult::KERNEL_DIDNT_EXIST;
    }
  }

  const KernelFunction& operator[](DispatchKey dispatchKey) const {
    return kernels_[static_cast<uint8_t>(dispatchKey)];
  }

  size_t size() const {
    return kernelCount_;
  }

private:
  std::array<KernelFunction, static_cast<uint8_t>(DispatchKey::NumDispatchKeys)> kernels_;
  size_t kernelCount_;
};
}

/**
 * Per-operator dispatch table.
 *
 * Given an operator specified by a FunctionSchema, this class records a dispatch
 * table for various kernels provided for this operator.  For example, if we
 * consider the operator add(Tensor, Tensor), the dispatch table for this
 * operator may contain implementations for various dynamic tensor types, such
 * as CPUTensorId, CUDATensorId, etc.
 */
class DispatchTable final {
 public:
  explicit DispatchTable(const FunctionSchema& schema)
  : kernels_()
  , catchallKernel_()
  , dispatchKeyExtractor_(DispatchKeyExtractor::make(schema))
  , operatorName_(toString(schema.operator_name())) {}

  /**
   * Register a kernel in the table at some dispatch key.
   * @param dispatch_key Dispatch key to define when this kernel is selected.
   * @param kernel Concrete kernel function implementation to register
   */
  void setKernel(DispatchKey dispatchKey, KernelFunction kernel) {
    auto result = kernels_.setKernel(dispatchKey, std::move(kernel));
    dispatchKeyExtractor_.setOperatorHasKernelForBackend(dispatchKey, true);
    if (result == impl::KernelFunctionTable::SetKernelResult::OVERWROTE_EXISTING_KERNEL) {
      TORCH_WARN("Registered a kernel for operator ", operatorName_, " with dispatch key ", toString(dispatchKey), " that overwrote a previously registered kernel with the same dispatch key for the same operator.");
    }
  }

  /**
   * Deregister the kernel for some dispatch key.
   *
   * @param dispatch_key Dispatch key to unregister.
   */
  void removeKernelIfExists(DispatchKey dispatchKey) {
    kernels_.removeKernelIfExists(dispatchKey);
    dispatchKeyExtractor_.setOperatorHasKernelForBackend(dispatchKey, false);
  }

  /**
   * Register a catch-all kernel that is called for this operator
   * independent of the inputs. An operator can have either
   * a catch-all kernel or a set of kernels with concrete
   * dispatch keys, not both.
   */
  void setCatchallKernel(KernelFunction kernel) {
    if (catchallKernel_.isValid()) {
      TORCH_WARN("Registered a catch-all kernel for operator ", operatorName_," that overwrote a previously registered catch-all kernel for the same operator.");
    }
    catchallKernel_ = std::move(kernel);
  }

  /**
   * Remove the catch-all kernel.
   */
  void removeCatchallKernel() {
    TORCH_INTERNAL_ASSERT(catchallKernel_.isValid(), "Tried to remove the catch-all kernel for operator ", operatorName_," but there is no catch-all kernel registered.");
    catchallKernel_ = {};
  }

  bool isEmpty() const {
    return !catchallKernel_.isValid() && kernels_.size() == 0;
  }

  std::string listAllDispatchKeys() const {
    std::ostringstream str;
    str << "[";

    bool has_kernels = false;
    for (uint8_t iter = 0; iter != static_cast<uint8_t>(DispatchKey::NumDispatchKeys); ++iter) {
      if (!kernels_[static_cast<DispatchKey>(iter)].isValid()) {
        continue;
      }
      if (has_kernels) {
        str << ", ";
      }
      str << toString(static_cast<DispatchKey>(iter));
      has_kernels = true;
    }

    if (catchallKernel_.isValid()) {
      if (has_kernels) {
        str << ", ";
      }
      str << "CATCH-ALL";
    }
    str << "]";
    return str.str();
  }

  const KernelFunction* lookup(DispatchKey dispatchKey) const {
    auto& slot = kernels_[dispatchKey];
    if (slot.isValid()) {
      return &slot;
    } else {
      return nullptr;
    }
  }

  const KernelFunction* lookupCatchallKernel() const {
    if (!catchallKernel_.isValid()) {
      return nullptr;
    }

    return &catchallKernel_;
  }

  const DispatchKeyExtractor& dispatchKeyExtractor() const {
    return dispatchKeyExtractor_;
  }

  const std::string& operatorName() const {
    return operatorName_;
  }

private:

  impl::KernelFunctionTable kernels_;
  KernelFunction catchallKernel_;
  DispatchKeyExtractor dispatchKeyExtractor_;
  std::string operatorName_;
};

} // namespace c10
