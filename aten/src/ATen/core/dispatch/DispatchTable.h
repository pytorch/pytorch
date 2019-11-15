#pragma once

#include <ATen/core/function_schema.h>
#include <c10/util/LeftRight.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/either.h>
#include <c10/core/TensorTypeId.h>
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
  DispatchTable(const FunctionSchema& schema)
  : dispatchKeyExtractor_(DispatchKeyExtractor::make(schema))
  , operatorName_(toString(schema.operator_name())) {}

  /**
   * Register a kernel in the table at some dispatch key.
   * @param dispatch_key Dispatch key to define when this kernel is selected.
   * @param kernel Concrete kernel function implementation to register
   */
  void setKernel(TensorTypeId dispatchKey, const KernelFunction& kernel) {
    TORCH_INTERNAL_ASSERT(dispatchKey != TensorTypeId::UndefinedTensorId);
    // The following assertion is disabled because we're codegenerating
    // autograd kernels for operators without tensor arguments even though
    // they are never called. These, however, register kernels for
    // VariableTensorId.
    // TODO Stop generating those kernels and re-enable this assertion here.
    auto& slot = kernels_[(uint8_t)dispatchKey];
    if (slot.isValid()) {
      TORCH_WARN("Registered a kernel for operator ", operatorName_," with dispatch key ", toString(dispatchKey), " that overwrote a previously registered kernel with the same dispatch key for the same operator.");
    }
    slot = kernel;
    ++kernelCount_;
  }

  /**
   * Deregister the kernel for some dispatch key.
   *
   * @param dispatch_key Dispatch key to unregister.
   */
  void removeKernelIfExists(TensorTypeId dispatchKey) {
    auto& slot = kernels_[(uint8_t)dispatchKey];
    if (slot.isValid()) {
      --kernelCount_;
      slot = {};
    }
  }

  /**
   * Register a catch-all kernel that is called for this operator
   * independent of the inputs. An operator can have either
   * a catch-all kernel or a set of kernels with concrete
   * dispatch keys, not both.
   */
  void setCatchallKernel(const KernelFunction& kernel) {
    if (catchallKernel_.isValid()) {
      TORCH_WARN("Registered a catch-all kernel for operator ", operatorName_," that overwrote a previously registered catch-all kernel for the same operator.");
    }
    catchallKernel_ = kernel;
  }

  /**
   * Remove the catch-all kernel.
   */
  void removeCatchallKernel() {
    TORCH_INTERNAL_ASSERT(catchallKernel_.isValid(), "Tried to remove the catch-all kernel for operator ", operatorName_," but there is no catch-all kernel registered.");
    catchallKernel_ = {};
  }

  bool isEmpty() const {
   return !catchallKernel_.isValid() && kernelCount_ == 0;
  }

  std::string listAllDispatchKeys() const {
    std::ostringstream str;
    str << "[";

    if (kernelCount_ != 0) {
      bool first = true;
      for (uint8_t iter = 0; iter != (uint8_t)TensorTypeId::NumTensorIds; ++iter) {
        if (!kernels_[iter].isValid()) {
          continue;
        }
        if (!first) {
          str << ", ";
        }
        str << toString((TensorTypeId)iter);
        first = false;
      }
    }
    if (catchallKernel_.isValid()) {
      if (kernelCount_ != 0) {
        str << ", ";
      }
      str << "CATCH-ALL";
    }
    str << "]";
    return str.str();
  }

  const KernelFunction* lookup(TensorTypeId dispatchKey) const {
    auto& slot = kernels_[(uint8_t)dispatchKey];
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

  std::array<KernelFunction, (uint8_t)TensorTypeId::NumTensorIds> kernels_;
  size_t kernelCount_ = 0;
  KernelFunction catchallKernel_;
  DispatchKeyExtractor dispatchKeyExtractor_;
  std::string operatorName_;

};

} // namespace c10
