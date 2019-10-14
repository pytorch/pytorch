#pragma once

#include <ATen/core/function_schema.h>
#include <c10/util/LeftRight.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/either.h>
#include <c10/core/TensorTypeId.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/ATenDispatch.h>

#include <array>
#include <atomic>
#include <iostream>
#include <mutex>
#include <type_traits>
#include <sstream>
#include <unordered_map>
#include <functional>

namespace c10 {

namespace detail {

class KernelTable_ final {
 public:
  void set(TensorTypeId key, const KernelFunction& value, const std::string& operator_name) {
    auto emplaced = map_.emplace(key, value);
    if (!emplaced.second) {
      // Element already existed. Overwrite it.
      emplaced.first->second = value;
      TORCH_WARN("Registered a kernel for operator ", operator_name," with dispatch key ", toString(key), " that overwrote a previously registered kernel with the same dispatch key for the same operator.");
    }
  }

  void removeIfExists(TensorTypeId key, const std::string& operator_name) {
    auto num_removed = map_.erase(key);
    TORCH_INTERNAL_ASSERT(num_removed <= 1); // This is not a multi-map
  }

  const KernelFunction* lookup(TensorTypeId key) const {
    auto found = map_.find(key);
    if (found != map_.end()) {
      return &found->second;
    } else {
      return nullptr;
    }
  }

  size_t size() const {
    return map_.size();
  }

  std::string list_all_dispatch_keys() const {
    if (map_.size() == 0) {
      return "[]";
    }
    std::ostringstream str;
    str << "[" << toString(map_.begin()->first);
    for (auto iter = ++map_.begin(); iter != map_.end(); ++iter) {
      str << ", " << toString(iter->first);
    }
    str << "]";
    return str.str();
  }

 private:
   ska::flat_hash_map<TensorTypeId, KernelFunction> map_;
};
} // namespace detail

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
  : kernels_()
  , catchall_kernel_(c10::nullopt)
  , dispatch_strategy_(get_dispatch_strategy_(schema))
  , operator_name_(schema.name()) {}

  /**
   * Register a kernel in the table at some dispatch key.
   * @param dispatch_key Dispatch key to define when this kernel is selected.
   * @param kernel Concrete kernel function implementation to register
   */
  void setKernel(
      TensorTypeId dispatch_key,
      const KernelFunction& kernel) {
    TORCH_INTERNAL_ASSERT(dispatch_key != TensorTypeId::UndefinedTensorId);
    // The following assertion is disabled because we're codegenerating
    // autograd kernels for operators without tensor arguments even though
    // they are never called. These, however, register kernels for
    // VariableTensorId.
    // TODO Stop generating those kernels and re-enable this assertion here.
    //TORCH_CHECK(dispatch_strategy_.is_valid_, "Tried to register a kernel with dispatch key ", toString(dispatch_key), " for operator ", operator_name_, " that doesn't have tensor arguments.");
    kernels_.set(dispatch_key, kernel, operator_name_);
  }

  /**
   * Deregister the kernel for some dispatch key.
   *
   * @param dispatch_key Dispatch key to unregister.
   */
  void removeKernelIfExists(TensorTypeId dispatch_key) {
    kernels_.removeIfExists(dispatch_key, operator_name_);
  }

  /**
   * Register a catch-all kernel that is called for this operator
   * independent of the inputs. An operator can have either
   * a catch-all kernel or a set of kernels with concrete
   * dispatch keys, not both.
   */
  void setCatchallKernel(const KernelFunction& kernel) {
    if (catchall_kernel_.has_value()) {
      TORCH_WARN("Registered a catch-all kernel for operator ", operator_name_," that overwrote a previously registered catch-all kernel for the same operator.");
    }
    catchall_kernel_ = kernel;
  }

  /**
   * Remove the catch-all kernel.
   */
  void removeCatchallKernel() {
    TORCH_INTERNAL_ASSERT(catchall_kernel_.has_value(), "Tried to remove the catch-all kernel for operator ", operator_name_," but there is no catch-all kernel registered.");
    catchall_kernel_ = c10::nullopt;
  }

  /**
   * Perform a dynamic dispatch on this table and find the kernel to call
   * for the given arguments.
   *
   * @param args Arguments to invoke the function with
   * @return Kernel function pointing to the right kernel for the given arguments.
   */
   const KernelFunction& lookup(const Stack* stack) const {
     return lookup_([=] () -> c10::optional<TensorTypeId> {
       if (!dispatch_strategy_.is_valid_) {
         return c10::nullopt;
       }
       return dispatch_strategy_.get_dispatch_key(stack, operator_name_);
     });
   }

   const KernelFunction& lookup(TensorTypeId dispatchKey) const {
     return lookup_([=] () -> c10::optional<TensorTypeId> { return dispatchKey;});
   }

   bool isEmpty() const {
     return !catchall_kernel_.has_value() && kernels_.size() == 0;
   }

   std::string listAllDispatchKeys() const {
     std::string result = kernels_.list_all_dispatch_keys();
     if (catchall_kernel_.has_value()) {
       result += ", CATCH-ALL";
     }
     return result;
   }

private:
  struct DispatchStrategy final {
    // this is caching the index so we don't have to parse the schema inputs
    // again and again for each dispatcher lookup.
    // num_args_ is allowed to be zero; that just means you must do the
    // fallthrough
    // TODO: a potential optimization is to store a bitfield of arg locations,
    size_t num_args_;

    // An invalid dispatch strategy means we can't dispatch any kernels.
    // You're able to create a dispatch table with an invalid dispatch strategy,
    // but adding kernels to it will fail.
    // This is used to allow creating operators with empty argument lists
    // as long as they only have fallback kernels and no dispatched kernels.
    bool is_valid_;

    TensorTypeId get_dispatch_key(const Stack* stack, const std::string& operator_name) const {

      TensorTypeSet ts;
      for (const auto& ivalue : torch::jit::last(*stack, num_args_)) {
        if (C10_LIKELY(ivalue.isTensor())) {
          // NB: Take care not to introduce a refcount bump (there's
          // no safe toTensorRef method, alas)
          ts = ts | ivalue.unsafeToTensorImpl()->type_set();
        } else if (C10_UNLIKELY(ivalue.isTensorList())) {
          for (const auto& tensor : ivalue.toTensorListRef()) {
            ts = ts | tensor.type_set();
          }
        }
      }
      // TODO: Don't use legacy extractor; blocked on c10 understanding
      // variable
      return c10::legacyExtractTypeId(ts);
    }
  };

  static DispatchStrategy get_dispatch_strategy_(const FunctionSchema& schema) {
    bool is_valid = false;
    for (size_t i = 0; i < schema.arguments().size(); ++i) {
      const auto& type = schema.arguments()[i].type();
      if (type->isSubtypeOf(TensorType::get())) {
        is_valid = true;
        break;
      }
      if (type->isSubtypeOf(ListType::ofTensors())) {
        is_valid = true;
        break;
      }
    }

    return {schema.arguments().size(), is_valid};
  }

  template<class GetDispatchKeyFunc>
  const KernelFunction& lookup_(const GetDispatchKeyFunc& getDispatchKey) const {
      c10::optional<TensorTypeId> dispatch_key = getDispatchKey();
      if (dispatch_key.has_value()) {
        const auto* found = kernels_.lookup(*dispatch_key);

        if (nullptr != found) {
          return *found;
        }
      }

      if (catchall_kernel_.has_value()) {
        return *catchall_kernel_;
      }

      if (!dispatch_key.has_value() || *dispatch_key == TensorTypeId::UndefinedTensorId) {
        TORCH_CHECK(false,
              "There were no tensor arguments to this function (e.g., you passed an "
              "empty list of Tensors), but no fallback function is registered for schema ", operator_name_,
              ".  This usually means that this function requires a non-empty list of Tensors.  "
              "Available functions are ", listAllDispatchKeys())
      }

      const std::string dispatch_key_str = dispatch_key.has_value() ? toString(*dispatch_key) : "None";
      TORCH_CHECK(false, "Didn't find kernel to dispatch to for operator '", operator_name_,
               "'. Tried to look up kernel for dispatch key '", dispatch_key_str,
               "'. Registered dispatch keys are: ", listAllDispatchKeys());
  }

  detail::KernelTable_ kernels_;
  c10::optional<KernelFunction> catchall_kernel_;
  DispatchStrategy dispatch_strategy_;
  std::string operator_name_;
};

} // namespace c10
