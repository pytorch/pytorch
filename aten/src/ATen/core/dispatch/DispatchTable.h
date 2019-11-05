#pragma once

#include <ATen/core/function_schema.h>
#include <c10/util/LeftRight.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/either.h>
#include <c10/core/TensorTypeId.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/Variadic.h>

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

// Take a TensorTypeSet for a Tensor, and combine it with the current thread
// local valid (implemented) and enabled (not implemented) TensorTypeSets
// to determine what the actual dispatch TensorTypeId should be.  Unlike
// Tensor::type_set(), the value of this on a tensor can change depending
// on TLS.
//
// NB: I didn't make this take a Tensor to avoid header include shenanigans.
//
// TODO: I'm not sure if this should live in this header or not; the operant
// question is whether or not we have access to all the relevant TLS at this
// point.
static inline TensorTypeId dispatchTypeId(TensorTypeSet ts) {
  c10::impl::LocalTensorTypeSet local = c10::impl::tls_local_tensor_type_set();
  return ((ts | local.included_) - local.excluded_).highestPriorityTypeId();
}

}

namespace detail {
  struct MultiDispatchTensorTypeSet : at::IterArgs<MultiDispatchTensorTypeSet> {
    TensorTypeSet ts;
    void operator()(const at::Tensor& x) {
      ts = ts | x.type_set();
    }
    void operator()(TensorOptions x) {
      ts = ts | x.type_set();
    }
    void operator()(at::ArrayRef<at::Tensor> xs) {
      for (const auto& x : xs) {
        ts = ts | x.type_set();
      }
    }
    template <typename T>
    void operator()(const T& x) {
      // do nothing
    }
  };

  // NB: take by const reference (Don't do universal forwarding here! You
  // don't want to move into this function!)
  template <typename... Args>
  TensorTypeSet multi_dispatch_tensor_type_set(const Args&... args) {
    return MultiDispatchTensorTypeSet().apply(args...).ts;
  }
}

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
   * @param stack Stack with arguments to invoke the kernel function with
   * @return Kernel function pointing to the right kernel for the given arguments.
   */
   const KernelFunction& lookupBoxed(const Stack* stack) const {
     return lookup_([&] () -> c10::optional<TensorTypeId> {
       return dispatch_strategy_.get_dispatch_key_boxed(stack);
     });
   }

   /**
    * Perform a dynamic dispatch on this table and find the kernel to call
    * for the given arguments.
    *
    * @param args Arguments to invoke the kernel function with
    * @return Kernel function pointing to the right kernel for the given arguments.
    */
   template<class... Args>
   const KernelFunction& lookupUnboxed(const Args&... args) const {
     return lookup_([&] () -> c10::optional<TensorTypeId> {
       return dispatch_strategy_.get_dispatch_key_unboxed<Args...>(args...);
     });
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

    c10::optional<TensorTypeId> get_dispatch_key_boxed(const Stack* stack) const {
      // TODO Unboxed dispatch supports TensorOptions (i.e. ScalarType/Device/Layout) arguments
      //      but boxed doesn't yet. These should be aligned and do the same thing.
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
      if (ts.empty()) {
        return c10::nullopt;
      }
      // TODO: Don't use legacy extractor; blocked on c10 understanding
      // variable
      return c10::legacyExtractTypeId(ts);
    }

    template<class... Args>
    c10::optional<TensorTypeId> get_dispatch_key_unboxed(const Args&... args) const {
      auto type_set = detail::multi_dispatch_tensor_type_set(args...);
      if (type_set.empty()) {
        return c10::nullopt;
      }
      return impl::dispatchTypeId(type_set);
    }
  };

  static DispatchStrategy get_dispatch_strategy_(const FunctionSchema& schema) {
    return {schema.arguments().size()};
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

      const std::string dispatch_key_str = toString(*dispatch_key);
      TORCH_CHECK(false, "Could not run '", operator_name_, "' with arguments",
                  " from the '", dispatch_key_str, "' backend. '",
                  operator_name_, "' is only available for these backends: ",
                  listAllDispatchKeys(), ".");
  }

  detail::KernelTable_ kernels_;
  c10::optional<KernelFunction> catchall_kernel_;
  DispatchStrategy dispatch_strategy_;
  std::string operator_name_;
};

} // namespace c10
