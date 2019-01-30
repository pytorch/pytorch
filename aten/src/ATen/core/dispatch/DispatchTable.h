#pragma once

#include <ATen/core/function_schema.h>
#include <c10/util/LeftRight.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/flat_hash_map.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/dispatch/KernelFunction.h>

#include <array>
#include <atomic>
#include <iostream>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace c10 {

/**
 * The type of a user-supplied function to initialize the kernel cache.
 * this is stored together with the kernel function in the dispatch table
 * so we can create a new cache instance when a kernel is looked up
 * from the dispatch table.
 */
using KernelCacheCreatorFunction = std::unique_ptr<c10::KernelCache> ();
/**
 * The dispatch table stores a pointer to a kernel function and a pointer
 * to a function initializing a cache for the kernel. If the kernel wants
 * to use the cache, they supply the state initializer when the kernel
 * is registered. When a kernel is looked up from the dispatcher, a new
 * cache instance is created for it and each call to that kernel will get
 * this same cache instance.
 */
struct DispatchTableEntry final {
  KernelFunction* kernel_func;
  KernelCacheCreatorFunction* cache_creator_func;
};

namespace details {
/// Kernel implementations in a thread-safe hash table.
class ThreadsafeOperatorTable_ final {
 public:
  void emplace(TensorTypeId key, const DispatchTableEntry& value) {
    bool res = map_.write([&](ska::flat_hash_map<TensorTypeId, DispatchTableEntry>& map) -> bool {
      auto result = map.emplace(key, value);
      return result.second;
    });
    if (!res) {
      AT_ERROR("Tried to register conflicting kernels to the dispatcher: ", key);
    }
  }

  void erase(TensorTypeId key) {
    auto num_removed =
        map_.write([&](ska::flat_hash_map<TensorTypeId, DispatchTableEntry>& map) -> size_t {
          return map.erase(key);
        });
    assert(num_removed <= 1); // This is not a multi-map
    if (num_removed == 0) {
      AT_ERROR("Tried to deregister a kernel that isn't registered.");
    }
  }

  const DispatchTableEntry* lookup(TensorTypeId key) const {
    return map_.read([&](const ska::flat_hash_map<TensorTypeId, DispatchTableEntry>& map) -> const DispatchTableEntry* {
      auto found = map.find(key);
      if (found != map.end()) {
        return &found->second;
      } else {
        return nullptr;
      }
    });
  }

 private:
  LeftRight<ska::flat_hash_map<TensorTypeId, DispatchTableEntry>> map_;
};
} // namespace details

/**
 * Per-operator dispatch table.
 *
 * Given an operator specified by 'OpSchemaDef', this class records a dispatch
 * table for various kernels provided for this operator.  For example, if we
 * consider the operator add(Tensor, Tensor), the dispatch table for this
 * operator may contain implementations for various dynamic tensor types, such
 * as (CPUFloatTensor, CPUFloatTensor), (CUDAFloatTensor, CUDAFloatTensor), etc.
 */
class DispatchTable final {
 public:
  DispatchTable(FunctionSchema schema)
  : schema_(std::move(schema))
  , kernels_()
  , index_of_first_tensor_arg_(get_index_of_first_tensor_arg_(schema)) {}

  /**
   * Register a kernel in the table at some dispatch key.
   * @param func Concrete kernel function implementation to register
   * @param dispatch_key Dispatch key to define when this kernel is selected
   */
  void registerKernel(
      TensorTypeId dispatch_key,
      const DispatchTableEntry& kernel) {
    kernels_.emplace(dispatch_key, kernel);
  }

  /**
   * Deregister the kernel for some dispatch key.
   *
   * @param dispatch_key Dispatch key to unregister.
   */
  // TODO: This isn't going to work so well when we get more complicated
  // override patterns! In this case, an operator will show up in multiple
  // slots, and erasing them one-by-one is probably not such a good idea.
  void deregisterKernel(TensorTypeId dispatch_key) {
    kernels_.erase(dispatch_key);
  }

  /**
   * Perform a dynamic dispatch on this table and find the kernel to call
   * for the given arguments.
   *
   * @param args Arguments to invoke the function with
   * @return Kernel function pointing to the right kernel for the given arguments
   */
   const DispatchTableEntry& lookup(const Stack* stack) const {
     TensorTypeId dispatch_key = torch::jit::peek(
       *stack,
       index_of_first_tensor_arg_,
       schema_.arguments().size()
     ).toTensor().type_id();
     const DispatchTableEntry* found = kernels_.lookup(dispatch_key);
     if (found == nullptr) {
       // TODO Better error message - include op name and dispatch key (i.e.
       // argument types)
       AT_ERROR("Didn't find kernel to dispatch to for operator '", schema_.name(), "'");
     }
     return *found;
   }

   const FunctionSchema& schema() const {
     return schema_;
   }

 private:
  size_t get_index_of_first_tensor_arg_(const FunctionSchema& schema) {
    for (size_t i = 0; i < schema.arguments().size(); ++i) {
      if (schema.arguments()[i].type()->isSubtypeOf(DynamicType::get())) {  // DynamicType means it's a tensor
        return i;
      }
    }

    throw std::logic_error("Tried to create dispatch table for operator schema " + schema_.name() + " that doesn't have tensor arguments.");
  }


  FunctionSchema schema_;
  details::ThreadsafeOperatorTable_ kernels_;

  // this is caching the index so we don't have to parse the schema inputs
  // again and again for each dispatcher lookup.
  size_t index_of_first_tensor_arg_;
};

} // namespace c10
