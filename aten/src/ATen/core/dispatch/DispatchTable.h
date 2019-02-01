#pragma once

#include <ATen/core/dispatch/OpSchema.h>
#include <c10/util/LeftRight.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/flat_hash_map.h>
#include <ATen/core/ivalue.h>

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
using KernelStateCreatorFunction = std::unique_ptr<c10::KernelState> ();

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
  KernelStateCreatorFunction* state_creator_func;
};

namespace details {
/// Kernel implementations in a thread-safe hash table.
template <class Key>
class ThreadsafeOperatorTable_ final {
 public:
  template <class Key_>
  void emplace(Key_&& key, const DispatchTableEntry& value) {
    bool res = map_.write([&](ska::flat_hash_map<Key, DispatchTableEntry>& map) -> bool {
      auto result = map.emplace(std::forward<Key>(key), value);
      return result.second;
    });
    if (!res) {
      AT_ERROR("Tried to register conflicting kernels to the dispatcher: ", key);
    }
  }

  void erase(const Key& key) {
    auto num_removed =
        map_.write([&](ska::flat_hash_map<Key, DispatchTableEntry>& map) -> size_t {
          return map.erase(key);
        });
    assert(num_removed <= 1); // This is not a multi-map
    if (num_removed == 0) {
      AT_ERROR("Tried to deregister a kernel that isn't registered.");
    }
  }

  const DispatchTableEntry* lookup(const Key& key) const {
    return map_.read([&](const ska::flat_hash_map<Key, DispatchTableEntry>& map) -> const DispatchTableEntry* {
      auto found = map.find(key);
      if (found != map.end()) {
        return &found->second;
      } else {
        return nullptr;
      }
    });
  }

 private:
  LeftRight<ska::flat_hash_map<Key, DispatchTableEntry>> map_;
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
 *
 * @tparam OpSchemaDef The operator signature this dispatch table encodes.
 */
// TODO: Support dispatch for meta-operators (which apply to all dynamic types)
template <class OpSchemaDef>
class DispatchTable final {
 private:
  using Schema = OpSchema<OpSchemaDef>;

 public:
  DispatchTable() : kernels_() {}

  /**
   * Register a kernel in the table at some dispatch key.
   * @param func Concrete kernel function implementation to register
   * @param dispatch_key Dispatch key to define when this kernel is selected
   */
  void registerKernel(
      typename Schema::dispatch::dispatch_key_type dispatch_key,
      const DispatchTableEntry& kernel) {
    kernels_.emplace(std::move(dispatch_key), kernel);
  }

  /**
   * Deregister the kernel for some dispatch key.
   *
   * @param dispatch_key Dispatch key to unregister.
   */
  // TODO: This isn't going to work so well when we get more complicated
  // override patterns! In this case, an operator will show up in multiple
  // slots, and erasing them one-by-one is probably not such a good idea.
  void deregisterKernel(
      const typename Schema::dispatch::dispatch_key_type& dispatch_key) {
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
     auto dispatch_key = Schema::dispatch::dispatch_key(stack);
     const DispatchTableEntry* found = kernels_.lookup(dispatch_key);
     if (found == nullptr) {
       // TODO Better error message - include op name and dispatch key (i.e.
       // argument types)
       AT_ERROR("Didn't find kernel to dispatch to for operator '", Schema::metadata::name(), "'");
     }
     return *found;
   }

 private:


  details::ThreadsafeOperatorTable_<
      typename Schema::dispatch::dispatch_key_type>
      kernels_;
};

} // namespace c10

/*
 * Use this to access the dispatch table singleton for a given op schema.
 * It has an implementation for each op schema def in a cpp file, because
 * we can't rely on the one-definition-rule.
 */
template <class OpSchemaDef>
C10_API c10::DispatchTable<OpSchemaDef>& c10_dispatch_table();
