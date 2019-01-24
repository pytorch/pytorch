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

namespace details {
/// Kernel implementations in a thread-safe hash table.
template <class Key>
class ThreadsafeOperatorTable_ final {
 public:
  template <class Key_>
  void emplace(Key_&& key, KernelFunction* value) {
    bool res = map_.write([&](ska::flat_hash_map<Key, KernelFunction*>& map) -> bool {
      auto result = map.emplace(std::forward<Key>(key), std::move(value));
      return result.second;
    });
    if (!res) {
      AT_ERROR("Tried to register conflicting kernels to the dispatcher: ", key);
    }
  }

  void erase(const Key& key) {
    auto num_removed =
        map_.write([&](ska::flat_hash_map<Key, KernelFunction*>& map) -> size_t {
          return map.erase(key);
        });
    assert(num_removed <= 1); // This is not a multi-map
    if (num_removed == 0) {
      AT_ERROR("Tried to deregister a kernel that isn't registered.");
    }
  }

  KernelFunction* lookup(const Key& key) const {
    return map_.read([&](const ska::flat_hash_map<Key, KernelFunction*>& map) -> KernelFunction* {
      auto found = map.find(key);
      if (found != map.end()) {
        return found->second;
      } else {
        return nullptr;
      }
    });
  }

 private:
  LeftRight<ska::flat_hash_map<Key, KernelFunction*>> map_;
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
      KernelFunction* func,
      typename Schema::dispatch::dispatch_key_type dispatch_key) {
    kernels_.emplace(std::move(dispatch_key), func);
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
   KernelFunction* lookup(ArrayRef<IValue> args) const {
     auto dispatch_key = Schema::dispatch::dispatch_key(args);
     KernelFunction* found = kernels_.lookup(dispatch_key);
     if (found == nullptr) {
       // TODO Better error message - include op name and dispatch key (i.e.
       // argument types)
       AT_ERROR("Didn't find kernel to dispatch to for operator '", Schema::metadata::name(), "'");
     }
     return found;
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
