#pragma once

#include "caffe2/core/dispatch/OpSchema.h"
#include <c10/util/LeftRight.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/flat_hash_map.h>

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
  void emplace(Key_&& key, void* value) {
    bool res = map_.write([&](ska::flat_hash_map<Key, void*>& map) -> bool {
      auto result = map.emplace(std::forward<Key>(key), value);
      return result.second;
    });
    if (!res) {
      std::ostringstream msg;
      msg << "Tried to register conflicting kernels to the dispatcher: " << key;
      throw std::logic_error(msg.str());
    }
  }

  void erase(const Key& key) {
    auto num_removed =
        map_.write([&](ska::flat_hash_map<Key, void*>& map) -> size_t {
          return map.erase(key);
        });
    assert(num_removed <= 1); // This is not a multi-map
    if (num_removed == 0) {
      throw std::logic_error(
          "Tried to deregister a kernel that isn't registered.");
    }
  }

  void* lookup(const Key& key) const {
    return map_.read([&](const ska::flat_hash_map<Key, void*>& map) -> void* {
      auto found = map.find(key);
      if (found != map.end()) {
        return found->second;
      } else {
        return nullptr;
      }
    });
  }

 private:
  LeftRight<ska::flat_hash_map<Key, void*>> map_;
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
      typename Schema::signature::func_type* func,
      typename Schema::dispatch::dispatch_key_type dispatch_key) {
    kernels_.emplace(std::move(dispatch_key), reinterpret_cast<void*>(func));
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
   * Perform a dynamic dispatch on this table.
   *
   * @tparam Args Perfect forwarding template arguments to the dispatch
   * @param args Arguments to invoke the function with
   * @return Returned value of the operator
   */
  template <class... Args>
  typename Schema::signature::return_type call(Args&&... args) const {
    // TODO Better error message, but need to take care that reference arguments
    // match non-reference arguments and so on.
    //      static_assert(std::is_same<typename Schema::return_type (Args...),
    //      typename Schema::func_type>::value, "Argument types don't match
    //      operator signature");
    auto kernel_func = lookupKernelFunc_(args...);
    return kernel_func(std::forward<Args>(args)...);
  }

 private:
  template <class... Args>
  typename Schema::signature::func_type* lookupKernelFunc_(
      const Args&... args) const {
    auto dispatch_key = Schema::dispatch::dispatch_key(args...);
    void* found = kernels_.lookup(dispatch_key);
    if (found == nullptr) {
      // TODO Better error message - include op name and dispatch key (i.e.
      // argument types)
      throw std::logic_error(
          std::string() + "Didn't find kernel to dispatch to for operator '" +
          Schema::metadata::name() + "'");
    }
    return reinterpret_cast<typename Schema::signature::func_type*>(found);
  }

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
c10::DispatchTable<OpSchemaDef>& c10_dispatch_table();
