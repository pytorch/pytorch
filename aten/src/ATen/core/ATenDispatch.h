#pragma once

#include <c10/core/TensorTypeSet.h>
#include <c10/core/Backend.h>
#include <c10/core/impl/LocalTensorTypeSet.h>
#include <unordered_map>
#include <unordered_set>
#include <ATen/core/OpsAlreadyMovedToC10.h>
#include <ATen/core/Variadic.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/EnableNamedTensor.h>
#include <c10/util/C++17.h>
#include <memory>
#include <mutex>
#include <ATen/core/interned_strings.h>
#include <ATen/core/stack.h>
#include <torch/csrc/jit/script/function_schema_parser.h>

// TODO: Rewrite this comment
//
// This dispatch class serves as a replacement for our previous dispatch
// mechanism, in which all functions were members of a Type class. A derived
// class existed for each backend (and Variable), and the vtable was used to
// dispatch to the correct implementation. This class is to be replaced by
// the c10 dispatcher when it supports all argument and return types.
// This implementation opts to store implementations in a table of void*.

namespace at {

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
  struct MultiDispatchTensorTypeSet : IterArgs<MultiDispatchTensorTypeSet> {
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

} // namespace at
