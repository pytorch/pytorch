#pragma once

#include <c10/core/TensorTypeSet.h>
#include <c10/core/Backend.h>
#include <c10/core/impl/LocalTensorTypeSet.h>
#include <unordered_map>
#include <unordered_set>
#include <ATen/core/OpsAlreadyMovedToC10.h>
#include <ATen/core/Variadic.h>
#include <ATen/core/TensorBody.h>
#include <c10/util/C++17.h>
#include <memory>
#include <mutex>
#include <ATen/core/interned_strings.h>
#include <ATen/core/stack.h>

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
  return (ts - c10::impl::tls_excluded_tensor_type_set()).highestPriorityTypeId();
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

using FallbackBoxedFunction = void(const char* schema, torch::jit::Stack*);

// Assume T is decayed
template <typename T>
using not_ok_to_box =
  c10::guts::disjunction<
    c10::guts::negation<
      c10::guts::disjunction<
        std::is_constructible<IValue, T>,
        // TensorOptions are not directly constructible into IValue,
        // but torch::jit::push knows how to handle them
        std::is_same<TensorOptions, T>
      >>,
    // some constructors are templated (and therefore pass
    // is_constructible), but do not actually work with all
    // template arguments, so we must blacklist them explicitly
    // TODO: The correct fix is to sfinae based on is_constructible of T
    std::is_same<optional<ArrayRef<Dimname>>, T>
  >;

template <class Result, class... Args>
using supports_boxed_fallback =
  c10::guts::negation<c10::guts::disjunction<
    std::is_lvalue_reference<Result>,
    not_ok_to_box<Result>,
    std::is_same<IntArrayRef, Result>,
    not_ok_to_box<guts::decay_t<Args>>...
  >>;

// ATenOpTable stores the implementations for each backend, in addition to
// an implementation for variables.
class CAFFE2_API ATenOpTable {
 public:
  ATenOpTable(std::string schema)
    : schema_(std::move(schema)) {}

  // NB: No universal forwarding
  template<class Result, class... Args>
  Result callUnboxed(Args... args) const;

 private:

  void registerOp(TensorTypeId tid, void* fn) {
    TORCH_CHECK(function_table_[static_cast<int64_t>(tid)] == nullptr,
        "Attempting to register function for schema ", schema_,
        " and tensor type ", toString(tid),
        " but there is already a function registered");
    function_table_[static_cast<int64_t>(tid)] = fn;
  }

  C10_NORETURN void reportError(TensorTypeId tid) const;

  friend class ATenDispatch;

  std::string schema_;
  void* function_table_[static_cast<int64_t>(TensorTypeId::NumTensorIds)] = {nullptr};
};

class CAFFE2_API ATenDispatch {
 public:
  template<class FuncType>
  ATenDispatch& registerOp(TensorTypeId id, const char* schema, FuncType* fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (op_tables_.find(schema) == op_tables_.end()) {
      op_tables_.insert(std::make_pair(schema, ATenOpTable(schema)));
    }
    op_tables_.at(schema).registerOp(id, reinterpret_cast<void*>(fn));
    return *this;
  }

  ATenDispatch& registerFallbackBoxedOp(TensorTypeId id, FallbackBoxedFunction* fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    boxed_fallback_table_[static_cast<size_t>(id)] = fn;
    return *this;
  }

  const ATenOpTable* getOpTable(const char* schema) const {
    auto iter = op_tables_.find(schema);
    TORCH_CHECK(iter != op_tables_.end(),
        "No functions are registered for schema ", schema);
    return &iter->second;
  }

  const FallbackBoxedFunction* getFallbackBoxedOp(TensorTypeId tid) const {
    return boxed_fallback_table_[static_cast<size_t>(tid)];
  }

 private:
  std::unordered_map<std::string, ATenOpTable> op_tables_;
  FallbackBoxedFunction* boxed_fallback_table_[static_cast<int64_t>(TensorTypeId::NumTensorIds)] = {nullptr};
  std::mutex mutex_;
};

CAFFE2_API ATenDispatch& globalATenDispatch();

template<
  class Result, class... Args,
  typename c10::guts::enable_if_t<
    !supports_boxed_fallback<Result, Args...>::value,
    std::nullptr_t
  > = nullptr>
Result callBoxedFallback(const char* schema, const FallbackBoxedFunction* boxed_fallback_fn, Args&&... args) {
  // This is dead because we test the SFINAE condition before calling
  // boxed_fallback_fn.  A more functional way of writing this with
  // optional<Result> return value works poorly when void is involved.
  TORCH_INTERNAL_ASSERT(0);
}

template<
  class Result, class... Args,
  typename c10::guts::enable_if_t<
    supports_boxed_fallback<Result, Args...>::value,
    std::nullptr_t
  > = nullptr>
Result callBoxedFallback(const char* schema, const FallbackBoxedFunction* boxed_fallback_fn, Args&&... args) {
  torch::jit::Stack stack;
  torch::jit::push(stack, std::forward<Args>(args)...);
  boxed_fallback_fn(schema, &stack);
  TORCH_INTERNAL_ASSERT(stack.size() == 1);
  return torch::jit::pop(stack).to<Result>();
}

// NB: No universal forwarding
template<class Result, class... Args>
Result ATenOpTable::callUnboxed(Args... args) const {
  using FuncType = Result(Args...);
  // NB: No universal forwarding (takes const& only)
  TensorTypeSet ts = detail::multi_dispatch_tensor_type_set(args...);
  TensorTypeId tid = impl::dispatchTypeId(ts);

  // You might think we can eliminate the second branch by maintaining a
  // bitmask of registered operator keys, so we don't select dispatch ids
  // which don't have implementations here.  But the net effect is that if you
  // get a Variable CPUTensor, if there is no variable registration, you'll
  // fall back to the CPU implementation.  Is this what you want?  Unlikely...

  auto* unboxed_fn = reinterpret_cast<FuncType*>(function_table_[static_cast<int64_t>(tid)]);
  if (C10_LIKELY(unboxed_fn != nullptr)) {
    return (*unboxed_fn)(std::forward<Args>(args)...);
  }

  // The supports_boxed_fallback condition test, and the SFINAE on
  // callBoxedFallback, do the same thing.  But we perform this (compile-time)
  // test twice so that we can take advantage of the fact that return
  // func_returns_void() is OK.  If we eliminated this condition in exchange
  // for having callBoxedFallback return an optional, we can't conveniently
  // handle the Result=void case anymore.
  //
  // (The SFINAE in callBoxedFallback, of course, is necessary to
  // prevent us from attempting to typecheck code that won't typecheck.)
  auto* boxed_fallback_fn = globalATenDispatch().getFallbackBoxedOp(tid);
  if (C10_UNLIKELY(boxed_fallback_fn)) {
    if (supports_boxed_fallback<Result, Args...>::value) {
      return callBoxedFallback<Result, Args...>(schema_.c_str(), boxed_fallback_fn, std::forward<Args>(args)...);
    } else {
      TORCH_INTERNAL_ASSERT(0, schema_, " does not support boxed fallback, but boxed fallback for ", tid, " was available");
    }
  }

  auto* unboxed_fallback_fn = reinterpret_cast<FuncType*>(function_table_[static_cast<int64_t>(TensorTypeId::UndefinedTensorId)]);
  if (C10_LIKELY(unboxed_fallback_fn != nullptr)) {
    return (*unboxed_fallback_fn)(std::forward<Args>(args)...);
  }

  reportError(tid);
  TORCH_INTERNAL_ASSERT(0);
}

} // namespace at
