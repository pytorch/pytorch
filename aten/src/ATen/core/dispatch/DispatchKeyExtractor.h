#pragma once

#include <ATen/core/Variadic.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/Bitset.h>
#include <c10/util/irange.h>
#include <cstdint>

namespace c10 {

namespace impl {

// Take a DispatchKeySet for a Tensor and determine what the actual dispatch
// DispatchKey should be, taking into account TLS, and skipping backends which
// fall through.
//
// Unlike Tensor::key_set(), the value of this on a tensor can change depending
// on TLS.
//
// NB: If there is no valid dispatch key, this will return Undefined
inline DispatchKeySet computeDispatchKeySet(
    DispatchKeySet ks,
    // The key mask lets us eliminate (by zero entries) keys which should not
    // be considered for dispatch.  There are two cases when we use this:
    //
    // - If an operator's dispatch table contains a fallthrough entry, we
    //   should bypass it entirely when finding the key
    // - If a user invokes with redispatch, the mask lets us
    //   zero out the key the user asked us to stop.
    //
    // These excluded backends are NOT tracked in the TLS, but must be applied
    // AFTER TLS (since the backend may have been introduced for consideration
    // by the included TLS), which is why you have to pass them in to this
    // function (as opposed to just applying it to the input 'ks').
    DispatchKeySet key_mask) {
  c10::impl::LocalDispatchKeySet local =
      c10::impl::tls_local_dispatch_key_set();
  // TODO: It's a bit irritating that we have to do logical ORs here, it would
  // be nice to only do one.  Can always_included be folded into the TLS?  Well,
  // it's a bit troublesome, because fastpath TLS access requires the type of
  // the TLS in question to be zero-initialized, so you don't actually win
  // anything in that case.
  return (((ks | local.included_) - local.excluded_) & key_mask);
}

} // namespace impl

namespace detail {
// A small gadget to extract the DispatchKeySet from types which are known
// to have it.  Used to extract dispatch keys from unboxed calls.
struct MultiDispatchKeySet : at::IterArgs<MultiDispatchKeySet> {
  DispatchKeySet ts;
  void operator()(const at::Tensor& x) {
    ts = ts | x.key_set();
  }
  void operator()(const std::optional<at::Tensor>& x) {
    if (x.has_value()) {
      ts = ts | x->key_set();
    }
  }
  void operator()(at::ArrayRef<at::Tensor> xs) {
    for (const auto& x : xs) {
      ts = ts | x.key_set();
    }
  }
  // Tensor?[] translates to this case.
  void operator()(const c10::List<std::optional<at::Tensor>>& xs) {
    for (std::optional<at::Tensor> x : xs) {
      if (x.has_value()) {
        ts = ts | x.value().key_set();
      }
    }
  }
  // Structured Tensor[] translates to this case
  void operator()(const at::ITensorListRef& xs) {
    for (const auto& x : xs) {
      ts = ts | x.key_set();
    }
  }
  [[noreturn]] void operator()(at::ArrayRef<std::optional<at::Tensor>>) {
    // Just checking that the handling of Tensor?[] didn't change.
    TORCH_INTERNAL_ASSERT(false);
  }
  void operator()(const at::Generator& gen) {
    if (gen.defined()) {
      ts = ts | gen.key_set();
    }
  }
  void operator()(const std::optional<at::Generator>& gen) {
    if (gen.has_value() && gen->defined()) {
      ts = ts | gen->key_set();
    }
  }
  template <typename T>
  void operator()(const T&) {
    // do nothing
  }
};

// NB: take by const reference (Don't do universal forwarding here! You
// don't want to move into this function!)
template <typename... Args>
DispatchKeySet multi_dispatch_key_set(const Args&... args) {
  return MultiDispatchKeySet().apply(args...).ts;
}
} // namespace detail

/**
 * An instance of DispatchKeyExtractor knows how to get a dispatch key given
 * a list of arguments for an operator call.
 *
 * The instance is specific for a certain operator as:
 *  - In boxed dispatch, different operators have different ways to extract
 *    the dispatch key (e.g. different numbers of arguments), and we precompute
 *    the stack locations we should look at; and
 *  - In all dispatch, some backends should be excluded from dispatch because
 *    they have been registered as fallthrough.  The set of excluded backends
 *    varies from operator, as some operators may have overridden the
 *    fallthrough with custom behavior.
 *
 *   Note - this should maintain identical impl to the py dispatcher key
 * extraction logic at pytorch/torch/dispatcher.py
 */
struct TORCH_API DispatchKeyExtractor final {
 public:
  static DispatchKeyExtractor make(const FunctionSchema& schema) {
    return DispatchKeyExtractor(makeBitsetForDispatchArgs(schema));
  }

  static DispatchKeyExtractor makeUninitialized() {
    return DispatchKeyExtractor(c10::utils::bitset());
  }

  void registerSchema(const FunctionSchema& schema) {
    TORCH_INTERNAL_ASSERT(dispatch_arg_indices_reverse_.is_entirely_unset());
    dispatch_arg_indices_reverse_ = makeBitsetForDispatchArgs(schema);
  }
  void deregisterSchema() {
    dispatch_arg_indices_reverse_ = c10::utils::bitset();
  }

  DispatchKeySet getDispatchKeySetBoxed(const torch::jit::Stack* stack) const {
    DispatchKeySet ks;
    dispatch_arg_indices_reverse_.for_each_set_bit([&](size_t
                                                           reverse_arg_index) {
      const auto& ivalue = torch::jit::peek(*stack, 0, reverse_arg_index + 1);
      if (C10_LIKELY(ivalue.isTensor())) {
        // NB: Take care not to introduce a refcount bump (there's
        // no safe toTensorRef method, alas)
        ks = ks | ivalue.unsafeToTensorImpl()->key_set();
      } else if (C10_UNLIKELY(ivalue.isTensorList())) {
        // NB: use toListRef as it doesn't induce refcount bumps
        // (toTensorListRef is not a thing)
        for (const auto& nv : ivalue.toListRef()) {
          auto* tensor = nv.unsafeToTensorImpl();
          ks = ks | tensor->key_set();
        }
      }
      // Tensor?[] translates to a c10::List<IValue> so we need to peek inside
      else if (C10_UNLIKELY(ivalue.isList())) {
        for (const auto& elt : ivalue.toListRef()) {
          if (elt.isTensor()) {
            ks = ks | elt.toTensor().key_set();
          }
        }
      }
    });
    // Keys that are fallthrough should be skipped
    if (requiresBitsetPerBackend_) {
      c10::impl::LocalDispatchKeySet tls =
          c10::impl::tls_local_dispatch_key_set();
      auto backend_idx =
          ((ks | tls.included_) - tls.excluded_).getBackendIndex();
      return impl::computeDispatchKeySet(
          ks, nonFallthroughKeysPerBackend_[backend_idx]);
    } else {
      return impl::computeDispatchKeySet(ks, nonFallthroughKeys_);
    }
  }

  template <class... Args>
  DispatchKeySet getDispatchKeySetUnboxed(const Args&... args) const {
    auto ks = detail::multi_dispatch_key_set(args...);
    // Keys that are fallthrough should be skipped
    if (requiresBitsetPerBackend_) {
      c10::impl::LocalDispatchKeySet tls =
          c10::impl::tls_local_dispatch_key_set();
      auto backend_idx =
          ((ks | tls.included_) - tls.excluded_).getBackendIndex();
      return impl::computeDispatchKeySet(
          ks, nonFallthroughKeysPerBackend_[backend_idx]);
    } else {
      return impl::computeDispatchKeySet(ks, nonFallthroughKeys_);
    }
  }

  void setOperatorHasFallthroughForKey(DispatchKey k, bool has_fallthrough);

  std::string dumpState() const;
  void checkInvariants(const FunctionSchema& schema) const;

 private:
  static bool isDispatchType(const Type& type) {
    // Checking isSubtypeOf on a DynamicType heap-allocates a
    // DynamicType version of the argument if it's not a DynamicType
    // already, and this has measurable overhead during startup.
#ifdef C10_MOBILE
    struct CachedTypes {
      DynamicTypePtr listOfTensors;
      DynamicTypePtr listOfOptionalTensors;
      DynamicTypePtr optionalOfTensor;
    };
    static const CachedTypes ct = {
        DynamicType::create(*ListType::ofTensors()),
        DynamicType::create(*ListType::ofOptionalTensors()),
        DynamicType::create(*OptionalType::ofTensor())};
    return type.isSubtypeOf(c10::TypeFactory::get<TensorType>()) ||
        type.isSubtypeOf(ct.listOfTensors) ||
        type.isSubtypeOf(ct.listOfOptionalTensors) ||
        type.isSubtypeOf(ct.optionalOfTensor);
#else // C10_MOBILE
    return type.isSubtypeOf(*TensorType::get()) ||
        type.isSubtypeOf(*ListType::ofTensors()) ||
        type.isSubtypeOf(*ListType::ofOptionalTensors()) ||
        type.isSubtypeOf(*OptionalType::ofTensor());
#endif // C10_MOBILE
  }
  static c10::utils::bitset makeBitsetForDispatchArgs(
      const FunctionSchema& schema) {
    TORCH_CHECK(
        schema.arguments().size() <= c10::utils::bitset::NUM_BITS(),
        "The function schema has ",
        schema.arguments().size(),
        " arguments but this PyTorch build only supports ",
        c10::utils::bitset::NUM_BITS());
    c10::utils::bitset dispatch_arg_indices_reverse;
    for (const auto index : c10::irange(schema.arguments().size())) {
      if (isDispatchType(*schema.arguments()[index].type())) {
        dispatch_arg_indices_reverse.set(schema.arguments().size() - 1 - index);
      }
    }
    return dispatch_arg_indices_reverse;
  }

  explicit DispatchKeyExtractor(c10::utils::bitset dispatch_arg_indices_reverse)
      : dispatch_arg_indices_reverse_(dispatch_arg_indices_reverse),
        nonFallthroughKeys_(DispatchKeySet::FULL) {
    for (const auto i : c10::irange(nonFallthroughKeysPerBackend_.size())) {
      nonFallthroughKeysPerBackend_[i] = DispatchKeySet::FULL;
    }
  }

  // this is a bitset that has ones for each argument index which has to be
  // considered for dispatch. This avoids having to iterate over the stack
  // to find all the tensors. The bits are stored in reverse order, i.e.
  // dispatch_arg_indices_reverse_[i] == true, then the i-th argument from
  // the top of the stack (i.e. the i-th last argument of the function)
  // is relevant for dispatch.
  // dispatch_arg_indices_reverse_ is allowed to have zero bits set; that just
  // means you must do the fallthrough
  c10::utils::bitset dispatch_arg_indices_reverse_;

  // Set of functionality keys for which the operator does NOT have fallthrough
  // kernel.
  DispatchKeySet nonFallthroughKeys_;
  // Set of functionality keys for which the operator does NOT have fallthrough
  // kernel, defined PER BACKEND. This is only needed if we know that the
  // operator has a different set of fallthroughs defined for some backends.
  std::array<DispatchKeySet, num_backends> nonFallthroughKeysPerBackend_;
  // Flag to tell us if we can use the single set of nonFallthroughKeys_ (fast
  // path), or if we need to fall back to the slower path and check
  // nonFallthroughKeysPerBackend_
  bool requiresBitsetPerBackend_{false};
};

} // namespace c10
