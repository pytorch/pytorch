#pragma once

#include <cstdint>
#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <c10/core/DispatchKeySet.h>
#include <ATen/core/Variadic.h>
#include <ATen/core/stack.h>

namespace c10 {

namespace impl {

// Take a DispatchKeySet for a Tensor, and combine it with the current thread
// local valid (implemented) and enabled (not implemented) DispatchKeySets
// to determine what the actual dispatch DispatchKey should be.  Unlike
// Tensor::key_set(), the value of this on a tensor can change depending
// on TLS.
//
// NB: I didn't make this take a Tensor to avoid header include shenanigans.
//
// TODO: I'm not sure if this should live in this header or not; the operant
// question is whether or not we have access to all the relevant TLS at this
// point.
static inline DispatchKey dispatchTypeId(
    DispatchKeySet ts,
    // This argument is used to mask out fallthrough keys, so we don't
    // consider them for dispatch.  It's taken in the weird "inverted" form so
    // that we can do it in one instruction (x & y) rather than two (x & ~y).
    // These exclusions are NOT tracked in the TLS, but must be applied AFTER
    // TLS, which is why you have to pass them in to this function (as opposed
    // to just applying it to the input 'ts').
    DispatchKeySet non_fallthrough_mask
) {
  c10::impl::LocalDispatchKeySet local = c10::impl::tls_local_dispatch_key_set();
  return (((ts | local.included_) - local.excluded_) & non_fallthrough_mask).highestPriorityTypeId();
}

}

namespace detail {
  struct MultiDispatchKeySet : at::IterArgs<MultiDispatchKeySet> {
    DispatchKeySet ts;
    void operator()(const at::Tensor& x) {
      ts = ts | x.key_set();
    }
    void operator()(const TensorOptions& x) {
      ts = ts | x.key_set();
    }
    void operator()(at::ArrayRef<at::Tensor> xs) {
      for (const auto& x : xs) {
        ts = ts | x.key_set();
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
  DispatchKeySet multi_dispatch_key_set(const Args&... args) {
    return MultiDispatchKeySet().apply(args...).ts;
  }
}

/**
 * An instance of DispatchKeyExtractor knows how to get a dispatch key given
 * a list of arguments for an operator call. The instance is specific for
 * a certain operator as different operators have different ways to extract
 * the dispatch key (e.g. different numbers of arguments).
 */
struct CAFFE2_API DispatchKeyExtractor final {
public:
  static DispatchKeyExtractor make(const FunctionSchema& schema) {
    return DispatchKeyExtractor(schema.arguments().size());
  }

  c10::optional<DispatchKey> getDispatchKeyBoxed(DispatchKeySet nonFallthroughMask, const torch::jit::Stack* stack) const {
    // TODO Unboxed dispatch supports TensorOptions (i.e. ScalarType/Device/Layout) arguments
    //      but boxed doesn't yet. These should be aligned and do the same thing.

    DispatchKeySet ts;
    for (const auto& ivalue : torch::jit::last(*stack, num_args_)) {
      if (C10_LIKELY(ivalue.isTensor())) {
        // NB: Take care not to introduce a refcount bump (there's
        // no safe toTensorRef method, alas)
        ts = ts | ivalue.unsafeToTensorImpl()->key_set();
      } else if (C10_UNLIKELY(ivalue.isTensorList())) {
        for (const at::Tensor& tensor : ivalue.toTensorList()) {
          ts = ts | tensor.key_set();
        }
      }
    }
    return dispatchKeySetToDispatchKey_(nonFallthroughMask, ts);
  }

  template<class... Args>
  c10::optional<DispatchKey> getDispatchKeyUnboxed(DispatchKeySet nonFallthroughMask, const Args&... args) const {
    auto key_set = detail::multi_dispatch_key_set(args...);
    return dispatchKeySetToDispatchKey_(nonFallthroughMask, key_set);
  }

  // Used by DispatchTable to maintain the fallthrough invariant, see
  // docs on nonFallthroughKernels_
  void setIsOperatorOverridden(DispatchKey k, bool is_overridden);

private:
  c10::optional<DispatchKey> dispatchKeySetToDispatchKey_(DispatchKeySet nonFallthroughMask, const DispatchKeySet& keySet) const {
    if (C10_UNLIKELY(keySet.empty())) {
      return c10::nullopt;
    }

    return impl::dispatchTypeId(keySet, nonFallthroughMask | perOperatorOverriddenKernels_);
  }

  explicit DispatchKeyExtractor(size_t num_args)
  : num_args_(num_args)
  , perOperatorOverriddenKernels_() {}

  // this is caching the index so we don't have to parse the schema inputs
  // again and again for each dispatcher lookup.
  // num_args_ is allowed to be zero; that just means you must do the
  // fallthrough
  // TODO: a potential optimization is to store a bitfield of arg locations,
  size_t num_args_;

  // We must NOT respect the passed in nonFallthroughMask if an operator has
  // specifically overridden the backend, since that means we've opted to
  // not fallthrough and instead apply some specific behavior (which we
  // must dispatch to).  For now, we assume that operators NEVER are the
  // fallthrough kernel (see https://github.com/pytorch/pytorch/issues/32454)
  // which means we can just unconditionally fill in the mask when the
  // operator tells us to.
  //
  // This scheme doesn't work if you want to also apply fallthrough on a
  // per-op basis, but while we could fix this by maintaining a second
  // DispatchKeySet, it doesn't seem that there is any actual use case,
  // so we are deferring it for 32454.
  DispatchKeySet perOperatorOverriddenKernels_;
};

}
