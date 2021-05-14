#pragma once

#include <cstdint>
#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <c10/util/Bitset.h>
#include <c10/core/DispatchKeySet.h>
#include <ATen/core/Variadic.h>
#include <ATen/core/stack.h>

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
static inline DispatchKeySet computeDispatchKeySet(
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
    DispatchKeySet key_mask
) {
  c10::impl::LocalDispatchKeySet local = c10::impl::tls_local_dispatch_key_set();
  // TODO: It's a bit irritating that we have to do logical ORs here, it would
  // be nice to only do one.  Can always_included be folded into the TLS?  Well,
  // it's a bit troublesome, because fastpath TLS access requires the type of
  // the TLS in question to be zero-initialized, so you don't actually win
  // anyting in that case.
  return (((ks | local.included_) - local.excluded_) & key_mask);
}

}

namespace detail {
  // A small gadget to extract the DispatchKeySet from types which are known
  // to have it.  Used to extract dispatch keys from unboxed calls.
  struct MultiDispatchKeySet : at::IterArgs<MultiDispatchKeySet> {
    DispatchKeySet ts;
    void operator()(const at::Tensor& x) {
      ts = ts | x.key_set();
    }
    void operator()(c10::optional<at::Tensor> x) {
      if (x.has_value()) {
        ts = ts | x->key_set();
      }
    }
    void operator()(at::ArrayRef<at::Tensor> xs) {
      for (const auto& x : xs) {
        ts = ts | x.key_set();
      }
    }
    void operator()(at::Generator gen) {
      if (gen.defined()) {
        ts = ts | gen.key_set();
      }
    }
    void operator()(c10::optional<at::Generator> gen) {
      if (gen.has_value() && gen->defined()) {
        ts = ts | gen->key_set();
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
    dispatch_arg_indices_reverse_.for_each_set_bit([&] (size_t reverse_arg_index) {
      const auto& ivalue = torch::jit::peek(*stack, 0, reverse_arg_index + 1);
      if (C10_LIKELY(ivalue.isTensor())) {
        // NB: Take care not to introduce a refcount bump (there's
        // no safe toTensorRef method, alas)
        ks = ks | ivalue.unsafeToTensorImpl()->key_set();
      } else if (C10_UNLIKELY(ivalue.isTensorList())) {
        for (const at::Tensor tensor : ivalue.toTensorList()) {
          ks = ks | tensor.key_set();
        }
      }
    });
    // Keys that are fallthrough should be skipped
    return impl::computeDispatchKeySet(ks, nonFallthroughKeys_);
  }

  template<class... Args>
  DispatchKeySet getDispatchKeySetUnboxed(DispatchKeySet eligibleKeys, const Args&... args) const {
    auto ks = detail::multi_dispatch_key_set(args...);
    // Keys that are fallthrough should be skipped
    return impl::computeDispatchKeySet(ks, nonFallthroughKeys_ & eligibleKeys);
  }

  void setOperatorHasFallthroughForKey(DispatchKey k, bool has_fallthrough);

  std::string dumpState() const;
  void checkInvariants(const FunctionSchema& schema) const;

private:
  static c10::utils::bitset makeBitsetForDispatchArgs(const FunctionSchema& schema) {
    TORCH_CHECK(schema.arguments().size() <= c10::utils::bitset::NUM_BITS(),
        "The function schema has ", schema.arguments().size(),
        " arguments but this PyTorch build only supports ", c10::utils::bitset::NUM_BITS());
    c10::utils::bitset dispatch_arg_indices_reverse;
    for (size_t index = 0; index < schema.arguments().size(); ++index) {
      if (schema.arguments()[index].type()->isSubtypeOf(TensorType::get()) ||
          schema.arguments()[index].type()->isSubtypeOf(
              ListType::ofTensors()) ||
          schema.arguments()[index].type()->isSubtypeOf(
              OptionalType::ofTensor())) {
        dispatch_arg_indices_reverse.set(schema.arguments().size() - 1 - index);
      }
    }
    return dispatch_arg_indices_reverse;
  }

  explicit DispatchKeyExtractor(c10::utils::bitset dispatch_arg_indices_reverse)
  : dispatch_arg_indices_reverse_(dispatch_arg_indices_reverse)
  , nonFallthroughKeys_(DispatchKeySet::FULL) {}

  // this is a bitset that has ones for each argument index which has to be
  // considered for dispatch. This avoids having to iterate over the stack
  // to find all the tensors. The bits are stored in reverse order, i.e.
  // dispatch_arg_indices_reverse_[i] == true, then the i-th argument from
  // the top of the stack (i.e. the i-th last argument of the function)
  // is relevant for dispatch.
  // dispatch_arg_indices_reverse_ is allowed to have zero bits set; that just means you must do the
  // fallthrough
  c10::utils::bitset dispatch_arg_indices_reverse_;

  // Set of keys for which the operator does NOT have fallthrough kernel.
  DispatchKeySet nonFallthroughKeys_;
};

}
