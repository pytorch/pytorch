#pragma once

#include <cstdint>
#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <c10/core/DispatchKeySet.h>
#include <ATen/core/Variadic.h>

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
static inline DispatchKey dispatchTypeId(DispatchKeySet ts) {
  c10::impl::LocalDispatchKeySet local = c10::impl::tls_local_dispatch_key_set();
  return ((ts | local.included_) - local.excluded_).highestPriorityTypeId();
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
    void operator()(at::Generator* gen) {
      if (gen != nullptr) {
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
 * a list of arguments for an operator call. The instance is specific for
 * a certain operator as different operators have different ways to extract
 * the dispatch key (e.g. different numbers of arguments).
 */
struct DispatchKeyExtractor final {
public:
  static DispatchKeyExtractor make(const FunctionSchema& schema) {
    return DispatchKeyExtractor(schema.arguments().size());
  }

  c10::optional<DispatchKey> getDispatchKeyBoxed(const Stack* stack) const {
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
    return dispatchKeySetToDispatchKey_(ts);
  }

  template<class... Args>
  c10::optional<DispatchKey> getDispatchKeyUnboxed(const Args&... args) const {
    auto key_set = detail::multi_dispatch_key_set(args...);
    return dispatchKeySetToDispatchKey_(key_set);
  }

private:
  static c10::optional<DispatchKey> dispatchKeySetToDispatchKey_(const DispatchKeySet& keySet) {
    if (C10_UNLIKELY(keySet.empty())) {
      return c10::nullopt;
    }

    return impl::dispatchTypeId(keySet);
  }

  explicit DispatchKeyExtractor(size_t num_args)
  : num_args_(num_args) {}

  // this is caching the index so we don't have to parse the schema inputs
  // again and again for each dispatcher lookup.
  // num_args_ is allowed to be zero; that just means you must do the
  // fallthrough
  // TODO: a potential optimization is to store a bitfield of arg locations,
  size_t num_args_;
};

}
