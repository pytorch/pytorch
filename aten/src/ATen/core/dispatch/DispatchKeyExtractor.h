#pragma once

#include <cstdint>
#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <c10/core/TensorTypeSet.h>
#include <ATen/core/Variadic.h>

namespace c10 {

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
  struct MultiDispatchTensorTypeSet : at::IterArgs<MultiDispatchTensorTypeSet> {
    TensorTypeSet ts;
    void operator()(const at::Tensor& x) {
      ts = ts | x.type_set();
    }
    void operator()(const TensorOptions& x) {
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

  c10::optional<TensorTypeId> getDispatchKeyBoxed(const Stack* stack) const {
    // TODO Unboxed dispatch supports TensorOptions (i.e. ScalarType/Device/Layout) arguments
    //      but boxed doesn't yet. These should be aligned and do the same thing.

    TensorTypeSet ts;
    for (const auto& ivalue : torch::jit::last(*stack, num_args_)) {
      if (C10_LIKELY(ivalue.isTensor())) {
        // NB: Take care not to introduce a refcount bump (there's
        // no safe toTensorRef method, alas)
        ts = ts | ivalue.unsafeToTensorImpl()->type_set();
      } else if (C10_UNLIKELY(ivalue.isTensorList())) {
        for (const auto& tensor : ivalue.toTensorListRef()) {
          ts = ts | tensor.type_set();
        }
      }
    }
    if (C10_UNLIKELY(ts.empty())) {
      return c10::nullopt;
    }

    // TODO: Don't use legacy extractor; blocked on c10 understanding variable
    return c10::legacyExtractTypeId(ts);
  }

  template<class... Args>
  c10::optional<TensorTypeId> getDispatchKeyUnboxed(const Args&... args) const {
    auto type_set = detail::multi_dispatch_tensor_type_set(args...);

    if (C10_UNLIKELY(type_set.empty())) {
      return c10::nullopt;
    }

    return impl::dispatchTypeId(type_set);
  }

private:
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
