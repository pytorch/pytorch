#pragma once

#include <cstdint>
#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <c10/core/TensorTypeSet.h>

namespace c10 {

/**
 * An instance of DispatchKeyExtractor knows how to get a dispatch key given
 * a list of arguments for an operator call. The instance is specific for
 * a certain operator as different operators have different ways to extract
 * the dispatch key (e.g. different numbers of arguments).
 */
struct DispatchKeyExtractor final {
public:
  static DispatchKeyExtractor make(const FunctionSchema& schema) {
    bool is_valid = false;
    for (size_t i = 0; i < schema.arguments().size(); ++i) {
      const auto& type = schema.arguments()[i].type();
      if (type->isSubtypeOf(TensorType::get())) {
        is_valid = true;
        break;
      }
      if (type->isSubtypeOf(ListType::ofTensors())) {
        is_valid = true;
        break;
      }
    }

    return DispatchKeyExtractor(schema.arguments().size(), is_valid);
  }

  c10::optional<TensorTypeId> getDispatchKeyBoxed(const Stack* stack) const {
    if (C10_UNLIKELY(!is_valid_)) {
      return c10::nullopt;
    }

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
    // TODO: Don't use legacy extractor; blocked on c10 understanding variable
    return c10::legacyExtractTypeId(ts);
  }

  template<class... Args>
  c10::optional<TensorTypeId> getDispatchKeyUnboxed(const Args&... args) const {
    // nb: assertion not enabled because this is on the hot inlined path
    // AT_ASSERT(sizeof...(args) == num_args_, "Wrong number of arguments. Schema says ", num_args_, " but was called with ", sizeof...(args));
    if (C10_UNLIKELY(!is_valid_)) {
      return c10::nullopt;
    }

    return at::impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(args...));
  }

private:
  explicit DispatchKeyExtractor(size_t num_args, bool is_valid)
  : num_args_(num_args), is_valid_(is_valid) {}

  // this is caching the index so we don't have to parse the schema inputs
  // again and again for each dispatcher lookup.
  // num_args_ is allowed to be zero; that just means you must do the
  // fallthrough
  // TODO: a potential optimization is to store a bitfield of arg locations,
  size_t num_args_;

  // An invalid dispatch extractor means we can't dispatch any kernels.
  // You're able to create a dispatch table with an invalid dispatch strategy,
  // but adding kernels to it will fail.
  // This is used to allow creating operators without Tensor arguments
  // as long as they only have fallback kernels and no dispatched kernels.
  bool is_valid_;
};

}
