#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/FunctionalInverses.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#include <ATen/NativeFunctions.h>
#else
// needed for the meta tensor calls to get stride info in functionalization
#include <ATen/ops/empty_strided_native.h>
// needed for special handling of copy_().
// See Note [functionalizating copy_() and not preserving strides]
#include <ATen/ops/to_ops.h>
#include <ATen/ops/expand_copy_ops.h>

$ops_headers
#endif

namespace at {
namespace functionalization {

// This keyset is used by functionalization when it calls into meta kernels
// to accurately propagate stride metadata.
// Exclude any modes: the purpose of calling into meta kernels is only as an implementation
// detail to perform shape inference, and we don't want any modal keys to run.
// Specifically, we want to prevent functionalization and Python modes from running.
constexpr auto exclude_keys_for_meta_dispatch =
    c10::functorch_transforms_ks |
    c10::DispatchKeySet({
        c10::DispatchKey::FuncTorchDynamicLayerBackMode,
        c10::DispatchKey::FuncTorchDynamicLayerFrontMode,
        c10::DispatchKey::Python,
        // The idea here is that this exclude set is used in conjunction
        // with adding MetaBit to the TLS include keyset.
        // That means that we will *always* dispatch to meta kernels,
        // and so we want to skip BackendSelect entirely (which will do the wrong thing).
        c10::DispatchKey::BackendSelect
    });


inline Tensor to_meta(const Tensor& t) {
    return at::native::empty_strided_symint_meta(t.sym_sizes(), t.sym_strides(),
/*dtype=*/c10::make_optional(t.scalar_type()), /*layout=*/c10::make_optional(t.layout()),
/*device=*/c10::make_optional(c10::Device(kMeta)), /*pin_memory=*/c10::nullopt);
}

inline c10::optional<Tensor> to_meta(const c10::optional<Tensor>& t) {
  if (t.has_value()) {
    return c10::make_optional<Tensor>(to_meta(*t));
  }
  return c10::nullopt;
}

inline std::vector<Tensor> to_meta(const TensorList& t_list) {
  std::vector<Tensor> outputs(t_list.size());
  for (const auto i : c10::irange(t_list.size())) {
    outputs[i] = to_meta(t_list[i]);
  }
  return outputs;
}

inline c10::List<Tensor> to_meta(const c10::List<Tensor>& t_list) {
  c10::List<Tensor> outputs;
  outputs.reserve(t_list.size());
  for (const auto i : c10::irange(t_list.size())) {
    outputs.push_back(to_meta(t_list[i]));
  }
  return outputs;
}

inline c10::List<c10::optional<Tensor>> to_meta(const c10::List<c10::optional<Tensor>>& t_list) {
  c10::List<c10::optional<Tensor>> outputs;
  outputs.reserve(t_list.size());
  for (const auto i : c10::irange(t_list.size())) {
    outputs.push_back(to_meta(t_list[i]));
  }
  return outputs;
}

inline c10::DispatchKeySet getKeySet(const Tensor& t) {
    return t.key_set();
}

inline c10::DispatchKeySet getKeySet(const c10::optional<Tensor>& t) {
  if (t.has_value()) {
    return getKeySet(*t);
  }
  return c10::DispatchKeySet();
}

inline c10::DispatchKeySet getKeySet(const TensorList& t_list) {
  auto out = DispatchKeySet();
  for (const auto i : c10::irange(t_list.size())) {
    out = out | getKeySet(t_list[i]);
  }
  return out;
}

inline c10::DispatchKeySet getKeySet(const c10::List<Tensor>& t_list) {
  auto out = DispatchKeySet();
  for (const auto i : c10::irange(t_list.size())) {
    out = out | getKeySet(t_list[i]);
  }
  return out;
}

inline c10::DispatchKeySet getKeySet(const c10::List<c10::optional<Tensor>>& t_list) {
  auto out = DispatchKeySet();
  for (const auto i : c10::irange(t_list.size())) {
    out = out | getKeySet(t_list[i]);
  }
  return out;
}


${func_definitions}

}  // namespace functionalization

namespace {

TORCH_LIBRARY_IMPL(aten, Functionalize, m) {
  ${func_registrations};
}

}  // namespace

} // namespace at
