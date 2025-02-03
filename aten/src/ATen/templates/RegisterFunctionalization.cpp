#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/EmptyTensor.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/FunctionalInverses.h>
#include <ATen/MemoryOverlap.h>
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
        c10::DispatchKey::PreDispatch,

    });

// Helper around at::has_internal_overlap.
// The ATen util is used in hot-path eager mode: it's always fast,
// but might return TOO_HARD sometimes.
// During functionalization, we're ok taking a bit longer
// to detect memory overlap.
inline bool has_internal_overlap_helper(const at::Tensor t) {
  auto has_overlap = at::has_internal_overlap(t);
  if (has_overlap == at::MemOverlap::Yes) return true;
  if (has_overlap == at::MemOverlap::No) return false;
  return false;
}


inline Tensor to_meta(const Tensor& t) {
    if (!t.defined()) return t;
    return at::native::empty_strided_meta_symint(t.sym_sizes(), t.sym_strides(),
/*dtype=*/t.scalar_type(), /*layout=*/t.layout(),
/*device=*/c10::Device(kMeta), /*pin_memory=*/std::nullopt);
}

inline std::optional<Tensor> to_meta(const std::optional<Tensor>& t) {
  if (t.has_value()) {
    return to_meta(*t);
  }
  return std::nullopt;
}

inline std::vector<Tensor> to_meta(at::ITensorListRef t_list) {
  std::vector<Tensor> outputs;
  outputs.reserve(t_list.size());
  for (const auto& tensor : t_list) {
    outputs.push_back(to_meta(tensor));
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

inline c10::List<::std::optional<Tensor>> to_meta(const c10::List<::std::optional<Tensor>>& t_list) {
  c10::List<::std::optional<Tensor>> outputs;
  outputs.reserve(t_list.size());
  for (const auto i : c10::irange(t_list.size())) {
    outputs.push_back(to_meta(t_list[i]));
  }
  return outputs;
}


${func_definitions}

}  // namespace functionalization

namespace {

TORCH_LIBRARY_IMPL(aten, Functionalize, m) {
  ${func_registrations};
}

}  // namespace

} // namespace at
