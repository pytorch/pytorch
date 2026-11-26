#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/distributed/fsdp/CollectiveCopy.h>
#include <torch/library.h>

namespace torch::distributed::fsdp {

namespace {

// Bumps versions, as the generated kernels of ATen in-place ops do. Outputs
// cached by an inference-mode forward are inference tensors, which have no
// version counter.
void all_gather_copy_out_ad_inplace_or_view(
    c10::DispatchKeySet ks,
    at::TensorList out,
    const at::Tensor& input,
    c10::SymIntArrayRef split_sizes,
    c10::SymIntArrayRef outer_sizes,
    int64_t num_chunks) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("fsdp::_all_gather_copy_out_", "")
                       .typed<void(
                           at::TensorList,
                           const at::Tensor&,
                           c10::SymIntArrayRef,
                           c10::SymIntArrayRef,
                           int64_t)>();
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    op.redispatch(
        ks & c10::after_ADInplaceOrView_keyset,
        out,
        input,
        split_sizes,
        outer_sizes,
        num_chunks);
  }
  for (const auto& tensor : out) {
    if (!tensor.is_inference()) {
      torch::autograd::impl::bump_version(tensor);
    }
  }
}

} // namespace

void check_all_gather_copy_out_inputs(
    at::TensorList out,
    const at::Tensor& input,
    c10::SymIntArrayRef split_sizes,
    c10::SymIntArrayRef outer_sizes,
    int64_t num_chunks) {
  TORCH_CHECK_VALUE(
      num_chunks > 0, "expected positive num_chunks, but got ", num_chunks);
  TORCH_CHECK_VALUE(
      input.layout() == at::kStrided,
      "expected a strided input, but got ",
      input.layout());
  TORCH_CHECK_VALUE(
      split_sizes.size() == out.size() && outer_sizes.size() == out.size(),
      "expected one split size and outer size per output, but got ",
      out.size(),
      " outputs, ",
      split_sizes.size(),
      " split sizes and ",
      outer_sizes.size(),
      " outer sizes");
  c10::SymInt chunk_numel = 0;
  for (const auto i : c10::irange(out.size())) {
    const auto& split_size = split_sizes[i];
    const auto& outer_size = outer_sizes[i];
    TORCH_CHECK_VALUE(
        outer_size > 0 && split_size >= 0 && split_size % outer_size == 0,
        "expected output ",
        i,
        " to have a non-negative split size divisible by a positive "
        "outer size, but got split size ",
        split_size,
        " and outer size ",
        outer_size);
    TORCH_CHECK_VALUE(
        out[i].layout() == at::kStrided && out[i].is_contiguous(),
        "expected contiguous strided outputs, but output ",
        i,
        " is not");
    TORCH_CHECK_VALUE(
        out[i].device() == input.device(),
        "expected outputs on the input device ",
        input.device(),
        ", but output ",
        i,
        " is on ",
        out[i].device());
    TORCH_CHECK_TYPE(
        out[i].scalar_type() == input.scalar_type(),
        "expected outputs with the input dtype ",
        input.scalar_type(),
        ", but output ",
        i,
        " has dtype ",
        out[i].scalar_type());
    const auto& output_numel = out[i].sym_numel();
    TORCH_CHECK_VALUE(
        output_numel == split_size * num_chunks,
        "expected output ",
        i,
        " to hold split size * num_chunks = ",
        split_size * num_chunks,
        " input elements, but it holds ",
        output_numel);
    chunk_numel += split_size;
  }
  TORCH_CHECK_VALUE(
      chunk_numel * num_chunks == input.sym_numel(),
      "expected split sizes to sum to the input size ",
      input.sym_numel(),
      " divided by num_chunks ",
      num_chunks,
      ", but they sum to ",
      chunk_numel);
}

void all_gather_copy_out(
    at::TensorList out,
    const at::Tensor& input,
    c10::SymIntArrayRef split_sizes,
    c10::SymIntArrayRef outer_sizes,
    int64_t num_chunks) {
  check_all_gather_copy_out_inputs(
      out, input, split_sizes, outer_sizes, num_chunks);
  const auto chunks =
      input.view_symint({num_chunks, input.sym_numel() / num_chunks});
  c10::SymInt offset = 0;
  for (const auto i : c10::irange(out.size())) {
    const auto& outer_size = outer_sizes[i];
    const auto inner_size = split_sizes[i] / outer_size;
    // One strided copy per output keeps traced graphs linear in outputs.
    out[i]
        .view_symint({outer_size, num_chunks, inner_size})
        .copy_(chunks.narrow_symint(1, offset, split_sizes[i])
                   .view_symint({num_chunks, outer_size, inner_size})
                   .transpose(0, 1));
    offset += split_sizes[i];
  }
}

TORCH_LIBRARY_FRAGMENT(fsdp, m) {
  m.def(
      "_all_gather_copy_out_(Tensor(a!)[] self, Tensor src, SymInt[] split_sizes, SymInt[] outer_sizes, int num_chunks) -> ()");
}

TORCH_LIBRARY_IMPL(fsdp, CompositeExplicitAutograd, m) {
  m.impl("_all_gather_copy_out_", TORCH_FN(all_gather_copy_out));
}

TORCH_LIBRARY_IMPL(fsdp, Functionalize, m) {
  m.impl("_all_gather_copy_out_", TORCH_FN(all_gather_copy_out));
}

TORCH_LIBRARY_IMPL(fsdp, ADInplaceOrView, m) {
  m.impl(
      "_all_gather_copy_out_",
      TORCH_FN(all_gather_copy_out_ad_inplace_or_view));
}

} // namespace torch::distributed::fsdp
