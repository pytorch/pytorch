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
    int64_t num_chunks,
    at::OptionalSymIntArrayRef output_inner_sizes) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("fsdp::_all_gather_copy_out_", "")
                       .typed<void(
                           at::TensorList,
                           const at::Tensor&,
                           c10::SymIntArrayRef,
                           c10::SymIntArrayRef,
                           int64_t,
                           at::OptionalSymIntArrayRef)>();
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    op.redispatch(
        ks & c10::after_ADInplaceOrView_keyset,
        out,
        input,
        split_sizes,
        outer_sizes,
        num_chunks,
        output_inner_sizes);
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
    int64_t num_chunks,
    at::OptionalSymIntArrayRef output_inner_sizes) {
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
  TORCH_CHECK_VALUE(
      !output_inner_sizes.has_value() ||
          output_inner_sizes->size() == out.size(),
      "expected one output inner size per output, but got ",
      out.size(),
      " outputs and ",
      output_inner_sizes.has_value() ? output_inner_sizes->size() : 0,
      " output inner sizes");
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
    if (!output_inner_sizes.has_value()) {
      TORCH_CHECK_VALUE(
          output_numel == split_size * num_chunks,
          "expected output ",
          i,
          " to hold split size * num_chunks = ",
          split_size * num_chunks,
          " input elements, but it holds ",
          output_numel);
    } else {
      const auto& output_inner_size = (*output_inner_sizes)[i];
      const auto max_output_inner_size = split_size / outer_size * num_chunks;
      TORCH_CHECK_VALUE(
          output_inner_size >= 0 && output_inner_size <= max_output_inner_size,
          "expected output ",
          i,
          " to have an output inner size in [0, split size / outer size * "
          "num_chunks = ",
          max_output_inner_size,
          "], but got ",
          output_inner_size);
      // Rows after the first are strided by the output inner size, so only a
      // single-row output may be larger than the rows it receives
      if (outer_size == 1) {
        TORCH_CHECK_VALUE(
            output_numel >= output_inner_size,
            "expected output ",
            i,
            " to hold at least its output inner size ",
            output_inner_size,
            " elements, but it holds ",
            output_numel);
      } else {
        TORCH_CHECK_VALUE(
            output_numel == outer_size * output_inner_size,
            "expected output ",
            i,
            " to hold outer size * output inner size = ",
            outer_size * output_inner_size,
            " elements, but it holds ",
            output_numel);
      }
    }
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
    int64_t num_chunks,
    at::OptionalSymIntArrayRef output_inner_sizes) {
  check_all_gather_copy_out_inputs(
      out, input, split_sizes, outer_sizes, num_chunks, output_inner_sizes);
  const auto chunks =
      input.view_symint({num_chunks, input.sym_numel() / num_chunks});
  c10::SymInt offset = 0;
  for (const auto i : c10::irange(out.size())) {
    const auto& outer_size = outer_sizes[i];
    const auto inner_size = split_sizes[i] / outer_size;
    // (num_chunks, outer_size, inner_size): each rank's rows of this output
    const auto src = chunks.narrow_symint(1, offset, split_sizes[i])
                         .view_symint({num_chunks, outer_size, inner_size});
    offset += split_sizes[i];
    if (!output_inner_sizes.has_value()) {
      // One strided copy per output keeps traced graphs linear in outputs.
      out[i]
          .view_symint({outer_size, num_chunks, inner_size})
          .copy_(src.transpose(0, 1));
      continue;
    }
    const auto& output_inner_size = (*output_inner_sizes)[i];
    if (output_inner_size == 0) {
      continue; // nothing to copy; also covers an empty split
    }
    // Each destination row takes the complete chunks of the first num_full
    // ranks, then a prefix of the next rank's chunk. Later ranks' chunks lie
    // past the row and are not copied. At most two strided copies per output
    // keep traced graphs linear in outputs.
    const auto dst = out[i]
                         .view(-1)
                         .narrow_symint(0, 0, outer_size * output_inner_size)
                         .view_symint({outer_size, output_inner_size});
    const auto num_full = output_inner_size / inner_size;
    const auto tail_size = output_inner_size - num_full * inner_size;
    dst.narrow_symint(1, 0, num_full * inner_size)
        .view_symint({outer_size, num_full, inner_size})
        .copy_(src.narrow_symint(0, 0, num_full).transpose(0, 1));
    if (tail_size > 0) {
      // tail_size > 0 implies num_full < num_chunks
      dst.narrow_symint(1, num_full * inner_size, tail_size)
          .copy_(src.select_symint(0, num_full).narrow_symint(1, 0, tail_size));
    }
  }
}

TORCH_LIBRARY_FRAGMENT(fsdp, m) {
  // output_inner_sizes[i], when given, is the number of elements in each of
  // out[i]'s outer_sizes[i] rows; see check_all_gather_copy_out_inputs.
  m.def(
      "_all_gather_copy_out_(Tensor(a!)[] self, Tensor src, SymInt[] split_sizes, SymInt[] outer_sizes, int num_chunks, *, SymInt[]? output_inner_sizes=None) -> ()");
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
