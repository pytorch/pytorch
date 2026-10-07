#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/distributed/fsdp/CollectiveCopy.hpp>
#include <torch/library.h>

#include <algorithm>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_chunk_cat.h>
#endif

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

at::Tensor& reduce_scatter_copy_in_ad_inplace_or_view(
    c10::DispatchKeySet ks,
    at::Tensor& out,
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("fsdp::_reduce_scatter_copy_in_", "")
          .typed<at::Tensor&(
              at::Tensor&, at::TensorList, at::IntArrayRef, int64_t)>();
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    op.redispatch(
        ks & c10::after_ADInplaceOrView_keyset,
        out,
        tensors,
        num_leading_dims,
        num_chunks);
  }
  torch::autograd::impl::bump_version(out);
  return out;
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

void check_reduce_scatter_copy_in_inputs(
    const at::Tensor& out,
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks) {
  TORCH_CHECK_VALUE(
      num_chunks > 0, "expected positive num_chunks, but got ", num_chunks);
  TORCH_CHECK_VALUE(!tensors.empty(), "expected a non-empty input tensor list");
  TORCH_CHECK_TYPE(
      c10::canCast(tensors[0].scalar_type(), out.scalar_type()),
      "expected inputs castable to the output dtype ",
      out.scalar_type(),
      ", but got ",
      tensors[0].scalar_type());
  TORCH_CHECK_VALUE(
      tensors.size() == num_leading_dims.size(),
      "expected one leading dimension count per input, but got ",
      tensors.size(),
      " inputs and ",
      num_leading_dims.size(),
      " counts");
  TORCH_CHECK_VALUE(
      out.layout() == at::kStrided,
      "expected a strided output, but got ",
      out.layout());
  c10::SymInt chunk_numel = 0;
  bool has_input = false;
  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    const auto dim = num_leading_dims[i];
    TORCH_CHECK_VALUE(
        dim >= 0 && dim < tensor.dim(),
        "expected a leading dimension count in [0, ",
        tensor.dim(),
        ") for input ",
        i,
        ", but got ",
        dim);
    TORCH_CHECK_VALUE(
        tensor.layout() == at::kStrided && (dim == 0 || tensor.is_contiguous()),
        "expected strided inputs, contiguous if they have leading "
        "dimensions, but input ",
        i,
        " is not");
    TORCH_CHECK_TYPE(
        tensor.scalar_type() == tensors[0].scalar_type(),
        "expected inputs with the same dtype, but input ",
        i,
        " has dtype ",
        tensor.scalar_type(),
        " and input 0 has dtype ",
        tensors[0].scalar_type());
    TORCH_CHECK_VALUE(
        tensor.device() == out.device(),
        "expected inputs on the output device ",
        out.device(),
        ", but input ",
        i,
        " is on ",
        tensor.device());
    const auto sizes = tensor.sym_sizes();
    const auto outer_size = c10::multiply_integers(sizes.slice(0, dim));
    TORCH_CHECK_VALUE(
        dim == 0 || sizes[dim] % num_chunks == 0,
        "expected input ",
        i,
        " to have a size divisible by num_chunks ",
        num_chunks,
        " in dim ",
        dim,
        ", but got shape ",
        sizes);
    if (tensor.sym_numel() == 0) {
      TORCH_CHECK_VALUE(
          dim > 0 && outer_size == 0,
          "expected input ",
          i,
          " to be non-empty unless a leading dimension is empty, but got "
          "shape ",
          sizes);
    } else {
      has_input = true;
    }
    chunk_numel += (sizes[dim] + num_chunks - 1) / num_chunks * outer_size *
        c10::multiply_integers(sizes.slice(dim + 1));
  }
  TORCH_CHECK_VALUE(has_input, "expected at least one non-empty input");
  TORCH_CHECK_VALUE(
      out.sym_numel() == chunk_numel * num_chunks,
      "expected an output with ",
      chunk_numel * num_chunks,
      " elements, but got ",
      out.sym_numel());
}

at::Tensor& reduce_scatter_copy_in(
    at::Tensor& out,
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks) {
  check_reduce_scatter_copy_in_inputs(
      out, tensors, num_leading_dims, num_chunks);
  auto chunks = out.view({num_chunks, -1});
  if (std::all_of(
          num_leading_dims.begin(), num_leading_dims.end(), [](int64_t dim) {
            return dim == 0;
          })) {
    at::_chunk_cat_out(chunks, tensors, 0, num_chunks);
    return out;
  }
  c10::SymInt offset = 0;
  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    const auto dim = num_leading_dims[i];
    const auto sizes = tensor.sym_sizes();
    const auto outer_size = c10::multiply_integers(sizes.slice(0, dim));
    const auto inner_size = (sizes[dim] + num_chunks - 1) / num_chunks *
        c10::multiply_integers(sizes.slice(dim + 1));
    auto chunk = chunks.narrow_symint(1, offset, outer_size * inner_size);
    if (dim == 0) {
      chunk.copy_(at::_chunk_cat(tensor, 0, num_chunks));
    } else {
      // One strided copy per input keeps traced graphs linear in inputs.
      chunk.view_symint({num_chunks, outer_size, inner_size})
          .copy_(tensor.view_symint({outer_size, num_chunks, inner_size})
                     .transpose(0, 1));
    }
    offset += outer_size * inner_size;
  }
  return out;
}

TORCH_LIBRARY_FRAGMENT(fsdp, m) {
  m.def(
      "_all_gather_copy_out_(Tensor(a!)[] self, Tensor src, SymInt[] split_sizes, SymInt[] outer_sizes, int num_chunks) -> ()");
  m.def(
      "_reduce_scatter_copy_in_(Tensor(a!) self, Tensor[] tensors, int[] num_leading_dims, int num_chunks) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(fsdp, CompositeExplicitAutograd, m) {
  m.impl("_all_gather_copy_out_", TORCH_FN(all_gather_copy_out));
  m.impl("_reduce_scatter_copy_in_", TORCH_FN(reduce_scatter_copy_in));
}

TORCH_LIBRARY_IMPL(fsdp, Functionalize, m) {
  m.impl("_all_gather_copy_out_", TORCH_FN(all_gather_copy_out));
  m.impl("_reduce_scatter_copy_in_", TORCH_FN(reduce_scatter_copy_in));
}

TORCH_LIBRARY_IMPL(fsdp, ADInplaceOrView, m) {
  m.impl(
      "_all_gather_copy_out_",
      TORCH_FN(all_gather_copy_out_ad_inplace_or_view));
  m.impl(
      "_reduce_scatter_copy_in_",
      TORCH_FN(reduce_scatter_copy_in_ad_inplace_or_view));
}

} // namespace torch::distributed::fsdp
