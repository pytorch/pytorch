#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/distributed/fsdp/ChunkCat.h>
#include <torch/library.h>

namespace torch::distributed::fsdp {

namespace {

// Bumps out's version, as the generated kernels of ATen out= ops do.
void chunk_cat_mixed_dtype_ad_inplace_or_view(
    c10::DispatchKeySet ks,
    at::TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    at::Tensor& out,
    at::OptionalIntArrayRef num_leading_dims) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("fsdp::chunk_cat_mixed_dtype", "")
                       .typed<void(
                           at::TensorList,
                           int64_t,
                           int64_t,
                           at::Tensor&,
                           at::OptionalIntArrayRef)>();
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    op.redispatch(
        ks & c10::after_ADInplaceOrView_keyset,
        tensors,
        dim,
        num_chunks,
        out,
        num_leading_dims);
  }
  torch::autograd::impl::bump_version(out);
}

void chunk_cat_with_leading_dims(
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks,
    at::Tensor& out) {
  check_chunk_cat_leading_dims_inputs(
      tensors, num_leading_dims, num_chunks, out);
  auto chunks = out.view({num_chunks, -1});
  c10::SymInt offset = 0;
  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    const auto dim = num_leading_dims[i];
    const auto sizes = tensor.sym_sizes();
    const auto outer_size = c10::multiply_integers(sizes.slice(0, dim));
    const auto inner_size = (sizes[dim] + num_chunks - 1) / num_chunks *
        c10::multiply_integers(sizes.slice(dim + 1));
    auto chunk = chunks.narrow_symint(1, offset, outer_size * inner_size);
    // copy_ casts to out's dtype
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
}

} // namespace

bool has_leading_dims(at::OptionalIntArrayRef num_leading_dims) {
  const auto dims = num_leading_dims.value_or(at::IntArrayRef{});
  return std::any_of(
      dims.begin(), dims.end(), [](int64_t dim) { return dim != 0; });
}

void check_chunk_cat_leading_dims_inputs(
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks,
    const at::Tensor& out) {
  TORCH_CHECK_VALUE(
      num_chunks > 0, "expected positive num_chunks, but got ", num_chunks);
  TORCH_CHECK_VALUE(!tensors.empty(), "expected a non-empty input tensor list");
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
        c10::canCast(tensor.scalar_type(), out.scalar_type()),
        "expected inputs castable to the output dtype ",
        out.scalar_type(),
        ", but input ",
        i,
        " has dtype ",
        tensor.scalar_type());
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

void chunk_cat_mixed_dtype(
    at::TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    at::Tensor& out,
    at::OptionalIntArrayRef num_leading_dims) {
  if (has_leading_dims(num_leading_dims)) {
    TORCH_CHECK_VALUE(
        dim == 0, "expected dim 0 with num_leading_dims, but got dim ", dim);
    chunk_cat_with_leading_dims(tensors, *num_leading_dims, num_chunks, out);
    return;
  }
  // _chunk_cat takes same-dtype inputs as is and casts during its copy-in
  if (std::all_of(tensors.begin(), tensors.end(), [&](const at::Tensor& t) {
        return t.scalar_type() == tensors[0].scalar_type();
      })) {
    at::_chunk_cat_out(out, tensors, dim, num_chunks);
    return;
  }
  // _chunk_cat takes one input dtype, so this holds a cast copy of every
  // mismatched input until it returns. CUDA bf16 or fp16 + fp32, the common
  // case, takes the fused kernel in ChunkCat.cu instead, so only other devices
  // and dtype pairs pay for the copies.
  // TODO: Above a size threshold, copy each input into its slice of out, which
  // casts without a copy. Small inputs keep this path, since per-input copies
  // cost a launch each.
  std::vector<at::Tensor> inputs;
  inputs.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    TORCH_CHECK(
        c10::canCast(tensor.scalar_type(), out.scalar_type()),
        "chunk_cat_mixed_dtype: can't cast ",
        tensor.scalar_type(),
        " to ",
        out.scalar_type());
    inputs.push_back(tensor.to(out.scalar_type()));
  }
  at::_chunk_cat_out(out, inputs, dim, num_chunks);
}

TORCH_LIBRARY_FRAGMENT(fsdp, m) {
  // Like fsdp::chunk_cat, but inputs may have different dtypes. With dim 0, a
  // nonzero num_leading_dims[i] chunks tensors[i] along that dim instead,
  // keeping its leading dims in each chunk, as Shard(i) needs for i > 0.
  m.def(
      "chunk_cat_mixed_dtype(Tensor[] tensors, int dim, int num_chunks, *, Tensor(a!) out, int[]? num_leading_dims=None) -> ()");
}

TORCH_LIBRARY_IMPL(fsdp, CompositeExplicitAutograd, m) {
  m.impl("chunk_cat_mixed_dtype", TORCH_FN(chunk_cat_mixed_dtype));
}

TORCH_LIBRARY_IMPL(fsdp, Functionalize, m) {
  m.impl("chunk_cat_mixed_dtype", TORCH_FN(chunk_cat_mixed_dtype));
}

TORCH_LIBRARY_IMPL(fsdp, ADInplaceOrView, m) {
  m.impl(
      "chunk_cat_mixed_dtype",
      TORCH_FN(chunk_cat_mixed_dtype_ad_inplace_or_view));
}

} // namespace torch::distributed::fsdp
