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
    at::OptionalIntArrayRef num_leading_dims,
    at::OptionalIntArrayRef padded_dim_sizes) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("fsdp::chunk_cat_mixed_dtype", "")
                       .typed<void(
                           at::TensorList,
                           int64_t,
                           int64_t,
                           at::Tensor&,
                           at::OptionalIntArrayRef,
                           at::OptionalIntArrayRef)>();
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    op.redispatch(
        ks & c10::after_ADInplaceOrView_keyset,
        tensors,
        dim,
        num_chunks,
        out,
        num_leading_dims,
        padded_dim_sizes);
  }
  torch::autograd::impl::bump_version(out);
}

void chunk_cat_with_leading_dims(
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks,
    at::Tensor& out,
    at::OptionalIntArrayRef padded_dim_sizes) {
  check_chunk_cat_leading_dims_inputs(
      tensors, num_leading_dims, num_chunks, out, padded_dim_sizes);
  auto chunks = out.view({num_chunks, -1});
  c10::SymInt offset = 0;
  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    const auto dim = num_leading_dims[i];
    const auto sizes = tensor.sym_sizes();
    const auto outer_size = c10::multiply_integers(sizes.slice(0, dim));
    const auto& dim_size = sizes[dim];
    // Like torch.chunk, every rank's chunk spans ceil(dim_size / num_chunks)
    // rows of the chunked dim, or padded_dim_sizes[i] / num_chunks if given;
    // rows past the tensor are zero padding.
    const auto chunk_rows = padded_dim_sizes.has_value()
        ? c10::SymInt((*padded_dim_sizes)[i] / num_chunks)
        : (dim_size + num_chunks - 1) / num_chunks;
    const auto row_numel = c10::multiply_integers(sizes.slice(dim + 1));
    const auto inner_size = chunk_rows * row_numel;
    auto chunk = chunks.narrow_symint(1, offset, outer_size * inner_size);
    offset += outer_size * inner_size;
    // copy_ casts to out's dtype
    if (dim == 0 && !padded_dim_sizes.has_value()) {
      chunk.copy_(at::_chunk_cat(tensor, 0, num_chunks));
      continue;
    }
    // (num_chunks, outer_size, inner_size): each rank's rows of this input
    auto rank_chunks = chunk.view_symint({num_chunks, outer_size, inner_size});
    if (dim_size == chunk_rows * num_chunks) {
      // One strided copy per input keeps traced graphs linear in inputs.
      rank_chunks.copy_(tensor.view_symint({outer_size, num_chunks, inner_size})
                            .transpose(0, 1));
      continue;
    }
    // Uneven: the first num_full ranks get chunk_rows rows each, the next one
    // gets the remaining tail_rows rows followed by padding, and any later
    // ranks get only padding. chunk_rows > 0 here (zero rows divide evenly),
    // and num_full < num_chunks; num_full is 0 when padded_dim_sizes pads a
    // single chunk. A constant number of copies and fills per input keeps
    // traced graphs linear in inputs.
    const auto num_full = dim_size / chunk_rows;
    const auto tail_rows = dim_size - num_full * chunk_rows;
    const auto rows = tensor.view_symint({outer_size, dim_size * row_numel});
    rank_chunks.narrow_symint(0, 0, num_full)
        .copy_(rows.narrow_symint(1, 0, num_full * inner_size)
                   .view_symint({outer_size, num_full, inner_size})
                   .transpose(0, 1));
    auto num_written = num_full;
    if (tail_rows > 0) {
      const auto tail_numel = tail_rows * row_numel;
      auto tail_chunk = rank_chunks.select_symint(0, num_full);
      tail_chunk.narrow_symint(1, 0, tail_numel)
          .copy_(rows.narrow_symint(1, num_full * inner_size, tail_numel));
      tail_chunk.narrow_symint(1, tail_numel, inner_size - tail_numel).zero_();
      num_written += 1;
    }
    rank_chunks.narrow_symint(0, num_written, num_chunks - num_written).zero_();
  }
}

} // namespace

bool has_leading_dims(at::OptionalIntArrayRef num_leading_dims) {
  const auto dims = num_leading_dims.value_or(at::IntArrayRef{});
  return std::any_of(
      dims.begin(), dims.end(), [](int64_t dim) { return dim != 0; });
}

std::vector<int64_t> leading_dims_or_zeros(
    at::OptionalIntArrayRef num_leading_dims,
    size_t num_tensors) {
  if (num_leading_dims.has_value() && !num_leading_dims->empty()) {
    return num_leading_dims->vec();
  }
  return std::vector<int64_t>(num_tensors, 0);
}

void check_chunk_cat_leading_dims_inputs(
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks,
    const at::Tensor& out,
    at::OptionalIntArrayRef padded_dim_sizes) {
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
      !padded_dim_sizes.has_value() ||
          padded_dim_sizes->size() == tensors.size(),
      "expected one padded dim size per input, but got ",
      tensors.size(),
      " inputs and ",
      padded_dim_sizes.has_value() ? padded_dim_sizes->size() : 0,
      " sizes");
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
    // A chunked dim need not divide by num_chunks: like dim 0, it pads every
    // rank's chunk to ceil(sizes[dim] / num_chunks) rows, or to
    // padded_dim_sizes[i] / num_chunks rows if given.
    auto chunk_rows = (sizes[dim] + num_chunks - 1) / num_chunks;
    if (padded_dim_sizes.has_value()) {
      const auto padded_dim_size = (*padded_dim_sizes)[i];
      TORCH_CHECK_VALUE(
          padded_dim_size % num_chunks == 0 && sizes[dim] <= padded_dim_size,
          "expected a multiple of num_chunks (",
          num_chunks,
          ") of at least ",
          sizes[dim],
          " as padded dim size of input ",
          i,
          ", but got ",
          padded_dim_size);
      chunk_rows = padded_dim_size / num_chunks;
    }
    // With a padded dim size, an empty chunked dim is all padding
    if (tensor.sym_numel() == 0) {
      TORCH_CHECK_VALUE(
          (dim > 0 && outer_size == 0) || padded_dim_sizes.has_value(),
          "expected input ",
          i,
          " to be non-empty unless a leading dimension is empty or its "
          "padded dim size is given, but got shape ",
          sizes);
    } else {
      has_input = true;
    }
    chunk_numel +=
        chunk_rows * outer_size * c10::multiply_integers(sizes.slice(dim + 1));
  }
  TORCH_CHECK_VALUE(
      has_input || padded_dim_sizes.has_value(),
      "expected at least one non-empty input");
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
    at::OptionalIntArrayRef num_leading_dims,
    at::OptionalIntArrayRef padded_dim_sizes) {
  if (has_leading_dims(num_leading_dims) || padded_dim_sizes.has_value()) {
    TORCH_CHECK_VALUE(
        dim == 0,
        "expected dim 0 with num_leading_dims or padded_dim_sizes, but got dim ",
        dim);
    chunk_cat_with_leading_dims(
        tensors,
        leading_dims_or_zeros(num_leading_dims, tensors.size()),
        num_chunks,
        out,
        padded_dim_sizes);
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
  // keeping its leading dims in each chunk, as Shard(i) needs for i > 0. As
  // with dim 0, a dim that num_chunks does not divide is chunked like
  // torch.chunk and each rank's chunk is zero-padded to the same size.
  // padded_dim_sizes[i], a multiple of num_chunks, sets the size the chunked
  // dim of tensors[i] is zero-padded to, e.g. to pad a rank's own shard to the
  // padded sharded size with num_chunks=1.
  m.def(
      "chunk_cat_mixed_dtype(Tensor[] tensors, int dim, int num_chunks, *, Tensor(a!) out, int[]? num_leading_dims=None, int[]? padded_dim_sizes=None) -> ()");
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
