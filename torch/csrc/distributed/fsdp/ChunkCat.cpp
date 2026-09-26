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
    at::Tensor& out) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("fsdp::chunk_cat_mixed_dtype", "")
          .typed<void(at::TensorList, int64_t, int64_t, at::Tensor&)>();
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    op.redispatch(
        ks & c10::after_ADInplaceOrView_keyset, tensors, dim, num_chunks, out);
  }
  torch::autograd::impl::bump_version(out);
}

} // namespace

void chunk_cat_mixed_dtype(
    at::TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    at::Tensor& out) {
  // _chunk_cat takes same-dtype inputs as is and casts during its copy-in
  if (std::all_of(tensors.begin(), tensors.end(), [&](const at::Tensor& t) {
        return t.scalar_type() == tensors[0].scalar_type();
      })) {
    at::_chunk_cat_out(out, tensors, dim, num_chunks);
    return;
  }
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
  // Like fsdp::chunk_cat, but inputs may have different dtypes.
  m.def(
      "chunk_cat_mixed_dtype(Tensor[] tensors, int dim, int num_chunks, *, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(fsdp, CompositeExplicitAutograd, m) {
  m.impl("chunk_cat_mixed_dtype", TORCH_FN(chunk_cat_mixed_dtype));
}

TORCH_LIBRARY_IMPL(fsdp, ADInplaceOrView, m) {
  m.impl(
      "chunk_cat_mixed_dtype",
      TORCH_FN(chunk_cat_mixed_dtype_ad_inplace_or_view));
}

} // namespace torch::distributed::fsdp
