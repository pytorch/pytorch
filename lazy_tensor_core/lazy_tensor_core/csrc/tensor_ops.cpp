#include "lazy_tensor_core/csrc/tensor_ops.h"

#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/tensor_aten_ops.h"
#include "lazy_tensor_core/csrc/tensor_distributed.h"
#include "lazy_tensor_core/csrc/ts_backend/LazyLazyIr.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"
#include "torch/csrc/lazy/core/ir.h"
#include "torch/csrc/lazy/core/ir_metadata.h"

namespace torch_lazy_tensors {
namespace tensor_ops {
namespace {
using torch::lazy::ScopePusher;

// Returns the sub-tensor at the given index in the given dimension. Its rank
// is one less than the input, in other words the singleton dimension is
// squeezed out.
LazyTensor IndexAcrossDims(const LazyTensor& input, lazy_tensors::int64 dim,
                           lazy_tensors::int64 index) {
  return tensor_aten_ops::squeeze(
      tensor_aten_ops::slice(input, dim, index, index + 1, 1), dim);
}

}  // namespace

LazyTensor Cross(const LazyTensor& input, const LazyTensor& other,
                 c10::optional<lazy_tensors::int64> dim) {
  lazy_tensors::int64 canonical_dim;
  if (dim) {
    canonical_dim =
        Helpers::GetCanonicalDimensionIndex(*dim, input.shape().get().rank());
  } else {
    auto input_shape_ref = input.shape();
    auto dim_3_it = std::find((*input_shape_ref).dimensions().begin(),
                              (*input_shape_ref).dimensions().end(), 3);
    LTC_CHECK(dim_3_it != (*input_shape_ref).dimensions().end())
        << "No dimension of size 3 in input: " << (*input_shape_ref).ToString();
    canonical_dim = dim_3_it - (*input_shape_ref).dimensions().begin();
  }
  LTC_CHECK_EQ(input.size(canonical_dim), 3)
      << "Invalid cross argument: dimension " << canonical_dim
      << " does not have size 3";
  LTC_CHECK_LT(canonical_dim, input.shape().get().rank())
      << "Invalid cross argument: dimension " << canonical_dim
      << " out of range";
  // Extract the slices for each axis.
  LazyTensor u1 = IndexAcrossDims(input, canonical_dim, 0);
  LazyTensor v1 = IndexAcrossDims(other, canonical_dim, 0);
  LazyTensor u2 = IndexAcrossDims(input, canonical_dim, 1);
  LazyTensor v2 = IndexAcrossDims(other, canonical_dim, 1);
  LazyTensor u3 = IndexAcrossDims(input, canonical_dim, 2);
  LazyTensor v3 = IndexAcrossDims(other, canonical_dim, 2);
  // Compute the term for each axis.
  at::Scalar one(1);
  LazyTensor s1 = tensor_aten_ops::sub(tensor_aten_ops::mul(u2, v3),
                                       tensor_aten_ops::mul(u3, v2), one);
  LazyTensor s2 = tensor_aten_ops::sub(tensor_aten_ops::mul(u3, v1),
                                       tensor_aten_ops::mul(u1, v3), one);
  LazyTensor s3 = tensor_aten_ops::sub(tensor_aten_ops::mul(u1, v2),
                                       tensor_aten_ops::mul(u2, v1), one);
  // Stack the terms into one result tensor.
  return tensor_aten_ops::stack({s1, s2, s3}, canonical_dim);
}

LazyTensor KlDivBackward(const LazyTensor& grad_output, const LazyTensor& input,
                         const LazyTensor& target, ReductionMode reduction,
                         bool log_target) {
  auto input_shape_ref = input.shape();
  LazyTensor expanded_grad_output = tensor_aten_ops::expand(
      grad_output, lazy_tensors::util::ToVector<lazy_tensors::int64>(
                       input_shape_ref.get().dimensions()));
  LazyTensor grad_input;
  if (!log_target) {
    grad_input = tensor_aten_ops::where(
        tensor_aten_ops::gt(target, 0),
        tensor_aten_ops::neg(
            tensor_aten_ops::mul(target, expanded_grad_output)),
        tensor_aten_ops::full_like(input, 0, input.GetDevice(), c10::nullopt));
  } else {
    grad_input = tensor_aten_ops::neg(tensor_aten_ops::mul(
        tensor_aten_ops::exp(target), expanded_grad_output));
  }
  if (reduction == ReductionMode::kMean) {
    LazyTensor dims_size = lazy_tensor_distributed::get_dimensions_size(
        input, Helpers::GetAllDimensions(input_shape_ref));
    grad_input = tensor_aten_ops::div(grad_input, dims_size);
  }
  return grad_input;
}

LazyTensor MakeMatrixWithDiagonal(const LazyTensor& input,
                                  lazy_tensors::int64 diagonal) {
  lazy_tensors::int64 size = input.shape().get().dimensions(0);
  LazyTensor identity =
      tensor_aten_ops::eye(size, size, input.GetDevice(), input.dtype());
  auto padding =
      diagonal >= 0
          ? std::vector<lazy_tensors::int64>{diagonal, 0, 0, diagonal}
          : std::vector<lazy_tensors::int64>{0, -diagonal, -diagonal, 0};
  return tensor_aten_ops::constant_pad_nd(tensor_aten_ops::mul(identity, input),
                                          padding, 0);
}

LazyTensor Select(const LazyTensor& input, lazy_tensors::int64 dim,
                  lazy_tensors::int64 index) {
  auto shape = input.shape();
  dim = Helpers::GetCanonicalDimensionIndex(dim, shape.get().rank());
  LazyTensor result = tensor_aten_ops::narrow(input, dim, index, 1);
  auto new_dims = Helpers::DropDimensions(shape.get().dimensions(), {dim});
  return tensor_aten_ops::view(result, new_dims);
}

LazyTensor EmbeddingDenseBackward(const LazyTensor& grad_output,
                                  const LazyTensor& indices,
                                  lazy_tensors::int64 num_weights,
                                  lazy_tensors::int64 padding_idx,
                                  bool scale_grad_by_freq) {
  LTC_CHECK_EQ(indices.dtype(), at::ScalarType::Long)
      << "Embedding indices are expected to be of scalar type Long";
  auto indices_shape_ref = indices.shape();
  // The weight must be of rank 2, which means the rank of grad_output is one
  // more than the indices.
  LTC_CHECK_EQ(grad_output.shape().get().rank(),
               indices_shape_ref.get().rank() + 1);
  lazy_tensors::int64 numel =
      lazy_tensors::ShapeUtil::ElementsIn(indices_shape_ref.get());
  LazyTensor grad =
      tensor_aten_ops::view(grad_output, {numel, grad_output.size(-1)});
  LazyTensor grad_weight =
      tensor_aten_ops::full({num_weights, grad_output.size(-1)}, 0,
                            grad_output.GetDevice(), grad_output.dtype());
  LazyTensor indices_rank1 = tensor_aten_ops::view(indices, {numel});
  if (scale_grad_by_freq) {
    // Compute the histogram of index values.
    LazyTensor counts = tensor_aten_ops::full(
        {num_weights}, 0, indices.GetDevice(), indices.dtype());
    LazyTensor ones =
        tensor_aten_ops::full({numel}, 1, indices.GetDevice(), indices.dtype());
    tensor_aten_ops::index_put_(
        counts, counts, {indices_rank1}, /*start_dim=*/0,
        /*values=*/ones,
        /*accumulate=*/true, /*result_permutation=*/{0});
    LazyTensor grad_weights_scale =
        tensor_aten_ops::index(counts, {indices_rank1}, 0);
    // Scale the value of the gradient by the histogram.
    grad = tensor_aten_ops::div(
        grad, tensor_aten_ops::unsqueeze(grad_weights_scale, 1));
  }
  // Don't accumulate gradients for indices which are equal with the given
  // padding_idx.
  LazyTensor skip_padding = tensor_aten_ops::unsqueeze(
      tensor_aten_ops::ne(indices_rank1, static_cast<double>(padding_idx)), 1);
  skip_padding = tensor_aten_ops::expand(
      skip_padding, lazy_tensors::util::ToVector<lazy_tensors::int64>(
                        grad.shape().get().dimensions()));
  LazyTensor zero_grad =
      tensor_aten_ops::full_like(grad, 0, grad.GetDevice(), grad.dtype());
  return tensor_aten_ops::index_put(
      grad_weight, {indices_rank1},
      /*start_dim=*/0,
      /*values=*/tensor_aten_ops::where(skip_padding, grad, zero_grad),
      /*accumulate=*/true,
      /*result_permutation=*/{0, 1});
}

}  // namespace tensor_ops
}  // namespace torch_lazy_tensors
