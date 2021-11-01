#include "lazy_tensor_core/csrc/tensor_ops.h"

#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/tensor_aten_ops.h"
#include "lazy_tensor_core/csrc/tensor_distributed.h"
#include "lazy_tensor_core/csrc/ts_backend/LazyLazyIr.h"
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
LazyTensor IndexAcrossDims(const LazyTensor& input, int64_t dim,
                           int64_t index) {
  return lazy_tensor_aten_ops::squeeze(
      lazy_tensor_aten_ops::slice(input, dim, index, index + 1, 1), dim);
}

}  // namespace

LazyTensor Cross(const LazyTensor& input, const LazyTensor& other,
                 c10::optional<int64_t> dim) {
  int64_t canonical_dim;
  if (dim) {
    canonical_dim =
        Helpers::GetCanonicalDimensionIndex(*dim, input.shape().get().rank());
  } else {
    auto input_shape_ref = input.shape();
    auto dim_3_it = std::find((*input_shape_ref).dimensions().begin(),
                              (*input_shape_ref).dimensions().end(), 3);
    CHECK(dim_3_it != (*input_shape_ref).dimensions().end())
        << "No dimension of size 3 in input: " << (*input_shape_ref).ToString();
    canonical_dim = dim_3_it - (*input_shape_ref).dimensions().begin();
  }
  CHECK_EQ(input.size(canonical_dim), 3)
      << "Invalid cross argument: dimension " << canonical_dim
      << " does not have size 3";
  CHECK_LT(canonical_dim, input.shape().get().rank())
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
  LazyTensor s1 =
      lazy_tensor_aten_ops::sub(lazy_tensor_aten_ops::mul(u2, v3),
                                lazy_tensor_aten_ops::mul(u3, v2), one);
  LazyTensor s2 =
      lazy_tensor_aten_ops::sub(lazy_tensor_aten_ops::mul(u3, v1),
                                lazy_tensor_aten_ops::mul(u1, v3), one);
  LazyTensor s3 =
      lazy_tensor_aten_ops::sub(lazy_tensor_aten_ops::mul(u1, v2),
                                lazy_tensor_aten_ops::mul(u2, v1), one);
  // Stack the terms into one result tensor.
  return lazy_tensor_aten_ops::stack({s1, s2, s3}, canonical_dim);
}

LazyTensor Select(const LazyTensor& input, int64_t dim, int64_t index) {
  auto shape = input.shape();
  dim = Helpers::GetCanonicalDimensionIndex(dim, shape.get().rank());
  LazyTensor result = lazy_tensor_aten_ops::narrow(input, dim, index, 1);
  auto new_dims = Helpers::DropDimensions(shape.get().dimensions(), {dim});
  return lazy_tensor_aten_ops::view(result, new_dims);
}

}  // namespace tensor_ops
}  // namespace torch_lazy_tensors
