#pragma once

#include <ATen/native/DispatchStub.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>

namespace at {

class Tensor;

namespace native {

using mul_sparse_sparse_out_fn = void (*)(Tensor& res, const Tensor& x, const Tensor& y);
DECLARE_DISPATCH(mul_sparse_sparse_out_fn, mul_sparse_sparse_out_stub);

using sparse_mask_intersection_out_fn = void (*)(Tensor& res, const Tensor& x, const Tensor& y, const c10::optional<Tensor>& x_hash_opt);
DECLARE_DISPATCH(sparse_mask_intersection_out_fn, sparse_mask_intersection_out_stub);

using sparse_mask_projection_out_fn = void (*)(Tensor& res, const Tensor& x, const Tensor& y, const c10::optional<Tensor>& x_hash_opt, bool accumulate_matches);
DECLARE_DISPATCH(sparse_mask_projection_out_fn, sparse_mask_projection_out_stub);

using flatten_indices_fn = Tensor (*)(const Tensor& indices, IntArrayRef size);
DECLARE_DISPATCH(flatten_indices_fn, flatten_indices_stub);

}

}
