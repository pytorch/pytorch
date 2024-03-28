#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/record_function.h>

#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <vector>

#include <c10/util/typeid.h>

namespace at {
namespace native::xpu {
namespace impl {

static inline bool check_broadcast(
    const Tensor& src,
    const IntArrayRef& shape) {
  auto src_dim = src.dim();
  int64_t tgt_dim = shape.size();
  if (src_dim == 0 && src_dim < tgt_dim)
    return true;
  if (src_dim > tgt_dim)
    return false;
  do {
    src_dim--;
    tgt_dim--;
    auto size = src.size(src_dim);
    if (size != 1 && size != shape[tgt_dim])
      return false;
  } while (src_dim);
  return true;
}

/***** The helper function to get post binary(or sum) for onednn_matmul *****
In onednn, it supports: result = BinaryOP(alpha * (m1 @ m2 + bias), beta *
binary). Since the inputs/outputs shapes of Matmul are complicated,
this helper function is used to adjust binary tensor size according different
matmul cases.*/
static bool get_onednn_matmul_binary_attr(
    Tensor& result,
    onednn::Attr& attr,
    int dim_tensor1,
    int dim_tensor2,
    DimVector output_shape,
    bool t2_is_matrix = true,
    bool should_fold_tensor1 = false,
    bool should_fold_tensor2 = false) {
  onednn::Attr attr_update;
  for (size_t i = 0; i < attr.ops_params_.size(); ++i) {
    onednn::kind_t kind = attr.ops_params_[i].kind_;
    if (kind != onednn::kind_t::binary || !attr.ops_params_[i].binary_.defined()) {
      attr_update.ops_params_.push_back(attr.ops_params_[i]);
      continue;
    }

    float beta = attr.ops_params_[i].scale_;
    bool need_binary = attr.ops_params_[i].binary_.defined() && (beta != 0.f);
    if (!need_binary) {
      continue;
    }

    Tensor binary_final;
    std::vector<int64_t> compute_shape = result.sizes().vec();
    if (dim_tensor1 == 2 && dim_tensor2 == 1) {
      // case 2
      const auto binary =
          MaybeOwned<Tensor>::borrowed(attr.ops_params_[i].binary_);
      binary_final = binary->unsqueeze(1);
    } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
      // case 3
      const auto binary =
          MaybeOwned<Tensor>::borrowed(attr.ops_params_[i].binary_);
      binary_final = binary->unsqueeze(0);
    } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
      // case 4
      const auto binary =
          MaybeOwned<Tensor>::borrowed(attr.ops_params_[i].binary_);
      if (binary->dim() < (int64_t)output_shape.size())
        binary_final = binary->unsqueeze(0);
      else
        binary_final = *binary;
    } else if (should_fold_tensor1) {
      // case 5
      auto binary = t2_is_matrix
          ? MaybeOwned<Tensor>::borrowed(attr.ops_params_[i].binary_)
          : MaybeOwned<Tensor>::owned(
                attr.ops_params_[i].binary_.unsqueeze(-1));
      while (binary->dim() < (int64_t)output_shape.size())
        binary = MaybeOwned<Tensor>::owned(binary->unsqueeze(0));

      if (binary->dim() >= (int64_t)compute_shape.size()) {
        std::vector<int64_t> shape = binary->sizes().vec();
        auto shape_fold = DimVector(shape.begin(), shape.end() - 1);
        const auto first_dim = c10::multiply_integers(shape_fold);
        shape_fold = {first_dim, *(shape.end() - 1)};
        if (first_dim == compute_shape[0] || first_dim == 1) {
          binary_final = binary->contiguous().view(shape_fold);
        } else {
          auto expand_shape = output_shape;
          expand_shape[expand_shape.size() - 1] =
              shape_fold[shape_fold.size() - 1];
          std::vector<int64_t> acc_shape = {compute_shape[0], shape_fold[1]};
          binary_final =
              binary->expand(expand_shape).contiguous().view(acc_shape);
        }
      } else {
        binary_final = *binary;
      }

    } else if (should_fold_tensor2) {
      // case 6
      auto binary = t2_is_matrix
          ? MaybeOwned<Tensor>::borrowed(attr.ops_params_[i].binary_)
          : MaybeOwned<Tensor>::owned(
                attr.ops_params_[i].binary_.unsqueeze(-1));

      while (binary->dim() < (int64_t)output_shape.size())
        binary = MaybeOwned<Tensor>::owned(binary->unsqueeze(0));

      if (binary->dim() >= (int64_t)compute_shape.size()) {
        if (t2_is_matrix)
          binary = MaybeOwned<Tensor>::owned(binary->mT());

        std::vector<int64_t> shape = binary->sizes().vec();
        auto shape_fold = DimVector(shape.begin(), shape.end() - 1);
        const auto first_dim = c10::multiply_integers(shape_fold);
        shape_fold = {first_dim, *(shape.end() - 1)};
        if (first_dim == compute_shape[0] || first_dim == 1) {
          binary_final = binary->contiguous().view(shape_fold);
        } else {
          auto expand_shape = output_shape;
          expand_shape[expand_shape.size() - 1] =
              shape_fold[shape_fold.size() - 1];
          std::vector<int64_t> acc_shape = {compute_shape[0], shape_fold[1]};
          binary_final =
              binary->expand(expand_shape).contiguous().view(acc_shape);
        }
      } else {
        binary_final = *binary;
      }
    } else {
      // case 7
      auto binary = MaybeOwned<Tensor>::borrowed(attr.ops_params_[i].binary_);
      while (binary->dim() < (int64_t)output_shape.size())
        binary = MaybeOwned<Tensor>::owned(binary->unsqueeze(0));

      if (binary->dim() > 3) {
        std::vector<int64_t> shape = binary->sizes().vec();
        auto shape_fold = DimVector(shape.begin(), shape.end() - 2);
        const auto first_dim = c10::multiply_integers(shape_fold);
        shape_fold = {first_dim, *(shape.end() - 2), *(shape.end() - 1)};
        if (first_dim == compute_shape[0] || first_dim == 1) {
          binary_final = binary->reshape(shape_fold);
        } else {
          auto expand_shape = output_shape;
          expand_shape[expand_shape.size() - 1] =
              shape_fold[shape_fold.size() - 1];
          expand_shape[expand_shape.size() - 2] =
              shape_fold[shape_fold.size() - 2];
          std::vector<int64_t> acc_shape = {
              compute_shape[0], shape_fold[1], shape_fold[2]};
          binary_final =
              binary->expand(expand_shape).contiguous().view(acc_shape);
        }
      } else {
        binary_final = *binary;
      }
    }

    if (!onednn::binary_valid(result, binary_final, true)) {
      attr = onednn::Attr();
      return false;
    }

    auto algo = attr.ops_params_[i].algo_;
    if (algo == attr.kind_with_binary_add) {
      // result = (m1 x m2) + beta * binary
      // result = beta * (1.f / beta * (mat1 * mat2) + binary)
      // Since oneDNN only supports sum_scale=1.0 for non-int8 case,
      // we do this formula transformation.
      if (beta != 1.f) {
        attr_update.append_post_eltwise(
            1.f, 1.f / beta, 0.f, attr.kind_with_linear);
      }
      attr_update.append_post_binary(algo, binary_final);
      if (beta != 1.f) {
        attr_update.append_post_eltwise(1.f, beta, 0.f, attr.kind_with_linear);
      }
    } else {
      // binary_mul: result = (m1 x m2) * binary * beta;
      // binary_div: result = (m1 x m2) / binary * beta;
      // binary_min: result = min((m1 x m2), binary) * beta;
      // binary_max: result = max((m1 x m2), binary) * beta;
      // binary_eq: result = eq((m1 x m2), binary) * beta;
      // binary_ne: result = ne((m1 x m2), binary) * beta;
      // binary_ge: result = ge((m1 x m2), binary) * beta;
      // binary_gt: result = gt((m1 x m2), binary) * beta;
      // binary_le: result = le((m1 x m2), binary) * beta;
      // binary_lt: result = lt((m1 x m2), binary) * beta;
      attr_update.append_post_binary(algo, binary_final);
      if (beta != 1.f)
        attr_update.append_post_eltwise(1.f, beta, 0.f, attr.kind_with_linear);
    }
  }
  attr_update.q_scale_ = attr.q_scale_;
  attr_update.q_zero_point_ = attr.q_zero_point_;
  attr = attr_update;
  return true;
}

static bool should_fold(const Tensor& tensor1, const int64_t dim_tensor2) {
  const auto dim_tensor1 = tensor1.dim();
  if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
    const auto t1_sizes_ptr = tensor1.sizes().cbegin();
    const auto t1_strides = tensor1.strides();
    if (dim_tensor1 == 3 && dim_tensor2 == 2 && t1_strides.back() != 1 &&
        t1_strides.front() == t1_sizes_ptr[1] * t1_sizes_ptr[2]) {
      // First dim is slowest moving, and then the following two dims are //
      // transposed. This can happen for example by permute(0, 2, 1).      //
      // First 2 dims could be folded to use mm but would require permutation //
      // with actual data movement, which can be instead handled by BMM with
      // each      // GEMM transposed. This can be generalized to a tensor with
      // dim X + Y + Z where X, Y, and Z      // dims are contiguous, Y dims and
      // Z dims are transposed, and X, Y, Z > 0.      // For example, this can
      // happen by permute(0, 1, 5, 2, 3, 4), where X = 2, Y = 3, and Z = 1.
      return false;
    } else {
      return true;
    }
  } else {
    return false;
  }
}

/*
Matrix product of two Tensors.
The behavior depends on the dimensionality of the Tensors as follows:
- If both Tensors are 1-dimensional, (1d) the dot product (scalar) is
returned.
- If the arguments are 2D - 1D or 1D - 2D, the matrix-vector product is
returned.
- If both arguments are 2D, the matrix-matrix product is returned.
- If one of the arguments is ND with N >= 3 and the other is 1D or 2D, and
some conditions on the strides apply (see should_fold) we fold the first N-1
dimensions of the ND argument to form a matrix, call mm or mv, reshape it
back to ND and return it
- Otherwise, we return bmm, after broadcasting and folding the batched
dimensions if there's more than one
**/
static Tensor& matmul_fusion_variants(
    Tensor& output,
    const Tensor& tensor1,
    const Tensor& tensor2,
    bool trans,
    onednn::Attr& attr,
    bool& is_fused,
    Tensor bias = at::Tensor()) {
  const auto dim_tensor1 = tensor1.dim();
  const auto dim_tensor2 = tensor2.dim();
  // This is checked up here to simplify the logic below
  // Note that the strings are just evaluated on failure, so almost always we
  // just evaluate the condition and move on
  TORCH_CHECK(
      dim_tensor1 != 0 && dim_tensor2 != 0,
      "both arguments to matmul need to be at least 1D, but they are ",
      dim_tensor1,
      "D and ",
      dim_tensor2,
      "D");

  bool should_fold_tensor1 = should_fold(tensor1, dim_tensor2);
  bool should_fold_tensor2 = should_fold(tensor2, dim_tensor1);

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    // case1:
    // original size: [6] x [6] -> []
    is_fused = true;
    Tensor result = output.defined() ? output.view({1, 1})
                                     : at::empty({1, 1}, tensor1.options());
    onednn::matmul(
        result,
        tensor1.view({1, tensor1.size(0)}),
        tensor2.view({tensor2.size(0), 1}),
        bias,
        trans,
        attr);
    if (output.defined() && !output.is_alias_of(result)) {
      output.copy_(result);
    } else {
      output = result;
      output.resize_({});
    }
    return output;
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    // case2:
    // original sizes: [4, 2] x [2] -> [4]
    // onednn sizes: [4, 2] x [2, 1] -> [4, 1]
    DimVector output_shape({tensor1.size(0)});
    DimVector result_shape({tensor1.size(0), 1});
    Tensor result = output.defined()
        ? output.view(result_shape)
        : at::empty(result_shape, tensor1.options());
    Tensor t2 = tensor2.view({tensor2.size(0), 1});

    is_fused = get_onednn_matmul_binary_attr(
        result, attr, dim_tensor1, dim_tensor2, output_shape);
    onednn::matmul(result, tensor1, t2, bias, trans, attr);
    if (output.defined() && !output.is_alias_of(result)) {
      output.copy_(result);
    } else {
      output = result.view(output_shape);
    }
    return output;
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    // case3:
    // original sizes: [2] x [2, 6] -> [6]
    // onednn sizes: [1, 2] x [2, 6] -> [1, 6]
    DimVector output_shape({tensor2.size(1)});
    if (!trans)
      output_shape[0] = tensor2.size(0);
    Tensor t1 = tensor1.unsqueeze(0);
    DimVector result_shape({1, output_shape[0]});
    Tensor result = output.defined()
        ? output.view(result_shape)
        : at::empty(result_shape, tensor1.options());

    is_fused = get_onednn_matmul_binary_attr(
        result, attr, dim_tensor1, dim_tensor2, output_shape);
    onednn::matmul(result, t1, tensor2, bias, trans, attr);
    if (output.defined() && !output.is_alias_of(result)) {
      output.copy_(result);
    } else {
      output = result.view(output_shape);
    }
    return output;
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    // case4:
    // original sizes: [4, 2] x [2, 6] -> [4, 6]
    // onednn sizes: [4, 2] x [2, 6] -> [4, 6]
    DimVector output_shape({tensor1.size(0), tensor2.size(1)});
    if (!trans)
      output_shape[1] = tensor2.size(0);

    Tensor result =
        output.defined() ? output : at::empty(output_shape, tensor1.options());

    is_fused = get_onednn_matmul_binary_attr(
        result, attr, dim_tensor1, dim_tensor2, output_shape);
    onednn::matmul(result, tensor1, tensor2, bias, trans, attr);
    if (output.defined() && !output.is_alias_of(result)) {
      output.copy_(result);
    } else {
      output = result;
    }
    return output;
  } else if (should_fold_tensor1) {
    // dim_tensor1 >=3 && (dim_tensor2 == 1 || dim_tensor2 == 2)
    // case5-1:
    // original sizes: [3, 4, 2] x [2, 6] -> [3, 4, 6]
    // onednn sizes: [12, 2] x [2, 6] -> [12, 6]
    // case5-2:
    // original sizes: [3, 4, 2] x [2] -> [3, 4]
    // onednn sizes: [12, 2] x [2, 1] -> [12, 1]
    const auto t1_own = MaybeOwned<Tensor>::borrowed(tensor1);
    const auto t2_own = MaybeOwned<Tensor>::borrowed(tensor2);

    const auto sizes_1 = t1_own->sizes();
    auto output_shape = DimVector(sizes_1.begin(), sizes_1.end() - 1);
    const auto folded_dim1 = c10::multiply_integers(output_shape);
    const auto t1 = t1_own->reshape({folded_dim1, sizes_1.back()});
    const auto t2_is_matrix = t2_own->dim() == 2;
    Tensor t2 = t2_is_matrix ? *t2_own : t2_own->view({t2_own->size(0), 1});
    if (trans)
      output_shape.push_back(t2.size(1));
    else
      output_shape.push_back(t2.size(0));
    DimVector result_shape({t1.size(0), output_shape[output_shape.size() - 1]});
    Tensor result = output.defined()
        ? output.view(result_shape)
        : at::empty(result_shape, tensor1.options());
    is_fused = get_onednn_matmul_binary_attr(
        result,
        attr,
        dim_tensor1,
        dim_tensor2,
        output_shape,
        t2_is_matrix,
        should_fold_tensor1,
        should_fold_tensor2);
    onednn::matmul(result, t1, t2, bias, trans, attr);
    if (output.defined() && !output.is_alias_of(result)) {
      output.copy_(result);
    } else {
      output = at::_unsafe_view(result, output_shape);
    }
    output = t2_is_matrix ? output : output.squeeze(-1);
    return output;
  } else if (should_fold_tensor2) {
    // dim_tensor2 >=3 && (dim_tensor1 == 1 || dim_tensor1 == 2)
    // case6-1:
    // original sizes: [2] x [3, 2, 4] = [3, 4]
    // onednn sizes: [12, 2] x [2, 1] = [12, 1]
    // or
    // original sizes: [2] x [2, 3, 2, 4] = [2, 3, 4]
    // onednn sizes: [24, 2] x [2, 1] = [24, 1]

    // case6-2:
    // original sizes: [6, 2] x [3, 2, 4] = [3, 6, 4]
    // onednn sizes: [12, 2] x [2, 6] = [12, 6]
    // or
    // original sizes: [6, 2] x [2, 3, 2, 4] = [2, 3, 6, 4]
    // onednn sizes: [24, 2] x [2, 6] = [24, 6]

    const auto t1_own = trans
        ? MaybeOwned<Tensor>::owned(tensor2.mT())
        : MaybeOwned<Tensor>::owned(tensor2.transpose(-1, -2).mT());
    trans = true;
    const auto t2_own = dim_tensor1 == 2
        ? MaybeOwned<Tensor>::owned(tensor1.t())
        : MaybeOwned<Tensor>::borrowed(tensor1);

    const auto sizes_1 = t1_own->sizes();
    auto output_shape = DimVector(sizes_1.begin(), sizes_1.end() - 1);
    const auto folded_dim1 = c10::multiply_integers(output_shape);
    const auto t1 = t1_own->reshape({folded_dim1, sizes_1.back()});
    const auto t2_is_matrix = t2_own->dim() == 2;
    Tensor t2 = t2_is_matrix ? *t2_own : t2_own->view({t2_own->size(0), 1});
    output_shape.push_back(t2.size(1));
    DimVector result_shape({t1.size(0), t2.size(1)});
    Tensor result = output.defined()
        ? output.view(result_shape)
        : at::empty(result_shape, tensor1.options());

    is_fused = get_onednn_matmul_binary_attr(
        result,
        attr,
        dim_tensor1,
        dim_tensor2,
        output_shape,
        t2_is_matrix,
        should_fold_tensor1,
        should_fold_tensor2);
    onednn::matmul(result, t1, t2, bias, trans, attr);
    if (output.defined() && !output.is_alias_of(result)) {
      output.copy_(result);
    } else {
      output = at::_unsafe_view(result, output_shape);
    }
    output = t2_is_matrix ? output.mT().contiguous() : output.squeeze(-1);
    return output;
  } else {
    // dim_tensor1 >= 3 || dim_tensor2 >= 3
    // case7-1:
    // original sizes: [3, 4, 2] x [3, 2, 6] = [3, 4, 6]
    // onednn sizes: [3, 4, 2] x [3, 2, 6] = [3, 4, 6]
    // case7-2:
    // original sizes: [5, 1, 4, 2] x [3, 2, 6] = [5, 3, 4, 6]
    // onednn sizes: [15, 4, 2] x [15, 2, 6] = [15, 4, 6]
    const auto t2_own = trans
        ? MaybeOwned<Tensor>::borrowed(tensor2)
        : MaybeOwned<Tensor>::owned(tensor2.transpose(-1, -2));
    trans = true;

    const int64_t n = dim_tensor1 > 1 ? tensor1.sizes().cend()[-2] : 1LL;
    const int64_t m1 = tensor1.sizes().back();
    const IntArrayRef batch_tensor1(
        tensor1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0LL));
    const int64_t m2 =
        dim_tensor2 > 1 ? t2_own->sizes().cend()[-2] : t2_own->sizes().back();
    const int64_t p = dim_tensor2 > 1 ? t2_own->sizes().back() : 1LL;
    const IntArrayRef batch_tensor2(
        t2_own->sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0LL));
    auto output_shape = infer_size_dimvector(batch_tensor1, batch_tensor2);

    const auto tensor1_expand_size = [&output_shape, n, m1] {
      DimVector ret(output_shape);
      ret.append({n, m1});
      return ret;
    }();
    const auto tensor2_expand_size = [&output_shape, m2, p] {
      DimVector ret(output_shape);
      ret.append({m2, p});
      return ret;
    }();
    const int64_t expand_batch_product = c10::multiply_integers(output_shape);

    // flatten expanded batches
    const auto tensor1_expanded = tensor1.expand(tensor1_expand_size)
                                      .reshape({expand_batch_product, n, m1});
    const auto tensor2_expanded = t2_own->expand(tensor2_expand_size)
                                      .reshape({expand_batch_product, m2, p});
    if (dim_tensor1 > 1) {
      output_shape.push_back(n);
    }
    if (dim_tensor2 > 1) {
      output_shape.push_back(p);
    }
    DimVector result_shape({expand_batch_product, n, p});
    Tensor result = output.defined()
        ? output.view(result_shape)
        : at::empty(result_shape, tensor1.options());

    is_fused = get_onednn_matmul_binary_attr(
        result, attr, dim_tensor1, dim_tensor2, output_shape);
    onednn::matmul(
        result, tensor1_expanded, tensor2_expanded, bias, trans, attr);
    if (output.defined() && !output.is_alias_of(result)) {
      output.copy_(result);
    } else {
      output = at::_unsafe_view(result, output_shape);
    }
    return output;
  }
}

} // namespace impl
} // namespace native::xpu
} // namespace at
