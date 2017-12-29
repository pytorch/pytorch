#include "ATen/ATen.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include <functional>
#include <numeric>
#include <vector>

namespace at {
namespace native {

// For backward, we save svd.
// http://www.ics.forth.gr/cvrl/publications/conferences/2000_eccv_SVD_jacobian.pdf
// But instead of gesvd SVD A = U(A) Sig(A) V(A)^T, which doesn't specify signs
// of determinants of U and V, we consider det(A) = \prod Sig_(A), where
//   1. A = U_(A) Sig_(A) V(A)^T
//   2. Sig_(A) and U_(A) can be different in signs in first row/col from
//      their counterparts so that U_(A) * V_(A) have +1 determinant
std::tuple<Tensor, Tensor, Tensor, Tensor> _det_with_svd(const Tensor& self) {
  if (!at::isFloatingType(self.type().scalarType()) ||
      self.dim() != 2 || self.size(0) != self.size(1)) {
    std::ostringstream ss;
    ss << "det(" << self.type() << "{" << self.sizes() << "}): expected a 2D "
       << "square tensor of floating types";
    throw std::runtime_error(ss.str());
  }
  // check symmetric
  bool symmetric = self.equal(self.transpose(0, 1));

  auto svd = self.svd(true);
  auto sigma = std::get<1>(svd);
  auto u = std::get<0>(svd);
  auto v = std::get<2>(svd);
  auto det = sigma.prod();
  if (!symmetric) {
    auto qr = self.geqrf();
    auto a = std::get<0>(qr);
    auto tau = std::get<1>(qr);
    // non-zero values in tau represent Householder reflectors, which has -1 det
    int64_t num_reflectors = tau.nonzero().size(0);
    auto qr_det = a.diag().prod();
    if (num_reflectors % 2 == 1) {
      qr_det = -qr_det;
    }
    det = qr_det;  // QR is more stable than svd, so use it anyways
    if ((qr_det < 0).any() ^ (det < 0).any()) {  // if different sign
      u.narrow(1, 0, 1).mul_(-1);
      sigma.narrow(0, 0, 1).mul_(-1);
    }
  }
  return std::make_tuple(det, u, sigma, v);
}

Tensor det(const Tensor& self) {
  return std::get<0>(self._det_with_svd());
}

static Tensor maybeSqueeze(const Tensor & tensor, int64_t dim_tensor1, int64_t dim_tensor2) {
  if (dim_tensor1 == 1) {
    return tensor.squeeze(-2);
  } else if (dim_tensor2 == 1) {
    return tensor.squeeze(-1);
  } else {
    return tensor;
  }
}

/*
Matrix product of two Tensors.
The behavior depends on the dimensionality of the Tensors as follows:
- If both Tensors are 1-dimensional, the dot product (scalar) is returned.
- If both arguments are 2-dimensional, the matrix-matrix product is returned.
- If the first argument is 1-dimensional and the second argument is 2-dimensional,
  a 1 is prepended to its dimension for the purpose of the matrix multiply.
  After the matrix multiply, the prepended dimension is removed.
- If the first argument is 2-dimensional and the second argument is 1-dimensional,
  the matrix-vector product is returned.
- If both arguments are at least 1-dimensional and at least one argument is
  N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
  argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
  batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
  1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
  The non-matrix (i.e. batch) dimensions are broadcasted (and thus
  must be broadcastable).  For example, if tensor1 is a (j x 1 x n x m) Tensor
  and tensor2 is a (k x m x p) Tensor, the returned tensor will be an (j x k x n x p) Tensor.
*/
Tensor matmul(const Tensor & tensor1, const Tensor & tensor2) {
  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    return tensor1.dot(tensor2);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    return tensor1.mv(tensor2);
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    return tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    return tensor1.mm(tensor2);
  } else if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
    // optimization: use mm instead of bmm by folding tensor1's batch into
    // its leading matrix dimension.

    Tensor t2 = dim_tensor2 == 1 ? tensor2.unsqueeze(-1) : tensor2;
    auto size1 = tensor1.sizes();
    auto size2 = t2.sizes();
    std::vector<int64_t> output_size;
    output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
    output_size.insert(output_size.end(), size2.end() - 1, size2.end());

    // fold the batch into the first dimension
    Tensor t1 = tensor1.contiguous().view({-1, size1[size1.size() - 1]});

    auto output = t1.mm(t2).view(output_size);
    if (dim_tensor2 == 1) {
      output = output.squeeze(-1);
    }
    return output;
  } else if ((dim_tensor1 >= 1 && dim_tensor2 >= 1) && (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
    // We are multiplying b1 x n x m1 by x2 x m2 x p (where b1 can be a list);
    // we track m1 vs m2 separately even though they must match for nicer error messages
    int64_t n = dim_tensor1 > 1 ? tensor1.size(-2) : 1;
    int64_t m1 = tensor1.size(-1);
    IntList batch_tensor1(tensor1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0));
    int64_t m2 = dim_tensor2 > 1 ? tensor2.size(-2) : 1;
    int64_t p = tensor2.size(-1);
    IntList batch_tensor2(tensor2.sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0));

    // expand the batch portion (i.e. cut off matrix dimensions and expand rest)
    std::vector<int64_t> expand_batch_portion = infer_size(batch_tensor1, batch_tensor2);

    std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
    tensor1_expand_size.insert(tensor1_expand_size.end(), {n, m1});

    std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
    tensor2_expand_size.insert(tensor2_expand_size.end(), {m2, p});

    int expand_batch_product = std::accumulate(expand_batch_portion.begin(), expand_batch_portion.end(),
                                               1, std::multiplies<int64_t>());

    std::vector<int64_t> tensor1_bmm_view({expand_batch_product});
    tensor1_bmm_view.insert(tensor1_bmm_view.end(), {n, m1});

    std::vector<int64_t> tensor2_bmm_view({expand_batch_product});
    tensor2_bmm_view.insert(tensor2_bmm_view.end(), {m2, p});

    // flatten expanded batches
    Tensor tensor1_expanded = tensor1.expand(tensor1_expand_size).contiguous().view(tensor1_bmm_view);
    Tensor tensor2_expanded = tensor2.expand(tensor2_expand_size).contiguous().view(tensor2_bmm_view);

    Tensor output = tensor1_expanded.bmm(tensor2_expanded);

    // reshape batches back into result
    std::vector<int64_t> total_expansion(expand_batch_portion);
    total_expansion.insert(total_expansion.end(), {n, p});
    return maybeSqueeze(output.view(total_expansion), dim_tensor1, dim_tensor2);
  }

  runtime_error("both arguments to matmul need to be at least 1D, but they are %dD and %dD",
                dim_tensor1, dim_tensor2);

}

}
}
