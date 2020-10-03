#pragma once

#include <c10/core/ScalarType.h>
#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/TensorUtils.h>
#include <limits>
#include <sstream>
#include <cstring>

namespace at { namespace native {

/*
 * Clones a Tensor so that the following conditions hold:
 * If we think of a Tensor of having size (B, M, N), where B is any number
 * of batch dimensions, then:
 * - Each (M, N) matrix is in column major form
 * - Let Tensor P have size (B, M, N) and Q have size (B, M', N').
 *   Then when laid out in memory, the M by N matrix starting at
 *   P.data_ptr()[b * M * N] is of the same corresponding batch as the M' by N'
 *   matrix starting at Q.data_ptr()[b * M' * N'].
 */
static inline Tensor cloneBatchedColumnMajor(const Tensor& src) {
  // If src is already in batched column major format, then
  // this will be efficient (no reordering of the data will occur)
  // because the first transpose will make the tensor contiguous,
  // and cloning a contiguous tensor is fast.
  auto result = src.transpose(-2, -1).clone(at::MemoryFormat::Contiguous);
  result.transpose_(-2, -1);
  return result;
}

/*
 * Given batches of matrices with arbitrary batch dim,
 * computes the number of batches.
 */
static inline int64_t batchCount(const Tensor& batched_matrices) {
  int64_t result = 1;
  for (int64_t i = 0; i < batched_matrices.ndimension() - 2; i++) {
    result *= batched_matrices.size(i);
  }
  return result;
}

// Computes the number of elements of a matrix in a batched matrix tensor
static inline int64_t matrixStride(const Tensor& batched_matrices) {
  return batched_matrices.size(-1) * batched_matrices.size(-2);
}

// Returns the epsilon value for floating types except half
static inline double _get_epsilon(const ScalarType& sc_type) {
  switch (sc_type) {
    case at::ScalarType::Float:
      return static_cast<double>(std::numeric_limits<float>::epsilon());
    case at::ScalarType::Double:
      return std::numeric_limits<double>::epsilon();
    default:
      AT_ERROR("This function doesn't handle types other than float and double");
  }
}

// Validates input shapes and devices
// for linear solve methods (solve, cholesky_solve, lu_solve, triangular_solve)
static inline void linearSolveCheckInputs(const Tensor& self, const Tensor& A, const char* name) {
  TORCH_CHECK(self.device() == A.device(),
              "Expected b and A to be on the same device, but found b on ",
              self.device(), " and A on ", A.device(), " instead.");

  TORCH_CHECK(A.size(-1) == A.size(-2),
              "A must be batches of square matrices, "
              "but they are ", A.size(-1), " by ", A.size(-2), " matrices");

  TORCH_CHECK(A.size(-1) == self.size(-2),
              "Incompatible matrix sizes for ", name, ": each A "
              "matrix is ", A.size(-1), " by ", A.size(-1),
              " but each b matrix is ", self.size(-2), " by ", self.size(-1));
}

// Validates input shapes for operations on batches of square matrices (inverse, cholesky, symeig)
static inline void squareCheckInputs(const Tensor& self) {
  TORCH_CHECK(self.dim() >= 2, "Tensor of matrices must have at least 2 dimensions. ");
  TORCH_CHECK(self.size(-1) == self.size(-2),
              "A must be batches of square matrices, "
              "but they are ", self.size(-1), " by ", self.size(-2), " matrices");
}

/*
 * Given a vector of int64_t infos, obtained after a batch operations,
 * this function checks if the computation over all these batches has been
 * successful (info = 0) or not, and report in case of the latter.
 */
static inline void batchCheckErrors(std::vector<int64_t>& infos, const char* name, bool allow_singular=false) {
  for (size_t i = 0; i < infos.size(); i++) {
    auto info = infos[i];
    if (info < 0) {
      AT_ERROR(name, ": For batch ", i, ": Argument ", -info, " has illegal value");
    } else if (info > 0) {
      if (strstr(name, "svd")) {
        AT_ERROR(name, ": the updating process of SBDSDC did not converge (error: ", info, ")");
      } else if (strstr(name, "symeig")) {
        AT_ERROR(name, ": For batch ", i, ": the algorithm failed to converge; ", info,
                 " off-diagonal elements of an intermediate tridiagonal form did not converge to zero.");
      } else if (!allow_singular) {
        AT_ERROR(name, ": For batch ", i, ": U(", info, ",", info, ") is zero, singular U.");
      }
    }
  }
}

/*
 * This is an overloaded case of the previous function for a tensor of infos.
 */
static inline void batchCheckErrors(const Tensor& infos, const char* name, bool allow_singular=false) {
  auto batch_size = infos.numel();
  auto infos_cpu = infos.to(at::kCPU);
  auto infos_data = infos_cpu.data_ptr<int>();
  for (int64_t i = 0; i < batch_size; i++) {
    auto info = infos_data[i];
    if (info < 0) {
      AT_ERROR(name, ": For batch ", i, ": Argument ", -info, " has illegal value");
    } else if (!allow_singular && info > 0) {
      AT_ERROR(name, ": For batch ", i, ": U(", info, ",", info, ") is zero, singular U.");
    }
  }
}

/*
 * Given a info int, obtained after a single operation, this function check if the computation
 * has been successful (info = 0) or not, and report in case of the latter.
 */
static inline void singleCheckErrors(int64_t info, const char* name, bool allow_singular=false) {
  if (info < 0) {
    AT_ERROR(name, ": Argument ", -info, " has illegal value");
  } else if (info > 0) {
    if (strstr(name, "svd")) {
      AT_ERROR(name, ": the updating process of SBDSDC did not converge (error: ", info, ")");
    } else if (strstr(name, "symeig")) {
      AT_ERROR(name, ": the algorithm failed to converge; ", info,
               " off-diagonal elements of an intermediate tridiagonal form did not converge to zero.");
    } else if (!allow_singular) {
      AT_ERROR(name, ": U(", info, ",", info, ") is zero, singular U.");
    }
  }
}

// Checks if all the Tensors in a TensorList are of the same dimensions
static inline void checkAllSameDim(TensorList tensors, int64_t dim) {
  for (auto &t : tensors) {
    TORCH_CHECK(t.dim() == dim, "Tensor dimension is ", t.dim(), ", expected ", dim, " instead.");
  }
}

static inline std::tuple<Tensor,Tensor> _linalg_broadcast_batch_dims(const Tensor& arg1, const Tensor& arg2, const char* name) {
  linearSolveCheckInputs(arg1, arg2, name);

  // broadcast the batch dimensions of arg1 and arg2.
  IntArrayRef arg1_batch_sizes(arg1.sizes().data(), arg1.ndimension() - 2);
  IntArrayRef arg2_batch_sizes(arg2.sizes().data(), arg2.ndimension() - 2);
  std::vector<int64_t> expand_batch_portion = infer_size(arg1_batch_sizes, arg2_batch_sizes);

  std::vector<int64_t> arg1_expand_size({expand_batch_portion});
  arg1_expand_size.insert(arg1_expand_size.end(), { arg1.size(-2), arg1.size(-1) });

  std::vector<int64_t> arg2_expand_size({expand_batch_portion});
  arg2_expand_size.insert(arg2_expand_size.end(), { arg2.size(-2), arg2.size(-1) });

  Tensor arg1_broadcasted  = arg1.expand(arg1_expand_size);
  Tensor arg2_broadcasted = arg2.expand(arg2_expand_size);
  return std::make_tuple(arg1_broadcasted, arg2_broadcasted);
}

// Return a permutation with the given axes moved to the end.
static inline Tensor _move_to_end(const Tensor& self, IntArrayRef axes) {
  const std::vector<int64_t> a = axes.vec();
  const int64_t ndim = self.ndimension();
  std::vector<int64_t> perm;

  for (int64_t i = 0; i < ndim; i++) {
    auto it = std::find(a.begin(), a.end(), i);
    if (it == a.end()) {
       perm.push_back(i);
    }
  }
  for (auto i : a) {
    perm.push_back(i);
  }

  TORCH_CHECK((int64_t)perm.size() == ndim,
    "duplicate or invalid axis in 'dim' argument for tensor with ndim==", ndim);

  return self.permute(perm);
}

// Function to compute sizes, strides and the extra columns for the Q matrix in the QR Decomposition
static inline std::tuple<std::vector<int64_t>,
                         std::vector<int64_t>,
                         int64_t> _compute_geometry_for_Q(const Tensor& input, bool some) {
  int64_t m = input.size(-2), n = input.size(-1);
  int64_t n_columns_q;

  // We need to compute the required size of Q based on the `some` option
  auto q_sizes = input.sizes().vec();
  if (!some && m > n) {
    q_sizes[input.dim() - 1] = m;
    n_columns_q = m;
  } else {
    q_sizes[input.dim() - 1] = n;
    n_columns_q = std::min(m, n);
  }
  auto q_strides = at::detail::defaultStrides(q_sizes);

  // Q should be a column-major or a batch of column-major matrices
  // ... x m x n will have strides: ...., n, 1
  // We require: ...., 1, m
  q_strides[input.dim() - 1] = m;
  q_strides[input.dim() - 2] = 1;
  return std::make_tuple(q_sizes, q_strides, n_columns_q);
}

// Function to generate empty tensors of required size, strides and dtype for the SVD operation
static inline std::tuple<Tensor, Tensor, Tensor> _create_U_S_VT(const Tensor& input, bool some, bool compute_uv) {
  auto sizes = input.sizes().vec();
  int64_t m = input.size(-2), n = input.size(-1);

  sizes[input.dim() - 1] = (compute_uv && some) ? std::min(m, n) : m;
  auto strides = at::detail::defaultStrides(sizes);
  // U should be a column-major or a batch of column-major matrices
  // ... x m x ucol will have strides: ...., ucol, 1
  // We require: ...., 1, m
  strides[input.dim() - 1] = m;
  strides[input.dim() - 2] = 1;

  Tensor U_empty;
  if (!input.is_cuda()) {
    U_empty = at::empty_strided(sizes, strides, input.options());
  } else {
    // NB: U_empty is an empty tensor created on the CPU intentionally, because magma_(d/s)gesdd
    // (which is the driver routine for the divide and conquer SVD operation)
    // takes in arrays on the CPU as input. This routine is a hybrid CPU-GPU routine that
    // moves the inputs between devices internally.
    U_empty = at::empty_strided(sizes, strides, input.options().device(at::kCPU));
  }

  sizes[input.dim() - 2] = n;
  sizes[input.dim() - 1] = n;
  // VT should be a row-major or a batch of row-major matrices
  Tensor VT_empty;
  if (!input.is_cuda()) {
    VT_empty = at::empty(sizes, input.options());
  } else {
    // NB: VT_empty is an empty tensor created on the CPU intentionally, because magma_(d/s)gesdd
    // (which is the driver routine for the divide and conquer SVD operation)
    // takes in arrays on the CPU as input. This routine is a hybrid CPU-GPU routine that
    // moves the inputs between devices internally.
    VT_empty = at::empty(sizes, input.options().device(at::kCPU));
  }

  sizes.pop_back();
  sizes[input.dim() - 2] = std::min(m, n);
  Tensor S_empty;
  ScalarType dtype = toValueType(typeMetaToScalarType(input.dtype()));
  if (!input.is_cuda()) {
    S_empty = at::empty(sizes, input.options().dtype(dtype));
  } else {
    // NB: S_empty is an empty tensor created on the CPU intentionally, because magma_(d/s)gesdd
    // (which is the driver routine for the divide and conquer SVD operation)
    // takes in arrays on the CPU as input. This routine is a hybrid CPU-GPU routine that
    // moves the inputs between devices internally. 
    S_empty = at::empty(sizes, input.options().dtype(dtype).device(at::kCPU));
  }
  return std::tuple<Tensor, Tensor, Tensor>(U_empty, S_empty, VT_empty);
}

// Function used instead of .to so that the original strides are retained
// .to doesn't retain strides and make the output tensor contiguous
static inline Tensor same_stride_to(const Tensor& original_tensor, const at::TensorOptions& options) {
  auto strided_to = at::empty_strided(original_tensor.sizes(),
                                      original_tensor.strides(),
                                      options);
  strided_to.copy_(original_tensor);
  return strided_to;
}

// Creates a dimension permutation array that can be given to `at::permute()`, which will shift
// the two specified dimensions to the end of a tensor, without changing the order of
// the other dimensions. `dim1` will be placed at the very end, and `dim0` will be
// placed just to the left of it.
//
// For instance, given a 4-D tensor, dimensions 1 and 3 can be shifted to the end by
// calling `create_dim_backshift_permutation(1, 3, 4)`. The resulting vector will
// be `vec(0, 2, 1, 3)`.
static inline std::vector<int64_t> create_dim_backshift_permutation(int64_t dim0, int64_t dim1, int64_t ndim) {
  TORCH_CHECK(
    (dim0 != dim1) && (dim0 < ndim) && (dim0 >= 0) && (dim1 < ndim) && (dim1 >= 0),
    "duplicate or invalid dimensions");
  std::vector<int64_t> permutation(ndim);
  int64_t cur_permuted_dim = 0;
  for (int64_t dim_ind = 0; dim_ind < ndim; dim_ind++) {
    if ((dim_ind != dim0) && (dim_ind != dim1)) {
      permutation[cur_permuted_dim++] = dim_ind;
    }
  }
  permutation[cur_permuted_dim++] = dim0;
  permutation[cur_permuted_dim] = dim1;
  return permutation;
}

// Creates a dimension permutation array that can be given to `at::permute()`, which
// will reverse a given permutation.
// The reverse permutation array is created by swapping the indices and their
// associated values from the given permutation array.
static inline std::vector<int64_t> create_reverse_permutation(std::vector<int64_t> permutation) {
  int64_t ndim = permutation.size();
  std::vector<int64_t> reverse_permutation(ndim);
  for (int64_t dim_ind = 0; dim_ind < ndim; dim_ind++) {
    reverse_permutation[permutation[dim_ind]] = dim_ind;
  }
  return reverse_permutation;
}

}}  // namespace at::native
