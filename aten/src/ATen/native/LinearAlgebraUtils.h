#include "ATen/ATen.h"
#include <limits>

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
  auto result = src.transpose(-2, -1).clone();
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

// Validates input shapes for gesv
static inline void gesvCheckInputs(const Tensor& self, const Tensor& A) {
  AT_CHECK(A.size(-1) == A.size(-2),
           "A must be batches of square matrices, "
           "but they are ", A.size(-1), " by ", A.size(-2), " matrices");

  AT_CHECK(A.size(-1) == self.size(-2),
           "Incompatible matrix sizes for matmul: each A "
           "matrix is ", A.size(-1), " by ", A.size(-1),
           " but each b matrix is ", self.size(-2), " by ", self.size(-1));
}

// Validates input shapes for inverse
static inline void inverseCheckInputs(const Tensor& self) {
  AT_CHECK(self.size(-1) == self.size(-2),
           "A must be batches of square matrices, "
           "but they are ", self.size(-1), " by ", self.size(-2), " matrices");
}

/*
 * Given a vector of int64_t infos, obtained after a batch operations,
 * this function checks if the computation over all these batches has been
 * successful (info = 0) or not, and report in case of the latter.
 */ 
static inline void batchCheckErrors(std::vector<int64_t>& infos, const char* name) {
  for (size_t i = 0; i < infos.size(); i++) {
    auto info = infos[i];
    if (info < 0) {
      AT_ERROR(name, ": For batch ", i, ": Argument ", -info, " has illegal value");
    } else if (info > 0) {
      AT_ERROR(name, ": For batch ", i, ": U(", info, ",", info, ") is zero, singular U.");
    }
  }
}

#define GENERATE_LINALG_HELPER_1_ARGS(NAME, ARG, BACKEND) \
  Tensor _##NAME##_helper_##BACKEND(const Tensor& ARG) { \
    std::vector<int64_t> infos(batchCount(ARG), 0); \
    auto ARG##_working_copy = cloneBatchedColumnMajor(ARG); \
    AT_DISPATCH_FLOATING_TYPES(ARG.type(), #NAME, [&]{ \
      apply_##NAME<scalar_t>(ARG##_working_copy, infos); \
    }); \
    batchCheckErrors(infos, #NAME); \
    return ARG##_working_copy; \
  }

#define GENERATE_LINALG_HELPER_2_ARGS(NAME, ARG1, ARG2, BACKEND) \
  std::tuple<Tensor, Tensor> _##NAME##_helper_##BACKEND(const Tensor& ARG1, const Tensor& ARG2) { \
    std::vector<int64_t> infos(batchCount(ARG1), 0); \
    auto ARG1##_working_copy = cloneBatchedColumnMajor(ARG1); \
    auto ARG2##_working_copy = cloneBatchedColumnMajor(ARG2); \
    AT_DISPATCH_FLOATING_TYPES(ARG1.type(), #NAME, [&]{ \
      apply_##NAME<scalar_t>(ARG1##_working_copy, ARG2##_working_copy, infos); \
    }); \
    batchCheckErrors(infos, #NAME); \
    return std::tuple<Tensor, Tensor>(ARG1##_working_copy, ARG2##_working_copy); \
  }

// Checks if all the Tensors in a TensorList are of the same dimensions
static inline void checkAllSameDim(TensorList tensors, int64_t dim) {
  for (auto &t : tensors) {
    AT_CHECK(t.dim() == dim, "Tensor dimension is ", t.dim(), ", expected ", dim, " instead.");
  }
}

}}  // namespace at::native
