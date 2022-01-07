#include <ATen/native/LinearAlgebraUtils.h>

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <ATen/ExpandUtils.h>
#include <ATen/TensorUtils.h>

#include <cctype>
#include <cstring>
#include <limits>
#include <sstream>
#include <type_traits>

namespace at { namespace native {

// Transforms TransposeType into the BLAS / LAPACK format
char to_blas(TransposeType trans) {
  switch (trans) {
    case TransposeType::Transpose: return 'T';
    case TransposeType::NoTranspose: return 'N';
    case TransposeType::ConjTranspose: return 'C';
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}

c10::MaybeOwned<Tensor> expect_resolved_conj(const Tensor& tensor) {
  if (tensor.is_conj()) {
    return c10::MaybeOwned<Tensor>::owned(tensor.resolve_conj());
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
}

template<class Vec>
Vec contiguous_strides_template(const IntArrayRef sizes, const bool f_contig=false) {
  static_assert(std::is_same<IntArrayRef::value_type, typename Vec::value_type>::value,
                "Incompatible integral type of sizes and strides");
  // f_contig chooses between the strides of a batch of Fortran (F-contiguous) and C-contiguous matrices
  using Int = IntArrayRef::value_type;
  constexpr auto one = Int{1};
  const auto n = sizes.size();
  if (n == 0) {
    return Vec{};
  } else if (n == 1) {
    // Use initializer-list to initialize the vector
    return Vec{one};
  }
  // Now we have a matrix or batch of matrices
  auto strides = Vec(n);
  const auto last_idx = n - 1;
  const auto snd_last_idx = n - 2;
  // We'll fill the first two strides afterwards, otherwise the first step
  // in the for loop is wrong
  strides[snd_last_idx] = std::max<int64_t>(sizes[last_idx], one);
  for (int i = snd_last_idx - 1; i >= 0; --i) {
    strides[i] = strides[i + 1] * std::max(sizes[i + 1], one);
  }
  strides[last_idx] = f_contig ? std::max(sizes[snd_last_idx], one) : one;
  if (f_contig) {
    // We filled the wrong stride before so we correct it
    strides[snd_last_idx] = one;
  }
  return strides;
}

DimVector contiguous_strides(const IntArrayRef sizes, const bool f_contig) {
  return contiguous_strides_template<DimVector>(sizes, f_contig);
}

std::vector<int64_t> contiguous_strides_vec(const IntArrayRef sizes, const bool f_contig) {
  return contiguous_strides_template<std::vector<int64_t>>(sizes, f_contig);
}

/*
 * Clones a Tensor so that the following conditions hold:
 * If we think of a Tensor of having size (B, M, N), where B is any number
 * of batch dimensions, then:
 * - Each (M, N) matrix is in column major form
 * - Let Tensor P have size (B, M, N) and Q have size (B, M', N').
 *   Then when laid out in memory, the M by N matrix starting at
 *   P.data_ptr()[B * M * N] is of the same corresponding batch as the M' by N'
 *   matrix starting at Q.data_ptr()[B * M' * N'].
 */
Tensor cloneBatchedColumnMajor(const Tensor& src) {
  // If src is already in batched column major format, then
  // this will be efficient (no reordering of the data will occur)
  // because the first transpose will make the tensor contiguous,
  // and cloning a contiguous tensor is fast.
  auto result = src.mT().clone(at::MemoryFormat::Contiguous);
  result.transpose_(-2, -1);
  return result;
}

/*
 * contig chooses between C-contig (true) and F-contig (false)
 */
c10::MaybeOwned<Tensor> borrow_else_clone(const bool cond, const Tensor& borrow, const Tensor& clone, const bool contig) {
  return cond ? c10::MaybeOwned<Tensor>::borrowed(borrow)
              : c10::MaybeOwned<Tensor>::owned(contig ? clone.clone(MemoryFormat::Contiguous)
                                                      : cloneBatchedColumnMajor(clone));
}

/*
 * This method is designed to be a faster alternative to
 * `cloneBatchedColumnMajor` with some additional features,
 * namely:
 * 1. It uses `copy` instead of `clone` which could be much faster.
 * 2. `nrows` parameter used to create inputs with the number of rows larger
 *  than the original input, which is required for some LAPACK/MAGMA methods.
 * 3. `desired_batch_size` is used to create copies with the batch size
 *  which is either the original batch size of the input, or its larger
 *  broadcasted shape.
 */
Tensor copyBatchedColumnMajor(const Tensor& src, int64_t nrows, c10::optional<IntArrayRef> desired_batch_sizes) {
  nrows = (nrows == -1) ? src.size(-2) : nrows;
  auto copy_sizes = desired_batch_sizes.has_value()
    ? desired_batch_sizes.value().vec()
    : IntArrayRef(src.sizes().data(), src.dim() - 2).vec();
  copy_sizes.insert(copy_sizes.end(), {nrows, src.size(-1)});
  const auto copy_strides = contiguous_strides(copy_sizes, /*f-contig*/true);
  auto copy = at::empty_strided(copy_sizes, copy_strides, src.options());
  copy.narrow(-2, 0, src.size(-2)).copy_(src);
  return copy;
}

/*
 * Given batches of matrices with arbitrary batch dim,
 * computes the number of batches.
 */
int64_t batchCount(const Tensor& batched_matrices) {
  int64_t result = 1;
  for (int64_t i = 0; i < batched_matrices.ndimension() - 2; i++) {
    result *= batched_matrices.size(i);
  }
  return result;
}

// Computes the number of elements of a matrix in a batched matrix tensor
int64_t matrixStride(const Tensor& batched_matrices) {
  return batched_matrices.size(-1) * batched_matrices.size(-2);
}

// Returns the epsilon value for floating types except half
double _get_epsilon(const ScalarType& sc_type) {
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
void linearSolveCheckInputs(const Tensor& self, const Tensor& A, const char* name) {
  TORCH_CHECK(self.device() == A.device(),
              "Expected b and A to be on the same device, but found b on ",
              self.device(), " and A on ", A.device(), " instead.");

  TORCH_CHECK(self.scalar_type() == A.scalar_type(),
              "Expected b and A to have the same dtype, but found b of type ",
              self.scalar_type(), " and A of type ", A.scalar_type(), " instead.");

  TORCH_CHECK(A.size(-1) == A.size(-2),
              "A must be batches of square matrices, "
              "but they are ", A.size(-1), " by ", A.size(-2), " matrices");

  TORCH_CHECK(A.size(-1) == self.size(-2),
              "Incompatible matrix sizes for ", name, ": each A "
              "matrix is ", A.size(-1), " by ", A.size(-1),
              " but each b matrix is ", self.size(-2), " by ", self.size(-1));
}

// Validates input shapes for operations on batches of square matrices (inverse, cholesky, symeig, eig)
void squareCheckInputs(const Tensor& self, const char* const f_name) {
  TORCH_CHECK(self.dim() >= 2, f_name, ": The input tensor must have at least 2 dimensions.");
  TORCH_CHECK(self.size(-1) == self.size(-2),
              f_name,
              ": A must be batches of square matrices, "
              "but they are ", self.size(-1), " by ", self.size(-2), " matrices");
}

void checkFloatingOrComplex(const Tensor& t, const char* const f_name) {
  TORCH_CHECK((at::isFloatingType(t.scalar_type()) || at::isComplexType(t.scalar_type())),
              f_name, ": Expected a floating point or complex tensor as input. Got ", toString(t.scalar_type()));
}

void singleCheckErrors(int64_t info, const char* name, int64_t batch_id) {
  std::string batch_string{""};
  if (batch_id >= 0) {
    batch_string = ": (Batch element " + std::to_string(batch_id) + ")";
  }
  if (info < 0) {
    TORCH_INTERNAL_ASSERT(false, name, batch_string,
        ": Argument ", -info, " has illegal value. Most certainly there is a bug in the implementation calling the backend library.");
  } else if (info > 0) {
    if (strstr(name, "inv")) {
      // inv, inverse, cholesky_inverse, etc.
      TORCH_CHECK_LINALG(false, name, batch_string,
          ": The diagonal element ", info, " is zero, the inversion could not be completed because the input matrix is singular.");
    } else if (strstr(name, "solve")) {
      // solve, linalg_solve, cholesky_solve, etc.
      TORCH_CHECK_LINALG(false, name, batch_string,
          ": The diagonal element ", info, " is zero, the solve could not be completed because the input matrix is singular.");
    } else if (strstr(name, "cholesky")) {
      TORCH_CHECK_LINALG(false, name, batch_string,
          ": The factorization could not be completed because the input is not positive-definite (the leading minor of order ", info, " is not positive-definite).");
    } else if (strstr(name, "svd")) {
      TORCH_CHECK_LINALG(false, name, batch_string,
          ": The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated singular values (error code: ", info, ").");
    } else if (strstr(name, "eig") || strstr(name, "syevd")) {
      TORCH_CHECK_LINALG(false, name, batch_string,
          ": The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues (error code: ", info, ").");
    } else if (strstr(name, "lstsq")) {
      TORCH_CHECK_LINALG(false, name, batch_string,
          ": The least squares solution could not be computed because the input matrix does not have full rank (error code: ", info, ").");
    } else if (strstr(name, "lu_factor")) {
      TORCH_CHECK_LINALG(false, name, batch_string,
          ": U[", info, ",", info, "] is zero and using it on lu_solve would result in a division by zero. "
          "If you still want to perform the factorization, consider calling linalg.lu(A, pivot) or "
          "linalg.lu_factor_ex(A, pivot)");
    } else {
      TORCH_INTERNAL_ASSERT(false, name, ": Unknown error code: ", info, ".");
    }
  }
}

/*
 * Given a vector of int64_t infos, obtained after a batch operations,
 * this function checks if the computation over all these batches has been
 * successful (info = 0) or not, and report in case of the latter.
 */
void batchCheckErrors(const std::vector<int64_t>& infos, const char* name) {
  for (size_t i = 0; i < infos.size(); i++) {
    auto info = infos[i];
    singleCheckErrors(info, name, i);
  }
}

/*
 * This is an overloaded case of the previous function for a tensor of infos.
 */
void batchCheckErrors(const Tensor& infos, const char* name) {
  auto infos_cpu = infos.to(at::kCPU);
  auto infos_data = infos_cpu.data_ptr<int>();
  for (int64_t i = 0; i < infos.numel(); i++) {
    auto info = infos_data[i];
    singleCheckErrors(info, name, i);
  }
}

// Checks if all the Tensors in a TensorList are of the same dimensions
void checkAllSameDim(TensorList tensors, int64_t dim) {
  for (auto &t : tensors) {
    TORCH_CHECK(t.dim() == dim, "Tensor dimension is ", t.dim(), ", expected ", dim, " instead.");
  }
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> _linalg_broadcast_batch_dims(const Tensor& arg1, const Tensor& arg2) {
  // broadcast the batch dimensions of arg1 and arg2.
  IntArrayRef arg1_batch_sizes(arg1.sizes().data(), arg1.ndimension() - 2);
  IntArrayRef arg2_batch_sizes(arg2.sizes().data(), arg2.ndimension() - 2);
  std::vector<int64_t> expand_batch_portion = infer_size(arg1_batch_sizes, arg2_batch_sizes);

  std::vector<int64_t> arg1_expand_size({expand_batch_portion});
  arg1_expand_size.insert(arg1_expand_size.end(), { arg1.size(-2), arg1.size(-1) });

  std::vector<int64_t> arg2_expand_size({expand_batch_portion});
  arg2_expand_size.insert(arg2_expand_size.end(), { arg2.size(-2), arg2.size(-1) });
  return std::make_tuple(std::move(arg1_expand_size), std::move(arg2_expand_size));
}

std::tuple<Tensor,Tensor> _linalg_broadcast_batch_dims(const Tensor& arg1, const Tensor& arg2, const char* name) {
  // If there's no name we assume we don't want to check the errors
  if (name != nullptr) {
    linearSolveCheckInputs(arg1, arg2, name);
  }

  std::vector<int64_t> arg1_expand_size, arg2_expand_size;
  std::tie(arg1_expand_size, arg2_expand_size) = at::native::_linalg_broadcast_batch_dims(arg1, arg2);

  auto arg1_broadcasted  = arg1_expand_size == arg1.sizes() ? arg1 : arg1.expand(arg1_expand_size);
  auto arg2_broadcasted  = arg2_expand_size == arg2.sizes() ? arg2 : arg2.expand(arg2_expand_size);
  return std::make_tuple(arg1_broadcasted, arg2_broadcasted);
}

std::vector<int64_t> broadcast_batch_size(const Tensor& t1, const Tensor& t2, int64_t n_batch_dims) {
  IntArrayRef t1_batch_sizes(t1.sizes().data(), n_batch_dims);
  IntArrayRef t2_batch_sizes(t2.sizes().data(), n_batch_dims);
  auto broadcasted_batch_sizes = infer_size(t1_batch_sizes, t2_batch_sizes);
  return broadcasted_batch_sizes;
}

// Return a permutation with the given axes moved to the end.
Tensor _move_to_end(const Tensor& self, IntArrayRef axes) {
  const std::vector<int64_t> a = axes.vec();
  const int64_t ndim = self.ndimension();
  std::vector<int64_t> perm;

  for (const auto i : c10::irange(ndim)) {
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

// parse the "mode" param in linalg_qr: return a tuple of bools (compute_q, reduced)
std::tuple<bool, bool> _parse_qr_mode(c10::string_view mode) {
  bool compute_q;
  bool reduced;
  if (mode == "reduced") {
    compute_q = true;
    reduced = true;
  } else if (mode == "complete") {
    compute_q = true;
    reduced = false;
  } else if (mode == "r") {
    compute_q = false;
    reduced = true; // this is actually irrelevant in this mode
  } else {
      TORCH_CHECK(false, "qr received unrecognized mode '", mode,
                  "' but expected one of 'reduced' (default), 'r', or 'complete'");
  }
  return std::make_tuple(compute_q, reduced);
}

// Function to compute sizes, strides and the extra columns for the Q matrix in the QR Decomposition
std::tuple<std::vector<int64_t>,
                         std::vector<int64_t>,
                         int64_t> _compute_geometry_for_Q(const Tensor& input, bool reduced) {
  int64_t m = input.size(-2), n = input.size(-1);
  int64_t n_columns_q;

  // We need to compute the required size of Q based on the `reduced` option
  auto q_sizes = input.sizes().vec();
  if (!reduced && m > n) {
    q_sizes[input.dim() - 1] = m;
    n_columns_q = m;
  } else {
    q_sizes[input.dim() - 1] = n;
    n_columns_q = std::min(m, n);
  }
  auto q_strides = contiguous_strides_vec(q_sizes, /*f-contig*/true);
  return std::make_tuple(q_sizes, q_strides, n_columns_q);
}

// Function to generate empty tensors of required size, strides and dtype for the SVD operation
std::tuple<Tensor, Tensor, Tensor> _create_U_S_VT(const Tensor& input, bool some, bool compute_uv,
    const bool svd_use_cusolver) {

  // U, S, VT are initialized as empty tensors.
  // For CPU LAPACK and GPU MAGMA backend, the tensors are initialized on CPU.
  // For GPU cuSOLVER backend, the tensors are initialized on GPU.
  const auto usvt_device = svd_use_cusolver ? at::kCUDA : at::kCPU;

  auto sizes = input.sizes().vec();
  int64_t m = input.size(-2), n = input.size(-1);

  sizes[input.dim() - 1] = some ? std::min(m, n) : m;
  const auto u_strides = contiguous_strides(sizes, /*f-contig*/true);

  // cuSOLVER's gesvdjBatched fails with illegal memory access and
  // cuSOLVER's gesvdj fails with CUSOLVER_STATUS_EXECUTION_FAILED
  // if matrices for U and VT are not allocated
  // even though the result of computation is not used we need to allocate this memory

  Tensor U_empty = (compute_uv || svd_use_cusolver)
      ? at::empty_strided(sizes, u_strides, input.options().device(usvt_device))
      : at::empty({0}, input.options().device(usvt_device));

  // VT should be a column-major or a batch of column-major matrices
  sizes[input.dim() - 2] = some ? std::min(m, n) : n;
  sizes[input.dim() - 1] = n;
  const auto vt_strides = contiguous_strides(sizes, /*f-contig*/!svd_use_cusolver);
  Tensor VT_empty = (compute_uv || svd_use_cusolver)
      ? at::empty_strided(sizes, vt_strides, input.options().device(usvt_device))
      : at::empty({0}, input.options().device(usvt_device));

  // U and VT might not get filled in this case
  if (!some && compute_uv && input.numel() == 0) {
    U_empty.zero_();
    VT_empty.zero_();
    // make U and VT an identity matrix, because they should be orthogonal
    U_empty.diagonal(0, -2, -1).fill_(1);
    VT_empty.diagonal(0, -2, -1).fill_(1);
  }

  sizes.pop_back();
  sizes[input.dim() - 2] = std::min(m, n);
  ScalarType dtype = toValueType(input.scalar_type());
  Tensor S_empty = at::empty(sizes, input.options().dtype(dtype).device(usvt_device));

  return std::tuple<Tensor, Tensor, Tensor>(U_empty, S_empty, VT_empty);
}

// Function used instead of .to so that the original strides are retained
// .to doesn't retain strides and make the output tensor contiguous
Tensor same_stride_to(const Tensor& original_tensor, const at::TensorOptions& options) {
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
std::vector<int64_t> create_dim_backshift_permutation(int64_t dim0, int64_t dim1, int64_t ndim) {
  TORCH_CHECK(
    (dim0 != dim1) && (dim0 < ndim) && (dim0 >= 0) && (dim1 < ndim) && (dim1 >= 0),
    "duplicate or invalid dimensions");
  std::vector<int64_t> permutation(ndim);
  int64_t cur_permuted_dim = 0;
  for (const auto dim_ind : c10::irange(ndim)) {
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
std::vector<int64_t> create_reverse_permutation(std::vector<int64_t> permutation) {
  int64_t ndim = permutation.size();
  std::vector<int64_t> reverse_permutation(ndim);
  for (const auto dim_ind : c10::irange(ndim)) {
    reverse_permutation[permutation[dim_ind]] = dim_ind;
  }
  return reverse_permutation;
}

// Compute R-work array size for MAGMA/LAPACK cgesdd/zgesdd
// See https://github.com/Reference-LAPACK/lapack/blob/122506cd8b6ce050a200920c3d4c0b153b150fd8/SRC/cgesdd.f#L186
int64_t computeLRWorkDim(const char jobz, int64_t m, int64_t n) {
  auto mn = std::min(m, n);
  auto mx = std::max(m, n);
  if (jobz == 'N') {
#ifdef __APPLE__
    // According to `vecLib.framework/Headers/clapack.h` Accelerate.framework is based on LAPACK 3.2.1
    return 7 * mn;
#else
    // These setting is valid for on LAPACK 3.6+
    return 5 * mn;
#endif
  }
  if (mx > 10 * mn) {
    return 5 * mn * mn + 5 * mn;
  }
  return std::max(5 * mn * mn + 5 * mn, 2 * mx * mn + 2 * mn * mn + mn);
}

// This function checks whether the uplo argument input is valid
// Allowed strings are "u", "U", "l", "L"
void checkUplo(const c10::string_view uplo) {
  // To use std::toupper safely with plain chars (or signed chars), the argument should first be converted to unsigned char
  char uplo_uppercase = static_cast<char>(std::toupper(static_cast<unsigned char>(uplo[0])));
  TORCH_CHECK(uplo.size() == 1 && (uplo_uppercase == 'U' || uplo_uppercase == 'L'),
    "Expected UPLO argument to be 'L' or 'U', but got ", uplo);
}

void checkSameDevice(const std::string& fn_name, Tensor result, Tensor input, const std::string& result_name) {
  TORCH_CHECK(
      result.device() == input.device(),
      fn_name,
      ": Expected ", result_name, " and input tensors to be on the same device, but got ",
      result_name, " on ", result.device(), " and input on ", input.device());
}

// Check the dtype of result and input tensors (for _out variants).
// Most linear algebra functions have the same dtype for input and output
// (either floating or complex type input), so we can check whether input's dtype can be casted to result's dtype.
// According to https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-does-out-work-in-pytorch
// c10::canCast is used for checking the "safe copy" dtype requirements.
void checkLinalgCompatibleDtype(const std::string& fn_name, Tensor result, Tensor input, const std::string& result_name) {
  bool can_cast = c10::canCast(input.scalar_type(), result.scalar_type());
  TORCH_CHECK(
      can_cast,
      fn_name,
      ": Expected ", result_name, " to be safely castable from ", input.scalar_type(), " dtype, but got ",
      result_name, " with dtype ", result.scalar_type());
}

// Alternatively, we can check whether the specific expected output type (result_type) can be safely casted to out tensor dtype (out_type)
void checkLinalgCompatibleDtype(const std::string& fn_name, ScalarType out_type, ScalarType result_type, const std::string& out_name) {
  bool can_cast = c10::canCast(result_type, out_type);
  TORCH_CHECK(
      can_cast,
      fn_name,
      ": Expected ", out_name, " to be safely castable from ", result_type, " dtype, but got ",
      out_name, " with dtype ", out_type);
}

void checkNotComplexTolerance(const Tensor& tol, const c10::string_view f_name, const c10::string_view tol_name) {
  TORCH_CHECK(!at::isComplexType(tol.scalar_type()),
              f_name, ": ", tol_name, " tensor of complex type is not supported. Got ", tol.scalar_type());
}

/*
  Two types of 'other' tensors are supported when solving
  a system of linear equations matmul(input, x) = other:
  * 1-dimensional (1D) tensor or batch of 1D tensors (vector case)
  * 2-dimensional (2D) tensor or batch of 2D tensors (matrix case).
  The original torch.solve supported only the matrix case, while NumPy works for both cases.
  For the batched input we need to be able to distinguish them.
  Let input.shape = (batch_dimensions, m, n), then 'other' is of vector type if other.shape == (batch_dimensions, m).
  This rule is compatible with NumPy, see https://github.com/numpy/numpy/blob/v1.20.0/numpy/linalg/linalg.py#L384-L389
*/
bool linalg_solve_is_vector_rhs(const Tensor& input, const Tensor& other) {
  auto expected_batched_rhs_shape = IntArrayRef(input.sizes().data(), input.dim() - 1); // input.shape[:-1]
  bool vector_case = other.dim() == 1 || (input.dim() - 1 == other.dim() && other.sizes().equals(expected_batched_rhs_shape));
  return vector_case;
}

bool is_blas_compatible_column_major_order(const Tensor& input) {
  IntArrayRef input_strides = input.strides();
  IntArrayRef input_sizes = input.sizes();
  auto ndim = input.dim();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ndim == 2);
  auto leading_dimension = input_strides[ndim - 1];
  auto rows = input_sizes[ndim - 2];
  return (input_strides[ndim - 2] == 1) && (leading_dimension >= std::max<int64_t>(1, rows));
}

bool is_blas_compatible_row_major_order(const Tensor& input) {
  IntArrayRef input_strides = input.strides();
  IntArrayRef input_sizes = input.sizes();
  auto ndim = input.dim();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ndim == 2);
  auto leading_dimension = input_strides[ndim - 2];
  auto cols = input_sizes[ndim - 1];
  return (input_strides[ndim - 1] == 1) && (leading_dimension >= std::max<int64_t>(1, cols));
}

}}  // namespace at::native
