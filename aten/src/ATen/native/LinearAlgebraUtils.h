#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/irange.h>
#include <c10/util/Exception.h>
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/TensorIterator.h>
#include <limits>
#include <type_traits>
#include <sstream>
#include <cstring>
#include <cctype>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/zeros.h>
#endif

namespace at { namespace native {

static inline c10::MaybeOwned<Tensor> expect_resolved_conj(const Tensor& tensor) {
  if (tensor.is_conj()) {
    return c10::MaybeOwned<Tensor>::owned(tensor.resolve_conj());
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
}

template<class Vec>
static inline Vec contiguous_strides_template(const IntArrayRef sizes, const bool f_contig=false) {
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

static inline DimVector contiguous_strides(const IntArrayRef sizes, const bool f_contig=false) {
  return contiguous_strides_template<DimVector>(sizes, f_contig);
}

static inline std::vector<int64_t> contiguous_strides_vec(const IntArrayRef sizes, const bool f_contig=false) {
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
static inline Tensor cloneBatchedColumnMajor(const Tensor& src) {
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
static inline c10::MaybeOwned<Tensor> borrow_else_clone(const bool cond, const Tensor& borrow, const Tensor& clone, const bool contig) {
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
static inline Tensor copyBatchedColumnMajor(const Tensor& src, int64_t nrows = -1,
    c10::optional<IntArrayRef> desired_batch_sizes = c10::nullopt) {
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

// This function is designed to be used with linear algebra methods that minimize
// L(ax - b) = 0, where L is generally the identity map (`solve`, for example)
// or the L2 norm (`lstsq`).
// It is expected that `a` and `b` are contiguous tensors of column-major matrices
// (so that a.view({-1, a.size(-2), a.size(-1)}) succeeds, same for `b`),
// with the following additional properties:
//
// 1. a.dim() == b.dim()
// 2. a.shape[:-2] broadcasts over b.shape[:-2]
// 3. a.size(i) <= b.size(i) for i=0,..., a.dim() - 3 (only for batch dimensions)
//
// MAGMA/LAPACK modify tensor `a` in-place, and the main goal of this method
// is to be memory efficient, which means that if there exists an index i such that
// a.shape[i] < b.shape[i], 0 <= i <= a.dim() - 3,
// then instead of materializing copies of `a` in the broadcasted shape, we keep
// a buffer copy of `a` along with flags that check whether specific batch dimension
// indices for `a` were already accessed. If they were, we copy the data from the buffer
// into `a`. The number of copies does not exceed
// prod(max(a.shape[:-2], b.shape[:-2]) - a.shape[:-2] + 1)
// and this value is attained by tensors with non-empty batch dimensions.
//
// func_t `f` is a callable that is being supplied with
// scalar_t* a_working_ptr, scalar_t* b_working_ptr, int64_t a_linear_batch_idx.
// a_working_ptr and b_working_ptr can directly be passed to LAPACK/MAGMA routines,
// and a_linear_batch_idx is an index in the 3d representation which corresponds to
// the memory a_working_ptr points to, in other words:
// a_working_ptr == a.view({-1, a.size(-2), a.size(-1)}.select(0, a_linear_batch_idx).data_ptr<scalar_t>();
// a_linear_batch_idx is useful to store metadata related to `a`, such as, for example,
// its rank or singular values (see linalg_lstsq).
template<typename scalar_t, typename func_t>
void batch_iterator_with_broadcasting(const Tensor& a, const Tensor& b, const func_t& f) {
  IntArrayRef a_batch_sizes(a.sizes().data(), a.dim() - 2);
  IntArrayRef b_batch_sizes(b.sizes().data(), b.dim() - 2);

  auto a_linear_batch_idx = at::arange(batchCount(a)).view(a_batch_sizes);
  auto b_linear_batch_idx = at::arange(batchCount(b)).view(b_batch_sizes);

  TensorIterator iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_output(b_linear_batch_idx)
    .add_input(a_linear_batch_idx)
    .build();

  auto m = a.size(-2);
  auto n = a.size(-1);
  auto a_3d = a.view({batchCount(a), m, n});
  auto b_3d = b.view({batchCount(b), b.size(-2), b.size(-1)});

  auto a_broadcasts_over_b = (a_batch_sizes != b_batch_sizes);
  Tensor a_buffer, a_was_accessed, a_buffer_3d;
  std::function<void(int64_t)> check_if_copy_needed_for_a
    = [](int64_t a_curr_linear_batch_idx){};
  if (a_broadcasts_over_b) {
    a_buffer = at::empty_strided(a.sizes(), a.strides(), a.options())
      .copy_(a);
    a_was_accessed = at::zeros(batchCount(a), at::kBool);
    a_buffer_3d = a_buffer.view({batchCount(a), m, n});
    check_if_copy_needed_for_a = [&](int64_t a_curr_linear_batch_idx) {
      auto* a_was_accessed_flag = a_was_accessed
        .select(0, a_curr_linear_batch_idx)
        .data_ptr<bool>();
      if (!(*a_was_accessed_flag)) {
        *a_was_accessed_flag = true;
      }
      else {
        a_3d.select(0, a_curr_linear_batch_idx)
          .copy_(a_buffer_3d.select(0, a_curr_linear_batch_idx));
      }
    };
  }

  auto loop = [&](char** data, const int64_t* strides, int64_t nelems) {
    auto* b_batch_idx_ptr = data[0];
    auto* a_batch_idx_ptr = data[1];

    for (const auto elem : c10::irange(nelems)) {
      (void)elem; //Suppress unused variable warning
      auto b_curr_linear_batch_idx = *reinterpret_cast<int64_t*>(b_batch_idx_ptr);
      auto a_curr_linear_batch_idx = *reinterpret_cast<int64_t*>(a_batch_idx_ptr);

      check_if_copy_needed_for_a(a_curr_linear_batch_idx);

      auto* a_working_ptr = a_3d.select(0, a_curr_linear_batch_idx)
        .data_ptr<scalar_t>();
      auto* b_working_ptr = b_3d.select(0, b_curr_linear_batch_idx)
        .data_ptr<scalar_t>();
      f(a_working_ptr, b_working_ptr, a_curr_linear_batch_idx);

      b_batch_idx_ptr += strides[0];
      a_batch_idx_ptr += strides[1];
    }
  };
  iter.serial_for_each(loop, {0, batchCount(b)});
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
static inline void squareCheckInputs(const Tensor& self, const char* const f_name) {
  TORCH_CHECK(self.dim() >= 2, f_name, ": The input tensor must have at least 2 dimensions.");
  TORCH_CHECK(self.size(-1) == self.size(-2),
              f_name,
              ": A must be batches of square matrices, "
              "but they are ", self.size(-1), " by ", self.size(-2), " matrices");
}

static inline void checkFloatingOrComplex(const Tensor& t, const char* const f_name) {
  TORCH_CHECK((at::isFloatingType(t.scalar_type()) || at::isComplexType(t.scalar_type())),
              f_name, ": Expected a floating point or complex tensor as input. Got ", toString(t.scalar_type()));
}

/*
 * Given a info int, obtained after a single operation, this function check if the computation
 * has been successful (info = 0) or not, and report in case of the latter.
 */
static inline void singleCheckErrors(int64_t info, const c10::string_view name, int64_t batch_id=-1) {
  std::string batch_string{""};
  if (batch_id >= 0) {
    batch_string = ": (Batch element " + std::to_string(batch_id) + ")";
  }
  if (info < 0) {
    TORCH_INTERNAL_ASSERT(false, name, batch_string,
        ": Argument ", -info, " has illegal value. Most certainly there is a bug in the implementation calling the backend library.");
  } else if (info > 0) {
    if (name.find("inv") != name.npos) {
      // inv, inverse, cholesky_inverse, etc.
      TORCH_CHECK_LINALG(false, name, batch_string,
          ": The diagonal element ", info, " is zero, the inversion could not be completed because the input matrix is singular.");
    } else if (name.find("solve") != name.npos) {
      // solve, linalg_solve, cholesky_solve, etc.
      TORCH_CHECK_LINALG(false, name, batch_string,
          ": The diagonal element ", info, " is zero, the solve could not be completed because the input matrix is singular.");
    } else if (name.find("cholesky") != name.npos) {
      TORCH_CHECK_LINALG(false, name, batch_string,
          ": The factorization could not be completed because the input is not positive-definite (the leading minor of order ", info, " is not positive-definite).");
    } else if (name.find("svd") != name.npos) {
      TORCH_CHECK_LINALG(false, name, batch_string,
          ": The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated singular values (error code: ", info, ").");
    } else if (name.find("eig") || name.find("syevd")) {
      TORCH_CHECK_LINALG(false, name, batch_string,
          ": The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues (error code: ", info, ").");
    } else if (name.find("lstsq")) {
      TORCH_CHECK_LINALG(false, name, batch_string,
          ": The least squares solution could not be computed because the input matrix does not have full rank (error code: ", info, ").");
    } else if (name.find("lu_factor")) {
      TORCH_CHECK(false, name, batch_string,
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
static inline void batchCheckErrors(const std::vector<int64_t>& infos, const c10::string_view name) {
  for (size_t i = 0; i < infos.size(); i++) {
    auto info = infos[i];
    singleCheckErrors(info, name, i);
  }
}

/*
 * This is an overloaded case of the previous function for a tensor of infos.
 */
static inline void batchCheckErrors(const Tensor& infos, const c10::string_view name) {
  auto infos_cpu = infos.to(at::kCPU);
  auto infos_data = infos_cpu.data_ptr<int>();
  for (int64_t i = 0; i < infos.numel(); i++) {
    auto info = infos_data[i];
    singleCheckErrors(info, name, i);
  }
}

// Checks if all the Tensors in a TensorList are of the same dimensions
static inline void checkAllSameDim(TensorList tensors, int64_t dim) {
  for (auto &t : tensors) {
    TORCH_CHECK(t.dim() == dim, "Tensor dimension is ", t.dim(), ", expected ", dim, " instead.");
  }
}

static inline std::tuple<std::vector<int64_t>, std::vector<int64_t>> _linalg_broadcast_batch_dims(const Tensor& arg1, const Tensor& arg2) {
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

static inline std::tuple<Tensor,Tensor> _linalg_broadcast_batch_dims(const Tensor& arg1, const Tensor& arg2, const char* name) {
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

static inline std::vector<int64_t> broadcast_batch_size(const Tensor& t1, const Tensor& t2, int64_t n_batch_dims) {
  IntArrayRef t1_batch_sizes(t1.sizes().data(), n_batch_dims);
  IntArrayRef t2_batch_sizes(t2.sizes().data(), n_batch_dims);
  auto broadcasted_batch_sizes = infer_size(t1_batch_sizes, t2_batch_sizes);
  return broadcasted_batch_sizes;
}

// Return a permutation with the given axes moved to the end.
static inline Tensor _move_to_end(const Tensor& self, IntArrayRef axes) {
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
static inline std::tuple<bool, bool> _parse_qr_mode(c10::string_view mode) {
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
static inline std::tuple<std::vector<int64_t>,
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
static inline std::tuple<Tensor, Tensor, Tensor> _create_U_S_VT(const Tensor& input, bool some, bool compute_uv,
    const bool svd_use_cusolver=false) {

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
static inline std::vector<int64_t> create_reverse_permutation(std::vector<int64_t> permutation) {
  int64_t ndim = permutation.size();
  std::vector<int64_t> reverse_permutation(ndim);
  for (const auto dim_ind : c10::irange(ndim)) {
    reverse_permutation[permutation[dim_ind]] = dim_ind;
  }
  return reverse_permutation;
}

// Compute R-work array size for MAGMA/LAPACK cgesdd/zgesdd
// See https://github.com/Reference-LAPACK/lapack/blob/122506cd8b6ce050a200920c3d4c0b153b150fd8/SRC/cgesdd.f#L186
static inline int64_t computeLRWorkDim(const char jobz, int64_t m, int64_t n) {
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
static inline void checkUplo(const c10::string_view uplo) {
  // To use std::toupper safely with plain chars (or signed chars), the argument should first be converted to unsigned char
  char uplo_uppercase = static_cast<char>(std::toupper(static_cast<unsigned char>(uplo[0])));
  TORCH_CHECK(uplo.size() == 1 && (uplo_uppercase == 'U' || uplo_uppercase == 'L'),
    "Expected UPLO argument to be 'L' or 'U', but got ", uplo);
}

static inline void checkSameDevice(const std::string& fn_name, Tensor result, Tensor input, const std::string& result_name = "result") {
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
static inline void checkLinalgCompatibleDtype(const std::string& fn_name, Tensor result, Tensor input, const std::string& result_name = "result") {
  bool can_cast = c10::canCast(input.scalar_type(), result.scalar_type());
  TORCH_CHECK(
      can_cast,
      fn_name,
      ": Expected ", result_name, " to be safely castable from ", input.scalar_type(), " dtype, but got ",
      result_name, " with dtype ", result.scalar_type());
}

// Alternatively, we can check whether the specific expected output type (result_type) can be safely casted to out tensor dtype (out_type)
static inline void checkLinalgCompatibleDtype(const std::string& fn_name, ScalarType out_type, ScalarType result_type, const std::string& out_name = "result") {
  bool can_cast = c10::canCast(result_type, out_type);
  TORCH_CHECK(
      can_cast,
      fn_name,
      ": Expected ", out_name, " to be safely castable from ", result_type, " dtype, but got ",
      out_name, " with dtype ", out_type);
}

static inline void checkNotComplexTolerance(const Tensor& tol, const c10::string_view f_name, const c10::string_view tol_name) {
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
static inline bool linalg_solve_is_vector_rhs(const Tensor& input, const Tensor& other) {
  auto expected_batched_rhs_shape = IntArrayRef(input.sizes().data(), input.dim() - 1); // input.shape[:-1]
  bool vector_case = other.dim() == 1 || (input.dim() - 1 == other.dim() && other.sizes().equals(expected_batched_rhs_shape));
  return vector_case;
}

static inline bool is_blas_compatible_column_major_order(const Tensor& input) {
  IntArrayRef input_strides = input.strides();
  IntArrayRef input_sizes = input.sizes();
  auto ndim = input.dim();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ndim == 2);
  auto leading_dimension = input_strides[ndim - 1];
  auto rows = input_sizes[ndim - 2];
  return (input_strides[ndim - 2] == 1) && (leading_dimension >= std::max<int64_t>(1, rows));
}

static inline bool is_blas_compatible_row_major_order(const Tensor& input) {
  IntArrayRef input_strides = input.strides();
  IntArrayRef input_sizes = input.sizes();
  auto ndim = input.dim();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ndim == 2);
  auto leading_dimension = input_strides[ndim - 2];
  auto cols = input_sizes[ndim - 1];
  return (input_strides[ndim - 1] == 1) && (leading_dimension >= std::max<int64_t>(1, cols));
}

}}  // namespace at::native
