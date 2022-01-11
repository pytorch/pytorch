#pragma once

#include <ATen/ATen.h>
#include <c10/util/irange.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/string_view.h>
#include <c10/util/Optional.h>
#include <ATen/core/DimVector.h>
#include <ATen/native/TensorIterator.h>

#include <functional>
#include <string>
#include <tuple>
#include <vector>

namespace at { namespace native {

// Used as an interface between the different BLAS-like libraries
enum class TransposeType {
  NoTranspose,
  Transpose,
  ConjTranspose,
};

char to_blas(TransposeType trans);

TORCH_API c10::MaybeOwned<Tensor> expect_resolved_conj(const Tensor& tensor);

TORCH_API DimVector contiguous_strides(const IntArrayRef sizes, const bool f_contig=false);

TORCH_API std::vector<int64_t> contiguous_strides_vec(const IntArrayRef sizes, const bool f_contig=false);

TORCH_API Tensor cloneBatchedColumnMajor(const Tensor& src);

c10::MaybeOwned<Tensor> borrow_else_clone(const bool cond, const Tensor& borrow, const Tensor& clone, const bool contig);

Tensor copyBatchedColumnMajor(const Tensor& src, int64_t nrows = -1,
    c10::optional<IntArrayRef> desired_batch_sizes = c10::nullopt);

TORCH_API int64_t batchCount(const Tensor& batched_matrices);

TORCH_API int64_t matrixStride(const Tensor& batched_matrices);

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
double _get_epsilon(const ScalarType& sc_type);

void linearSolveCheckInputs(const Tensor& self, const Tensor& A, const char* name);

void squareCheckInputs(const Tensor& self, const char* const f_name);

void checkFloatingOrComplex(const Tensor& t, const char* const f_name);

TORCH_API void singleCheckErrors(int64_t info, const char* name, int64_t batch_id=-1);

TORCH_API void batchCheckErrors(const std::vector<int64_t>& infos, const char* name);

TORCH_API void batchCheckErrors(const Tensor& infos, const char* name);

void checkAllSameDim(TensorList tensors, int64_t dim);

std::tuple<std::vector<int64_t>, std::vector<int64_t>> _linalg_broadcast_batch_dims(const Tensor& arg1, const Tensor& arg2);

std::tuple<Tensor,Tensor> _linalg_broadcast_batch_dims(const Tensor& arg1, const Tensor& arg2, const char* name);

std::vector<int64_t> broadcast_batch_size(const Tensor& t1, const Tensor& t2, int64_t n_batch_dims);

Tensor _move_to_end(const Tensor& self, IntArrayRef axes);

TORCH_API std::tuple<bool, bool> _parse_qr_mode(c10::string_view mode);

TORCH_API std::tuple<std::vector<int64_t>,
                     std::vector<int64_t>,
                     int64_t> _compute_geometry_for_Q(const Tensor& input, bool reduced);

TORCH_API std::tuple<Tensor, Tensor, Tensor> _create_U_S_VT(const Tensor& input, bool some, bool compute_uv, const bool svd_use_cusolver=false);

TORCH_API Tensor same_stride_to(const Tensor& original_tensor, const at::TensorOptions& options);

std::vector<int64_t> create_dim_backshift_permutation(int64_t dim0, int64_t dim1, int64_t ndim);

std::vector<int64_t> create_reverse_permutation(std::vector<int64_t> permutation);

TORCH_API int64_t computeLRWorkDim(const char jobz, int64_t m, int64_t n);

void checkUplo(const c10::string_view uplo);

void checkSameDevice(const std::string& fn_name, Tensor result, Tensor input, const std::string& result_name = "result");

void checkLinalgCompatibleDtype(const std::string& fn_name, Tensor result, Tensor input, const std::string& result_name = "result");

void checkLinalgCompatibleDtype(const std::string& fn_name, ScalarType out_type, ScalarType result_type, const std::string& out_name = "result");

void checkNotComplexTolerance(const Tensor& tol, const c10::string_view f_name, const c10::string_view tol_name);

TORCH_API bool linalg_solve_is_vector_rhs(const Tensor& input, const Tensor& other);

TORCH_API bool is_blas_compatible_column_major_order(const Tensor& input);

TORCH_API bool is_blas_compatible_row_major_order(const Tensor& input);

}}  // namespace at::native
