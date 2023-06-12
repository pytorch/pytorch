#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/SpmmReduceKernel.h>
#include <ATen/native/cpu/ReduceUtils.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_native.h>
#endif

namespace at { namespace native {

namespace {

template <typename scalar_t, typename index_t, ReductionType reduce>
void spmm_reduce_kernel_impl(
    const Tensor& out,
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& values,
    const Tensor& other_) {

  int64_t nnz = values.numel();
  if (nnz == 0) {
    return;
  }

  auto other = other_.contiguous();

  // access `crow_indices`, `col_indices` and `values` via TessorAccessor
  scalar_t* out_data = out.data_ptr<scalar_t>();
  auto csr_data = crow_indices.accessor<index_t, 1>();
  auto col_data = col_indices.accessor<index_t, 1>();
  auto val_data = values.accessor<scalar_t, 1>();
  scalar_t* other_data = other.data_ptr<scalar_t>();

  int64_t M = crow_indices.numel() - 1;
  int64_t K = other.size(-1);

  using Vec = vec::Vectorized<scalar_t>;
  utils::parallel_sparse_csr(csr_data, M, nnz, [&](int64_t begin, int64_t end) {
    int64_t row_start, row_end, c;
    for (const auto m : c10::irange(begin, end)) {
      row_start = csr_data[m];
      row_end = csr_data[m + 1];

      scalar_t* out_ptr = out_data + m * K;

      constexpr int64_t kVecSize = Vec::size();
      constexpr int64_t kVLEN = kVecSize * 4;
      constexpr int64_t CHUNK_SIZE = 16;

      // step 1: reinit the output row for reduce type 'amax' and 'amin'
      int64_t count = row_end - row_start;
      if (count != 0) {
        init<scalar_t, reduce>(out_ptr, K, /*include_self*/false);
      }

      // step 2: reduce, do blocking on rowwise to reduce write memory bandwidth
      for (int64_t e0 = row_start; e0 < row_end; e0 += CHUNK_SIZE) {
        int64_t e1 = std::min(e0 + CHUNK_SIZE, row_end);

        int64_t k = 0;
        for (; k < K - (K % kVLEN); k += kVLEN) {
          Vec out_vec0 = Vec::loadu(out_ptr + k);
          Vec out_vec1 = Vec::loadu(out_ptr + k + kVecSize);
          Vec out_vec2 = Vec::loadu(out_ptr + k + kVecSize * 2);
          Vec out_vec3 = Vec::loadu(out_ptr + k + kVecSize * 3);
          for (const auto e : c10::irange(e0, e1)) {
            c = col_data[e];
            scalar_t val = val_data[e];
            scalar_t* other_ptr = other_data + c * K + k;

            out_vec0 = update<Vec, reduce>(out_vec0, Vec::loadu(other_ptr) * Vec(val));
            out_vec1 = update<Vec, reduce>(out_vec1, Vec::loadu(other_ptr + kVecSize) * Vec(val));
            out_vec2 = update<Vec, reduce>(out_vec2, Vec::loadu(other_ptr + kVecSize * 2) * Vec(val));
            out_vec3 = update<Vec, reduce>(out_vec3, Vec::loadu(other_ptr + kVecSize * 3) * Vec(val));
          }
          out_vec0.store(out_ptr + k);
          out_vec1.store(out_ptr + k + kVecSize);
          out_vec2.store(out_ptr + k + kVecSize * 2);
          out_vec3.store(out_ptr + k + kVecSize * 3);
        }
        for (; k < K - (K % kVecSize); k += kVecSize) {
          Vec out_vec = Vec::loadu(out_ptr + k);
          for (const auto e : c10::irange(e0, e1)) {
            c = col_data[e];
            scalar_t val = val_data[e];
            scalar_t* other_ptr = other_data + c * K;
            out_vec = update<Vec, reduce>(out_vec, Vec::loadu(other_ptr + k) * Vec(val));
          }
          out_vec.store(out_ptr + k);
        }
        for (; k < K; k++) {
          scalar_t out_val = out_ptr[k];
          for (const auto e : c10::irange(e0, e1)) {
            c = col_data[e];
            scalar_t val = val_data[e];
            scalar_t* other_ptr = other_data + c * K;
            out_val = update<scalar_t, reduce>(out_val, other_ptr[k] * val);
          }
          out_ptr[k] = out_val;
        }
      }

      // step 3: finalize
      write<scalar_t, reduce>(out_ptr, count, K);
    }
  });
}

// update both val and arg, used for `amin` and `amax`
// it is a little troublesome to vectorize it since `scalar_t` and `index_t`
// might have different vector length, for example, each vector holds 8 floats
// and 4 int64_t.
template <typename scalar_t, typename index_t, ReductionType reduce>
inline void update_with_index(scalar_t *val, scalar_t new_val, index_t *arg, index_t new_arg) {
  if ((reduce == ReductionType::MIN && new_val < *val) ||
      (reduce == ReductionType::MAX && new_val > *val) ||
      at::_isnan<scalar_t>(new_val)) {
    *val = new_val;
    *arg = new_arg;
  }
}

template <typename scalar_t, typename index_t, ReductionType reduce>
void spmm_reduce_arg_kernel_impl(
    const Tensor& out,
    const Tensor& arg_out,
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& values,
    const Tensor& other_) {

  TORCH_CHECK(reduce == ReductionType::MAX || reduce == ReductionType::MIN);
  int64_t nnz = values.numel();
  if (nnz == 0) {
    return;
  }

  auto other = other_.contiguous();

  scalar_t* out_data = out.data_ptr<scalar_t>();
  index_t* arg_out_data = arg_out.data_ptr<index_t>();
  auto csr_data = crow_indices.accessor<index_t, 1>();
  auto col_data = col_indices.accessor<index_t, 1>();
  auto val_data = values.accessor<scalar_t, 1>();
  scalar_t* other_data = other.data_ptr<scalar_t>();

  int64_t M = crow_indices.numel() - 1;
  int64_t K = other.size(-1);

  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    int64_t row_start, row_end, c;
    for (const auto m : c10::irange(begin, end)) {
      row_start = csr_data[m];
      row_end = csr_data[m + 1];

      scalar_t* out_ptr = out_data + m * K;
      index_t* arg_out_ptr = arg_out_data + m * K;

      if (row_end != row_start) {
        init<scalar_t, reduce>(out_ptr, K, /*include_self*/false);
        for (const auto e : c10::irange(row_start, row_end)) {
          c = col_data[e];
          scalar_t val = val_data[e];

          scalar_t* other_ptr = other_data + c * K;
          for (const auto k : c10::irange(K)) {
            update_with_index<scalar_t, index_t, reduce>(
                &out_ptr[k], val *  other_ptr[k], &arg_out_ptr[k], index_t(e));
          };
        }
      }
    }
  });
}

template <typename scalar_t, typename index_t, ReductionType reduce>
void spmm_reduce_backward_input_kernel_impl(
    const Tensor& grad_self,
    const Tensor& grad_out_,
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& other_,
    const Tensor& row_indices) {

  int64_t nnz = grad_self._nnz();
  if (nnz == 0) {
    return;
  }

  auto grad_out = grad_out_.contiguous();
  auto other = other_.contiguous();

  auto values = grad_self.values();
  auto grad_values_data = values.accessor<scalar_t, 1>();
  scalar_t* grad_out_data = grad_out.data_ptr<scalar_t>();
  auto crow_data = crow_indices.accessor<index_t, 1>();
  auto col_data = col_indices.accessor<index_t, 1>();
  scalar_t* other_data = other.data_ptr<scalar_t>();
  auto row_data = row_indices.accessor<index_t, 1>();

  int64_t K = grad_out.size(1);

  using Vec = vec::Vectorized<vec::vec_scalar_t<scalar_t>>;
  at::parallel_for(0, nnz, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      index_t row = row_data[i], col = col_data[i];

      scalar_t val = vec::map2_reduce_all<scalar_t>(
          [](Vec x, Vec y) { return x * y; },
          [](Vec x, Vec y) { return x + y; },
          other_data + col * K,
          grad_out_data + row * K,
          K);

      if (reduce == ReductionType::MEAN) {
        index_t row_start = crow_data[row], row_end = crow_data[row + 1];
        val /= (row_end - row_start);
      }

      grad_values_data[i] = val;
    }
  });
}

// backward for reduce type 'amax' or 'amin'
template <typename scalar_t, typename index_t>
void spmm_reduce_backward_input_arg_kernel_impl(
    const Tensor& grad_self,
    const Tensor& grad_out_,
    const Tensor& col_indices,
    const Tensor& other_,
    const Tensor& arg_out_) {

  int64_t nnz = grad_self._nnz();
  if (nnz == 0) {
    return;
  }

  auto grad_out = grad_out_.contiguous();
  auto other = other_.contiguous();
  auto arg_out = arg_out_.contiguous();

  auto grad_values = grad_self.values();
  auto grad_values_data = grad_values.accessor<scalar_t, 1>();
  scalar_t* grad_out_data = grad_out.data_ptr<scalar_t>();
  auto col_data = col_indices.accessor<index_t, 1>();
  scalar_t* other_data = other.data_ptr<scalar_t>();
  index_t* arg_out_data = arg_out.data_ptr<index_t>();

  int64_t M = grad_out.size(0);
  int64_t K = grad_out.size(1);
  auto grad = at::empty({M, K}, grad_out.options());
  scalar_t* grad_data = grad.mutable_data_ptr<scalar_t>();

  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    for (const auto m : c10::irange(begin, end)) {
      scalar_t* grad_out_ptr = grad_out_data + m * K;
      scalar_t* grad_ptr = grad_data + m * K;
      index_t* arg_out_ptr = arg_out_data + m * K;

      for (const auto k : c10::irange(K)) {
        if (arg_out_ptr[k] == index_t(nnz)) {
          grad_ptr[k] = scalar_t(0);
        } else {
          // collect weight at max/min indices
          index_t col = col_data[arg_out_data[m * K + k]];
          grad_ptr[k] = other_data[col * K + k] * grad_out_ptr[k];
        }
      }
    }
  });

  // scatter_add, consider to parallel this with atomic
  for (const auto i : c10::irange(M * K)) {
    index_t ind = arg_out_data[i];
    if (ind != index_t(nnz)) {
      grad_values_data[ind] += grad_data[i];
    }
  }
}

template <typename scalar_t, typename index_t>
void spmm_reduce_normalize_values_kernel_impl(
    const Tensor& normalized_values,
    const Tensor& values,
    const Tensor& crow_indices,
    const Tensor& row_indices) {

  int64_t nnz = values.numel();
  if (nnz == 0) {
    return;
  }

  auto normalized_values_data = normalized_values.accessor<scalar_t, 1>();
  auto values_data = values.accessor<scalar_t, 1>();
  auto crow_data = crow_indices.accessor<index_t, 1>();
  auto row_data = row_indices.accessor<index_t, 1>();

  at::parallel_for(0, nnz, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      index_t row = row_data[i];
      index_t row_start = crow_data[row], row_end = crow_data[row + 1];
      // Note that when the row index row is listed in row_indices,
      // then crow_indices[row+1] > crow_indices[row] holds
      normalized_values_data[i] = values_data[i] / (row_end - row_start);
    }
  });
}

template <typename scalar_t, typename index_t>
void spmm_reduce_backward_other_arg_kernel_impl(
    const Tensor& grad_other,
    const Tensor& grad_out_,
    const Tensor& col_indices,
    const Tensor& values,
    const Tensor& arg_out_) {

  int64_t nnz = values.numel();
  if (nnz == 0) {
    return;
  }

  auto grad_out = grad_out_.contiguous();
  auto arg_out = arg_out_.contiguous();

  scalar_t* grad_other_data = grad_other.data_ptr<scalar_t>();
  scalar_t* grad_out_data = grad_out.data_ptr<scalar_t>();
  auto col_data = col_indices.accessor<index_t, 1>();
  auto values_data = values.accessor<scalar_t, 1>();
  index_t* arg_out_data = arg_out.data_ptr<index_t>();

  int64_t M = grad_out.size(0);
  int64_t K = grad_out.size(1);
  auto grad = at::empty({M, K}, grad_out.options());
  scalar_t* grad_data = grad.mutable_data_ptr<scalar_t>();

  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    for (const auto m : c10::irange(begin, end)) {
      scalar_t* grad_out_ptr = grad_out_data + m * K;
      scalar_t* grad_ptr = grad_data + m * K;
      index_t* arg_out_ptr = arg_out_data + m * K;

      for (const auto k : c10::irange(K)) {
        if (arg_out_ptr[k] == index_t(nnz)) {
          grad_ptr[k] = scalar_t(0);
        } else {
          grad_ptr[k] = values_data[arg_out_ptr[k]] * grad_out_ptr[k];
        }
      }
    }
  });

  // scatter_add, consider to parallel this with atomic
  for (const auto m : c10::irange(M)) {
    for (const auto k : c10::irange(K)) {
      index_t ind = arg_out_data[m * K + k];
      if (ind != index_t(nnz)) {
        index_t col = col_data[ind];
        grad_other_data[col * K + k] += grad_data[m * K + k];
      }
    }
  }
}

void spmm_reduce_kernel(
    const Tensor& out,
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& values,
    const Tensor& other,
    ReductionType reduce_op) {
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, values.scalar_type(), "spmm_reduce_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "spmm_reduce_indices", [&]() {
      AT_DISPATCH_REDUCTION_TYPES(reduce_op, [&]() {
        spmm_reduce_kernel_impl<scalar_t, index_t, reduce>(
            out, crow_indices, col_indices, values, other);
      });
    });
  });
}

void spmm_reduce_arg_kernel(
    const Tensor& out,
    const Tensor& arg_out,
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& values,
    const Tensor& other,
    ReductionType reduce_op) {
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, values.scalar_type(), "spmm_reduce_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "spmm_reduce_indices", [&]() {
      AT_DISPATCH_REDUCTION_TYPES(reduce_op, [&]() {
        spmm_reduce_arg_kernel_impl<scalar_t, index_t, reduce>(
            out, arg_out, crow_indices, col_indices, values, other);
      });
    });
  });
}

void spmm_reduce_backward_input_kernel(
    const Tensor& grad_self,
    const Tensor& grad_out,
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& other,
    const Tensor& row_indices,
    ReductionType reduce_op) {
  TORCH_CHECK(reduce_op == ReductionType::SUM || reduce_op == ReductionType::MEAN);
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, other.scalar_type(), "spmm_reduce_backward_input_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "spmm_reduce_backward_input_indices", [&]() {
      AT_DISPATCH_REDUCTION_TYPES(reduce_op, [&]() {
        spmm_reduce_backward_input_kernel_impl<scalar_t, index_t, reduce>(
            grad_self, grad_out, crow_indices, col_indices, other, row_indices);
      });
    });
  });
}

void spmm_reduce_backward_input_arg_kernel(
    const Tensor& grad_self,
    const Tensor& grad_out,
    const Tensor& col_indices,
    const Tensor& other,
    const Tensor& arg_out,
    ReductionType reduce_op) {
  TORCH_CHECK(reduce_op == ReductionType::MAX || reduce_op == ReductionType::MIN);
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, other.scalar_type(), "spmm_reduce_backward_input_arg_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "spmm_reduce_backward_input_arg_indices", [&]() {
      spmm_reduce_backward_input_arg_kernel_impl<scalar_t, index_t>(
          grad_self, grad_out, col_indices, other, arg_out);
    });
  });
}

void spmm_reduce_normalize_values_kernel(
    const Tensor& normalized_values,
    const Tensor& values,
    const Tensor& crow_indices,
    const Tensor& row_indices) {
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, values.scalar_type(), "spmm_reduce_normalize_values_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(crow_indices.scalar_type(), "spmm_reduce_normalize_values_indices", [&]() {
      spmm_reduce_normalize_values_kernel_impl<scalar_t, index_t>(
          normalized_values, values, crow_indices, row_indices);
    });
  });
}

void spmm_reduce_backward_other_kernel(
    const Tensor& grad_other,
    const Tensor& grad_out,
    const Tensor& crow_indices,
    const Tensor& values,
    const Tensor& row_indices,
    const Tensor& ccol_indices,
    const Tensor& csr2csc,
    ReductionType reduce_op) {
  TORCH_CHECK(reduce_op == ReductionType::SUM || reduce_op == ReductionType::MEAN);
  // need to permute row_indices to CSC order
  auto row = row_indices.index_select(0, csr2csc);

  Tensor val;
  if (reduce_op == ReductionType::MEAN) {
    // for reduce type "mean", need to normalize the values
    // with rowcount for each of the nonzero element.
    Tensor normalized_values = at::empty(values.sizes(), values.options());
    spmm_reduce_normalize_values_kernel(normalized_values, values, crow_indices, row_indices);
    val = normalized_values.index_select(0, csr2csc);
  } else {
    val = values.index_select(0, csr2csc);
  }

  spmm_reduce_kernel(grad_other, ccol_indices, row, val, grad_out, ReductionType::SUM);
}

void spmm_reduce_backward_other_arg_kernel(
    const Tensor& grad_other,
    const Tensor& grad_out,
    const Tensor& col_indices,
    const Tensor& values,
    const Tensor& arg_out,
    ReductionType reduce_op) {
  TORCH_CHECK(reduce_op == ReductionType::MAX || reduce_op == ReductionType::MIN);
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, values.scalar_type(), "spmm_reduce_backward_other_arg_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "spmm_reduce_backward_other_arg_indices", [&]() {
      spmm_reduce_backward_other_arg_kernel_impl<scalar_t, index_t>(
          grad_other, grad_out, col_indices, values, arg_out);
    });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(spmm_reduce_stub, &spmm_reduce_kernel);
REGISTER_DISPATCH(spmm_reduce_arg_stub, &spmm_reduce_arg_kernel);
REGISTER_DISPATCH(spmm_reduce_backward_input_stub, &spmm_reduce_backward_input_kernel);
REGISTER_DISPATCH(spmm_reduce_backward_input_arg_stub, &spmm_reduce_backward_input_arg_kernel);
REGISTER_DISPATCH(spmm_reduce_backward_other_stub, &spmm_reduce_backward_other_kernel);
REGISTER_DISPATCH(spmm_reduce_backward_other_arg_stub, &spmm_reduce_backward_other_arg_kernel);

}} // at::native
