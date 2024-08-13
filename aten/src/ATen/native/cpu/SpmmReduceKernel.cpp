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
#include <ATen/OpMathType.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

namespace {

template <typename scalar_t, typename index_t, ReductionType reduce>
inline void _update(at::opmath_type<scalar_t>* out_ptr, int64_t e, int64_t c, const scalar_t val, const scalar_t* other_data, int64_t K) {
  using opmath_t = at::opmath_type<scalar_t>;
  using Vec = vec::Vectorized<scalar_t>;
  using aVec = VecType<scalar_t>;
  constexpr int64_t kVecSize = Vec::size();
  constexpr int64_t kVLEN = kVecSize * 4;

  int64_t k = 0;
  aVec val_vec = aVec((opmath_t)val);
  const scalar_t* other_ptr = other_data + c * K;

  for (; k < K - (K % kVLEN); k += kVLEN) {
    aVec out_vec0 = aVec::loadu(out_ptr + k);
    aVec out_vec1 = aVec::loadu(out_ptr + k + kVecSize);
    aVec out_vec2 = aVec::loadu(out_ptr + k + kVecSize * 2);
    aVec out_vec3 = aVec::loadu(out_ptr + k + kVecSize * 3);

    out_vec0 = update<aVec, reduce>(out_vec0, aVec::loadu(other_ptr + k) * val_vec);
    out_vec1 = update<aVec, reduce>(out_vec1, aVec::loadu(other_ptr + k + kVecSize) * val_vec);
    out_vec2 = update<aVec, reduce>(out_vec2, aVec::loadu(other_ptr + k + kVecSize * 2) * val_vec);
    out_vec3 = update<aVec, reduce>(out_vec3, aVec::loadu(other_ptr + k + kVecSize * 3) * val_vec);

    out_vec0.store(out_ptr + k);
    out_vec1.store(out_ptr + k + kVecSize);
    out_vec2.store(out_ptr + k + kVecSize * 2);
    out_vec3.store(out_ptr + k + kVecSize * 3);
  }
  for (; k < K - (K % kVecSize); k += kVecSize) {
    aVec out_vec = aVec::loadu(out_ptr + k);
    out_vec = update<aVec, reduce>(out_vec, aVec::loadu(other_ptr + k) * val_vec);
    out_vec.store(out_ptr + k);
  }
  for (; k < K; k++) {
    opmath_t out_val = opmath_t(out_ptr[k]);
    out_val = update<opmath_t, reduce>(out_val, opmath_t(other_ptr[k]) * opmath_t(val));
    out_ptr[k] = out_val;
  }
}

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

  // access `crow_indices`, `col_indices` and `values` via TensorAccessor
  scalar_t* out_data = out.data_ptr<scalar_t>();
  auto csr_data = crow_indices.accessor<const index_t, 1>();
  auto col_data = col_indices.accessor<const index_t, 1>();
  auto val_data = values.accessor<const scalar_t, 1>();
  const scalar_t* other_data = other.const_data_ptr<scalar_t>();

  int64_t M = crow_indices.numel() - 1;
  int64_t K = other.size(-1);

  int num_threads = at::get_num_threads();
  using opmath_t = at::opmath_type<scalar_t>;
  Tensor buffer;
  opmath_t* buffer_data = nullptr;
  static constexpr bool need_acc = is_reduced_floating_point_v<scalar_t>;
  if constexpr (need_acc) {
    auto acc_type = at::toAccumulateType(out.scalar_type(), /*is_cuda=*/true);
    buffer = at::zeros({num_threads, K}, out.options().dtype(acc_type));
    buffer_data = buffer.data_ptr<opmath_t>();
  }

  utils::parallel_sparse_csr(csr_data, M, nnz, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads,
                "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    opmath_t* buffer_ptr = nullptr;

    int64_t row_start = 0, row_end = 0;
    for (const auto m : c10::irange(begin, end)) {
      row_start = csr_data[m];
      row_end = csr_data[m + 1];

      scalar_t* out_ptr = out_data + m * K;
      if constexpr (need_acc) {
        buffer_ptr = buffer_data + tid * K;
      } else {
        buffer_ptr = reinterpret_cast<opmath_t*>(out_ptr);
      }

      // step 1: reinit the output row for reduce type 'amax' and 'amin'
      int64_t count = row_end - row_start;
      if (count != 0) {
        _init<scalar_t, reduce>(out_ptr, buffer_ptr, K, /*include_self*/false);
      }

      // step 2: reduce, do blocking on rowwise to reduce write memory bandwidth
      constexpr int64_t CHUNK_SIZE = 16;
      for (int64_t e0 = row_start; e0 < row_end; e0 += CHUNK_SIZE) {
        int64_t e1 = std::min(e0 + CHUNK_SIZE, row_end);
        for (const auto e : c10::irange(e0, e1)) {
          int64_t c = col_data[e];
          scalar_t val = val_data[e];
          _update<scalar_t, index_t, reduce>(buffer_ptr, e, c, val, other_data, K);
        }
      }
      if constexpr (need_acc) {
        if (count != 0) {
          vec::convert(buffer_ptr, out_ptr, K);
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
  auto csr_data = crow_indices.accessor<const index_t, 1>();
  auto col_data = col_indices.accessor<const index_t, 1>();
  auto val_data = values.accessor<const scalar_t, 1>();
  const scalar_t* other_data = other.const_data_ptr<scalar_t>();

  int64_t M = crow_indices.numel() - 1;
  int64_t K = other.size(-1);

  int num_threads = at::get_num_threads();
  using opmath_t = at::opmath_type<scalar_t>;
  Tensor buffer;
  opmath_t* buffer_data = nullptr;
  static constexpr bool need_acc = is_reduced_floating_point_v<scalar_t>;
  if constexpr (need_acc) {
    auto acc_type = at::toAccumulateType(out.scalar_type(), /*is_cuda=*/true);
    buffer = at::zeros({num_threads, K}, out.options().dtype(acc_type));
    buffer_data = buffer.data_ptr<opmath_t>();
  }

  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads,
                "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    opmath_t* buffer_ptr = nullptr;

    int64_t row_start = 0, row_end = 0, c = 0;
    for (const auto m : c10::irange(begin, end)) {
      row_start = csr_data[m];
      row_end = csr_data[m + 1];

      scalar_t* out_ptr = out_data + m * K;
      index_t* arg_out_ptr = arg_out_data + m * K;
      if constexpr (need_acc) {
        buffer_ptr = buffer_data + tid * K;
      } else {
        buffer_ptr = reinterpret_cast<opmath_t*>(out_ptr);
      }

      if (row_end != row_start) {
        _init<scalar_t, reduce>(out_ptr, buffer_ptr, K, /*include_self*/false);
        for (const auto e : c10::irange(row_start, row_end)) {
          c = col_data[e];
          opmath_t val = opmath_t(val_data[e]);

          const scalar_t* other_ptr = other_data + c * K;
          for (const auto k : c10::irange(K)) {
            update_with_index<opmath_t, index_t, reduce>(
                &buffer_ptr[k], opmath_t(val *  other_ptr[k]), &arg_out_ptr[k], index_t(e));
          };
        }
      }
      if constexpr (need_acc) {
        if (row_end != row_start) {
          vec::convert(buffer_ptr, out_ptr, K);
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
  const scalar_t* grad_out_data = grad_out.const_data_ptr<scalar_t>();
  auto crow_data = crow_indices.accessor<const index_t, 1>();
  auto col_data = col_indices.accessor<const index_t, 1>();
  const scalar_t* other_data = other.const_data_ptr<scalar_t>();
  auto row_data = row_indices.accessor<const index_t, 1>();

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
  const scalar_t* grad_out_data = grad_out.const_data_ptr<scalar_t>();
  auto col_data = col_indices.accessor<const index_t, 1>();
  const scalar_t* other_data = other.const_data_ptr<scalar_t>();
  index_t* arg_out_data = arg_out.data_ptr<index_t>();

  int64_t M = grad_out.size(0);
  int64_t K = grad_out.size(1);
  auto grad = at::empty({M, K}, grad_out.options());
  scalar_t* grad_data = grad.mutable_data_ptr<scalar_t>();

  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    for (const auto m : c10::irange(begin, end)) {
      const scalar_t* grad_out_ptr = grad_out_data + m * K;
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
  const scalar_t* grad_out_data = grad_out.const_data_ptr<scalar_t>();
  auto col_data = col_indices.accessor<const index_t, 1>();
  auto values_data = values.accessor<const scalar_t, 1>();
  const index_t* arg_out_data = arg_out.const_data_ptr<index_t>();

  int64_t M = grad_out.size(0);
  int64_t K = grad_out.size(1);
  auto grad = at::empty({M, K}, grad_out.options());
  scalar_t* grad_data = grad.mutable_data_ptr<scalar_t>();

  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    for (const auto m : c10::irange(begin, end)) {
      const scalar_t* grad_out_ptr = grad_out_data + m * K;
      scalar_t* grad_ptr = grad_data + m * K;
      const index_t* arg_out_ptr = arg_out_data + m * K;

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

} // at::native
