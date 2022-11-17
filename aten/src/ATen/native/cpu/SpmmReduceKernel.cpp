#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/SpmmReduceKernel.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_native.h>
#endif

namespace at { namespace native {

namespace {

template <typename scalar_t, SPMM_REDUCE_OP reduce>
struct Reducer {
  static inline void init(scalar_t* ptr, int64_t size) {
    using acc_t = vec::vec_scalar_t<scalar_t>;
    using Vec = vec::Vectorized<acc_t>;

    acc_t val;
    if (reduce == SPMM_MAX) {
      val = std::numeric_limits<acc_t>::lowest();
    } else if (reduce == SPMM_MIN) {
      val = std::numeric_limits<acc_t>::max();
    } else {
      return;
    }

    vec::map<scalar_t>(
        [val](Vec x) { return Vec(val); },
        ptr,
        ptr,
        size);
  }

  static inline void update(scalar_t& out, const scalar_t data) {
    if (reduce == SPMM_SUM || reduce == SPMM_MEAN) {
      out += data;
    } else if (reduce == SPMM_MAX) {
      out = std::max(out, data);
    } else {
      out = std::min(out, data);
    }
  }

  static inline void update(
      vec::Vectorized<scalar_t>& out_vec,
      const vec::Vectorized<scalar_t>& data_vec) {
    if (reduce == SPMM_SUM || reduce == SPMM_MEAN) {
      out_vec += data_vec;
    } else if (reduce == SPMM_MAX) {
      out_vec = vec::maximum(out_vec, data_vec);
    } else {
      out_vec = vec::minimum(out_vec, data_vec);
    }
  }
};

template <typename scalar_t, typename index_t, SPMM_REDUCE_OP reduce>
void spmm_reduce_kernel_impl(
    const Tensor& out,
    const Tensor& crow_indices_,
    const Tensor& col_indices_,
    const Tensor& values_,
    const Tensor& weight_) {

  int64_t nnz = values_.numel();
  if (nnz == 0) {
    return;
  }

  auto crow_indices = crow_indices_.contiguous();
  auto col_indices = col_indices_.contiguous();
  auto values = values_.contiguous();
  auto weight = weight_.contiguous();

  scalar_t* out_data = out.data_ptr<scalar_t>();
  index_t* csr_data = crow_indices.data_ptr<index_t>();
  index_t* col_data = col_indices.data_ptr<index_t>();
  scalar_t* val_data = values.data_ptr<scalar_t>();
  scalar_t* weight_data = weight.data_ptr<scalar_t>();

  int64_t M = crow_indices.numel() - 1;
  int64_t K = weight.size(-1);

  // directly parallel on `M` may lead to load imbalance,
  // statically determine thread partition here to average payload
  // for each thread.
  int num_threads = at::get_num_threads();
  std::vector<int64_t> thread_splits(num_threads + 1, M);

  int64_t thread_averge_payload = std::min((int64_t)1, nnz / num_threads);

  thread_splits[0] = 0;
  int64_t sum = 0;
  int64_t t = 1;
  for (const auto m : c10::irange(M)) {
    int64_t row_start = csr_data[m];
    int64_t row_end = csr_data[m + 1];
    sum += row_end - row_start;
    if (sum > t * thread_averge_payload) {
      thread_splits[t] = m;
      t++;
    }
  }
  // need to restore the last index,
  // due to rounding error when calculating `thread_averge_payload`.
  thread_splits[num_threads] = M;

  using Vec = vec::Vectorized<scalar_t>;
  at::parallel_for(0, num_threads, 1, [&](int64_t cbegin, int64_t cend) {
    int tid = at::get_thread_num();
    int64_t begin = thread_splits[tid];
    int64_t end = thread_splits[tid + 1];

    int64_t row_start, row_end, c;
    for (const auto m : c10::irange(begin, end)) {
      row_start = csr_data[m];
      row_end = csr_data[m + 1];

      scalar_t* out_ptr = out_data + m * K;

      constexpr int64_t kVecSize = Vec::size();
      constexpr int64_t kVLEN = kVecSize * 4;
      constexpr int64_t CHUNK_SIZE = 16;

      // reinit the output row for reduce type 'max' and 'min'
      Reducer<scalar_t, reduce>::init(out_ptr, K);

      // blocking on rowwise to reduce write memory bandwidth
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
            scalar_t* weight_ptr = weight_data + c * K + k;

            Reducer<scalar_t, reduce>::update(out_vec0, Vec::loadu(weight_ptr) * Vec(val));
            Reducer<scalar_t, reduce>::update(out_vec1, Vec::loadu(weight_ptr + kVecSize) * Vec(val));
            Reducer<scalar_t, reduce>::update(out_vec2, Vec::loadu(weight_ptr + kVecSize * 2) * Vec(val));
            Reducer<scalar_t, reduce>::update(out_vec3, Vec::loadu(weight_ptr + kVecSize * 3) * Vec(val));
          }
          out_vec0.store(out_ptr + k);
          out_vec1.store(out_ptr + k + kVecSize);
          out_vec2.store(out_ptr + k + kVecSize * 2);
          out_vec3.store(out_ptr + k + kVecSize * 3);
        }
        for (; k < K - (K % Vec::size()); k += Vec::size()) {
          Vec out_vec = Vec::loadu(out_ptr + k);
          for (const auto e : c10::irange(e0, e1)) {
            c = col_data[e];
            scalar_t val = val_data[e];
            scalar_t* weight_ptr = weight_data + c * K;
            Reducer<scalar_t, reduce>::update(out_vec, Vec::loadu(weight_ptr + k) * Vec(val));
          }
          out_vec.store(out_ptr + k);
        }
        for (; k < K; k++) {
          scalar_t out_val = out_ptr[k];
          for (const auto e : c10::irange(e0, e1)) {
            c = col_data[e];
            scalar_t val = val_data[e];
            scalar_t* weight_ptr = weight_data + c * K;
            Reducer<scalar_t, reduce>::update(out_val, weight_ptr[k] * val);
          }
          out_ptr[k] = out_val;
        }
      }

      if (reduce == SPMM_MEAN) {
        int64_t count = row_end - row_start;
        if (count != 0) {
          int64_t k = 0;
          for (; k < K - (K % Vec::size()); k += Vec::size()) {
            Vec out_vec = Vec::loadu(out_ptr + k);
            out_vec /= Vec(count);
            out_vec.store(out_ptr + k);
          }
          for (; k < K; k++) {
            out_ptr[k] /= count;
          }
        }
      }
    }
  });
}

template <typename scalar_t, typename index_t, SPMM_REDUCE_OP reduce>
inline void update(scalar_t *val, scalar_t new_val, index_t *arg, index_t new_arg) {
  if ((reduce == SPMM_MIN && new_val < *val) ||
      (reduce == SPMM_MAX && new_val > *val)) {
    *val = new_val;
    *arg = new_arg;
  }
}

template <typename scalar_t, typename index_t, SPMM_REDUCE_OP reduce>
void spmm_reduce_arg_kernel_impl(
    const Tensor& out,
    const Tensor& arg_out,
    const Tensor& crow_indices_,
    const Tensor& col_indices_,
    const Tensor& values_,
    const Tensor& weight_) {

  TORCH_CHECK(reduce == SPMM_MAX || reduce == SPMM_MIN);
  int64_t nnz = values_.numel();
  if (nnz == 0) {
    return;
  }

  auto crow_indices = crow_indices_.contiguous();
  auto col_indices = col_indices_.contiguous();
  auto values = values_.contiguous();
  auto weight = weight_.contiguous();

  scalar_t* out_data = out.data_ptr<scalar_t>();
  index_t* arg_out_data = arg_out.data_ptr<index_t>();
  index_t* csr_data = crow_indices.data_ptr<index_t>();
  index_t* col_data = col_indices.data_ptr<index_t>();
  scalar_t* val_data = values.data_ptr<scalar_t>();
  scalar_t* weight_data = weight.data_ptr<scalar_t>();

  int64_t M = crow_indices.numel() - 1;
  int64_t K = weight.size(-1);

  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    int64_t row_start, row_end, c;
    for (const auto m : c10::irange(begin, end)) {
      row_start = csr_data[m];
      row_end = csr_data[m + 1];

      scalar_t* out_ptr = out_data + m * K;
      index_t* arg_out_ptr = arg_out_data + m * K;

      Reducer<scalar_t, reduce>::init(out_ptr, K);
      for (const auto e : c10::irange(row_start, row_end)) {
        c = col_data[e];
        scalar_t val = val_data[e];

        scalar_t* weight_ptr = weight_data + c * K;
        for (const auto k : c10::irange(K)) {
          update<scalar_t, index_t, reduce>(
              &out_ptr[k], val *  weight_ptr[k], &arg_out_ptr[k], index_t(e));
        }
      }
    }
  });
}

template <typename scalar_t, typename index_t, SPMM_REDUCE_OP reduce>
void spmm_reduce_backward_input_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_out_,
    const Tensor& crow_indices_,
    const Tensor& col_indices_,
    const Tensor& weight_,
    const Tensor& row_indices_) {

  int64_t nnz = grad_input._nnz();
  if (nnz == 0) {
    return;
  }

  auto grad_out = grad_out_.contiguous();
  auto crow_indices = crow_indices_.contiguous();
  auto col_indices = col_indices_.contiguous();
  auto weight = weight_.contiguous();
  auto row_indices = row_indices_.contiguous();

  scalar_t* grad_values_data = grad_input.values().data_ptr<scalar_t>();
  scalar_t* grad_out_data = grad_out.data_ptr<scalar_t>();
  index_t* crow_data = crow_indices.data_ptr<index_t>();
  index_t* col_data = col_indices.data_ptr<index_t>();
  scalar_t* weight_data = weight.data_ptr<scalar_t>();
  index_t* row_data = row_indices.data_ptr<index_t>();

  int64_t K = grad_out.size(1);

  using Vec = vec::Vectorized<vec::vec_scalar_t<scalar_t>>;
  at::parallel_for(0, nnz, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      index_t row = row_data[i], col = col_data[i];

      scalar_t val = vec::map2_reduce_all<scalar_t>(
          [](Vec x, Vec y) { return x * y; },
          [](Vec x, Vec y) { return x + y; },
          weight_data + col * K,
          grad_out_data + row * K,
          K);

      if (reduce == SPMM_MEAN) {
        index_t row_start = crow_data[row], row_end = crow_data[row + 1];
        val /= std::max((index_t)1, row_end - row_start);
      }

      grad_values_data[i] = val;
    }
  });
}

// backward for reduce type 'max' or 'min'
template <typename scalar_t, typename index_t>
void spmm_reduce_backward_input_arg_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_out_,
    const Tensor& col_indices_,
    const Tensor& weight_,
    const Tensor& arg_out_) {

  int64_t nnz = grad_input._nnz();
  if (nnz == 0) {
    return;
  }

  auto grad_out = grad_out_.contiguous();
  auto col_indices = col_indices_.contiguous();
  auto weight = weight_.contiguous();
  auto arg_out = arg_out_.contiguous();

  scalar_t* grad_values_data = grad_input.values().data_ptr<scalar_t>();
  scalar_t* grad_out_data = grad_out.data_ptr<scalar_t>();
  index_t* col_data = col_indices.data_ptr<index_t>();
  scalar_t* weight_data = weight.data_ptr<scalar_t>();
  index_t* arg_out_data = arg_out.data_ptr<index_t>();

  int64_t M = grad_out.size(0);
  int64_t K = grad_out.size(1);
  auto grad = at::empty({M, K}, grad_out.options());
  scalar_t* grad_data = grad.data_ptr<scalar_t>();

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
          grad_ptr[k] = weight_data[col * K + k] * grad_out_ptr[k];
        }
      }
    }
  });

  // scatter_add, consider to parallel this with atomic
  for (const auto i : c10::irange(M * K)) {
    index_t ind = arg_out_data[i];
    grad_values_data[ind] += grad_data[i];
  }
}

template <typename scalar_t, typename index_t>
void spmm_reduce_update_values_kernel_impl(
    const Tensor& updated_values,
    const Tensor& values_,
    const Tensor& crow_indices_,
    const Tensor& row_indices_) {

  int64_t nnz = values_.numel();
  if (nnz == 0) {
    return;
  }

  auto values = values_.contiguous();
  auto crow_indices = crow_indices_.contiguous();
  auto row_indices = row_indices_.contiguous();

  scalar_t* updated_values_data = updated_values.data_ptr<scalar_t>();
  scalar_t* values_data = values.data_ptr<scalar_t>();
  index_t* crow_data = crow_indices.data_ptr<index_t>();
  index_t* row_data = row_indices.data_ptr<index_t>();

  at::parallel_for(0, nnz, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      index_t row = row_data[i];
      index_t row_start = crow_data[row], row_end = crow_data[row + 1];
      updated_values_data[i] = values_data[i] / std::max((index_t)1, row_end - row_start);
    }
  });
}

template <typename scalar_t, typename index_t>
void spmm_reduce_backward_weight_arg_kernel_impl(
    const Tensor& grad_weight,
    const Tensor& grad_out_,
    const Tensor& col_indices_,
    const Tensor& values_,
    const Tensor& arg_out_) {

  int64_t nnz = values_.numel();
  if (nnz == 0) {
    return;
  }

  auto grad_out = grad_out_.contiguous();
  auto col_indices = col_indices_.contiguous();
  auto values = values_.contiguous();
  auto arg_out = arg_out_.contiguous();

  scalar_t* grad_weight_data = grad_weight.data_ptr<scalar_t>();
  scalar_t* grad_out_data = grad_out.data_ptr<scalar_t>();
  index_t* col_data = col_indices.data_ptr<index_t>();
  scalar_t* values_data = values.data_ptr<scalar_t>();
  index_t* arg_out_data = arg_out.data_ptr<index_t>();

  int64_t M = grad_out.size(0);
  int64_t K = grad_out.size(1);
  auto grad = at::empty({M, K}, grad_out.options());
  scalar_t* grad_data = grad.data_ptr<scalar_t>();

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
      index_t col = col_data[arg_out_data[m * K + k]];
      grad_weight_data[col * K + k] += grad_data[m * K + k];
    }
  }
}

void spmm_reduce_kernel(
    const Tensor& out,
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& values,
    const Tensor& weight,
    SPMM_REDUCE_OP reduce_op) {
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, values.scalar_type(), "spmm_reduce_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "spmm_reduce_indices", [&]() {
      AT_DISPATCH_REDUCTION_TYPES(reduce_op, [&]() {
        spmm_reduce_kernel_impl<scalar_t, index_t, reduce>(
            out, crow_indices, col_indices, values, weight);
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
    const Tensor& weight,
    SPMM_REDUCE_OP reduce_op) {
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, values.scalar_type(), "spmm_reduce_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "spmm_reduce_indices", [&]() {
      AT_DISPATCH_REDUCTION_TYPES(reduce_op, [&]() {
        spmm_reduce_arg_kernel_impl<scalar_t, index_t, reduce>(
            out, arg_out, crow_indices, col_indices, values, weight);
      });
    });
  });
}

void spmm_reduce_backward_input_kernel(
    const Tensor& grad_input,
    const Tensor& grad_out,
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& weight,
    const Tensor& row_indices,
    SPMM_REDUCE_OP reduce_op) {
  TORCH_CHECK(reduce_op == SPMM_SUM || reduce_op == SPMM_MEAN);
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, weight.scalar_type(), "spmm_reduce_backward_input_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "spmm_reduce_backward_input_indices", [&]() {
      AT_DISPATCH_REDUCTION_TYPES(reduce_op, [&]() {
        spmm_reduce_backward_input_kernel_impl<scalar_t, index_t, reduce>(
            grad_input, grad_out, crow_indices, col_indices, weight, row_indices);
      });
    });
  });
}

void spmm_reduce_backward_input_arg_kernel(
    const Tensor& grad_input,
    const Tensor& grad_out,
    const Tensor& col_indices,
    const Tensor& weight,
    const Tensor& arg_out,
    SPMM_REDUCE_OP reduce_op) {
  TORCH_CHECK(reduce_op == SPMM_MAX || reduce_op == SPMM_MIN);
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, weight.scalar_type(), "spmm_reduce_backward_input_arg_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "spmm_reduce_backward_input_arg_indices", [&]() {
      spmm_reduce_backward_input_arg_kernel_impl<scalar_t, index_t>(
          grad_input, grad_out, col_indices, weight, arg_out);
    });
  });
}

void spmm_reduce_update_values_kernel(
    const Tensor& updated_values,
    const Tensor& values,
    const Tensor& crow_indices,
    const Tensor& row_indices) {
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, values.scalar_type(), "spmm_reduce_update_values_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(crow_indices.scalar_type(), "spmm_reduce_update_values_indices", [&]() {
      spmm_reduce_update_values_kernel_impl<scalar_t, index_t>(
          updated_values, values, crow_indices, row_indices);
    });
  });
}

void spmm_reduce_backward_weight_kernel(
    const Tensor& grad_weight,
    const Tensor& grad_out,
    const Tensor& crow_indices,
    const Tensor& values,
    const Tensor& row_indices,
    const Tensor& ccol_indices,
    const Tensor& csr2csc,
    SPMM_REDUCE_OP reduce_op) {
  TORCH_CHECK(reduce_op == SPMM_SUM || reduce_op == SPMM_MEAN);
  // need to permute row_indices to CSC order
  auto row = row_indices.index_select(0, csr2csc);

  Tensor val;
  if (reduce_op == SPMM_MEAN) {
    // for reduce type "mean", need to update the values
    // with rowcount for each of the nonzero element.
    Tensor updated_values = at::empty(values.sizes(), values.options());
    spmm_reduce_update_values_kernel(updated_values, values, crow_indices, row_indices);
    val = updated_values.index_select(0, csr2csc);
  } else {
    val = values.index_select(0, csr2csc);
  }

  if (reduce_op == SPMM_SUM || reduce_op == SPMM_MEAN) {
    spmm_reduce_kernel(grad_weight, ccol_indices, row, val, grad_out, SPMM_SUM);
  }
}

void spmm_reduce_backward_weight_arg_kernel(
    const Tensor& grad_weight,
    const Tensor& grad_out,
    const Tensor& col_indices,
    const Tensor& values,
    const Tensor& arg_out,
    SPMM_REDUCE_OP reduce_op) {
  TORCH_CHECK(reduce_op == SPMM_MAX || reduce_op == SPMM_MIN);
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, values.scalar_type(), "spmm_reduce_backward_weight_arg_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "spmm_reduce_backward_weight_arg_indices", [&]() {
      spmm_reduce_backward_weight_arg_kernel_impl<scalar_t, index_t>(
          grad_weight, grad_out, col_indices, values, arg_out);
    });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(spmm_reduce_stub, &spmm_reduce_kernel);
REGISTER_DISPATCH(spmm_reduce_arg_stub, &spmm_reduce_arg_kernel);
REGISTER_DISPATCH(spmm_reduce_backward_input_stub, &spmm_reduce_backward_input_kernel);
REGISTER_DISPATCH(spmm_reduce_backward_input_arg_stub, &spmm_reduce_backward_input_arg_kernel);
REGISTER_DISPATCH(spmm_reduce_backward_weight_stub, &spmm_reduce_backward_weight_kernel);
REGISTER_DISPATCH(spmm_reduce_backward_weight_arg_stub, &spmm_reduce_backward_weight_arg_kernel);

}} // at::native
