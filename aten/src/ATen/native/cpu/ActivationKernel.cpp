#include "ATen/native/cpu/ActivationKernel.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>

#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/cpu/vec256/functional.h"
#include "ATen/cpu/vec256/vec256.h"
#include "ATen/optional.h"

// NOTE: In general we avoid calls into cmath for code compiled with AVX/AVX2
// This is because of SSE-AVX transitions and a bug in Glibc2.23
// See https://bugs.launchpad.net/ubuntu/+source/glibc/+bug/1663280

namespace at { namespace native {
namespace {

static tbb::affinity_partitioner ap;

template <int64_t size>
inline int64_t _leftover(int64_t x, int64_t y) {
  if (x + size > y)
    return y - x;
  return size;
}

template <typename scalar_t>
inline scalar_t
_vec_sum_mul(int64_t dim_size, scalar_t* input_data, scalar_t* input_data2) {
  using Vec = vec256::Vec256<scalar_t>;
  int64_t d = 0;
  Vec sum_vec(0);
  scalar_t sum = 0;
  for (; d < dim_size; d += Vec::size) {
    if (d + Vec::size > dim_size) {
      for (int i = d; i < dim_size; i++) {
        sum += input_data[i] * input_data2[i];
      }
    } else {
      Vec value(0);
      Vec value2(0);
      value.load(input_data + d);
      value2.load(input_data2 + d);
      value = value * value2;
      sum_vec = sum_vec + value;
    }
  }
  scalar_t sum_arr[Vec::size];
  sum_vec.store(sum_arr);
  for (int64_t i = 0; i < Vec::size; i++) {
    sum += sum_arr[i];
  }
  return sum;
}

template <typename scalar_t>
inline void _vec_mul_scalarsub_write(
    scalar_t* grad_input_data,
    scalar_t* grad_data,
    scalar_t* output_data,
    int64_t dim_size,
    scalar_t sum) {
  using Vec = vec256::Vec256<scalar_t>;
  Vec sum_vec(sum);
  int64_t d = 0;
  for (; d < dim_size; d += Vec::size) {
    Vec output(0);
    Vec grad(0);
    int64_t leftover = _leftover<Vec::size>(d, dim_size);
    output.load(output_data + d, leftover);
    grad.load(grad_data + d, leftover);
    grad = grad - sum_vec;
    grad = grad * output;

    grad.store(grad_input_data + d, leftover);
  }
}

template <typename scalar_t>
inline void _vec_sub_exp_scalarmul_write(
    scalar_t* grad_input_data,
    scalar_t* grad_data,
    scalar_t* output_data,
    int64_t dim_size,
    scalar_t sum) {
  using Vec = vec256::Vec256<scalar_t>;
  Vec sum_vec(sum);
  int64_t d = 0;
  for (; d < dim_size; d += Vec::size) {
    Vec output(0);
    Vec grad(0);
    int64_t leftover = _leftover<Vec::size>(d, dim_size);
    output.load(output_data + d, leftover);
    grad.load(grad_data + d, leftover);
    output = output.exp();
    output = output * sum_vec;
    grad = grad - output;

    grad.store(grad_input_data + d, leftover);
  }
}

template <typename scalar_t>
inline void _vec_norm_exp_write(
    scalar_t* output_data,
    int64_t dim_size,
    scalar_t* input_data,
    scalar_t max_input) {
  using Vec = vec256::Vec256<scalar_t>;
  int64_t d = 0;
  Vec max_input_vec(max_input);
  for (; d < dim_size; d += Vec::size) {
    int64_t leftover = _leftover<Vec::size>(d, dim_size);
    Vec value(0);
    value.load(input_data + d, leftover);
    value = value - max_input_vec;
    value = value.exp();
    value.store(output_data + d, leftover);
  }
}

template <typename scalar_t>
inline void _vec_log_softmax_lastdim(
    scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t dim_size) {
  using Vec = vec256::Vec256<scalar_t>;
  static constexpr int64_t CHUNK_SIZE = (128 / sizeof(scalar_t)) * Vec::size;
  int64_t grain_size = internal::TBB_GRAIN_SIZE / (dim_size * CHUNK_SIZE);
  if (grain_size < CHUNK_SIZE)
    grain_size = CHUNK_SIZE;

  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, outer_size, grain_size),
      [&](const tbb::blocked_range<int64_t>& r) {
        for (int64_t ii = r.begin(); ii < r.end(); ii += CHUNK_SIZE) {
          scalar_t tmp_sum_scalar[CHUNK_SIZE];
          scalar_t max_input_arr[CHUNK_SIZE];
          int64_t inner_loop_end = CHUNK_SIZE;
          if (ii + CHUNK_SIZE > r.end())
            inner_loop_end = r.end() - ii;
          for (int64_t j = 0; j < inner_loop_end; j++) {
            int64_t i = ii + j;
            scalar_t* input_data = input_data_base + i * dim_size;
            max_input_arr[j] = vec256::reduce_all<scalar_t>(
                [](Vec& x, Vec& y) { return vec256::max(x, y); },
                input_data,
                dim_size,
                (scalar_t)0);
          }
          for (int64_t j = 0; j < inner_loop_end; j++) {
            int64_t i = ii + j;
            tmp_sum_scalar[j] = 0;
            scalar_t* input_data = input_data_base + i * dim_size;
            Vec tmpsum(0);
            Vec max_input_vec(max_input_arr[j]);
            for (int64_t d = 0; d < dim_size; d += Vec::size) {
              Vec value(0);
              int64_t leftover = _leftover<Vec::size>(d, dim_size);
              value.load(input_data + d, leftover);
              value = value - max_input_vec;
              value = value.exp();
              // Need to do a partial add
              if (leftover < Vec::size) {
                scalar_t value_arr[Vec::size];
                value.store(value_arr);
                for (int64_t i = 0; i < leftover; i++)
                  tmp_sum_scalar[j] += value_arr[i];
              } else {
                tmpsum = tmpsum + value;
              }
            }
            scalar_t tmp_sum_arr[Vec::size];
            tmpsum.store(tmp_sum_arr);
            for (int64_t i = 0; i < Vec::size; i++)
              tmp_sum_scalar[j] += tmp_sum_arr[i];
          }
          for (int64_t j = 0; j < inner_loop_end; j += Vec::size) {
            int64_t leftover = _leftover<Vec::size>(j, inner_loop_end);
            Vec tmp_sum_scalar_vec;
            tmp_sum_scalar_vec.load(tmp_sum_scalar + j, leftover);
            tmp_sum_scalar_vec = tmp_sum_scalar_vec.log();
            Vec max_input_vec;
            max_input_vec.load(max_input_arr + j, leftover);
            tmp_sum_scalar_vec = max_input_vec + tmp_sum_scalar_vec;
            tmp_sum_scalar_vec.store(tmp_sum_scalar + j, leftover);
          }
          for (int64_t j = 0; j < inner_loop_end; j++) {
            int64_t i = ii + j;
            scalar_t* input_data = input_data_base + i * dim_size;
            scalar_t* output_data = output_data_base + i * dim_size;
            Vec sub_val_vec(tmp_sum_scalar[j]);
            for (int64_t d = 0; d < dim_size; d += Vec::size) {
              Vec value;
              int64_t leftover = _leftover<Vec::size>(d, dim_size);
              value.load(input_data + d, leftover);
              value = value - sub_val_vec;
              value.store(output_data + d, leftover);
            }
          }
        }
      },
      ap);
}

template <typename scalar_t>
inline void _vec_softmax_lastdim(
    scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t dim_size) {
  int64_t grain_size = internal::TBB_GRAIN_SIZE / dim_size;
  if (grain_size < 1)
    grain_size = 1;

  using Vec = vec256::Vec256<scalar_t>;
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, outer_size, grain_size),
      [&](const tbb::blocked_range<int64_t>& r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          scalar_t* input_data = input_data_base + i * dim_size;
          scalar_t* output_data = output_data_base + i * dim_size;

          scalar_t max_input = vec256::reduce_all<scalar_t>(
              [](Vec& x, Vec& y) { return vec256::max(x, y); },
              input_data,
              dim_size,
              (scalar_t)0);

          _vec_norm_exp_write(output_data, dim_size, input_data, max_input);

          scalar_t tmp_sum = vec256::reduce_all<scalar_t>(
              [](Vec& x, Vec& y) { return x + y; },
              output_data,
              dim_size,
              (scalar_t)0);

          tmp_sum = 1 / tmp_sum;
          Vec sub_val_vec(tmp_sum);
          for (int64_t d = 0; d < dim_size; d += Vec::size) {
            Vec value;
            int64_t leftover = _leftover<Vec::size>(d, dim_size);
            value.load(output_data + d, leftover);
            value = value * sub_val_vec;
            value.store(output_data + d, leftover);
          }
        }
      },
      ap);
}

template <typename scalar_t>
inline void _vec_log_softmax_backward_lastdim(
    scalar_t* grad_input_data_base,
    scalar_t* grad_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t dim_size) {
  using Vec = vec256::Vec256<scalar_t>;
  int64_t grain_size = internal::TBB_GRAIN_SIZE / dim_size;
  if (grain_size < 1)
    grain_size = 1;

  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, outer_size, grain_size),
      [&](const tbb::blocked_range<int64_t>& r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          scalar_t* grad_input_data = grad_input_data_base + i * dim_size;
          scalar_t* grad_data = grad_data_base + i * dim_size;
          scalar_t* output_data = output_data_base + i * dim_size;

          scalar_t sum = vec256::reduce_all<scalar_t>(
              [](Vec& x, Vec& y) { return x + y; },
              grad_data,
              dim_size,
              (scalar_t)0);

          _vec_sub_exp_scalarmul_write(
              grad_input_data, grad_data, output_data, dim_size, sum);
        }
      },
      ap);
}

template <typename scalar_t>
inline void _vec_softmax_backward_lastdim(
    scalar_t* grad_input_data_base,
    scalar_t* grad_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t dim_size) {
  using Vec = vec256::Vec256<scalar_t>;
  static constexpr int64_t CHUNK_SIZE = (128 / sizeof(scalar_t)) * Vec::size;
  int64_t grain_size = internal::TBB_GRAIN_SIZE / (dim_size * CHUNK_SIZE);
  if (grain_size < CHUNK_SIZE)
    grain_size = CHUNK_SIZE;

  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, outer_size, grain_size),
      [&](const tbb::blocked_range<int64_t>& r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          scalar_t* grad_input_data = grad_input_data_base + i * dim_size;
          scalar_t* grad_data = grad_data_base + i * dim_size;
          scalar_t* output_data = output_data_base + i * dim_size;

          scalar_t sum = _vec_sum_mul(dim_size, grad_data, output_data);
          _vec_mul_scalarsub_write(
              grad_input_data, grad_data, output_data, dim_size, sum);
        }
      },
      ap);
}

template <typename scalar_t, bool LogSoftMax>
struct vec_host_softmax_lastdim {
  static void apply(Tensor& output, const Tensor& input) {
    internal::init_tbb_num_threads();

    int64_t outer_size = 1;
    int64_t dim_size = input.size(input.ndimension() - 1);
    for (int64_t i = 0; i < input.ndimension() - 1; ++i)
      outer_size *= input.size(i);
    scalar_t* input_data_base = input.data<scalar_t>();
    scalar_t* output_data_base = output.data<scalar_t>();
    if (LogSoftMax) {
      _vec_log_softmax_lastdim(
          input_data_base, output_data_base, outer_size, dim_size);
    } else {
      _vec_softmax_lastdim(
          input_data_base, output_data_base, outer_size, dim_size);
    }
  }
};

template <typename scalar_t, bool LogSoftMax>
struct vec_host_softmax_backward_lastdim {
  static void
  apply(Tensor& grad_input, const Tensor& grad, const Tensor& output) {
    internal::init_tbb_num_threads();

    int64_t outer_size = 1;
    int64_t dim_size = grad.size(grad.ndimension() - 1);
    for (int64_t i = 0; i < grad.ndimension() - 1; ++i)
      outer_size *= grad.size(i);
    scalar_t* grad_input_data_base = grad_input.data<scalar_t>();
    scalar_t* grad_data_base = grad.data<scalar_t>();
    scalar_t* output_data_base = output.data<scalar_t>();
    if (LogSoftMax) {
      _vec_log_softmax_backward_lastdim(
          grad_input_data_base,
          grad_data_base,
          output_data_base,
          outer_size,
          dim_size);
    } else {
      _vec_softmax_backward_lastdim(
          grad_input_data_base,
          grad_data_base,
          output_data_base,
          outer_size,
          dim_size);
    }
  }
};

static void softmax_lastdim_kernel_impl(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "softmax_lastdim_kernel_impl", [&] {
    vec_host_softmax_lastdim<scalar_t, false>::apply(result, self);
  });
}

static void log_softmax_lastdim_kernel_impl(
    Tensor& result,
    const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(
      self.type(), "log_softmax_lastdim_kernel_impl", [&] {
        vec_host_softmax_lastdim<scalar_t, true>::apply(result, self);
      });
}

static void softmax_backward_lastdim_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad,
    const Tensor& output) {
  AT_DISPATCH_FLOATING_TYPES(
      grad.type(), "softmax_backward_lastdim_kernel_impl", [&] {
        vec_host_softmax_backward_lastdim<scalar_t, false>::apply(
            grad_input, grad, output);
      });
}

static void log_softmax_backward_lastdim_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad,
    const Tensor& output) {
  AT_DISPATCH_FLOATING_TYPES(
      grad.type(), "log_softmax_backward_lastdim_kernel_impl", [&] {
        vec_host_softmax_backward_lastdim<scalar_t, true>::apply(
            grad_input, grad, output);
      });
}

} // anonymous namespace

REGISTER_DISPATCH(softmax_lastdim_kernel, &softmax_lastdim_kernel_impl);
REGISTER_DISPATCH(log_softmax_lastdim_kernel, &log_softmax_lastdim_kernel_impl);
REGISTER_DISPATCH(
    softmax_backward_lastdim_kernel,
    &softmax_backward_lastdim_kernel_impl);
REGISTER_DISPATCH(
    log_softmax_backward_lastdim_kernel,
    &log_softmax_backward_lastdim_kernel_impl);

}} // namespace at::native
