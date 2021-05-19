#include <ATen/native/cpu/SoftmaxKernel.h>

#include <algorithm>
#include <iterator>
#include <numeric>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/cpu/vec256/vec256.h>
#include <c10/util/Optional.h>

// [Note AVX-SSE transitions] In general we avoid calls into cmath for code
// compiled with AVX/AVX2 This is because of SSE-AVX transitions and a bug in
// Glibc2.23 See https://bugs.launchpad.net/ubuntu/+source/glibc/+bug/1663280
//
// On grainsize: The grainsize is chosen to roughly get GRAIN_SIZE number of
// computations per task. Each task works across dim_size elements. 16 should be
// a very rough approximation of the number of computations per dim_size element
// by counting simple computations (*, +, -) as 1 and exp or log as 4.

namespace at { namespace native {
namespace {

template <typename scalar_t>
inline void _vec_log_softmax_lastdim(
    scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t dim_size) {
  using Vec = vec256::Vec256<scalar_t>;
  static constexpr int64_t CHUNK_SIZE = (128 / sizeof(scalar_t)) * Vec::size();
  int64_t grain_size = internal::GRAIN_SIZE / (16 * dim_size * CHUNK_SIZE);
  if (grain_size < CHUNK_SIZE)
    grain_size = CHUNK_SIZE;

  parallel_for(
      0,
      outer_size,
      grain_size,
      [&](int64_t begin, int64_t end) {
        for (int64_t ii = begin; ii < end; ii += CHUNK_SIZE) {
          // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
          scalar_t tmp_sum_scalar[CHUNK_SIZE];
          // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
          scalar_t max_input_arr[CHUNK_SIZE];
          int64_t loop_end = CHUNK_SIZE;
          if (ii + CHUNK_SIZE > end)
            loop_end = end - ii;
          for (int64_t j = 0; j < loop_end; j++) {
            int64_t i = ii + j;
            scalar_t* input_data = input_data_base + i * dim_size;
            max_input_arr[j] = vec256::reduce_all<scalar_t>(
                [](Vec& x, Vec& y) { return vec256::maximum(x, y); },
                input_data,
                dim_size);
          }
          for (int64_t j = 0; j < loop_end; j++) {
            int64_t i = ii + j;
            scalar_t* input_data = input_data_base + i * dim_size;
            scalar_t max_input = max_input_arr[j];
            tmp_sum_scalar[j] = vec256::map_reduce_all<scalar_t>(
                [max_input](Vec x) { return (x - Vec(max_input)).exp(); },
                [](Vec x, Vec y) { return x + y; },
                input_data,
                dim_size);
          }
          // See [Note AVX-SSE transitions] for why this should call the
          // vectorized version (aside from perf improvements).
          vec256::map(
              [](Vec x) { return x.log(); },
              tmp_sum_scalar,
              tmp_sum_scalar,
              loop_end);
          for (int64_t j = 0; j < loop_end; j++) {
            int64_t i = ii + j;
            scalar_t* input_data = input_data_base + i * dim_size;
            scalar_t* output_data = output_data_base + i * dim_size;
            scalar_t tmp_sum = tmp_sum_scalar[j];
            scalar_t max_input = max_input_arr[j];

            // It's necessary to keep the order of the operations below.
            // In some cases that input is large digits and the difference
            // is small, if we compute `max_input` plus `tmp_sum` before,
            // there would be a numerical problem. See an example in
            // https://github.com/pytorch/pytorch/issues/11752#issuecomment-422883379
            vec256::map(
                [tmp_sum, max_input](Vec x) { return x - Vec(max_input) - Vec(tmp_sum); },
                output_data,
                input_data,
                dim_size);
          }
        }
      });
}

template <typename scalar_t>
inline void _vec_softmax_lastdim(
    scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t dim_size) {
  using Vec = vec256::Vec256<scalar_t>;
  int64_t grain_size = internal::GRAIN_SIZE / (16 * dim_size);
  if (grain_size < 1)
    grain_size = 1;

  parallel_for(
      0,
      outer_size,
      grain_size,
      [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          scalar_t* input_data = input_data_base + i * dim_size;
          scalar_t* output_data = output_data_base + i * dim_size;
          scalar_t max_input = vec256::reduce_all<scalar_t>(
              [](Vec& x, Vec& y) { return vec256::maximum(x, y); },
              input_data,
              dim_size);
          vec256::map(
              [max_input](Vec x) { return (x - Vec(max_input)).exp(); },
              output_data,
              input_data,
              dim_size);
          scalar_t tmp_sum = vec256::reduce_all<scalar_t>(
              [](Vec x, Vec y) { return x + y; }, output_data, dim_size);
          tmp_sum = 1 / tmp_sum;
          vec256::map(
              [tmp_sum](Vec x) { return x * Vec(tmp_sum); },
              output_data,
              output_data,
              dim_size);
        }
      });
}

template <typename scalar_t, bool log_softmax>
inline void _vec_host_softmax_backward_lastdim(
    scalar_t* grad_input_data_base,
    scalar_t* grad_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t dim_size) {
  using Vec = vec256::Vec256<scalar_t>;
  int64_t grain_size = internal::GRAIN_SIZE / (16 * dim_size);
  if (grain_size < 1)
    grain_size = 1;

  parallel_for(
      0,
      outer_size,
      grain_size,
      [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          scalar_t* grad_input_data = grad_input_data_base + i * dim_size;
          scalar_t* grad_data = grad_data_base + i * dim_size;
          scalar_t* output_data = output_data_base + i * dim_size;
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
          scalar_t sum;
          if (log_softmax) {
            sum = vec256::reduce_all<scalar_t>(
                [](Vec& x, Vec& y) { return x + y; }, grad_data, dim_size);
          } else {
            sum = vec256::map2_reduce_all<scalar_t>(
                [](Vec x, Vec y) { return x * y; },
                [](Vec x, Vec y) { return x + y; },
                grad_data,
                output_data,
                dim_size);
          }
          if (log_softmax) {
            vec256::map2(
                [sum](Vec x, Vec y) { return x - ((y.exp()) * Vec(sum)); },
                grad_input_data,
                grad_data,
                output_data,
                dim_size);
          } else {
            vec256::map2(
                [sum](Vec x, Vec y) { return (x - Vec(sum)) * y; },
                grad_input_data,
                grad_data,
                output_data,
                dim_size);
          }
        }
      });
}

template <typename scalar_t, bool LogSoftMax>
struct vec_host_softmax_lastdim {
  static void apply(Tensor& output, const Tensor& input) {
    int64_t outer_size = 1;
    int64_t dim_size = input.size(input.ndimension() - 1);
    for (int64_t i = 0; i < input.ndimension() - 1; ++i)
      outer_size *= input.size(i);
    scalar_t* input_data_base = input.data_ptr<scalar_t>();
    scalar_t* output_data_base = output.data_ptr<scalar_t>();
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
    int64_t outer_size = 1;
    int64_t dim_size = grad.size(grad.ndimension() - 1);
    for (int64_t i = 0; i < grad.ndimension() - 1; ++i)
      outer_size *= grad.size(i);
    scalar_t* grad_input_data_base = grad_input.data_ptr<scalar_t>();
    scalar_t* grad_data_base = grad.data_ptr<scalar_t>();
    scalar_t* output_data_base = output.data_ptr<scalar_t>();
    _vec_host_softmax_backward_lastdim<scalar_t, LogSoftMax>(
        grad_input_data_base,
        grad_data_base,
        output_data_base,
        outer_size,
        dim_size);
  }
};

static void softmax_lastdim_kernel_impl(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "softmax_lastdim_kernel_impl", [&] {
    vec_host_softmax_lastdim<scalar_t, false>::apply(result, self);
  });
}

static void log_softmax_lastdim_kernel_impl(
    Tensor& result,
    const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, self.scalar_type(),
      "log_softmax_lastdim_kernel_impl",
      [&] { vec_host_softmax_lastdim<scalar_t, true>::apply(result, self); });
}

static void softmax_backward_lastdim_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad,
    const Tensor& output) {
  AT_DISPATCH_FLOATING_TYPES(
      grad.scalar_type(), "softmax_backward_lastdim_kernel_impl", [&] {
        vec_host_softmax_backward_lastdim<scalar_t, false>::apply(
            grad_input, grad, output);
      });
}

static void log_softmax_backward_lastdim_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad,
    const Tensor& output) {
  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, grad.scalar_type(),
      "log_softmax_backward_lastdim_kernel_impl", [&] {
        vec_host_softmax_backward_lastdim<scalar_t, true>::apply(
            grad_input, grad, output);
      });
}

} // anonymous namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(softmax_lastdim_kernel, &softmax_lastdim_kernel_impl);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(log_softmax_lastdim_kernel, &log_softmax_lastdim_kernel_impl);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(
    softmax_backward_lastdim_kernel,
    &softmax_backward_lastdim_kernel_impl);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(
    log_softmax_backward_lastdim_kernel,
    &log_softmax_backward_lastdim_kernel_impl);

}} // namespace at::native
