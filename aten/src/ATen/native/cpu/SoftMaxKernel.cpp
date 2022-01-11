#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cpu/SoftmaxKernel.h>

#include <algorithm>
#include <iterator>
#include <numeric>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>

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
  using Vec = vec::Vectorized<vec::vec_scalar_t<scalar_t>>;
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
          for (const auto j : c10::irange(loop_end)) {
            int64_t i = ii + j;
            scalar_t* input_data = input_data_base + i * dim_size;
            max_input_arr[j] = vec::reduce_all<scalar_t>(
                [](Vec& x, Vec& y) { return vec::maximum(x, y); },
                input_data,
                dim_size);
          }
          for (const auto j : c10::irange(loop_end)) {
            int64_t i = ii + j;
            scalar_t* input_data = input_data_base + i * dim_size;
            scalar_t max_input = max_input_arr[j];
            tmp_sum_scalar[j] = vec::map_reduce_all<scalar_t>(
                [max_input](Vec x) { return (x - Vec(max_input)).exp(); },
                [](Vec x, Vec y) { return x + y; },
                input_data,
                dim_size);
          }
          // See [Note AVX-SSE transitions] for why this should call the
          // vectorized version (aside from perf improvements).
          vec::map(
              [](Vec x) { return x.log(); },
              tmp_sum_scalar,
              tmp_sum_scalar,
              loop_end);
          for (const auto j : c10::irange(loop_end)) {
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
            vec::map(
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
  using Vec = vec::Vectorized<vec::vec_scalar_t<scalar_t>>;
  int64_t grain_size = internal::GRAIN_SIZE / (16 * dim_size);
  if (grain_size < 1)
    grain_size = 1;

  parallel_for(
      0,
      outer_size,
      grain_size,
      [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(begin, end)) {
          scalar_t* input_data = input_data_base + i * dim_size;
          scalar_t* output_data = output_data_base + i * dim_size;
          scalar_t max_input = vec::reduce_all<scalar_t>(
              [](Vec& x, Vec& y) { return vec::maximum(x, y); },
              input_data,
              dim_size);
          vec::map(
              [max_input](Vec x) { return (x - Vec(max_input)).exp(); },
              output_data,
              input_data,
              dim_size);
          scalar_t tmp_sum = vec::reduce_all<scalar_t>(
              [](Vec x, Vec y) { return x + y; }, output_data, dim_size);
          tmp_sum = 1 / tmp_sum;
          vec::map(
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
  using Vec = vec::Vectorized<vec::vec_scalar_t<scalar_t>>;
  int64_t grain_size = internal::GRAIN_SIZE / (16 * dim_size);
  if (grain_size < 1)
    grain_size = 1;

  parallel_for(
      0,
      outer_size,
      grain_size,
      [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(begin, end)) {
          scalar_t* grad_input_data = grad_input_data_base + i * dim_size;
          scalar_t* grad_data = grad_data_base + i * dim_size;
          scalar_t* output_data = output_data_base + i * dim_size;
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
          scalar_t sum;
          if (log_softmax) {
            sum = vec::reduce_all<scalar_t>(
                [](Vec& x, Vec& y) { return x + y; }, grad_data, dim_size);
          } else {
            sum = vec::map2_reduce_all<scalar_t>(
                [](Vec x, Vec y) { return x * y; },
                [](Vec x, Vec y) { return x + y; },
                grad_data,
                output_data,
                dim_size);
          }
          if (log_softmax) {
            vec::map2(
                [sum](Vec x, Vec y) { return x - ((y.exp()) * Vec(sum)); },
                grad_input_data,
                grad_data,
                output_data,
                dim_size);
          } else {
            vec::map2(
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
  static void apply(const Tensor& output, const Tensor& input) {
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

inline void _vec_softmax(
    BFloat16* input_data_base,
    BFloat16* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_size) {
  using Vec = vec::Vectorized<float>;
  using Vec_bf16 = vec::Vectorized<BFloat16>;
  int64_t dim_stride = inner_size;
  int64_t outer_stride = dim_size * dim_stride;
  int64_t grain_size = std::min(internal::GRAIN_SIZE / dim_size, (int64_t)1);
  int vectorized_step = Vec_bf16().size(); // Currently, we only support BFloat16 in this special implementation
  parallel_for(
      0, outer_size * inner_size, grain_size, [&](int64_t begin, int64_t end) {
        int64_t idx = begin;
        std::unique_ptr<float[]> temp_vec_input(new float[dim_size*vectorized_step*2]());
        std::unique_ptr<float[]> temp_vec_output(new float[dim_size*vectorized_step*2]());
        float* temp_vec_input_data = temp_vec_input.get();
        float* temp_vec_output_data = temp_vec_output.get();
        while (idx < end) {
          int64_t outer_idx = idx / inner_size;
          int64_t inner_idx = idx % inner_size;
          if (((inner_idx + vectorized_step) <= inner_size) && ((idx + vectorized_step) <= end)) {
            // Vectorization
            BFloat16* input_data =
                input_data_base + outer_idx * outer_stride + inner_idx;
            BFloat16* output_data =
                output_data_base + outer_idx * outer_stride + inner_idx;
            // Step 1: Get max Score
            Vec_bf16 max_vec_bf16 = Vec_bf16::loadu(input_data);
            std::tuple<vec::Vectorized<float>, vec::Vectorized<float>> convert_result = convert_bfloat16_float(max_vec_bf16);
            Vec max_vec_o1 = std::get<0>(convert_result);
            Vec max_vec_o2 = std::get<1>(convert_result);
            std::get<0>(convert_result).store(temp_vec_input_data);
            std::get<1>(convert_result).store(temp_vec_input_data + vectorized_step);
            for (const auto d : c10::irange(1, dim_size)) {
              Vec_bf16 input_vec_bf16 = Vec_bf16::loadu(input_data + d * dim_stride);
              convert_result = convert_bfloat16_float(input_vec_bf16);
              max_vec_o1 = vec::maximum(max_vec_o1, std::get<0>(convert_result));
              max_vec_o2 = vec::maximum(max_vec_o2, std::get<1>(convert_result));
              std::get<0>(convert_result).store(temp_vec_input_data + d*vectorized_step*2);
              std::get<1>(convert_result).store(temp_vec_input_data + d*vectorized_step*2 + vectorized_step);
            }
            // Step2: Calculate sum
            Vec sum_vec_o1 = Vec(0.0);
            Vec sum_vec_o2 = Vec(0.0);
            for (const auto d : c10::irange(dim_size)) {
              Vec output_vec_o1 = Vec::loadu(temp_vec_input_data + d*vectorized_step*2);
              Vec output_vec_o2 = Vec::loadu(temp_vec_input_data + d*vectorized_step*2 + vectorized_step);
              output_vec_o1 = (output_vec_o1 - max_vec_o1).exp();
              output_vec_o2 = (output_vec_o2 - max_vec_o2).exp();
              output_vec_o1.store(temp_vec_output_data + d*vectorized_step*2);
              output_vec_o2.store(temp_vec_output_data + d*vectorized_step*2 + vectorized_step);

              sum_vec_o1 = sum_vec_o1 + output_vec_o1;
              sum_vec_o2 = sum_vec_o2 + output_vec_o2;
            }
            // Step3: Unify
            for (const auto d : c10::irange(dim_size)) {
              Vec output_vec_o1 = Vec::loadu(temp_vec_output_data + d*vectorized_step*2);
              Vec output_vec_o2 = Vec::loadu(temp_vec_output_data + d*vectorized_step*2 + vectorized_step);
              output_vec_o1 = output_vec_o1/sum_vec_o1;
              output_vec_o2 = output_vec_o2/sum_vec_o2;
              Vec_bf16 output_vec_bf16 = convert_float_bfloat16(output_vec_o1, output_vec_o2);
              output_vec_bf16.store(output_data + d * dim_stride);
            }
            idx += vectorized_step;
          } else {
            // Tail case(Scalar): it is exactly same logic as host_softmax
            // inside aten/src/ATen/native/SoftMax.cpp. There are 2 kind of
            // cases which will fall through this part:
            // Case 1: For the idx at the end of total chunk for each thread, there are not enough numbers for parallization.
            // Case 2: For the idx at the end of each inner_size inside thread, there are not enough numbers for parallization.
            int64_t tail_number = ((idx+vectorized_step) > end) ? /*Case1*/ (end - idx) : /*Case2*/ (inner_size - inner_idx);
            for (const auto i : c10::irange(tail_number)) {
              outer_idx = (idx + i) / inner_size;
              inner_idx = (idx + i) % inner_size;
              BFloat16* input_data =
                  input_data_base + outer_idx * outer_stride + inner_idx;
              BFloat16* output_data =
                  output_data_base + outer_idx * outer_stride + inner_idx;
              // Step1: Get max score
              float max_input = float(input_data[0]);
              for (const auto d : c10::irange(1, dim_size)) {
                max_input = std::max(max_input, float(input_data[d * dim_stride]));
              }
              // Step2: Calculate the Sum
              float sum_data = 0.0;
              float temp_output_data = 0.0;
              for (const auto d : c10::irange(dim_size)) {
                temp_output_data = std::exp(input_data[d * dim_stride] - max_input);
                sum_data += temp_output_data;
                output_data[d * dim_stride] = c10::BFloat16(temp_output_data);
              }
              // Step3: Unify
              for (const auto d : c10::irange(dim_size)) {
                output_data[d * dim_stride] =
                    c10::BFloat16(float(output_data[d * dim_stride])/sum_data);
              }
            }
            idx += tail_number;
          }
        }
      });
}

template <typename scalar_t>
inline void _vec_softmax(
    scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_size) {
  using Vec = vec::Vectorized<scalar_t>;
  int64_t dim_stride = inner_size;
  int64_t outer_stride = dim_size * dim_stride;
  int64_t grain_size = std::min(internal::GRAIN_SIZE / dim_size, (int64_t)1);
  int vectorized_step = Vec().size();
  parallel_for(
      0, outer_size * inner_size, grain_size, [&](int64_t begin, int64_t end) {
        int64_t idx = begin;
        while (idx < end) {
          int64_t outer_idx = idx / inner_size;
          int64_t inner_idx = idx % inner_size;
          if (((inner_idx + vectorized_step) <= inner_size) && ((idx + vectorized_step) <= end)) {
            // Vectorization
            scalar_t* input_data =
                input_data_base + outer_idx * outer_stride + inner_idx;
            scalar_t* output_data =
                output_data_base + outer_idx * outer_stride + inner_idx;
            // Step 1: Get max Score
            Vec max_vec = Vec::loadu(input_data);
            for (const auto d : c10::irange(1, dim_size)) {
              Vec input_vec = Vec::loadu(input_data + d * dim_stride);
              max_vec = vec::maximum(max_vec, input_vec);
            }
            // Step2: Calculate sum
            Vec sum_vec = Vec(0.0);
            for (const auto d : c10::irange(dim_size)) {
              Vec output_vec =
                  (Vec::loadu(input_data + d * dim_stride) - max_vec).exp();
              output_vec.store(output_data + d * dim_stride);
              sum_vec = sum_vec + output_vec;
            }
            // Step3: Unify
            for (const auto d : c10::irange(dim_size)) {
              Vec output_vec =
                  Vec::loadu(output_data + d * dim_stride) / sum_vec;
              output_vec.store(output_data + d * dim_stride);
            }
            idx += vectorized_step;
          } else {
            // Tail case(Scalar): it is exactly same logic as host_softmax
            // inside aten/src/ATen/native/SoftMax.cpp. There are 2 kind of
            // cases which will fall through this part:
            // Case 1: For the idx at the end of total chunk for each thread, there are not enough numbers for parallization.
            // Case 2: For the idx at the end of each inner_size inside thread, there are not enough numbers for parallization.
            int64_t tail_number = ((idx+vectorized_step) > end) ? /*Case1*/ (end - idx) : /*Case2*/ (inner_size - inner_idx);
            for (const auto i : c10::irange(tail_number)) {
              outer_idx = (idx + i) / inner_size;
              inner_idx = (idx + i) % inner_size;
              scalar_t* input_data =
                  input_data_base + outer_idx * outer_stride + inner_idx;
              scalar_t* output_data =
                  output_data_base + outer_idx * outer_stride + inner_idx;
              // Step1: Get max score
              scalar_t max_input = input_data[0];
              for (const auto d : c10::irange(1, dim_size)) {
                max_input = std::max(max_input, input_data[d * dim_stride]);
              }
              // Step2: Calculate the Sum
              scalar_t sum_data = 0;
              for (const auto d : c10::irange(dim_size)) {
                output_data[d * dim_stride] =
                    std::exp(input_data[d * dim_stride] - max_input);
                sum_data += output_data[d * dim_stride];
              }
              // Step3: Unify
              for (const auto d : c10::irange(dim_size)) {
                output_data[d * dim_stride] =
                    output_data[d * dim_stride]/sum_data;
              }
            }
            idx += tail_number;
          }
        }
      });
}

template <typename scalar_t, bool LogSoftMax>
struct vec_softmax {
  static void apply(const Tensor& output, const Tensor& input, int64_t dim) {
    int64_t outer_size = 1;
    int64_t dim_size = input.size(dim);
    int64_t inner_size = 1;
    for (const auto i : c10::irange(dim))outer_size *= input.size(i);
    for (int64_t i = dim + 1; i < input.dim(); ++i)
      inner_size *= input.size(i);
    scalar_t* input_data_base = input.data_ptr<scalar_t>();
    scalar_t* output_data_base = output.data_ptr<scalar_t>();
    if (LogSoftMax) {
      AT_ERROR("vec_softmax not implemented for LogSoftMax");
    } else {
      _vec_softmax(
          input_data_base, output_data_base, outer_size, inner_size, dim_size);
    }
  }
};

template <typename scalar_t, bool LogSoftMax>
struct vec_host_softmax_backward_lastdim {
  static void
  apply(const Tensor& grad_input, const Tensor& grad, const Tensor& output) {
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

static void softmax_lastdim_kernel_impl(
    const Tensor& result,
    const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, self.scalar_type(),
      "softmax_lastdim_kernel_impl",
      [&] { vec_host_softmax_lastdim<scalar_t, false>::apply(result, self); });
}

static void softmax_kernel_impl(const Tensor& result, const Tensor& self, int64_t dim) {
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, self.scalar_type(),
    "softmax_kernel_impl",
    [&] { vec_softmax<scalar_t, false>::apply(result, self, dim); });
}

static void log_softmax_lastdim_kernel_impl(
    const Tensor& result,
    const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, self.scalar_type(),
      "log_softmax_lastdim_kernel_impl",
      [&] { vec_host_softmax_lastdim<scalar_t, true>::apply(result, self); });
}

static void softmax_backward_lastdim_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad,
    const Tensor& output) {
  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, grad.scalar_type(),
      "softmax_backward_lastdim_kernel_impl", [&] {
        vec_host_softmax_backward_lastdim<scalar_t, false>::apply(
            grad_input, grad, output);
      });
}

static void log_softmax_backward_lastdim_kernel_impl(
    const Tensor& grad_input,
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

REGISTER_DISPATCH(softmax_lastdim_kernel, &softmax_lastdim_kernel_impl);
REGISTER_DISPATCH(log_softmax_lastdim_kernel, &log_softmax_lastdim_kernel_impl);
REGISTER_DISPATCH(
    softmax_backward_lastdim_kernel,
    &softmax_backward_lastdim_kernel_impl);
REGISTER_DISPATCH(
    log_softmax_backward_lastdim_kernel,
    &log_softmax_backward_lastdim_kernel_impl);

REGISTER_DISPATCH(softmax_kernel, &softmax_kernel_impl);

}} // namespace at::native
