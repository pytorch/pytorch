#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/batch_norm.h>

#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/ones.h>
#endif

namespace at { namespace native {
namespace {

using namespace vec;

template<typename scalar_t>
void batch_norm_cpu_collect_linear_and_constant_terms(
    scalar_t* alpha, scalar_t* beta, int64_t n_channel,
    const Tensor& weight /* optional */, const Tensor& bias /* optional */,
    const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {

  const scalar_t* weight_data = weight.defined() ? weight.data_ptr<scalar_t>() : nullptr;
  const scalar_t* bias_data = bias.defined() ? bias.data_ptr<scalar_t>() : nullptr;

  auto save_mean_a = conditional_accessor_1d<scalar_t>(save_mean);
  auto save_invstd_a = conditional_accessor_1d<scalar_t>(save_invstd);
  auto running_mean_a = conditional_accessor_1d<scalar_t>(running_mean);
  auto running_var_a = conditional_accessor_1d<scalar_t>(running_var);

  /// Collect the linear and constant terms regarding the input.
  /// output(n, c, h, w)
  ///     = (input(n, c, h, w) - mean(c)) / sqrt(var(c) + eps) * weight(c)
  ///         + bias(c)
  ///     = input(n, c, h, w) * inv_var(c) * weight(c)
  ///         - mean(c) * inv_var(c) * weight(c) + bias(c),
  /// where inv_var(c) = 1 / sqrt(var(c) + eps).
  /// So the linear term, alpha(c) = inv_var(c) * weight(c),
  ///   the constant term beta(c) = bias(c) - mean(c) * inv_var(c) * weight(c)
  /// Note that this is only a good idea if (input_size >> c), in degenerate
  /// cases where image_size == 1 && batch_size == 1, it is slow.
  for (const auto c : c10::irange(n_channel)) {
    scalar_t mean, invstd;
    if (train) {
      mean = save_mean_a[c];
      invstd = save_invstd_a[c];
    } else {
      mean = running_mean_a[c];
      invstd = 1 / std::sqrt(running_var_a[c] + static_cast<scalar_t>(eps));
    }
    scalar_t weight_v = weight_data ? weight_data[c] : 1;
    scalar_t bias_v = bias_data ? bias_data[c] : 0;
    alpha[c] = invstd * weight_v;
    beta[c] = bias_v - mean * alpha[c];
  }
}

/// A fast path for CPU inference and training forward when all tensors are contiguous.
template<typename scalar_t>
void batch_norm_cpu_contiguous_impl(Tensor& output, const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {

  using Vec = Vectorized<scalar_t>;
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;

  Tensor alpha = at::empty({n_channel}, input.options());
  Tensor beta = at::empty({n_channel}, input.options());
  scalar_t* alpha_data = alpha.data_ptr<scalar_t>();
  scalar_t* beta_data = beta.data_ptr<scalar_t>();

  batch_norm_cpu_collect_linear_and_constant_terms<scalar_t>(
     alpha_data, beta_data, n_channel, weight, bias,
     save_mean, save_invstd, running_mean, running_var, train, eps);

  scalar_t* output_data = output.data_ptr<scalar_t>();
  const scalar_t* input_data = input.data_ptr<scalar_t>();

  // Apply the linear terms to the input,
  // output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)
  if (image_size != 1) {
    const int64_t loop_size = image_size - (image_size % Vec::size());
    at::parallel_for(0, n_batch * n_channel, 1, [&](int64_t begin, int64_t end) {
      int64_t n = 0;
      int64_t c = 0;
      data_index_init(begin, n, n_batch, c, n_channel);

      for (const auto i : c10::irange(begin, end)) {
        const Vec alpha_vec(alpha_data[c]);
        const Vec beta_vec(beta_data[c]);
        int64_t offset = i * image_size;
        int64_t d = 0;
        for (; d < loop_size; d += Vec::size()) {
          Vec data_vec = Vec::loadu(input_data + offset + d);
          Vec output_vec = data_vec * alpha_vec + beta_vec;
          output_vec.store(output_data + offset + d);
        }
        if (image_size - d > 0) {
          Vec data_vec = Vec::loadu(input_data + offset + d, image_size - d);
          Vec output_vec = data_vec * alpha_vec + beta_vec;
          output_vec.store(output_data + offset + d, image_size - d);
        }
        // move on to next index
        data_index_step(n, n_batch, c, n_channel);
      }
    });
  } else {
    // image_size == 1
    const int64_t loop_size = n_channel - (n_channel % Vec::size());
    at::parallel_for(0, n_batch, 1, [&](int64_t begin, int64_t end) {
      for (const auto n : c10::irange(begin, end)) {
        int64_t offset = n * n_channel;
        int64_t d = 0;
        for (; d < loop_size; d += Vec::size()) {
          Vec alpha_vec = Vec::loadu(alpha_data + d);
          Vec beta_vec = Vec::loadu(beta_data + d);
          Vec data_vec = Vec::loadu(input_data + offset + d);
          Vec output_vec = data_vec * alpha_vec + beta_vec;
          output_vec.store(output_data + offset + d);
        }
        if (n_channel - d > 0) {
          Vec alpha_vec = Vec::loadu(alpha_data + d, n_channel - d);
          Vec beta_vec = Vec::loadu(beta_data + d, n_channel - d);
          Vec data_vec = Vec::loadu(input_data + offset + d, n_channel - d);
          Vec output_vec = data_vec * alpha_vec + beta_vec;
          output_vec.store(output_data + offset + d, n_channel - d);
        }
      }
    });
  }
}

template <typename scalar_t>
void batch_norm_cpu_channels_last_impl(Tensor& output, const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& runnning_var, bool train, double eps) {

  using Vec = Vectorized<scalar_t>;
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;

  Tensor alpha = at::empty({n_channel}, input.options());
  Tensor beta = at::empty({n_channel}, input.options());
  scalar_t* alpha_data = alpha.data_ptr<scalar_t>();
  scalar_t* beta_data = beta.data_ptr<scalar_t>();

  batch_norm_cpu_collect_linear_and_constant_terms<scalar_t>(
      alpha_data, beta_data, n_channel, weight, bias,
      save_mean, save_invstd, running_mean, runnning_var, train, eps);

  scalar_t* output_data = output.data_ptr<scalar_t>();
  const scalar_t* input_data = input.data_ptr<scalar_t>();

  // Apply the linear terms to the input,
  // output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)
  const int64_t loop_size = n_channel - (n_channel % Vec::size());
  at::parallel_for(0, n_batch * image_size, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      int64_t offset = i * n_channel;
      int64_t d = 0;
      // vectorize on channel dimension, for normal batch_norm input size,
      // alpha/beta should fit in L1 cache, otherwise consider blocking.
      for (; d < loop_size; d += Vec::size()) {
        Vec alpha_vec = Vec::loadu(alpha_data + d);
        Vec beta_vec = Vec::loadu(beta_data + d);
        Vec data_vec = Vec::loadu(input_data + offset + d);
        Vec output_vec = data_vec * alpha_vec + beta_vec;
        output_vec.store(output_data + offset + d);
      }
      if (n_channel - d > 0) {
        Vec alpha_vec = Vec::loadu(alpha_data + d, n_channel - d);
        Vec beta_vec = Vec::loadu(beta_data + d, n_channel - d);
        Vec data_vec = Vec::loadu(input_data + offset + d, n_channel - d);
        Vec output_vec = data_vec * alpha_vec + beta_vec;
        output_vec.store(output_data + offset + d, n_channel - d);
      }
    }
  });
}

template <typename scalar_t>
void batch_norm_cpu_collect_stats_contiguous_impl(
    Tensor& mean, Tensor& var_sum, const Tensor& input) {

  using accscalar_t = at::acc_type<scalar_t, false>;
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;
  int64_t N = input.numel() / n_channel;

  const scalar_t* input_data = input.data_ptr<scalar_t>();
  scalar_t* mean_data = mean.data_ptr<scalar_t>();
  scalar_t* var_sum_data = var_sum.data_ptr<scalar_t>();

  // parallel dim reduce on 'channel'
  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      // compute mean per input
      accscalar_t sum = 0;
      for (const auto n : c10::irange(n_batch)) {
        for (const auto i : c10::irange(image_size)) {
          auto offset = n * n_channel * image_size + c * image_size + i;
          sum += input_data[offset];
        }
      }
      scalar_t mean = sum / N;
      mean_data[c] = mean;

      // compute variance per input
      accscalar_t _var_sum = 0;
      for (const auto n : c10::irange(n_batch)) {
        for (const auto i : c10::irange(image_size)) {
          auto offset = n * n_channel * image_size + c * image_size + i;
          auto x = input_data[offset];
          _var_sum += (x - mean) * (x - mean);
        }
      }
      var_sum_data[c] = _var_sum;
    }
  });
}

template <typename scalar_t>
void batch_norm_cpu_collect_stats_channels_last_impl(
    Tensor& mean, Tensor& var_sum, const Tensor& input) {

  using Vec = Vectorized<scalar_t>;
  using accscalar_t = at::acc_type<scalar_t, false>;
  int64_t n_channel = input.size(1);
  int64_t N = input.numel() / n_channel;

  const scalar_t* input_data = input.data_ptr<scalar_t>();
  scalar_t* mean_data = mean.data_ptr<scalar_t>();
  scalar_t* var_sum_data = var_sum.data_ptr<scalar_t>();

  // Typical vertical reduce from shape of {NHW, C} to {C}.
  // Apply two path parallel reduction:
  // First path: allocate an immediate buffer of size {max_threads, C}, parallel along dim0,
  //    {NHW, C} => {max_threads, C}
  //
  // Second path: parallel along dim1 of the immediate buffer,
  //    {max_threads, C} => {C}
  //
  // Normal size of C should fit in L1, otherwise consider blocking on C.
  //
  int num_threads = at::get_num_threads();
  Tensor buffer = at::empty({num_threads, n_channel}, input.options()).zero_();
  scalar_t* buffer_data = buffer.data_ptr<scalar_t>();

  // compute mean per input
  at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads,
                "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    scalar_t* buffer_ptr = buffer_data + tid * n_channel;
    for (const auto i : c10::irange(begin, end)) {
      const scalar_t* x_ptr = input_data + i * n_channel;
      vec::map2<scalar_t>(
          [](Vec x, Vec y) { return x + y; },
          buffer_ptr,
          x_ptr,
          buffer_ptr,
          n_channel);
    }
  });

  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      accscalar_t sum = 0;
      for (const auto t : c10::irange(num_threads)) {
        sum += buffer_data[t * n_channel + c];
      }
      scalar_t mean = sum / N;
      mean_data[c] = mean;
    }
  });

  // compute variance per input, reuse the immediate buffer
  buffer.zero_();
  at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    scalar_t* buffer_ptr = buffer_data + tid * n_channel;
    for (const auto i : c10::irange(begin, end)) {
      const scalar_t* x_ptr = input_data + i * n_channel;
      vec::map3<scalar_t>(
          [](Vec x, Vec y, Vec mean) { return y + (x - mean) * (x - mean); },
          buffer_ptr,
          x_ptr,
          buffer_ptr,
          mean_data,
          n_channel);
    }
  });

  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      accscalar_t _var_sum = 0;
      for (const auto t : c10::irange(num_threads)) {
        _var_sum += buffer_data[t * n_channel + c];
      }
      var_sum_data[c] = _var_sum;
    }
  });
}

template <typename scalar_t>
void batch_norm_cpu_backward_contiguous_impl(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps) {

  using Vec = Vectorized<scalar_t>;
  using accscalar_t = at::acc_type<scalar_t, false>;
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;
  int64_t N = input.numel() / n_channel;

  const scalar_t* grad_output_data = grad_output.data_ptr<scalar_t>();
  const scalar_t* input_data = input.data_ptr<scalar_t>();

  scalar_t* grad_input_data = grad_input.defined() ? grad_input.data_ptr<scalar_t>() : nullptr;
  scalar_t* grad_weight_data = grad_weight.defined() ? grad_weight.data_ptr<scalar_t>() : nullptr;
  scalar_t* grad_bias_data = grad_bias.defined() ? grad_bias.data_ptr<scalar_t>() : nullptr;
  const bool grad_input_null = grad_input_data == nullptr;
  const bool grad_weight_null = grad_weight_data == nullptr;
  const bool grad_bias_null = grad_bias_data == nullptr;

  auto weight_a = conditional_accessor_1d<scalar_t>(weight);
  auto save_mean_a = conditional_accessor_1d<scalar_t>(save_mean);
  auto save_invstd_a = conditional_accessor_1d<scalar_t>(save_invstd);
  auto running_mean_a = conditional_accessor_1d<scalar_t>(running_mean);
  auto running_var_a = conditional_accessor_1d<scalar_t>(running_var);

  // parallel dim reduce on 'channel'
  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      scalar_t w = weight.defined() ? weight_a[c] : 1;

      scalar_t mean, invstd;
      if (train) {
        mean = save_mean_a[c];
        invstd = save_invstd_a[c];
      } else {
        mean = running_mean_a[c];
        invstd = 1 / std::sqrt(running_var_a[c] + eps);
      }

      // reduce over grad_output in feature plane
      // compute 1) sum; 2) dot product of Q(X) and dY.
      // fuse into a single loop to reuse dY
      //
      accscalar_t sum = 0;
      accscalar_t dotp = 0;
      for (const auto n : c10::irange(n_batch)) {
        const scalar_t* x_ptr = input_data + n * n_channel * image_size + c * image_size;
        const scalar_t* dy_ptr = grad_output_data + n * n_channel * image_size + c * image_size;

        sum += vec::reduce_all<scalar_t>(
            [](Vec& x, Vec& y) { return x + y; },
            dy_ptr,
            image_size);

        dotp += vec::map2_reduce_all<scalar_t>(
            [mean](Vec x, Vec dy) { return (x - Vec(mean)) * dy; },
            [](Vec x, Vec y) { return x + y; },
            x_ptr,
            dy_ptr,
            image_size);
      }

      if (!grad_input_null) {
        if (train) {
          scalar_t k = (scalar_t) dotp * invstd * invstd / N;
          scalar_t grad_mean = sum / N;

          for (const auto n : c10::irange(n_batch)) {
            const scalar_t* x_ptr = input_data + n * n_channel * image_size + c * image_size;
            scalar_t* dx_ptr = grad_input_data + n * n_channel * image_size + c * image_size;
            const scalar_t* dy_ptr = grad_output_data + n * n_channel * image_size + c * image_size;

            // Scalar math:
            // for (const auto j : c10::irange(image_size)) {
            //   scalar_t dx = (x_ptr[j] - mean) * k;
            //   dx_ptr[j] = (dy_ptr[j] - grad_mean - dx) * invstd * w;
            // }
            vec::map2<scalar_t>(
                [=](Vec x, Vec dy) {
                  Vec dx = (x - Vec(mean)) * Vec(k);
                  return (dy - Vec(grad_mean) - dx) * Vec(invstd) * Vec(w);
                },
                dx_ptr,
                x_ptr,
                dy_ptr,
                image_size);
          }
        } else { // evaluation mode
          for (const auto n : c10::irange(n_batch)) {
            scalar_t* dx_ptr = grad_input_data + n * n_channel * image_size + c * image_size;
            const scalar_t* dy_ptr = grad_output_data + n * n_channel * image_size + c * image_size;

            // Scalar math:
            // for (const auto j : c10::irange(image_size)) {
            //   dx_ptr[j] = dy_ptr[j] * invstd * w;
            // }
            vec::map<scalar_t>(
                [=](Vec dy) { return dy * Vec(invstd) * Vec(w); },
                dx_ptr,
                dy_ptr,
                image_size);
          }
        }
      }

      if (!grad_weight_null) {
        grad_weight_data[c] = dotp * invstd;
      }

      if (!grad_bias_null) {
        grad_bias_data[c] = sum;
      }
    }
  });
}

template <typename scalar_t>
void batch_norm_cpu_backward_channels_last_impl(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps) {

  using Vec = Vectorized<scalar_t>;
  using accscalar_t = at::acc_type<scalar_t, false>;
  int64_t n_channel = input.size(1);
  int64_t N = input.numel() / n_channel;

  const scalar_t* grad_output_data = grad_output.data_ptr<scalar_t>();
  const scalar_t* input_data = input.data_ptr<scalar_t>();

  scalar_t* grad_input_data = grad_input.defined() ? grad_input.data_ptr<scalar_t>() : nullptr;
  scalar_t* grad_weight_data = grad_weight.defined() ? grad_weight.data_ptr<scalar_t>() : nullptr;
  scalar_t* grad_bias_data = grad_bias.defined() ? grad_bias.data_ptr<scalar_t>() : nullptr;

  scalar_t* save_mean_data = conditional_data_ptr<scalar_t>(save_mean);
  scalar_t* save_invstd_data = conditional_data_ptr<scalar_t>(save_invstd);
  scalar_t* running_mean_data = conditional_data_ptr<scalar_t>(running_mean);
  scalar_t* running_var_data = conditional_data_ptr<scalar_t>(running_var);

  Tensor weight_ = weight.defined() ? weight : at::ones({n_channel}, input.options());
  const scalar_t* weight_data = weight_.data_ptr<scalar_t>();

  scalar_t* mean_ptr = nullptr;
  scalar_t* invstd_ptr = nullptr;
  Tensor invstd = at::empty({0}, input.options());
  if (train) {
    mean_ptr = save_mean_data;
    invstd_ptr = save_invstd_data;
  } else {
    mean_ptr = running_mean_data;

    invstd.resize_({n_channel});
    invstd_ptr = invstd.data_ptr<scalar_t>();
    for (const auto c : c10::irange(n_channel)) {
      invstd_ptr[c] = 1 / std::sqrt(running_var_data[c] + eps);
    }
  }

  // Typical vertical reduce from shape of {NHW, C} to {C}.
  // Apply two path parallel reduction:
  // First path: allocate an immediate buffer of size {2, max_threads, C}, parallel along dim0,
  //    sum = buffer[0], dotp = buffer[2]
  //
  // Second path: parallel along dim1 of the immediate buffer.
  //
  int num_threads = at::get_num_threads();
  Tensor buffer = at::empty({2, num_threads, n_channel}, input.options()).zero_();
  scalar_t* sum_data = buffer.data_ptr<scalar_t>();
  scalar_t* dotp_data = sum_data + num_threads * n_channel;

  // compute sum and dotp per feature plain,
  // fuse into a single loop to reuse grad_output in L1.
  at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    scalar_t* sum_ptr = sum_data + tid * n_channel;
    scalar_t* dotp_ptr = dotp_data + tid * n_channel;
    for (const auto i : c10::irange(begin, end)) {
      const scalar_t* x_ptr = input_data + i * n_channel;
      const scalar_t* dy_ptr = grad_output_data + i * n_channel;

      vec::map2<scalar_t>(
          [](Vec sum, Vec dy) { return sum + dy; },
          sum_ptr,
          sum_ptr,
          dy_ptr,
          n_channel);

      vec::map4<scalar_t>(
          [](Vec dotp, Vec x, Vec mean, Vec dy) { return dotp + (x - mean) * dy; },
          dotp_ptr,
          dotp_ptr,
          x_ptr,
          mean_ptr,
          dy_ptr,
          n_channel);
    }
  });

  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      // store the final result of sum and dotp in the 1st lane of immediate buffer,
      // so that we won't need to allocate anther buffer to store the temp values.
      accscalar_t _sum = 0;
      for (const auto t : c10::irange(num_threads)) {
        _sum += sum_data[t * n_channel + c];
      }
      sum_data[/* 0 * n_channel + */c] = _sum;

      accscalar_t _dotp = 0;
      for (const auto t : c10::irange(num_threads)) {
        _dotp += dotp_data[t * n_channel + c];
      }
      dotp_data[/* 0 * n_channel + */c] = _dotp;
    }
  });

  // compute grad_input
  const int64_t loop_size = n_channel - (n_channel % Vec::size());
  if (grad_input.defined()) {
    at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        scalar_t* dx_ptr = grad_input_data + i * n_channel;
        const scalar_t* x_ptr = input_data + i * n_channel;
        const scalar_t* dy_ptr = grad_output_data + i * n_channel;
        if (train) {
          int64_t d = 0;
          for (; d < loop_size; d += Vec::size()) {
            Vec x = Vec::loadu(x_ptr + d);
            Vec mean = Vec::loadu(mean_ptr + d);
            Vec dotp = Vec::loadu(dotp_data + d);
            Vec invstd = Vec::loadu(invstd_ptr + d);
            Vec k = dotp * invstd * invstd / Vec(N);
            Vec dx = (x - mean) * k;
            Vec dy = Vec::loadu(dy_ptr + d);
            Vec grad_mean = Vec::loadu(sum_data + d) / Vec(N);
            Vec w = Vec::loadu(weight_data + d);
            dx = (dy - grad_mean - dx) * invstd * w;
            dx.store(dx_ptr + d);
          }
          if (n_channel - d > 0) {
            Vec x = Vec::loadu(x_ptr + d, n_channel - d);
            Vec mean = Vec::loadu(mean_ptr + d, n_channel - d);
            Vec dotp = Vec::loadu(dotp_data + d, n_channel - d);
            Vec invstd = Vec::loadu(invstd_ptr + d, n_channel - d);
            Vec k = dotp * invstd * invstd / Vec(N);
            Vec dx = (x - mean) * k;
            Vec dy = Vec::loadu(dy_ptr + d, n_channel - d);
            Vec grad_mean = Vec::loadu(sum_data + d, n_channel - d) / Vec(N);
            Vec w = Vec::loadu(weight_data + d, n_channel - d);
            dx = (dy - grad_mean - dx) * invstd * w;
            dx.store(dx_ptr + d, n_channel - d);
          }
        } else { // evaluation mode
          int64_t d = 0;
          for (; d < loop_size; d += Vec::size()) {
            Vec dy = Vec::loadu(dy_ptr + d);
            Vec invstd = Vec::loadu(invstd_ptr + d);
            Vec w = Vec::loadu(weight_data + d);
            Vec dx = dy * invstd * w;
            dx.store(dx_ptr + d);
          }
          if (n_channel - d > 0) {
            Vec dy = Vec::loadu(dy_ptr + d, n_channel - d);
            Vec invstd = Vec::loadu(invstd_ptr + d, n_channel - d);
            Vec w = Vec::loadu(weight_data + d, n_channel - d);
            Vec dx = dy * invstd * w;
            dx.store(dx_ptr + d, n_channel - d);
          }
        }
      }
    });
  }

  if (grad_weight.defined()) {
    // grad_weight = dotp * invstd
    vec::map2<scalar_t>(
        [](Vec dotp, Vec invstd) { return dotp * invstd; },
        grad_weight_data,
        dotp_data,
        invstd_ptr,
        n_channel);
  }

  // grad_bias = sum
  if (grad_bias.defined()) {
    vec::map<scalar_t>(
        [](Vec sum) { return sum; },
        grad_bias_data,
        sum_data,
        n_channel);
  }
}

void batch_norm_cpu_kernel(Tensor& output, const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& save_mean,  const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {
  if (input.is_contiguous()) {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batch_norm_cpu_contiguous", [&] {
      batch_norm_cpu_contiguous_impl<scalar_t>(output, input, weight, bias,
          save_mean, save_invstd, running_mean, running_var, train, eps);
    });
  } else if (input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batch_norm_cpu_channels_last", [&] {
      batch_norm_cpu_channels_last_impl<scalar_t>(output, input, weight, bias,
          save_mean, save_invstd, running_mean, running_var, train, eps);
    });
  } else {
    TORCH_CHECK(false, "batch_norm_cpu_kernel: expecting input to be contiguous.");
  }
}

void batch_norm_cpu_collect_stats_kernel(
    Tensor& mean, Tensor& var_sum, const Tensor& input) {
  int64_t image_size = input.numel() / input.size(0) / input.size(1);
  if (input.is_contiguous()) {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batch_norm_cpu_collect_stats_contiguous", [&] {
      if (image_size == 1) { // NC11 is also channels last
        batch_norm_cpu_collect_stats_channels_last_impl<scalar_t>(mean, var_sum, input);
      } else {
        batch_norm_cpu_collect_stats_contiguous_impl<scalar_t>(mean, var_sum, input);
      }
    });
  } else if (input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batch_norm_cpu_collect_stats_channels_last", [&] {
      batch_norm_cpu_collect_stats_channels_last_impl<scalar_t>(mean, var_sum, input);
    });
  } else {
    TORCH_CHECK(false, "batch_norm_cpu_collect_stats_kernel: expecting input to be contiguous.");
  }
}

void batch_norm_cpu_backward_kernel(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps) {
  int64_t image_size = input.numel() / input.size(0) / input.size(1);
  if (input.is_contiguous()) {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batch_norm_cpu_backward_contiguous", [&] {
      if (image_size == 1) { // NC11 is also channels last
        batch_norm_cpu_backward_channels_last_impl<scalar_t>(grad_input, grad_weight, grad_bias,
            grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
      } else {
        batch_norm_cpu_backward_contiguous_impl<scalar_t>(grad_input, grad_weight, grad_bias,
            grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
      }
    });
  } else if (input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batch_norm_cpu_backward_channels_last", [&] {
      batch_norm_cpu_backward_channels_last_impl<scalar_t>(grad_input, grad_weight, grad_bias,
          grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
    });
  } else {
    TORCH_CHECK(false, "batch_norm_cpu_backward_kernel: expecting input to be contiguous.");
  }
}

}// anonymous namespace

REGISTER_DISPATCH(batch_norm_cpu_stub, &batch_norm_cpu_kernel);
REGISTER_DISPATCH(batch_norm_cpu_collect_stats_stub, &batch_norm_cpu_collect_stats_kernel);
REGISTER_DISPATCH(batch_norm_cpu_backward_stub, &batch_norm_cpu_backward_kernel);

}} // namespace at::native
