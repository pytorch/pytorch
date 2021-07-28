#include <ATen/native/batch_norm.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>

namespace at { namespace native {
namespace {

using namespace vec;

template<typename scalar_t, typename acc_t=scalar_t>
void batch_norm_cpu_collect_linear_and_constant_terms(
    acc_t* alpha, acc_t* beta, int64_t n_channel,
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
  for (int64_t c = 0; c < n_channel; c++) {
    acc_t mean, invstd;
    if (train) {
      mean = acc_t(save_mean_a[c]);
      invstd = acc_t(save_invstd_a[c]);
    } else {
      mean = acc_t(running_mean_a[c]);
      invstd = 1 / std::sqrt(running_var_a[c] + static_cast<acc_t>(eps));
    }
    acc_t weight_v = weight_data ? acc_t(weight_data[c]) : acc_t(1);
    acc_t bias_v = bias_data ? acc_t(bias_data[c]) : acc_t(0);
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

      for (int64_t i = begin; i < end; i++) {
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
      for (int64_t n = begin; n < end; n++) {
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
    for (int64_t i = begin; i < end; i++) {
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
    for (int64_t c = begin; c < end; c++) {
      // compute mean per input
      accscalar_t sum = 0;
      for (int64_t n = 0; n < n_batch; n++) {
        for (int64_t i = 0; i < image_size; i++) {
          auto offset = n * n_channel * image_size + c * image_size + i;
          sum += input_data[offset];
        }
      }
      scalar_t mean = sum / N;
      mean_data[c] = mean;

      // compute variance per input
      accscalar_t _var_sum = 0;
      for (int64_t n = 0; n < n_batch; n++) {
        for (int64_t i = 0; i < image_size; i++) {
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
    for (int64_t i = begin; i < end; i++) {
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
    for (int64_t c = begin; c < end; c++) {
      accscalar_t sum = 0;
      for (int64_t t = 0; t < num_threads; t++) {
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
    for (int64_t i = begin; i < end; i++) {
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
    for (int64_t c = begin; c < end; c++) {
      accscalar_t _var_sum = 0;
      for (int64_t t = 0; t < num_threads; t++) {
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
    for (int64_t c = begin; c < end; c++) {
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
      for (int64_t n = 0; n < n_batch; n++) {
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

          for (int64_t n = 0; n < n_batch; n++) {
            const scalar_t* x_ptr = input_data + n * n_channel * image_size + c * image_size;
            scalar_t* dx_ptr = grad_input_data + n * n_channel * image_size + c * image_size;
            const scalar_t* dy_ptr = grad_output_data + n * n_channel * image_size + c * image_size;

            // Scalar math:
            // for (int64_t j = 0; j < image_size; ++j) {
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
          for (int64_t n = 0; n < n_batch; n++) {
            scalar_t* dx_ptr = grad_input_data + n * n_channel * image_size + c * image_size;
            const scalar_t* dy_ptr = grad_output_data + n * n_channel * image_size + c * image_size;

            // Scalar math:
            // for (int64_t j = 0; j < image_size; ++j) {
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
    for (int64_t c = 0; c < n_channel; c++) {
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
    for (int64_t i = begin; i < end; i++) {
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
    for (int64_t c = begin; c < end; c++) {
      // store the final result of sum and dotp in the 1st lane of immediate buffer,
      // so that we won't need to allocate anther buffer to store the temp values.
      accscalar_t _sum = 0;
      for (int64_t t = 0; t < num_threads; t++) {
        _sum += sum_data[t * n_channel + c];
      }
      sum_data[/* 0 * n_channel + */c] = _sum;

      accscalar_t _dotp = 0;
      for (int64_t t = 0; t < num_threads; t++) {
        _dotp += dotp_data[t * n_channel + c];
      }
      dotp_data[/* 0 * n_channel + */c] = _dotp;
    }
  });

  // compute grad_input
  const int64_t loop_size = n_channel - (n_channel % Vec::size());
  if (grad_input.defined()) {
    at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; i++) {
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

// BFloat16 specification
template<>
void batch_norm_cpu_contiguous_impl<BFloat16>(Tensor& output, const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {

  using bVec = Vectorized<BFloat16>;
  using fVec = Vectorized<float>;
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;

  // store alpha and beta in float
  Tensor alpha = at::empty({n_channel}, input.options().dtype(kFloat));
  Tensor beta = at::empty({n_channel}, input.options().dtype(kFloat));
  float* alpha_data = alpha.data_ptr<float>();
  float* beta_data = beta.data_ptr<float>();

  if (weight.scalar_type() == ScalarType::BFloat16) {
    batch_norm_cpu_collect_linear_and_constant_terms<BFloat16, float>(
       alpha_data, beta_data, n_channel, weight, bias,
       save_mean, save_invstd, running_mean, running_var, train, eps);
  } else {
    batch_norm_cpu_collect_linear_and_constant_terms<float, float>(
       alpha_data, beta_data, n_channel, weight, bias,
       save_mean, save_invstd, running_mean, running_var, train, eps);
  }

  BFloat16* output_data = output.data_ptr<BFloat16>();
  const BFloat16* input_data = input.data_ptr<BFloat16>();

  if (image_size != 1) {
    const int64_t loop_size = image_size - (image_size % bVec::size());
    at::parallel_for(0, n_batch * n_channel, 1, [&](int64_t begin, int64_t end) {
      int64_t n = 0;
      int64_t c = 0;
      data_index_init(begin, n, n_batch, c, n_channel);

      for (int64_t i = begin; i < end; i++) {
        // convert alpha and beta vector to float
        const fVec alpha_fvec(alpha_data[c]);
        const fVec beta_fvec(beta_data[c]);
        int64_t offset = i * image_size;
        int64_t d = 0;
        for (; d < loop_size; d += bVec::size()) {
          bVec data_bvec = bVec::loadu(input_data + offset + d);
          fVec data_fvec0, data_fvec1;
          std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);

          fVec output_fvec0 = data_fvec0 * alpha_fvec + beta_fvec;
          fVec output_fvec1 = data_fvec1 * alpha_fvec + beta_fvec;
          bVec output_bvec = convert_float_bfloat16(output_fvec0, output_fvec1);
          output_bvec.store(output_data + offset + d);
        }
        for (; d < image_size; d++) {
          float data_val = input_data[offset + d];
          float alpha_val = alpha_data[c];
          float beta_val = beta_data[c];
          output_data[offset + d] = data_val * alpha_val + beta_val;
        }
        data_index_step(n, n_batch, c, n_channel);
      }
    });
  } else {
    // image_size == 1
    const int64_t loop_size = n_channel - (n_channel % bVec::size());
    at::parallel_for(0, n_batch, 1, [&](int64_t begin, int64_t end) {
      for (int64_t n = begin; n < end; n++) {
        int64_t offset = n * n_channel;
        int64_t d = 0;
        for (; d < loop_size; d += bVec::size()) {
          bVec alpha_bvec = bVec::loadu(alpha_data + d);
          fVec alpha_fvec0, alpha_fvec1;
          std::tie(alpha_fvec0, alpha_fvec1) = convert_bfloat16_float(alpha_bvec);
          bVec beta_bvec = bVec::loadu(beta_data + d);
          fVec beta_fvec0, beta_fvec1;
          std::tie(beta_fvec0, beta_fvec1) = convert_bfloat16_float(beta_bvec);
          bVec data_bvec = bVec::loadu(input_data + offset + d);
          fVec data_fvec0, data_fvec1;
          std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);

          fVec output_fvec0 = data_fvec0 * alpha_fvec0 + beta_fvec0;
          fVec output_fvec1 = data_fvec1 * alpha_fvec1 + beta_fvec1;
          bVec output_bvec = convert_float_bfloat16(output_fvec0, output_fvec1);
          output_bvec.store(output_data + offset + d);
        }
        for (; d < n_channel; d++) {
          float data_val = input_data[offset + d];
          float alpha_val = alpha_data[d];
          float beta_val = beta_data[d];
          output_data[offset + d] = data_val * alpha_val + beta_val;
        }
      }
    });
  }
}

template <>
void batch_norm_cpu_channels_last_impl<BFloat16>(Tensor& output, const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& runnning_var, bool train, double eps) {

  using bVec = Vectorized<BFloat16>;
  using fVec = Vectorized<float>;
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;

  Tensor alpha = at::empty({n_channel}, input.options().dtype(kFloat));
  Tensor beta = at::empty({n_channel}, input.options().dtype(kFloat));
  float* alpha_data = alpha.data_ptr<float>();
  float* beta_data = beta.data_ptr<float>();

  if (weight.scalar_type() == ScalarType::BFloat16) {
    batch_norm_cpu_collect_linear_and_constant_terms<BFloat16, float>(
        alpha_data, beta_data, n_channel, weight, bias,
        save_mean, save_invstd, running_mean, runnning_var, train, eps);
  } else {
    batch_norm_cpu_collect_linear_and_constant_terms<float, float>(
        alpha_data, beta_data, n_channel, weight, bias,
        save_mean, save_invstd, running_mean, runnning_var, train, eps);
  }

  BFloat16* output_data = output.data_ptr<BFloat16>();
  const BFloat16* input_data = input.data_ptr<BFloat16>();

  const int64_t loop_size = n_channel - (n_channel % bVec::size());
  at::parallel_for(0, n_batch * image_size, 1, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      int64_t offset = i * n_channel;
      int64_t d = 0;
      for (; d < loop_size; d += bVec::size()) {
        fVec alpha_fvec0 = fVec::loadu(alpha_data + d);
        fVec alpha_fvec1 = fVec::loadu(alpha_data + d + fVec::size());
        fVec beta_fvec0 = fVec::loadu(beta_data + d);
        fVec beta_fvec1 = fVec::loadu(beta_data + d + fVec::size());
        bVec data_bvec = bVec::loadu(input_data + offset + d);
        fVec data_fvec0, data_fvec1;
        std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);

        fVec output_fvec0 = data_fvec0 * alpha_fvec0 + beta_fvec0;
        fVec output_fvec1 = data_fvec1 * alpha_fvec1 + beta_fvec1;
        bVec output_bvec = convert_float_bfloat16(output_fvec0, output_fvec1);
        output_bvec.store(output_data + offset + d);
      }
      for (; d < n_channel; d++) {
        float data_val = input_data[offset + d];
        float alpha_val = alpha_data[d];
        float beta_val = beta_data[d];
        output_data[offset + d] = data_val * alpha_val + beta_val;
      }
    }
  });
}

template <typename scalar_t>
static inline scalar_t vec_acc(const Vectorized<scalar_t>& vec) {
  constexpr int N = Vectorized<scalar_t>::size();
  scalar_t vec_arr[N];
  vec.store(vec_arr);
  scalar_t acc = scalar_t(0);
  for (int i = 0; i < N; i++) {
    acc += vec_arr[i];
  }
  return acc;
}

template <>
void batch_norm_cpu_collect_stats_contiguous_impl<BFloat16>(
    Tensor& mean, Tensor& var_sum, const Tensor& input) {

  using bVec = Vectorized<BFloat16>;
  using fVec = Vectorized<float>;
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;
  int64_t N = input.numel() / n_channel;

  const BFloat16* input_data = input.data_ptr<BFloat16>();
  float* mean_data = mean.data_ptr<float>();
  float* var_sum_data = var_sum.data_ptr<float>();

  // Use float as accumulation type
  int64_t loop_size = image_size - (image_size % bVec::size());
  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; c++) {
      // compute mean per input
      float sum_val = float(0);
      fVec sum_fvec = fVec(float(0));
      for (int64_t n = 0; n < n_batch; n++) {
        auto offset = n * n_channel * image_size + c * image_size;
        int64_t d = 0;
        for (; d < loop_size; d += bVec::size()) {
          bVec data_bvec = bVec::loadu(input_data + offset + d);
          fVec data_fvec0, data_fvec1;
          std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
          sum_fvec += data_fvec0;
          sum_fvec += data_fvec1;
        }
        for (; d < image_size; d++) {
          sum_val += input_data[offset + d];
        }
      }
      sum_val += vec_acc(sum_fvec);
      float mean_val = sum_val / N;
      mean_data[c] = float(mean_val);

      fVec mean_fvec = fVec(mean_val);
      // compute variance per input
      float var_val = float(0);
      fVec var_fvec = fVec(float(0));
      for (int64_t n = 0; n < n_batch; n++) {
        auto offset = n * n_channel * image_size + c * image_size;
        int64_t d = 0;
        for (; d < loop_size; d += bVec::size()) {
          bVec data_bvec = bVec::loadu(input_data + offset + d);
          fVec data_fvec0, data_fvec1;
          std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
          var_fvec += (data_fvec0 - mean_fvec) * (data_fvec0 - mean_fvec);
          var_fvec += (data_fvec1 - mean_fvec) * (data_fvec1 - mean_fvec);
        }
        for (; d < image_size; d++) {
          float data_val = input_data[offset + d];
          var_val += (data_val - mean_val) * (data_val - mean_val);
        }
      }
      var_val += vec_acc(var_fvec);
      var_sum_data[c] = float(var_val);
    }
  });
}

template <>
void batch_norm_cpu_collect_stats_channels_last_impl<BFloat16>(
    Tensor& mean, Tensor& var_sum, const Tensor& input) {

  using bVec = Vectorized<BFloat16>;
  using fVec = Vectorized<float>;
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;
  int64_t N = input.numel() / n_channel;

  const BFloat16* input_data = input.data_ptr<BFloat16>();
  float* mean_data = mean.data_ptr<float>();
  float* var_sum_data = var_sum.data_ptr<float>();

  // temp buffer for vertical reduce, use float as accumulation type
  int num_threads = at::get_num_threads();
  Tensor buffer = at::empty({num_threads + 1, n_channel}, input.options().dtype(kFloat)).zero_();
  float* buffer_data = buffer.data_ptr<float>();

  // compute mean per input
  int64_t loop_size = n_channel - (n_channel % bVec::size());
  at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    float* buffer_ptr = buffer_data + tid * n_channel;
    for (int64_t i = begin; i < end; i++) {
      int64_t offset = i * n_channel;
      int64_t d = 0;
      for (; d < loop_size; d += bVec::size()) {
        bVec data_bvec = bVec::loadu(input_data + offset + d);
        fVec data_fvec0, data_fvec1;
        std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
        fVec sum_fvec0 = fVec::loadu(buffer_ptr + d) + data_fvec0;
        fVec sum_fvec1 = fVec::loadu(buffer_ptr + d + fVec::size()) + data_fvec1;
        sum_fvec0.store(buffer_ptr + d);
        sum_fvec1.store(buffer_ptr + d + fVec::size());
      }
      for (; d < n_channel; d++) {
        buffer_ptr[d] += input_data[offset + d];
      }
    }
  });

  // keep a copy of mean value in float32
  // so that we don't have to do bfloat16->float32 conversion when computing var
  Tensor meanf = at::empty({n_channel}, input.options().dtype(kFloat));
  float* meanf_data = meanf.data_ptr<float>();
  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; c++) {
      float sum_val = float(0);
      for (int64_t t = 0; t < num_threads; t++) {
        sum_val += buffer_data[t * n_channel + c];
      }
      float mean_val = sum_val / N;
      meanf_data[c] = mean_val;
      mean_data[c] = float(mean_val);
    }
  });

  // compute variance per input, reuse the immediate buffer
  buffer.zero_();
  at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    float* buffer_ptr = buffer_data + tid * n_channel;
    for (int64_t i = begin; i < end; i++) {
      int64_t offset = i * n_channel;
      int64_t d = 0;
      for (; d < loop_size; d += bVec::size()) {
        bVec data_bvec = bVec::loadu(input_data + offset + d);
        fVec data_fvec0, data_fvec1;
        std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
        fVec mean_fvec0 = fVec::loadu(meanf_data + d);
        fVec mean_fvec1 = fVec::loadu(meanf_data + d + fVec::size());
        fVec var_fvec0 = fVec::loadu(buffer_ptr + d);
        fVec var_fvec1 = fVec::loadu(buffer_ptr + d + fVec::size());
        var_fvec0 += (data_fvec0 - mean_fvec0) * (data_fvec0 - mean_fvec0);
        var_fvec1 += (data_fvec1 - mean_fvec1) * (data_fvec1 - mean_fvec1);
        var_fvec0.store(buffer_ptr + d);
        var_fvec1.store(buffer_ptr + d + fVec::size());
      }
      for (; d < n_channel; d++) {
        float data_val = float(input_data[offset + d]);
        buffer_ptr[d] += (data_val - meanf_data[d]) * (data_val - meanf_data[d]);
      }
    }
  });

  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; c++) {
      float var_val = float(0);
      for (int64_t t = 0; t < num_threads; t++) {
        var_val += buffer_data[t * n_channel + c];
      }
      var_sum_data[c] = float(var_val);
    }
  });
}

template <>
void batch_norm_cpu_backward_contiguous_impl<BFloat16>(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps) {

  using bVec = Vectorized<BFloat16>;
  using fVec = Vectorized<float>;
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;
  int64_t N = input.numel() / n_channel;

  const BFloat16* grad_output_data = grad_output.data_ptr<BFloat16>();
  const BFloat16* input_data = input.data_ptr<BFloat16>();

  BFloat16* grad_input_data = grad_input.defined() ? grad_input.data_ptr<BFloat16>() : nullptr;
  float* grad_weight_data = grad_weight.defined() ? grad_weight.data_ptr<float>() : nullptr;
  float* grad_bias_data = grad_bias.defined() ? grad_bias.data_ptr<float>() : nullptr;
  const bool grad_input_null = grad_input_data == nullptr;
  const bool grad_weight_null = grad_weight_data == nullptr;
  const bool grad_bias_null = grad_bias_data == nullptr;

  auto weight_a = conditional_accessor_1d<float>(weight);
  auto save_mean_a = conditional_accessor_1d<float>(save_mean);
  auto save_invstd_a = conditional_accessor_1d<float>(save_invstd);
  auto running_mean_a = conditional_accessor_1d<float>(running_mean);
  auto running_var_a = conditional_accessor_1d<float>(running_var);

  int64_t loop_size = image_size - (image_size % bVec::size());
  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; c++) {
      // explictly convert w, mean, invstd to float for following computation
      // so as to reduce rounding error
      float w = weight.defined() ? float(weight_a[c]) : float(1);

      float mean, invstd;
      if (train) {
        mean = save_mean_a[c];
        invstd = save_invstd_a[c];
      } else {
        mean = running_mean_a[c];
        invstd = 1 / std::sqrt(running_var_a[c] + eps);
      }

      // reduce over grad_output in feature plane
      // use float as accumulation type
      float sum_val{0}, dotp_val{0};
      fVec sum_fvec{0}, dotp_fvec{0};
      for (int64_t n = 0; n < n_batch; n++) {
        auto offset = n * n_channel * image_size + c * image_size;
        int64_t d = 0;
        for (; d < loop_size; d += bVec::size()) {
          bVec dy_bvec = bVec::loadu(grad_output_data + offset + d);
          fVec dy_fvec0, dy_fvec1;
          std::tie(dy_fvec0, dy_fvec1) = convert_bfloat16_float(dy_bvec);
          sum_fvec += dy_fvec0;
          sum_fvec += dy_fvec1;

          bVec x_bvec = bVec::loadu(input_data + offset + d);
          fVec x_fvec0, x_fvec1;
          std::tie(x_fvec0, x_fvec1) = convert_bfloat16_float(x_bvec);
          dotp_fvec += (x_fvec0 - fVec(mean)) * dy_fvec0;
          dotp_fvec += (x_fvec1 - fVec(mean)) * dy_fvec1;
        }
        for (; d < image_size; d++) {
          float dy = grad_output_data[offset + d];
          float x = input_data[offset + d];
          sum_val += dy;
          dotp_val += (x - mean) * dy;
        }
      }
      sum_val += vec_acc(sum_fvec);
      dotp_val += vec_acc(dotp_fvec);

      if (!grad_input_null) {
        if (train) {
          float k = dotp_val * invstd * invstd / N;
          float grad_mean = sum_val / N;

          for (int64_t n = 0; n < n_batch; n++) {
            auto offset = n * n_channel * image_size + c * image_size;
            int64_t d = 0;
            for (; d < loop_size; d += bVec::size()) {
              bVec x_bvec = bVec::loadu(input_data + offset + d);
              fVec x_fvec0, x_fvec1;
              std::tie(x_fvec0, x_fvec1) = convert_bfloat16_float(x_bvec);
              fVec dx_fvec0 = (x_fvec0 - fVec(mean)) * fVec(k);
              fVec dx_fvec1 = (x_fvec1 - fVec(mean)) * fVec(k);
              bVec dy_bvec = bVec::loadu(grad_output_data + offset + d);
              fVec dy_fvec0, dy_fvec1;
              std::tie(dy_fvec0, dy_fvec1) = convert_bfloat16_float(dy_bvec);
              dx_fvec0 = (dy_fvec0 - fVec(grad_mean) - dx_fvec0) * fVec(invstd) * fVec(w);
              dx_fvec1 = (dy_fvec1 - fVec(grad_mean) - dx_fvec1) * fVec(invstd) * fVec(w);
              bVec dx_bvec = convert_float_bfloat16(dx_fvec0, dx_fvec1);
              dx_bvec.store(grad_input_data + offset + d);
            }
            for (; d < image_size; d++) {
              float x_val = input_data[offset + d];
              float dy_val = grad_output_data[offset + d];
              float dx_val = (x_val - mean) * k;
              dx_val = (dy_val - grad_mean - dx_val) * invstd * w;
              grad_input_data[offset + d] = BFloat16(dx_val);
            }
          }
        } else { // evaluation mode
          for (int64_t n = 0; n < n_batch; n++) {
            auto offset = n * n_channel * image_size + c * image_size;
            int64_t d = 0;
            for (; d < loop_size; d += bVec::size()) {
              bVec dy_bvec = bVec::loadu(grad_output_data + offset + d);
              fVec dy_fvec0, dy_fvec1;
              std::tie(dy_fvec0, dy_fvec1) = convert_bfloat16_float(dy_bvec);
              fVec dx_fvec0 = dy_fvec0 * fVec(invstd) * fVec(w);
              fVec dx_fvec1 = dy_fvec1 * fVec(invstd) * fVec(w);
              bVec dx_bvec = convert_float_bfloat16(dx_fvec0, dx_fvec1);
              dx_bvec.store(grad_input_data + offset + d);
            }
            for (; d < image_size; d++) {
              float dy_val = grad_output_data[offset + d];
              float dx_val = dy_val * invstd * w;
              grad_input_data[offset + d] = BFloat16(dx_val);
            }
          }
        }
      }

      if (!grad_weight_null) {
        grad_weight_data[c] = float(dotp_val * invstd);
      }

      if (!grad_bias_null) {
        grad_bias_data[c] = float(sum_val);
      }
    }
  });
}

template <>
void batch_norm_cpu_backward_channels_last_impl<BFloat16>(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps) {

  using bVec = Vectorized<BFloat16>;
  using fVec = Vectorized<float>;
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;
  int64_t N = input.numel() / n_channel;

  const BFloat16* grad_output_data = grad_output.data_ptr<BFloat16>();
  const BFloat16* input_data = input.data_ptr<BFloat16>();

  BFloat16* grad_input_data = grad_input.defined() ? grad_input.data_ptr<BFloat16>() : nullptr;
  float* grad_weight_data = grad_weight.defined() ? grad_weight.data_ptr<float>() : nullptr;
  float* grad_bias_data = grad_bias.defined() ? grad_bias.data_ptr<float>() : nullptr;
  const bool grad_input_null = grad_input_data == nullptr;
  const bool grad_weight_null = grad_weight_data == nullptr;
  const bool grad_bias_null = grad_bias_data == nullptr;

  float* save_mean_data = conditional_data_ptr<float>(save_mean);
  float* save_invstd_data = conditional_data_ptr<float>(save_invstd);
  float* running_mean_data = conditional_data_ptr<float>(running_mean);
  float* running_var_data = conditional_data_ptr<float>(running_var);

  Tensor weight_ = weight.defined() ? weight : at::ones({n_channel}, input.options());
  const float* weight_ptr = weight_.data_ptr<float>();

  float* mean_ptr = nullptr;
  float* invstd_ptr = nullptr;
  Tensor invstd = at::empty({0}, input.options());
  if (train) {
    mean_ptr = save_mean_data;
    invstd_ptr = save_invstd_data;
  } else {
    mean_ptr = running_mean_data;

    invstd.resize_({n_channel});
    invstd_ptr = invstd.data_ptr<float>();
    for (int64_t c = 0; c < n_channel; c++) {
      invstd_ptr[c] = 1 / std::sqrt(running_var_data[c] + eps);
    }
  }

  // cache a copy of mean, invstd and weight in float32
  std::unique_ptr<float []> arr(new float[3 * n_channel]);
  float* meanf_ptr = arr.get();
  float* invstdf_ptr = arr.get() + n_channel;
  float* weightf_ptr = arr.get() + 2 * n_channel;
  // TODO: vectorize this
  convert(mean_ptr, meanf_ptr, n_channel);
  convert(invstd_ptr, invstdf_ptr, n_channel);
  convert(weight_ptr, weightf_ptr, n_channel);

  // use float as accumulation type
  int num_threads = at::get_num_threads();
  Tensor buffer = at::empty({2, num_threads, n_channel}, input.options().dtype(kFloat)).zero_();
  float* sum_data = buffer.data_ptr<float>();
  float* dotp_data = sum_data + num_threads * n_channel;

  // compute sum and dotp per feature plain
  // use vec256::mapN<BFloat16> functional here is not preferred, since:
  //   1. we want to keep the immediate data (sum, dotp) in float32,
  //      use vec256::map will introduce unnecessary dtype conversion.
  //   2. we can save fuse most inner loop so as to reduce dtype conversion.
  int64_t loop_size = n_channel - (n_channel % bVec::size());
  at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    float* sum_ptr = sum_data + tid * n_channel;
    float* dotp_ptr = dotp_data + tid * n_channel;

    for (int64_t i = begin; i < end; i++) {
      int64_t offset = i * n_channel;
      int64_t d = 0;
      for (; d < loop_size; d += bVec::size()) {
        bVec dy_bvec = bVec::loadu(grad_output_data + offset + d);
        fVec dy_fvec0, dy_fvec1;
        std::tie(dy_fvec0, dy_fvec1) = convert_bfloat16_float(dy_bvec);
        fVec sum_fvec0 = dy_fvec0 + fVec::loadu(sum_ptr + d);
        fVec sum_fvec1 = dy_fvec1 + fVec::loadu(sum_ptr + d + fVec::size());
        sum_fvec0.store(sum_ptr + d);
        sum_fvec1.store(sum_ptr + d + fVec::size());

        bVec x_bvec = bVec::loadu(input_data + offset + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = convert_bfloat16_float(x_bvec);
        fVec mean_fvec0 = fVec::loadu(meanf_ptr + d);
        fVec mean_fvec1 = fVec::loadu(meanf_ptr + d + fVec::size());
        fVec dotp_fvec0 = fVec::loadu(dotp_ptr + d);
        fVec dotp_fvec1 = fVec::loadu(dotp_ptr + d + fVec::size());
        dotp_fvec0 += (x_fvec0 - mean_fvec0) * dy_fvec0;
        dotp_fvec1 += (x_fvec1 - mean_fvec1) * dy_fvec1;
        dotp_fvec0.store(dotp_ptr + d);
        dotp_fvec1.store(dotp_ptr + d + fVec::size());
      }
      for (; d < n_channel; d++) {
        float dy_val = grad_output_data[offset + d];
        float x_val = input_data[offset + d];
        float mean_val = meanf_ptr[d];
        sum_ptr[d] += dy_val;
        dotp_ptr[d] += (x_val - mean_val) * dy_val;
      }
    }
  });

  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; c++) {
      float _sum = float(0);
      for (int64_t t = 0; t < num_threads; t++) {
        _sum += sum_data[t * n_channel + c];
      }
      sum_data[/* 0 * n_channel + */c] = _sum;

      float _dotp = float(0);
      for (int64_t t = 0; t < num_threads; t++) {
        _dotp += dotp_data[t * n_channel + c];
      }
      dotp_data[/* 0 * n_channel + */c] = _dotp;
    }
  });

  // compute grad_input
  if (grad_input.defined()) {
    at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; i++) {
        int64_t offset = i * n_channel;
        if (train) {
          int64_t d = 0;
          for (; d < loop_size; d += bVec::size()) {
            bVec x_bvec = bVec::loadu(input_data + offset + d);
            fVec x_fvec0, x_fvec1;
            std::tie(x_fvec0, x_fvec1) = convert_bfloat16_float(x_bvec);
            fVec mean_fvec0 = fVec::loadu(meanf_ptr + d);
            fVec mean_fvec1 = fVec::loadu(meanf_ptr + d + fVec::size());
            fVec dotp_fvec0 = fVec::loadu(dotp_data + d);
            fVec dotp_fvec1 = fVec::loadu(dotp_data + d + fVec::size());
            fVec invstd_fvec0 = fVec::loadu(invstdf_ptr + d);
            fVec invstd_fvec1 = fVec::loadu(invstdf_ptr + d + fVec::size());
            fVec k_fvec0 = dotp_fvec0 * invstd_fvec0 * invstd_fvec0 / fVec(N);
            fVec k_fvec1 = dotp_fvec1 * invstd_fvec1 * invstd_fvec1 / fVec(N);
            fVec dx_fvec0 = (x_fvec0 - mean_fvec0) * k_fvec0;
            fVec dx_fvec1 = (x_fvec1 - mean_fvec1) * k_fvec1;
            bVec dy_bvec = bVec::loadu(grad_output_data + offset + d);
            fVec dy_fvec0, dy_fvec1;
            std::tie(dy_fvec0, dy_fvec1) = convert_bfloat16_float(dy_bvec);
            fVec grad_mean_fvec0 = fVec::loadu(sum_data + d) / fVec(N);
            fVec grad_mean_fvec1 = fVec::loadu(sum_data + d + fVec::size()) / fVec(N);
            fVec w_fvec0 = fVec::loadu(weightf_ptr + d);
            fVec w_fvec1 = fVec::loadu(weightf_ptr + d + fVec::size());
            dx_fvec0 = (dy_fvec0 - grad_mean_fvec0 - dx_fvec0) * invstd_fvec0 * w_fvec0;
            dx_fvec1 = (dy_fvec1 - grad_mean_fvec1 - dx_fvec1) * invstd_fvec1 * w_fvec1;
            bVec dx_bvec = convert_float_bfloat16(dx_fvec0, dx_fvec1);
            dx_bvec.store(grad_input_data + offset + d);
          }
          for (; d < n_channel; d++) {
            float x_val = input_data[offset + d];
            float mean_val = meanf_ptr[d];
            float dotp_val = dotp_data[d];
            float invstd_val = invstdf_ptr[d];
            float k_val = dotp_val * invstd_val * invstd_val / N;
            float dx_val = (x_val - mean_val) * k_val;
            float dy_val = grad_output_data[offset + d];
            float grad_mean_val = sum_data[d] / N;
            float w_val = weightf_ptr[d];
            dx_val = (dy_val - grad_mean_val - dx_val) * invstd_val * w_val;
            grad_input_data[offset + d] = BFloat16(dx_val);
          }
        } else { // evaluation mode
          int64_t d = 0;
          for (; d < loop_size; d += bVec::size()) {
            bVec dy_bvec = bVec::loadu(grad_output_data + offset + d);
            fVec dy_fvec0, dy_fvec1;
            std::tie(dy_fvec0, dy_fvec1) = convert_bfloat16_float(dy_bvec);
            fVec invstd_fvec0 = fVec::loadu(invstdf_ptr + d);
            fVec invstd_fvec1 = fVec::loadu(invstdf_ptr + d + fVec::size());
            fVec w_fvec0 = fVec::loadu(weightf_ptr + d);
            fVec w_fvec1 = fVec::loadu(weightf_ptr + d + fVec::size());
            fVec dx_fvec0 = dy_fvec0 * invstd_fvec0 * w_fvec0;
            fVec dx_fvec1 = dy_fvec1 * invstd_fvec1 * w_fvec1;
            bVec dx_bvec = convert_float_bfloat16(dx_fvec0, dx_fvec1);
            dx_bvec.store(grad_input_data + offset + d);
          }
          for (; d < n_channel; d++) {
            float dy_val = grad_output_data[offset + d];
            float invstd_val = invstdf_ptr[d];
            float w_val = weightf_ptr[d];
            float dx_val = dy_val * invstd_val * w_val;
            grad_input_data[offset + d] = BFloat16(dx_val);
          }
        }
      }
    });
  }

  // grad_weight = dotp * invstd
  if (grad_weight.defined()) {
    int64_t d = 0;
    for (; d < n_channel; d++) {
      float grad_weight_val = dotp_data[d] * invstdf_ptr[d];
      grad_weight_data[d] = float(grad_weight_val);
    }
  }

  // grad_bias = sum
  if (grad_bias.defined()) {
    int64_t d = 0;
    for (; d < n_channel; d++) {
      float grad_bias_val = sum_data[d];
      grad_bias_data[d] = float(grad_bias_val);
    }
  }
}

void batch_norm_cpu_kernel(Tensor& output, const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& save_mean,  const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, input.scalar_type(), "batch_norm_cpu_contiguous", [&] {
        batch_norm_cpu_contiguous_impl<scalar_t>(output, input, weight, bias,
            save_mean, save_invstd, running_mean, running_var, train, eps);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, input.scalar_type(), "batch_norm_cpu_channels_last", [&] {
        batch_norm_cpu_channels_last_impl<scalar_t>(output, input, weight, bias,
            save_mean, save_invstd, running_mean, running_var, train, eps);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void batch_norm_cpu_collect_stats_kernel(
    Tensor& mean, Tensor& var_sum, const Tensor& input) {
  int64_t image_size = input.numel() / input.size(0) / input.size(1);
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, input.scalar_type(), "batch_norm_cpu_collect_stats_contiguous", [&] {
        if (image_size == 1) { // NC11 is also channels last
          batch_norm_cpu_collect_stats_channels_last_impl<scalar_t>(mean, var_sum, input);
        } else {
          batch_norm_cpu_collect_stats_contiguous_impl<scalar_t>(mean, var_sum, input);
        }
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, input.scalar_type(), "batch_norm_cpu_collect_stats_channels_last", [&] {
        batch_norm_cpu_collect_stats_channels_last_impl<scalar_t>(mean, var_sum, input);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void batch_norm_cpu_backward_kernel(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps) {
  int64_t image_size = input.numel() / input.size(0) / input.size(1);
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, input.scalar_type(), "batch_norm_cpu_backward_contiguous", [&] {
        if (image_size == 1) { // NC11 is also channels last
          batch_norm_cpu_backward_channels_last_impl<scalar_t>(grad_input, grad_weight, grad_bias,
              grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
        } else {
          batch_norm_cpu_backward_contiguous_impl<scalar_t>(grad_input, grad_weight, grad_bias,
              grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
        }
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, input.scalar_type(), "batch_norm_cpu_backward_channels_last", [&] {
        batch_norm_cpu_backward_channels_last_impl<scalar_t>(grad_input, grad_weight, grad_bias,
            grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

}// anonymous namespace

REGISTER_DISPATCH(batch_norm_cpu_stub, &batch_norm_cpu_kernel);
REGISTER_DISPATCH(batch_norm_cpu_collect_stats_stub, &batch_norm_cpu_collect_stats_kernel);
REGISTER_DISPATCH(batch_norm_cpu_backward_stub, &batch_norm_cpu_backward_kernel);

}} // namespace at::native
