#include <ATen/native/batch_norm.h>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/utils.h>

namespace at { namespace native {
namespace {

using namespace vec256;

template<typename scalar_t>
void batch_norm_cpu_inference_collect_linear_and_constant_terms(
    scalar_t* alpha, scalar_t* beta, int64_t n_channel,
    const Tensor& weight /* optional */, const Tensor& bias /* optional */,
    const Tensor& mean, const Tensor& variance, double eps) {

  const scalar_t* weight_data = weight.defined() ? weight.data_ptr<scalar_t>() : nullptr;
  const scalar_t* bias_data = bias.defined() ? bias.data_ptr<scalar_t>() : nullptr;
  auto mean_data = mean.accessor<scalar_t, 1>();
  auto var_data = variance.accessor<scalar_t, 1>();

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
    scalar_t inv_var = 1 / std::sqrt(var_data[c] + static_cast<scalar_t>(eps));
    scalar_t weight_v = weight_data ? weight_data[c] : 1;
    scalar_t bias_v = bias_data ? bias_data[c] : 0;
    alpha[c] = inv_var * weight_v;
    beta[c] = bias_v - mean_data[c] * alpha[c];
  }
}

/// A fast path for CPU inference when all tensors are contiguous.
template<typename scalar_t>
void batch_norm_cpu_inference_contiguous_impl(Tensor& output,
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& mean, const Tensor& variance, double eps) {

  using Vec = Vec256<scalar_t>;
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;

  Tensor alpha = at::empty({n_channel}, input.options());
  Tensor beta = at::empty({n_channel}, input.options());
  auto alpha_data = alpha.data_ptr<scalar_t>();
  auto beta_data = beta.data_ptr<scalar_t>();

  batch_norm_cpu_inference_collect_linear_and_constant_terms<scalar_t>(
     alpha_data, beta_data, n_channel, weight, bias, mean, variance, eps);

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
void batch_norm_cpu_inference_channels_last_impl(Tensor& output,
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& mean, const Tensor& variance, double eps) {

  using Vec = Vec256<scalar_t>;
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;

  Tensor alpha = at::empty({n_channel}, input.options());
  Tensor beta = at::empty({n_channel}, input.options());
  auto alpha_data = alpha.data_ptr<scalar_t>();
  auto beta_data = beta.data_ptr<scalar_t>();

  batch_norm_cpu_inference_collect_linear_and_constant_terms<scalar_t>(
      alpha_data, beta_data, n_channel, weight, bias, mean, variance, eps);

  scalar_t* output_data = output.data_ptr<scalar_t>();
  const scalar_t* input_data = input.data_ptr<scalar_t>();

  // Apply the linear terms to the input,
  // output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)
  if (n_channel != 1) {
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
  } else {
    // n_channel == 1
    const int64_t loop_size = image_size - (image_size % Vec::size());
    at::parallel_for(0, n_batch, 1, [&](int64_t begin, int64_t end) {
      const Vec alpha_vec(alpha_data[0]);
      const Vec beta_vec(beta_data[0]);
      for (int64_t n = begin; n < end; n++) {
        int64_t offset = n * image_size;
        int64_t d = 0;
        for (; d < loop_size; d += Vec::size()) {
          Vec data_vec = Vec::loadu(input_data + offset + d);
          Vec output_vec = data_vec * alpha_vec + beta_vec;
          output_vec.store(output_data + offset + d);
        }
        if (image_size - d > 0) {
          Vec data_vec = Vec::loadu(input_data + offset + d, n_channel - d);
          Vec output_vec = data_vec * alpha_vec + beta_vec;
          output_vec.store(output_data + offset + d, n_channel - d);
        }
      }
    });
  }
}

void batch_norm_cpu_inference_kernel(Tensor& output, const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& mean, const Tensor& variance, double eps) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batch_norm_cpu_inference_contiguous", [&] {
        batch_norm_cpu_inference_contiguous_impl<scalar_t>(output, input, weight, bias, mean, variance, eps);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batch_norm_cpu_inference_channels_last", [&] {
        batch_norm_cpu_inference_channels_last_impl<scalar_t>(output, input, weight, bias, mean, variance, eps);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}


}// anonymous namespace

REGISTER_DISPATCH(batch_norm_cpu_inference_stub, &batch_norm_cpu_inference_kernel);

}} // namespace at::native
