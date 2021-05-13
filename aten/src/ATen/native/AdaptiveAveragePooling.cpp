#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/native/xnnpack/Engine.h>


namespace at {
namespace native {

namespace {

  void adaptive_avg_pool2d_out_cpu_template(
    at::Tensor& output,
    at::Tensor const& input,
    IntArrayRef output_size)
  {
    TORCH_CHECK(output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");
    int64_t ndim = input.ndimension();
    for (int64_t i = 0; i < ndim; i++) {
      TORCH_CHECK(input.size(i) > 0,
        "adaptive_avg_pooling2d(): expected input to have non-empty spatial dimensions, "
        "but input has sizes ", input.sizes(), " with dimension ", i, " being "
        "empty");
    }

    TORCH_CHECK((ndim == 3 || ndim == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
    TORCH_CHECK(input.dtype() == output.dtype(),
      "expected dtype ", input.dtype(), " for `output` but got dtype ", output.dtype());

    int64_t channels  = input.size(-3);
    // NOLINTNEXTLINE(clang-diagnostic-unused-variable,clang-analyzer-deadcode.DeadStores)
    int64_t input_height = input.size(-2);
    // NOLINTNEXTLINE(clang-diagnostic-unused-variable,clang-analyzer-deadcode.DeadStores)
    int64_t input_width = input.size(-1);
    int64_t output_height = output_size[0];
    int64_t output_width = output_size[1];

    if (ndim == 3) {
      output.resize_({channels, output_height, output_width});
    } else {
      int64_t nbatch = input.size(0);
      output.resize_({nbatch, channels, output_height, output_width}, input.suggest_memory_format());
    }

    adaptive_avg_pool2d_kernel(kCPU, output, input, output_size);
  }

  Tensor& adaptive_avg_pool2d_backward_out_cpu_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input)
  {
    int64_t ndim = grad_output.ndimension();
    for (int64_t i = 0; i < ndim; i++) {
      TORCH_CHECK(grad_output.size(i) > 0,
        "adaptive_avg_pooling2d_backward(): expected grad_output to have non-empty spatial dimensions, "
        "but grad_output has sizes ", grad_output.sizes(), " with dimension ", i, " being "
        "empty");
    }

    TORCH_CHECK((ndim == 3 || ndim == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for grad_output");
    TORCH_CHECK(input.dtype() == grad_output.dtype(),
      "expected dtype ", input.dtype(), " for `grad_output` but got dtype ", grad_output.dtype());
    TORCH_CHECK(input.dtype() == grad_input.dtype(),
      "expected dtype ", input.dtype(), " for `grad_input` but got dtype ", grad_input.dtype());

    grad_input.resize_(input.sizes(), input.suggest_memory_format());
    grad_input.zero_();

    adaptive_avg_pool2d_backward_kernel(kCPU, grad_input, grad_output);
    return grad_input;
  }

} // namespace

  Tensor& adaptive_avg_pool2d_out_cpu(const Tensor& input,
    IntArrayRef output_size,
    Tensor& output)
  {
    adaptive_avg_pool2d_out_cpu_template(
      output, input, output_size);
    return output;
  }

  Tensor adaptive_avg_pool2d_cpu(
    at::Tensor const& input,
    IntArrayRef output_size)
  {
    auto output = at::empty({0}, input.options());
    adaptive_avg_pool2d_out_cpu_template(
      output, input, output_size);
    return output;
  }

  Tensor adaptive_avg_pool2d(at::Tensor const& input, IntArrayRef output_size) {
    TORCH_CHECK(output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");

    if (input.is_mkldnn()) {
      return at::mkldnn_adaptive_avg_pool2d(input, output_size);
    }

    if (!input.is_quantized() && output_size[0] == 1 && output_size[1] == 1) {
      // in this case, adaptive pooling is just computing mean over hw
      // dimensions, which can be done more efficiently
      #if defined(C10_MOBILE) && defined(USE_XNNPACK)
      if (xnnpack::use_global_average_pool(input)) {
        return xnnpack::global_average_pool(input);
      }
      #endif

      Tensor out = input.mean({-1, -2}, /* keepdim = */ true);
      if (input.suggest_memory_format() == at::MemoryFormat::ChannelsLast) {
        // assert ndim == 4, since ndim = 3 doesn't give channels_last
        const int n = input.size(0);
        const int c = input.size(1);
        out.as_strided_({n, c, 1, 1}, {c, 1, c, c});
      }
      return out;
    } else {
      return _adaptive_avg_pool2d(input, output_size);
    }
  }

  Tensor& adaptive_avg_pool2d_backward_out_cpu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input)
  {
    adaptive_avg_pool2d_backward_out_cpu_template(
      grad_input, grad_output, input);
    return grad_input;
  }

  Tensor adaptive_avg_pool2d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input)
  {
    auto grad_input = at::empty({0}, input.options());
    adaptive_avg_pool2d_backward_out_cpu_template(
      grad_input, grad_output, input);
    return grad_input;
  }

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(adaptive_avg_pool2d_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(adaptive_avg_pool2d_backward_kernel);

} // at::native
} // at
