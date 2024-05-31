#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cpu/MaxUnpoolKernel.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

#include <c10/util/Optional.h>

namespace at::native {

namespace {

template <typename scalar_t, bool is_3d = false>
void cpu_max_unpool(
    Tensor& output_,
    const Tensor& input,
    const Tensor& indices) {
  auto output = output_.contiguous();

  auto input_data = input.const_data_ptr<scalar_t>();
  auto indices_data = indices.const_data_ptr<int64_t>();
  auto output_data = output.data_ptr<scalar_t>();

  // NB: input tensor dimensions:
  // MaxUnpool2d:
  //    dim = 3: CHW
  //    dim = 4: NCHW
  // MaxUnpool3d:
  //    dim = 4: CDHW
  //    dim = 5: NCDHW

  int64_t numel = input.numel();
  int64_t ndim = input.ndimension();

  // treat batch size and channels as one dimension
  // and the feature map as another dimension
  int64_t channels, output_depth, output_height, output_width;
  if (is_3d) {
    TORCH_CHECK(ndim == 4 || ndim == 5, "MaxUnpool3d: expect input to be 4d or 5d tensor.");
    channels = ndim == 4 ? input.size(0) : input.size(0) * input.size(1);
    output_depth = output.size(-3);
    output_height = output.size(-2);
    output_width = output.size(-1);
  } else {
    TORCH_CHECK(ndim == 3 || ndim == 4, "MaxUnpool2d: expect input to be 3d or 4d tensor.");
    channels = ndim == 3 ? input.size(0) : input.size(0) * input.size(1);
    output_depth = 1;
    output_height = output.size(-2);
    output_width = output.size(-1);
  }
  int64_t input_image_size = numel / channels;
  int64_t output_image_size = output.numel() / channels;

  std::optional<int64_t> optional_error_index;

  // parallel on dim N, C, D, H, W: [channels, input_image_size]
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t c = 0;
    int64_t ip = 0;
    data_index_init(begin, c, channels, ip, input_image_size);

    for (const auto i : c10::irange(begin, end)) {
      scalar_t* output_ptr = output_data + c * output_image_size;

      int64_t maxp = indices_data[i];
      if (maxp < 0 || maxp >= output_image_size) {
        optional_error_index = maxp;
        std::atomic_thread_fence(std::memory_order_release);
      } else {
        output_ptr[maxp] = input_data[i];
      }

      // move on to next input index
      data_index_step(c, channels, ip, input_image_size);
    }
  });

  if (optional_error_index) {
    if (is_3d) {
      AT_ERROR("Found an invalid max index: ", optional_error_index.value(),
          " (output volumes are of size ", output_depth,
          "x", output_height, "x", output_width);
    } else {
      AT_ERROR("Found an invalid max index: ", optional_error_index.value(),
          " (output volumes are of size ", output_height,
          "x", output_width);
    }
  }

  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
}

template <typename scalar_t>
void cpu_max_unpool_channels_last(
    Tensor& output_,
    const Tensor& input,
    const Tensor& indices) {
  TORCH_CHECK(input.ndimension() == 4,
              "max_unpool2d with channels last format supports tensors with 4 dims");
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto output = output_.contiguous(memory_format);

  auto input_data = input.const_data_ptr<scalar_t>();
  auto indices_data = indices.const_data_ptr<int64_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  int64_t output_height = output.size(2);
  int64_t output_width = output.size(3);
  int64_t input_image_size = input_height * input_width;
  int64_t output_image_size = output_height * output_width;

  std::optional<int64_t> optional_error_index;

  // parallel on dim N, H, W
  at::parallel_for(0, nbatch * input_image_size, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t ip = 0;
    data_index_init(begin, n, nbatch, ip, input_image_size);

    for (const auto i : c10::irange(begin, end)) {
      const scalar_t* input_ptr = input_data + i * channels;
      const int64_t* indices_ptr = indices_data + i * channels;
      scalar_t* output_ptr = output_data + n * output_image_size * channels;

      // can't do scatter on avx2 (only available on avx512)
      for (const auto c : c10::irange(channels)) {
        int64_t maxp = indices_ptr[c];
        if (maxp < 0 || maxp >= output_image_size) {
          optional_error_index = maxp;
          std::atomic_thread_fence(std::memory_order_release);
        } else {
          output_ptr[maxp * channels + c] = input_ptr[c];
        }
      }

      // move on to next input index
      data_index_step(n, nbatch, ip, input_image_size);
    }
  });

  if (optional_error_index) {
    AT_ERROR("Found an invalid max index: ", optional_error_index.value(),
        " (output volumes are of size ", output_height,
        "x", output_width, ")");
  }

  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
}

template <typename scalar_t, bool is_3d = false>
void cpu_max_unpool_backward(
    Tensor& grad_input_,
    const Tensor& grad_output,
    const Tensor& indices) {
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto indices_data = indices.data_ptr<int64_t>();
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();

  int64_t numel = grad_input.numel();
  int64_t ndim = grad_output.ndimension();

  // treat batch size and channels as one dimension
  // and the feature map as another dimension
  int64_t channels, output_depth, output_height, output_width;
  if (is_3d) {
    TORCH_CHECK(ndim == 4 || ndim == 5, "MaxUnpool3d_backward: expect grad_output to be 4d or 5d tensor.");
    channels = ndim == 4 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
    output_depth = grad_output.size(-3);
    output_height = grad_output.size(-2);
    output_width = grad_output.size(-1);
  } else {
    TORCH_CHECK(ndim == 3 || ndim == 4, "MaxUnpool2d_backward: expect grad_output to be 3d or 4d tensor.");
    channels = ndim == 3 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
    output_depth = 1;
    output_height = grad_output.size(-2);
    output_width = grad_output.size(-1);
  }
  int64_t input_image_size = numel / channels;
  int64_t output_image_size = grad_output.numel() / channels;

  std::optional<int64_t> optional_error_index;

  // parallel on dim N, C, D, H, W
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t c = 0;
    int64_t ip = 0;
    data_index_init(begin, c, channels, ip, input_image_size);

    for (const auto i : c10::irange(begin, end)) {
      scalar_t* grad_output_ptr = grad_output_data + c * output_image_size;

      int64_t maxp = indices_data[i];
      if (maxp < 0 || maxp >= output_image_size) {
          optional_error_index = maxp;
          std::atomic_thread_fence(std::memory_order_release);
      } else {
        grad_input_data[i] = grad_output_ptr[maxp];
      }

      // move on to next input index
      data_index_step(c, channels, ip, input_image_size);
    }
  });

  if (optional_error_index) {
    if (is_3d) {
      AT_ERROR("invalid max index ", optional_error_index.value(),
          ", odepth= ", output_depth,
          ", owidth= ", output_width,
          ", oheight= ", output_height);
    } else {
      AT_ERROR("invalid max index ", optional_error_index.value(),
          ", owidth= ", output_width,
          ", oheight= ", output_height);
    }
  }

  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

void max_unpool2d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    const Tensor& indices) {
  switch(input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_unpool2d", [&] {
        cpu_max_unpool<scalar_t, /*is_3d*/false>(output, input, indices);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_unpool2d_channels_last", [&] {
        cpu_max_unpool_channels_last<scalar_t>(output, input, indices);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void max_unpool3d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    const Tensor& indices) {
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_unpool3d", [&] {
    cpu_max_unpool<scalar_t, /*is_3d*/true>(output, input, indices);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(max_unpool2d_kernel, &max_unpool2d_kernel_impl);
REGISTER_DISPATCH(max_unpool3d_kernel, &max_unpool3d_kernel_impl);

} // at::native
