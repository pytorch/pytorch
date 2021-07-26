#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/Padding.h>
#include <ATen/native/cpu/utils.h>

namespace at { namespace native {

namespace {

template <typename scalar_t, typename Indexr>
void cpu_padding(
    const Tensor& output_,
    const Tensor& input_,
    int p_left, int p_right,
    int p_top, int p_bottom,
    int p_front, int p_back,
    bool is_batch_mode) {
  auto input = input_.contiguous();
  auto output = output_.contiguous();

  scalar_t* input_data = input.data_ptr<scalar_t>();
  scalar_t* output_data = output.data_ptr<scalar_t>();

  PaddingParams p(input, output, is_batch_mode);

  Indexr width_indexr(p_left, p.input_width);
  Indexr height_indexr(p_top, p.input_height);
  Indexr depth_indexr(p_front, p.input_depth);

  // parallel on dim N, C
  at::parallel_for(0, p.nbatch * p.channels, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      scalar_t* input_ptr = input_data + i * p.input_depth * p.input_height * p.input_width;
      scalar_t* output_ptr = output_data + i * p.output_depth * p.output_height * p.output_width;

      for (int64_t od = 0; od < p.output_depth; od++) {
        int64_t id = depth_indexr.get(od);

        for (int64_t oh = 0; oh < p.output_height; oh++) {
          int64_t ih = height_indexr.get(oh);

          for (int64_t ow = 0; ow < p.output_width; ow++) {
            int64_t iw = width_indexr.get(ow);

            int64_t input_offset = id * p.input_height * p.input_width + ih * p.input_width + iw;
            int64_t output_offset = od * p.output_height * p.output_width + oh * p.output_width + ow;
            output_ptr[output_offset] = input_ptr[input_offset];
          }
        }
      }
    }
  });

  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
}

template <typename scalar_t, typename Indexr>
void cpu_padding_channels_last(
    const Tensor& output_,
    const Tensor& input_,
    int p_left, int p_right,
    int p_top, int p_bottom,
    int p_front, int p_back) {
  auto memory_format = input_.suggest_memory_format();

  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);

  scalar_t* input_data = input.data_ptr<scalar_t>();
  scalar_t* output_data = output.data_ptr<scalar_t>();

  PaddingParams p(input, output, /* is_batch_mode */ true);

  Indexr width_indexr(p_left, p.input_width);
  Indexr height_indexr(p_top, p.input_height);
  Indexr depth_indexr(p_front, p.input_depth);

  using Vec = vec::Vectorized<scalar_t>;
  // parallel on dim N, {D}, H, W
  at::parallel_for(0, p.nbatch * p.output_depth * p.output_height * p.output_width, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, p.nbatch, od, p.output_depth, oh, p.output_height, ow, p.output_width);

    int64_t size = p.channels;
    int64_t len = size - (size % Vec::size());
    for (int64_t i = begin; i < end; i++) {
      int64_t id = depth_indexr.get(od);
      int64_t ih = height_indexr.get(oh);
      int64_t iw = width_indexr.get(ow);

      scalar_t* in = input_data + (n * p.input_depth * p.input_height * p.input_width +
          id * p.input_height * p.input_width + ih * p.input_width + iw) * p.channels;
      scalar_t* out = output_data + i * p.channels;
      int64_t d = 0;
      for (; d < len; d += Vec::size()) {
        Vec out_vec = Vec::loadu(in + d);
        out_vec.store(out + d);
      }
      for (; d < size; d++) {
        out[d] = in[d];
      }

      // move on to next output index
      data_index_step(n, p.nbatch, od, p.output_depth, oh, p.output_height, ow, p.output_width);
    }
  });

  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
}

template <typename scalar_t, typename Indexr>
void cpu_padding_backward(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    int p_left, int p_right,
    int p_top, int p_bottom,
    int p_front, int p_back,
    bool is_batch_mode) {
  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  scalar_t* grad_output_data = grad_output.data_ptr<scalar_t>();
  scalar_t* grad_input_data = grad_input.data_ptr<scalar_t>();

  PaddingParams p(grad_input, grad_output, is_batch_mode);

  Indexr width_indexr(p_left, p.input_width);
  Indexr height_indexr(p_top, p.input_height);
  Indexr depth_indexr(p_front, p.input_depth);

  // parallel on dim N, C
  at::parallel_for(0, p.nbatch * p.channels, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      scalar_t* grad_input_ptr = grad_input_data + i * p.input_depth * p.input_height * p.input_width;
      scalar_t* grad_output_ptr = grad_output_data + i * p.output_depth * p.output_height * p.output_width;

      for (int64_t od = 0; od < p.output_depth; od++) {
        int64_t id = depth_indexr.get(od);

        for (int64_t oh = 0; oh < p.output_height; oh++) {
          int64_t ih = height_indexr.get(oh);

          for (int64_t ow = 0; ow < p.output_width; ow++) {
            int64_t iw = width_indexr.get(ow);

            int64_t input_offset = id * p.input_height * p.input_width + ih * p.input_width + iw;
            int64_t output_offset = od * p.output_height * p.output_width + oh * p.output_width + ow;
            grad_input_ptr[input_offset] += grad_output_ptr[output_offset];
          }
        }
      }
    }
  });

  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

template <typename scalar_t, typename Indexr>
void cpu_padding_backward_channels_last(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    int p_left, int p_right,
    int p_top, int p_bottom,
    int p_front, int p_back) {
  auto memory_format = grad_output_.suggest_memory_format();

  auto grad_output = grad_output_.contiguous(memory_format);
  auto grad_input = grad_input_.contiguous(memory_format);

  scalar_t* grad_output_data = grad_output.data_ptr<scalar_t>();
  scalar_t* grad_input_data = grad_input.data_ptr<scalar_t>();

  PaddingParams p(grad_input, grad_output, /* is_batch_mode */ true);

  Indexr width_indexr(p_left, p.input_width);
  Indexr height_indexr(p_top, p.input_height);
  Indexr depth_indexr(p_front, p.input_depth);

  using Vec = vec::Vectorized<scalar_t>;
  // parallel on dim N
  at::parallel_for(0, p.nbatch, 0, [&](int64_t begin, int64_t end) {
    int64_t size = p.channels;
    int64_t len = size - (size % Vec::size());
    for (int64_t n = begin; n < end; n++) {
      for (int64_t od = 0; od < p.output_depth; od++) {
        int64_t id = depth_indexr.get(od);

        for (int64_t oh = 0; oh < p.output_height; oh++) {
          int64_t ih = height_indexr.get(oh);

          for (int64_t ow = 0; ow < p.output_width; ow++) {
            int64_t iw = width_indexr.get(ow);

            scalar_t* gin = grad_input_data + (n * p.input_depth * p.input_height * p.input_width +
                id * p.input_height * p.input_width + ih * p.input_width + iw) * p.channels;
            scalar_t* gout = grad_output_data + (n * p.output_depth * p.output_height * p.output_width +
                od * p.output_height * p.output_width + oh * p.output_width + ow) * p.channels;
            int64_t d = 0;
            for (; d < len; d += Vec::size()) {
              Vec gin_vec = Vec::loadu(gin + d) + Vec::loadu(gout + d);
              gin_vec.store(gin + d);
            }
            for (; d < size; d++) {
              gin[d] += gout[d];
            }
          }
        }
      }
    }
  });

  if (!grad_input_.is_contiguous(memory_format)) {
    grad_input_.copy_(grad_input);
  }
}

void replication_pad1d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef padding_size) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "replication_pad1d", [&] {
    bool is_batch_mode = input.ndimension() == static_cast<int64_t>(3);
    cpu_padding<scalar_t, ReplicationPadIndexr>(
        output, input, padding_size[0], padding_size[1], /* p_top */ 0, /* p_bottom */ 0, /* p_front */ 0, /* p_back */ 0,
        is_batch_mode);
  });
}

void replication_pad1d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef padding_size) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(grad_output.scalar_type(), "replication_pad1d_backward", [&] {
    bool is_batch_mode = grad_output.ndimension() == static_cast<int64_t>(3);
    cpu_padding_backward<scalar_t, ReplicationPadIndexr>(
        grad_input, grad_output, padding_size[0], padding_size[1], /* p_top */ 0, /* p_bottom */ 0, /* p_front */ 0, /* p_back */ 0,
        is_batch_mode);
  });
}

void replication_pad2d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef padding_size) {
  switch(input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "replication_pad2d", [&] {
        bool is_batch_mode = input.ndimension() == static_cast<int64_t>(4);
        cpu_padding<scalar_t, ReplicationPadIndexr>(
            output, input, padding_size[0], padding_size[1], padding_size[2], padding_size[3], /* p_front */ 0, /* p_back */ 0,
            is_batch_mode);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "replication_pad2d_channels_last", [&] {
        cpu_padding_channels_last<scalar_t, ReplicationPadIndexr>(
            output, input, padding_size[0], padding_size[1], padding_size[2], padding_size[3], /* p_front */ 0, /* p_back */ 0);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void replication_pad2d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef padding_size) {
  switch(grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(grad_output.scalar_type(), "replication_pad2d_backward", [&] {
        bool is_batch_mode = grad_output.ndimension() == static_cast<int64_t>(4);
        cpu_padding_backward<scalar_t, ReplicationPadIndexr>(
            grad_input, grad_output, padding_size[0], padding_size[1], padding_size[2], padding_size[3], /* p_front */ 0, /* p_back */ 0,
            is_batch_mode);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(grad_output.scalar_type(), "replication_pad2d_backward_channels_last", [&] {
        cpu_padding_backward_channels_last<scalar_t, ReplicationPadIndexr>(
            grad_input, grad_output, padding_size[0], padding_size[1], padding_size[2], padding_size[3], /* p_front */ 0, /* p_back */ 0);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void replication_pad3d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef padding_size) {
  switch(input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "replication_pad3d", [&] {
        bool is_batch_mode = input.ndimension() == static_cast<int64_t>(5);
        cpu_padding<scalar_t, ReplicationPadIndexr>(
            output, input, padding_size[0], padding_size[1], padding_size[2], padding_size[3], padding_size[4], padding_size[5],
            is_batch_mode);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast3d: {
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "replication_pad3d_channels_last", [&] {
        cpu_padding_channels_last<scalar_t, ReplicationPadIndexr>(
            output, input, padding_size[0], padding_size[1], padding_size[2], padding_size[3], padding_size[4], padding_size[5]);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast3d, Contiguous");
  }
}

void replication_pad3d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef padding_size) {
  switch(grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(grad_output.scalar_type(), "replication_pad3d_backward", [&] {
        bool is_batch_mode = grad_output.ndimension() == static_cast<int64_t>(5);
        cpu_padding_backward<scalar_t, ReplicationPadIndexr>(
            grad_input, grad_output, padding_size[0], padding_size[1], padding_size[2], padding_size[3], padding_size[4], padding_size[5],
            is_batch_mode);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast3d: {
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(grad_output.scalar_type(), "replication_pad3d_backward_channels_last", [&] {
        cpu_padding_backward_channels_last<scalar_t, ReplicationPadIndexr>(
            grad_input, grad_output, padding_size[0], padding_size[1], padding_size[2], padding_size[3], padding_size[4], padding_size[5]);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast3d, Contiguous");
  }
}

} // anonymous namespace

REGISTER_DISPATCH(replication_pad1d_kernel, &replication_pad1d_kernel_impl);
REGISTER_DISPATCH(replication_pad1d_backward_kernel, &replication_pad1d_backward_kernel_impl);
REGISTER_DISPATCH(replication_pad2d_kernel, &replication_pad2d_kernel_impl);
REGISTER_DISPATCH(replication_pad2d_backward_kernel, &replication_pad2d_backward_kernel_impl);
REGISTER_DISPATCH(replication_pad3d_kernel, &replication_pad3d_kernel_impl);
REGISTER_DISPATCH(replication_pad3d_backward_kernel, &replication_pad3d_backward_kernel_impl);

}} // at::native
