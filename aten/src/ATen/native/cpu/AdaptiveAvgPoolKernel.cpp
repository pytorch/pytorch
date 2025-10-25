#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>
#include <ATen/OpMathType.h>

namespace at::native {

namespace {

template <typename scalar_t, typename accscalar_t>
void cpu_adaptive_avg_pool2d(
    Tensor& output_,
    const Tensor& input_,
    IntArrayRef output_size) {
  auto input = input_.contiguous();
  auto output = output_.contiguous();

  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t ndim = input.ndimension();
  // treat batch size and channels as one dimension
  int64_t channels = ndim == 3 ? input.size(0) : input.size(0) * input.size(1);
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  // parallel on dim of N, C
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      const scalar_t* input_ptr = input_data + c * input_height * input_width;
      scalar_t* output_ptr = output_data + c * output_height * output_width;

      for (const auto oh : c10::irange(output_height)) {
        int64_t ih0 = start_index(oh, output_height, input_height);
        int64_t ih1 = end_index(oh, output_height, input_height);
        int64_t kh = ih1 - ih0;

        for (const auto ow : c10::irange(output_width)) {
          int64_t iw0 = start_index(ow, output_width, input_width);
          int64_t iw1 = end_index(ow, output_width, input_width);
          int64_t kw = iw1 - iw0;

          // compute local average
          accscalar_t sum = 0;
          for (const auto ih : c10::irange(ih0, ih1)) {
            for (const auto iw : c10::irange(iw0, iw1)) {
              sum += accscalar_t(input_ptr[ih * input_width + iw]);
            }
          }
          output_ptr[oh * output_width + ow] = scalar_t(sum / kh / kw);
        }
      }
    }
  });

  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
}

template <typename scalar_t>
typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
cpu_adaptive_avg_pool2d_channels_last(
    Tensor& output_,
    const Tensor& input_,
    IntArrayRef output_size) {
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);

  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  using Vec = vec::Vectorized<scalar_t>;
  // parallel on dim N, H, W
  at::parallel_for(0, nbatch * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);

    for (const auto i : c10::irange(begin, end)) {
      int64_t ih0 = start_index(oh, output_height, input_height);
      int64_t ih1 = end_index(oh, output_height, input_height);
      int64_t kh = ih1 - ih0;

      int64_t iw0 = start_index(ow, output_width, input_width);
      int64_t iw1 = end_index(ow, output_width, input_width);
      int64_t kw = iw1 - iw0;

      scalar_t* out = output_data + i * channels;
      int64_t size = channels;

      // Note: For ordinary usage scenario, each out lane should
      //   fit in L1 cache; otherwise consider block dim C.
      // Pass I: zero the out lane
      int64_t d1 = 0;
      for (; d1 < size - (size % Vec::size()); d1 += Vec::size()) {
        Vec out_vec = Vec(scalar_t(0));
        out_vec.store(out + d1);
      }
      for (; d1 < size; d1++) {
        out[d1] = scalar_t(0);
      }
      // Pass II: compute local sum
      for (const auto ih : c10::irange(ih0, ih1)) {
        for (const auto iw : c10::irange(iw0, iw1)) {
          const scalar_t* in = input_data + n * input_height * input_width * channels +
              ih * input_width * channels + iw * channels;

          int64_t d2 = 0;
          for (; d2 < size - (size % Vec::size()); d2 += Vec::size()) {
            Vec out_vec = Vec::loadu(out + d2) + Vec::loadu(in + d2);
            out_vec.store(out + d2);
          }
          for (; d2 < size; d2++) {
            out[d2] += in[d2];
          }
        }
      }
      // Pass III: compute local average
      int64_t d3 = 0;
      for (; d3 < size - (size % Vec::size()); d3 += Vec::size()) {
        Vec out_vec = Vec::loadu(out + d3) / Vec(scalar_t(kh * kw));
        out_vec.store(out + d3);
      }
      for (; d3 < size; d3++) {
        out[d3] = out[d3] / kh / kw;
      }

      // move on to next output index
      data_index_step(n, nbatch, oh, output_height, ow, output_width);
    }
  });

  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
}

template <typename scalar_t>
typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
cpu_adaptive_avg_pool2d_channels_last(
    Tensor& output_,
    const Tensor& input_,
    IntArrayRef output_size) {
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);

  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  using bVec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  // parallel on dim N, H, W
  at::parallel_for(0, nbatch * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);

    // temp buffer for sum, use float as accumulation type
    // can't reuse output buffer to store sum since it is BFloat16/Half
    auto sum_arr = std::make_unique<float []>(channels);
    float* sum = sum_arr.get();

    for (const auto i : c10::irange(begin, end)) {
      int64_t ih0 = start_index(oh, output_height, input_height);
      int64_t ih1 = end_index(oh, output_height, input_height);
      int64_t kh = ih1 - ih0;

      int64_t iw0 = start_index(ow, output_width, input_width);
      int64_t iw1 = end_index(ow, output_width, input_width);
      int64_t kw = iw1 - iw0;

      scalar_t* out = output_data + i * channels;
      int64_t size = channels;

      // Pass I: zero the out lane
      int64_t d1 = 0;
      for (; d1 < size - (size % fVec::size()); d1 += fVec::size()) {
        fVec sum_fvec = fVec(float(0));
        sum_fvec.store(sum + d1);
      }
      for (; d1 < size; d1++) {
        sum[d1] = float(0);
      }
      // Pass II: compute local sum
      for (const auto ih : c10::irange(ih0, ih1)) {
        for (const auto iw : c10::irange(iw0, iw1)) {
          const scalar_t* in = input_data + n * input_height * input_width * channels +
              ih * input_width * channels + iw * channels;

          int64_t d2 = 0;
          for (; d2 < size - (size % bVec::size()); d2 += bVec::size()) {
            bVec data_bvec = bVec::loadu(in + d2);
            auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);

            fVec sum_fvec0 = fVec::loadu(sum + d2) + data_fvec0;
            fVec sum_fvec1 = fVec::loadu(sum + d2 + fVec::size()) + data_fvec1;
            sum_fvec0.store(sum + d2);
            sum_fvec1.store(sum + d2 + fVec::size());
          }
          for (; d2 < size; d2++) {
            sum[d2] += float(in[d2]);
          }
        }
      }
      // Pass III: compute local average
      int64_t d3 = 0;
      for (; d3 < size - (size % bVec::size()); d3 += bVec::size()) {
        fVec out_fvec0 = fVec::loadu(sum + d3) / fVec(float(kh * kw));
        fVec out_fvec1 = fVec::loadu(sum + d3 + fVec::size()) / fVec(float(kh * kw));

        bVec out_bvec = convert_from_float<scalar_t>(out_fvec0, out_fvec1);
        out_bvec.store(out + d3);
      }
      for (; d3 < size; d3++) {
        out[d3] = scalar_t(sum[d3] / kh / kw);
      }

      // move on to next output index
      data_index_step(n, nbatch, oh, output_height, ow, output_width);
    }
  });

  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
}

template <typename scalar_t>
void cpu_adaptive_avg_pool2d_backward(
    Tensor& grad_input_,
    const Tensor& grad_output_) {
  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();

  int64_t ndim = grad_output.ndimension();
  // treat batch size and channels as one dimension
  int64_t channels = ndim == 3 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
  int64_t input_height = grad_input.size(-2);
  int64_t input_width = grad_input.size(-1);
  int64_t output_height = grad_output.size(-2);
  int64_t output_width = grad_output.size(-1);

  // parallel on dim of N, C
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      scalar_t* grad_input_ptr = grad_input_data + c * input_height * input_width;
      const scalar_t* grad_output_ptr = grad_output_data + c * output_height * output_width;

      for (const auto oh : c10::irange(output_height)) {
        int64_t ih0 = start_index(oh, output_height, input_height);
        int64_t ih1 = end_index(oh, output_height, input_height);
        int64_t kh = ih1 - ih0;

        for (const auto ow : c10::irange(output_width)) {
          int64_t iw0 = start_index(ow, output_width, input_width);
          int64_t iw1 = end_index(ow, output_width, input_width);
          int64_t kw = iw1 - iw0;

          scalar_t grad_delta = grad_output_ptr[oh * output_width + ow] / kh / kw;
          for (const auto ih : c10::irange(ih0, ih1)) {
            for (const auto iw : c10::irange(iw0, iw1)) {
              grad_input_ptr[ih * input_width + iw] += grad_delta;
            }
          }
        }
      }
    }
  });

  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

template <typename scalar_t>
void cpu_adaptive_avg_pool2d_backward_channels_last(
    Tensor& grad_input_,
    const Tensor& grad_output_) {
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto grad_input = grad_input_.contiguous(memory_format);
  auto grad_output = grad_output_.contiguous(memory_format);

  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();

  int64_t nbatch = grad_input.size(0);
  int64_t channels = grad_input.size(1);
  int64_t input_height = grad_input.size(2);
  int64_t input_width = grad_input.size(3);
  int64_t output_height = grad_output.size(2);
  int64_t output_width = grad_output.size(3);

  using Vec = vec::Vectorized<scalar_t>;
  // parallel on dim N
  at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
    for (const auto n : c10::irange(begin, end)) {
      scalar_t* grad_input_ptr = grad_input_data + n * input_height * input_width * channels;
      const scalar_t* grad_output_ptr = grad_output_data + n * output_height * output_width * channels;

      for (const auto oh : c10::irange(output_height)) {
        int64_t ih0 = start_index(oh, output_height, input_height);
        int64_t ih1 = end_index(oh, output_height, input_height);
        int64_t kh = ih1 - ih0;

        for (const auto ow : c10::irange(output_width)) {
          int64_t iw0 = start_index(ow, output_width, input_width);
          int64_t iw1 = end_index(ow, output_width, input_width);
          int64_t kw = iw1 - iw0;

          const scalar_t* gout = grad_output_ptr + oh * output_width * channels + ow * channels;
          int64_t size = channels;
          for (const auto ih : c10::irange(ih0, ih1)) {
            for (const auto iw : c10::irange(iw0, iw1)) {
              scalar_t* gin = grad_input_ptr + ih * input_width * channels + iw * channels;

              int64_t d = 0;
              for (; d < size - (size % Vec::size()); d += Vec::size()) {
                Vec gin_vec = Vec::loadu(gin + d) + Vec::loadu(gout + d) / Vec(scalar_t(kh * kw));
                gin_vec.store(gin + d);
              }
              for (; d < size; d++) {
                gin[d] += gout[d] / kh / kw;
              }
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

void adaptive_avg_pool2d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "adaptive_avg_pool2d", [&] {
        using param_t = at::opmath_type<scalar_t>;
        cpu_adaptive_avg_pool2d<scalar_t, /*accscalar_t*/param_t>(output, input, output_size);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "adaptive_avg_pool2d_channels_last", [&]{
        cpu_adaptive_avg_pool2d_channels_last<scalar_t>(output, input, output_size);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void adapative_avg_pool2d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output) {
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "adaptive_avg_pool2d_backward", [&] {
        cpu_adaptive_avg_pool2d_backward<scalar_t>(grad_input, grad_output);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "adaptive_avg_pool2d_backward_channels_last", [&]{
        cpu_adaptive_avg_pool2d_backward_channels_last<scalar_t>(grad_input, grad_output);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}


template <typename scalar_t, typename accscalar_t>
void cpu_adaptive_avg_pool3d(
    Tensor& output_,
    const Tensor& input_,
    IntArrayRef output_size) {
  auto input = input_.contiguous();
  auto output = output_.contiguous();

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t ndim = input.ndimension();
  // treat batch size and channels as one dimension
  int64_t channels = ndim == 4 ? input.size(0) : input.size(0) * input.size(1);
  int64_t input_depth = input.size(-3);
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  // parallel on dim of N, C
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      scalar_t* input_ptr = input_data + c * input_depth * input_height * input_width;
      scalar_t* output_ptr = output_data + c * output_depth * output_height * output_width;

      for (const auto od : c10::irange(output_depth)) {
        int64_t id0 = start_index(od, output_depth, input_depth);
        int64_t id1 = end_index(od, output_depth, input_depth);
        int64_t kd = id1 - id0;

        for (const auto oh : c10::irange(output_height)) {
          int64_t ih0 = start_index(oh, output_height, input_height);
          int64_t ih1 = end_index(oh, output_height, input_height);
          int64_t kh = ih1 - ih0;

          for (const auto ow : c10::irange(output_width)) {
            int64_t iw0 = start_index(ow, output_width, input_width);
            int64_t iw1 = end_index(ow, output_width, input_width);
            int64_t kw = iw1 - iw0;

            // compute local average
            accscalar_t sum = 0;
            for (const auto id : c10::irange(id0, id1)) {
              for (const auto ih : c10::irange(ih0, ih1)) {
                for (const auto iw : c10::irange(iw0, iw1)) {
                  sum += accscalar_t(input_ptr[id * input_height * input_width + ih * input_width + iw]);
                }
              }
            }
            output_ptr[od * output_height * output_width + oh * output_width + ow] = scalar_t(sum / kd / kh / kw);
          }
        }
      }
    }
  });

  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
}


template <typename scalar_t>
typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
cpu_adaptive_avg_pool3d_channels_last(
    Tensor& output_,
    const Tensor& input_,
    IntArrayRef output_size) {
  auto memory_format = at::MemoryFormat::ChannelsLast3d;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_depth = input.size(2);
  int64_t input_height = input.size(3);
  int64_t input_width = input.size(4);
  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  using Vec = vec::Vectorized<scalar_t>;
  // parallel on dim N, H, W
  at::parallel_for(0, nbatch * output_depth * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, od, output_depth, oh, output_height, ow, output_width);

    for (const auto i : c10::irange(begin, end)) {
      int64_t id0 = start_index(od, output_depth, input_depth);
      int64_t id1 = end_index(od, output_depth, input_depth);
      int64_t kd = id1 - id0;

      int64_t ih0 = start_index(oh, output_height, input_height);
      int64_t ih1 = end_index(oh, output_height, input_height);
      int64_t kh = ih1 - ih0;

      int64_t iw0 = start_index(ow, output_width, input_width);
      int64_t iw1 = end_index(ow, output_width, input_width);
      int64_t kw = iw1 - iw0;

      scalar_t* out = output_data + i * channels;
      int64_t size = channels;

      // Note: For ordinary usage scenario, each out lane should
      //   fit in L1 cache; otherwise consider block dim C.
      // Pass I: zero the out lane
      int64_t d1 = 0;
      for (; d1 < size - (size % Vec::size()); d1 += Vec::size()) {
        Vec out_vec = Vec(scalar_t(0));
        out_vec.store(out + d1);
      }
      for (; d1 < size; d1++) {
        out[d1] = scalar_t(0);
      }
      // Pass II: compute local sum
      for (const auto id : c10::irange(id0, id1)) {
        for (const auto ih : c10::irange(ih0, ih1)) {
          for (const auto iw : c10::irange(iw0, iw1)) {
            scalar_t* in = input_data + n * input_depth * input_height * input_width * channels +
                id * input_height * input_width * channels + ih * input_width * channels + iw * channels;

            int64_t d2 = 0;
            for (; d2 < size - (size % Vec::size()); d2 += Vec::size()) {
              Vec out_vec = Vec::loadu(out + d2) + Vec::loadu(in + d2);
              out_vec.store(out + d2);
            }
            for (; d2 < size; d2++) {
              out[d2] += in[d2];
            }
          }
        }
      }
      // Pass III: compute local average
      int64_t d3 = 0;
      for (; d3 < size - (size % Vec::size()); d3 += Vec::size()) {
        Vec out_vec = Vec::loadu(out + d3) / Vec(scalar_t(kd * kh * kw));
        out_vec.store(out + d3);
      }
      for (; d3 < size; d3++) {
        out[d3] = out[d3] / kd / kh / kw;
      }

      // move on to next output index
      data_index_step(n, nbatch, od, output_depth, oh, output_height, ow, output_width);
    }
  });

  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
}

template <typename scalar_t>
typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
cpu_adaptive_avg_pool3d_channels_last(
    Tensor& output_,
    const Tensor& input_,
    IntArrayRef output_size) {
  auto memory_format = at::MemoryFormat::ChannelsLast3d;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_depth = input.size(2);
  int64_t input_height = input.size(3);
  int64_t input_width = input.size(4);
  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  using bVec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  // parallel on dim N,D, H, W
  at::parallel_for(0, nbatch * output_depth * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    int64_t od = 0;
    data_index_init(begin, n, nbatch, od, output_depth, oh, output_height, ow, output_width);

    // temp buffer for sum, use float as accumulation type
    // can't reuse output buffer to store sum since it is BFloat16/Half
    auto sum_arr = std::make_unique<float []>(channels);
    float* sum = sum_arr.get();

    for (const auto i : c10::irange(begin, end)) {
      int64_t id0 = start_index(od, output_depth, input_depth);
      int64_t id1 = end_index(od, output_depth, input_depth);
      int64_t kd = id1 - id0;

      int64_t ih0 = start_index(oh, output_height, input_height);
      int64_t ih1 = end_index(oh, output_height, input_height);
      int64_t kh = ih1 - ih0;

      int64_t iw0 = start_index(ow, output_width, input_width);
      int64_t iw1 = end_index(ow, output_width, input_width);
      int64_t kw = iw1 - iw0;

      scalar_t* out = output_data + i * channels;
      int64_t size = channels;

      // Pass I: zero the out lane
      int64_t d1 = 0;
      for (; d1 < size - (size % fVec::size()); d1 += fVec::size()) {
        fVec sum_fvec = fVec(float(0));
        sum_fvec.store(sum + d1);
      }
      for (; d1 < size; d1++) {
        sum[d1] = float(0);
      }
      // Pass II: compute local sum
      for (const auto id : c10::irange(id0, id1)) {
        for (const auto ih : c10::irange(ih0, ih1)) {
            for (const auto iw : c10::irange(iw0, iw1)) {
                scalar_t* in = input_data + n * input_depth * input_height * input_width * channels +
                    id * input_height * input_width * channels +
                    ih * input_width * channels + iw * channels;

                int64_t d2 = 0;
                for (; d2 < size - (size % bVec::size()); d2 += bVec::size()) {
                    bVec data_bvec = bVec::loadu(in + d2);
                    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);

                    fVec sum_fvec0 = fVec::loadu(sum + d2) + data_fvec0;
                    fVec sum_fvec1 = fVec::loadu(sum + d2 + fVec::size()) + data_fvec1;
                    sum_fvec0.store(sum + d2);
                    sum_fvec1.store(sum + d2 + fVec::size());
                }
                for (; d2 < size; d2++) {
                    sum[d2] += float(in[d2]);
                }
            }
        }
      }
      // Pass III: compute local average
      int64_t d3 = 0;
      for (; d3 < size - (size % bVec::size()); d3 += bVec::size()) {
        fVec out_fvec0 = fVec::loadu(sum + d3) / fVec(float(kd * kh * kw));
        fVec out_fvec1 = fVec::loadu(sum + d3 + fVec::size()) / fVec(float(kd * kh * kw));

        bVec out_bvec = convert_from_float<scalar_t>(out_fvec0, out_fvec1);
        out_bvec.store(out + d3);
      }
      for (; d3 < size; d3++) {
        out[d3] = scalar_t(sum[d3] / kd / kh / kw);
      }

      // move on to next output index
      data_index_step(n, nbatch, od, output_depth, oh, output_height, ow, output_width);
    }
  });

  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
}

template <typename scalar_t>
void cpu_adaptive_avg_pool3d_backward(
    Tensor& grad_input_,
    const Tensor& grad_output_) {
  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();

  int64_t ndim = grad_output.ndimension();
  // treat batch size and channels as one dimension
  int64_t channels = ndim == 4 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
  int64_t input_depth = grad_input.size(-3);
  int64_t input_height = grad_input.size(-2);
  int64_t input_width = grad_input.size(-1);
  int64_t output_depth = grad_output.size(-3);
  int64_t output_height = grad_output.size(-2);
  int64_t output_width = grad_output.size(-1);

  // parallel on dim of N, C
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      scalar_t* grad_input_ptr = grad_input_data + c * input_depth * input_height * input_width;
      scalar_t* grad_output_ptr = grad_output_data + c * output_depth * output_height * output_width;

      for (const auto od : c10::irange(output_depth)) {
        int64_t id0 = start_index(od, output_depth, input_depth);
        int64_t id1 = end_index(od, output_depth, input_depth);
        int64_t kd = id1 - id0;
        for (const auto oh : c10::irange(output_height)) {
          int64_t ih0 = start_index(oh, output_height, input_height);
          int64_t ih1 = end_index(oh, output_height, input_height);
          int64_t kh = ih1 - ih0;

          for (const auto ow : c10::irange(output_width)) {
            int64_t iw0 = start_index(ow, output_width, input_width);
            int64_t iw1 = end_index(ow, output_width, input_width);
            int64_t kw = iw1 - iw0;

            scalar_t grad_delta = grad_output_ptr[od * output_width * output_height + oh * output_width + ow] / kd / kh / kw;
            for (const auto id : c10::irange(id0, id1)) {
              for (const auto ih : c10::irange(ih0, ih1)) {
                for (const auto iw : c10::irange(iw0, iw1)) {
                  grad_input_ptr[id * input_height * input_width + ih * input_width + iw] += grad_delta;
                }
              }
            }
          }
        }
      }
    }
  });

  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

template <typename scalar_t>
void cpu_adaptive_avg_pool3d_backward_channels_last(
    Tensor& grad_input_,
    const Tensor& grad_output_) {
  auto memory_format = at::MemoryFormat::ChannelsLast3d;
  auto grad_input = grad_input_.contiguous(memory_format);
  auto grad_output = grad_output_.contiguous(memory_format);

  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
  auto grad_output_data = grad_output.data_ptr<scalar_t>();

  int64_t nbatch = grad_input.size(0);
  int64_t channels = grad_input.size(1);
  int64_t input_depth = grad_input.size(2);
  int64_t input_height = grad_input.size(3);
  int64_t input_width = grad_input.size(4);
  int64_t output_depth = grad_output.size(2);
  int64_t output_height = grad_output.size(3);
  int64_t output_width = grad_output.size(4);

  using Vec = vec::Vectorized<scalar_t>;
  // parallel on dim N
  at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
    for (const auto n : c10::irange(begin, end)) {
      scalar_t* grad_input_ptr = grad_input_data + n * input_depth * input_height * input_width * channels;
      scalar_t* grad_output_ptr = grad_output_data + n * output_depth * output_height * output_width * channels;

      for (const auto od : c10::irange(output_depth)) {
        int64_t id0 = start_index(od, output_depth, input_depth);
        int64_t id1 = end_index(od, output_depth, input_depth);
        int64_t kd = id1 - id0;
        for (const auto oh : c10::irange(output_height)) {
          int64_t ih0 = start_index(oh, output_height, input_height);
          int64_t ih1 = end_index(oh, output_height, input_height);
          int64_t kh = ih1 - ih0;

          for (const auto ow : c10::irange(output_width)) {
            int64_t iw0 = start_index(ow, output_width, input_width);
            int64_t iw1 = end_index(ow, output_width, input_width);
            int64_t kw = iw1 - iw0;

            scalar_t* gout = grad_output_ptr + od * output_depth * channels + oh * output_width * channels + ow * channels;
            int64_t size = channels;
            for (const auto id : c10::irange(id0, id1)) {
              for (const auto ih : c10::irange(ih0, ih1)) {
                for (const auto iw : c10::irange(iw0, iw1)) {
                  scalar_t* gin = grad_input_ptr + id * input_width * input_height * channels + ih * input_width * channels + iw * channels;

                  int64_t d = 0;
                  for (; d < size - (size % Vec::size()); d += Vec::size()) {
                    Vec gin_vec = Vec::loadu(gin + d) + Vec::loadu(gout + d) / Vec(scalar_t(kd * kh * kw));
                    gin_vec.store(gin + d);
                  }
                  for (; d < size; d++) {
                    gin[d] += gout[d] / kd / kh / kw;
                  }
                }
              }
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


void adaptive_avg_pool3d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "adaptive_avg_pool3d", [&] {
        using param_t = at::opmath_type<scalar_t>;
        cpu_adaptive_avg_pool3d<scalar_t, /*accscalar_t*/param_t>(output, input, output_size);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast3d: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "adaptive_avg_pool3d_channels_last", [&]{
        cpu_adaptive_avg_pool3d_channels_last<scalar_t>(output, input, output_size);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void adapative_avg_pool3d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output) {
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "adaptive_avg_pool3d_backward", [&] {
        cpu_adaptive_avg_pool3d_backward<scalar_t>(grad_input, grad_output);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast3d: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "adaptive_avg_pool3d_backward_channels_last", [&]{
        cpu_adaptive_avg_pool3d_backward_channels_last<scalar_t>(grad_input, grad_output);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

} // anonymous namespace

REGISTER_DISPATCH(adaptive_avg_pool2d_kernel, &adaptive_avg_pool2d_kernel_impl)
REGISTER_DISPATCH(adaptive_avg_pool2d_backward_kernel, &adapative_avg_pool2d_backward_kernel_impl)
REGISTER_DISPATCH(adaptive_avg_pool3d_kernel, &adaptive_avg_pool3d_kernel_impl)
REGISTER_DISPATCH(adaptive_avg_pool3d_backward_kernel, &adapative_avg_pool3d_backward_kernel_impl)

} // at::native
