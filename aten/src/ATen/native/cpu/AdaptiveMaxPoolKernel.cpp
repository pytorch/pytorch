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
void cpu_adaptive_max_pool(
    const Tensor& output_,
    const Tensor& indices_,
    const Tensor& input_,
    IntArrayRef output_size) {
  auto input = input_.contiguous();
  auto output = output_.contiguous();
  auto indices = indices_.contiguous();

  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();
  auto indices_data = indices.data_ptr<int64_t>();

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
      int64_t* indices_ptr = indices_data + c * output_height * output_width;

      for (const auto oh : c10::irange(output_height)) {
        int64_t ih0 = start_index(oh, output_height, input_height);
        int64_t ih1 = end_index(oh, output_height, input_height);

        for (const auto ow : c10::irange(output_width)) {
          int64_t iw0 = start_index(ow, output_width, input_width);
          int64_t iw1 = end_index(ow, output_width, input_width);

          // compute local max
          int64_t maxindex = ih0 * input_width + iw0;
          accscalar_t maxval = -std::numeric_limits<accscalar_t>::infinity();
          for (int64_t ih = ih0; ih < ih1; ih ++) {
            for (int64_t iw = iw0; iw < iw1; iw ++) {
              int64_t index = ih * input_width + iw;
              scalar_t val = input_ptr[index];
              if ((val > maxval) || std::isnan(val)) {
                maxval = val;
                maxindex = index;
              }
            }
          }

          // set output to local max and store location of max
          output_ptr[oh * output_width + ow] = maxval;
          indices_ptr[oh * output_width + ow] = scalar_t(maxindex);
        }
      }
    }
  });

  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
  if (!indices_.is_contiguous()) {
    indices_.copy_(indices);
  }
}

template <typename scalar_t>
typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
cpu_adaptive_max_pool_channels_last(
    const Tensor& output_,
    const Tensor& indices_,
    const Tensor& input_,
    IntArrayRef output_size) {
  TORCH_CHECK(input_.ndimension() == 4,
              "adaptive max pooling with channels last format supports tensors with 4 dims");
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);
  auto indices = indices_.contiguous(memory_format);

  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();
  auto indices_data = indices.data_ptr<int64_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  using Vec = vec::Vectorized<scalar_t>;
  using integer_t = vec::int_same_size_t<scalar_t>;
  using iVec = vec::Vectorized<integer_t>;
  // for the convenience of vectorization, use integer of the same size of scalar_t,
  //   e.g. int32_t for float, int64_t for double
  // need to make sure doesn't overflow
  TORCH_CHECK(input_height * input_width <= std::numeric_limits<integer_t>::max());

  // parallel on dim of N, H, W
  at::parallel_for(0, nbatch * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);

    int64_t size = channels;
    int64_t len = size - (size % Vec::size());
    // temp buffer holding index with integer_t
    auto index_buffer = std::make_unique<integer_t []>(len);

    for (const auto i : c10::irange(begin, end)) {
      int64_t ih0 = start_index(oh, output_height, input_height);
      int64_t ih1 = end_index(oh, output_height, input_height);

      int64_t iw0 = start_index(ow, output_width, input_width);
      int64_t iw1 = end_index(ow, output_width, input_width);

      scalar_t* out = output_data + i * channels;
      int64_t* ind = indices_data + i * channels;

      // Pass I: init out lane
      iVec index0_vec = iVec(ih0 * input_width + iw0);
      Vec out_vec = Vec(-std::numeric_limits<scalar_t>::infinity());
      int64_t d1 = 0;
      for (; d1 < len; d1 += Vec::size()) {
        index0_vec.store(index_buffer.get() + d1);
        out_vec.store(out + d1);
      }
      for (; d1 < size; d1++) {
        ind[d1] = ih0 * input_width + iw0;
        out[d1] = -std::numeric_limits<scalar_t>::infinity();
      }
      // Pass II: compute local max
      for (int64_t ih = ih0; ih < ih1; ih ++) {
        for (int64_t iw = iw0; iw < iw1; iw ++) {
          const scalar_t* in = input_data + n * input_height * input_width * channels +
              ih * input_width * channels + iw * channels;

          int64_t d2 = 0;
          for (; d2 < len; d2 += Vec::size()) {
            iVec index_vec = iVec(ih * input_width + iw);
            Vec val_vec = Vec::loadu(in + d2);
            iVec maxindex_vec = iVec::loadu(index_buffer.get() + d2);
            Vec maxval_vec = Vec::loadu(out + d2);

            // true = all ones, false = all zeros
            Vec mask = (val_vec > maxval_vec) | val_vec.isnan();
            iVec imask = vec::cast<integer_t>(mask);
            Vec out_vec = Vec::blendv(maxval_vec, val_vec, mask);
            iVec ind_vec = iVec::blendv(maxindex_vec, index_vec, imask);

            out_vec.store(out + d2);
            ind_vec.store(index_buffer.get() + d2);
          }
          for (; d2 < size; d2++) {
            int64_t index = ih * input_width + iw;
            scalar_t val = in[d2];
            int64_t maxindex = ind[d2];
            scalar_t maxval = out[d2];

            bool mask = (val > maxval) || std::isnan(val);
            out[d2] = mask ? val : maxval;
            ind[d2] = mask ? index : maxindex;
          }
        }
      }
      // convert indice data type
      vec::convert<integer_t, int64_t>(index_buffer.get(), ind, len);

      // move on to next output index
      data_index_step(n, nbatch, oh, output_height, ow, output_width);
    }
  });

  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
  if (!indices_.is_contiguous(memory_format)) {
    indices_.copy_(indices);
  }
}

template <typename scalar_t>
typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
cpu_adaptive_max_pool_channels_last(
    const Tensor& output_,
    const Tensor& indices_,
    const Tensor& input_,
    IntArrayRef output_size) {
  TORCH_CHECK(input_.ndimension() == 4,
              "adaptive max pooling with channels last format supports tensors with 4 dims");
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);
  auto indices = indices_.contiguous(memory_format);

  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();
  auto indices_data = indices.data_ptr<int64_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  using bVec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  using iVec = vec::Vectorized<int32_t>;
  // need to make sure doesn't overflow
  TORCH_CHECK(input_height * input_width <= std::numeric_limits<int32_t>::max());

  // parallel on dim of N, H, W
  at::parallel_for(0, nbatch * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);

    int64_t size = channels;
    int64_t len = size - (size % bVec::size());
    // temp buffer holding index with integer_t
    auto index_buffer = std::make_unique<int32_t []>(len);
    // temp buffer holding max value with float
    auto max_arr = std::make_unique<float []>(size);
    float* max = max_arr.get();

    for (const auto i : c10::irange(begin, end)) {
      int64_t ih0 = start_index(oh, output_height, input_height);
      int64_t ih1 = end_index(oh, output_height, input_height);

      int64_t iw0 = start_index(ow, output_width, input_width);
      int64_t iw1 = end_index(ow, output_width, input_width);

      scalar_t* out = output_data + i * channels;
      int64_t* ind = indices_data + i * channels;

      // Pass I: init out lane
      iVec index0_ivec = iVec(ih0 * input_width + iw0);
      fVec max_fvec = fVec(-std::numeric_limits<float>::infinity());
      int64_t d1 = 0;
      for (; d1 < len; d1 += fVec::size()) {
        index0_ivec.store(index_buffer.get() + d1);
        max_fvec.store(max + d1);
      }
      for (; d1 < size; d1++) {
        ind[d1] = ih0 * input_width + iw0;
        max[d1] = -std::numeric_limits<float>::infinity();
      }
      // Pass II: compute local max
      for (int64_t ih = ih0; ih < ih1; ih ++) {
        for (int64_t iw = iw0; iw < iw1; iw ++) {
          const scalar_t* in = input_data + n * input_height * input_width * channels +
              ih * input_width * channels + iw * channels;

          int64_t d2 = 0;
          for (; d2 < len; d2 += bVec::size()) {
            iVec index_ivec = iVec(ih * input_width + iw);
            bVec val_bvec = bVec::loadu(in + d2);
            fVec val_fvec0, val_fvec1;
            std::tie(val_fvec0, val_fvec1) = convert_to_float<scalar_t>(val_bvec);

            iVec maxindex_ivec0 = iVec::loadu(index_buffer.get() + d2);
            iVec maxindex_ivec1 = iVec::loadu(index_buffer.get() + d2 + iVec::size());
            fVec maxval_fvec0 = fVec::loadu(max + d2);
            fVec maxval_fvec1 = fVec::loadu(max + d2 + fVec::size());

            // true = all ones, false = all zeros
            fVec mask0 = (val_fvec0 > maxval_fvec0) | val_fvec0.isnan();
            fVec mask1 = (val_fvec1 > maxval_fvec1) | val_fvec1.isnan();
            iVec imask0 = vec::cast<int32_t>(mask0);
            iVec imask1 = vec::cast<int32_t>(mask1);

            fVec max_fvec0 = fVec::blendv(maxval_fvec0, val_fvec0, mask0);
            fVec max_fvec1 = fVec::blendv(maxval_fvec1, val_fvec1, mask1);
            iVec ind_ivec0 = iVec::blendv(maxindex_ivec0, index_ivec, imask0);
            iVec ind_ivec1 = iVec::blendv(maxindex_ivec1, index_ivec, imask1);

            max_fvec0.store(max + d2);
            max_fvec1.store(max + d2 + fVec::size());
            ind_ivec0.store(index_buffer.get() + d2);
            ind_ivec1.store(index_buffer.get() + d2 + iVec::size());
          }
          for (; d2 < size; d2++) {
            int64_t index = ih * input_width + iw;
            float val = float(in[d2]);
            int64_t maxindex = ind[d2];
            float maxval = max[d2];

            bool mask = (val > maxval) || std::isnan(val);
            max[d2] = mask ? val : maxval;
            ind[d2] = mask ? index : maxindex;
          }
        }
      }
      // Pass III: convert max values from float to bfloat16/Half
      int64_t d3 = 0;
      for (; d3 < len; d3 += bVec::size()) {
        fVec max_fvec0 = fVec::loadu(max + d3);
        fVec max_fvec1 = fVec::loadu(max + d3 + fVec::size());
        bVec max_bvec = convert_from_float<scalar_t>(max_fvec0, max_fvec1);
        max_bvec.store(out + d3);
      }
      for (; d3 < size; d3++) {
        out[d3] = scalar_t(max[d3]);
      }
      // convert indice data type
      vec::convert<int32_t, int64_t>(index_buffer.get(), ind, len);

      // move on to next output index
      data_index_step(n, nbatch, oh, output_height, ow, output_width);
    }
  });

  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
  if (!indices_.is_contiguous(memory_format)) {
    indices_.copy_(indices);
  }
}

template <typename scalar_t>
void cpu_adaptive_max_pool_backward(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    const Tensor& indices_) {
  auto grad_output = grad_output_.contiguous();
  auto indices = indices_.contiguous();
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  auto indices_data = indices.const_data_ptr<int64_t>();
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
      const int64_t* indices_ptr = indices_data + c * output_height * output_width;

      for (const auto oh : c10::irange(output_height)) {
        for (const auto ow : c10::irange(output_width)) {
          // retrieve position of max
          int64_t index = oh * output_width + ow;
          int64_t maxindex = indices_ptr[index];

          // update gradient
          grad_input_ptr[maxindex] += grad_output_ptr[index];
        }
      }
    }
  });

  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

template <typename scalar_t>
void cpu_adaptive_max_pool_backward_channels_last(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    const Tensor& indices_) {
  TORCH_CHECK(grad_output_.ndimension() == 4,
              "adaptive max pooling backward with channels last format supports tensors with 4 dims.");
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto grad_input = grad_input_.contiguous(memory_format);
  auto grad_output = grad_output_.contiguous(memory_format);
  auto indices = indices_.contiguous(memory_format);

  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  auto indices_data = indices.const_data_ptr<int64_t>();

  int64_t nbatch = grad_input.size(0);
  int64_t channels = grad_input.size(1);
  int64_t input_height = grad_input.size(2);
  int64_t input_width = grad_input.size(3);
  int64_t output_height = grad_output.size(2);
  int64_t output_width = grad_output.size(3);

  // parallel on dim N
  at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
    for (const auto n : c10::irange(begin, end)) {
      scalar_t* grad_input_ptr = grad_input_data + n * input_height * input_width * channels;
      const scalar_t* grad_output_ptr = grad_output_data + n * output_height * output_width * channels;
      const int64_t* indices_ptr = indices_data + n * output_height * output_width * channels;

      for (const auto oh : c10::irange(output_height)) {
        for (const auto ow : c10::irange(output_width)) {
          const scalar_t* gout = grad_output_ptr + oh * output_width * channels + ow * channels;
          const int64_t* ind = indices_ptr + oh * output_width * channels + ow * channels;
          // TODO: gcc vectorization
          for (const auto c : c10::irange(channels)) {
            int64_t maxindex = ind[c];
            grad_input_ptr[maxindex * channels + c] += gout[c];
          }
        }
      }
    }
  });

  if (!grad_input_.is_contiguous(memory_format)) {
    grad_input_.copy_(grad_input);
  }
}

void adaptive_max_pool2d_kernel_impl(
    const Tensor& output,
    const Tensor& indices,
    const Tensor& input,
    IntArrayRef output_size) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "adaptive_max_pool2d", [&] {
        using param_t = at::opmath_type<scalar_t>;
        cpu_adaptive_max_pool<scalar_t, /*accscalar_t*/param_t>(output, indices, input, output_size);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "adaptive_max_pool2d_channels_last", [&]{
        cpu_adaptive_max_pool_channels_last<scalar_t>(output, indices, input, output_size);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void adaptive_max_pool2d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& indices) {
  // can't use grad_output memory format to switch here since grad_output might be NC11
  switch (grad_input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "adaptive_max_pool2d_backward", [&] {
        cpu_adaptive_max_pool_backward<scalar_t>(grad_input, grad_output, indices);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "adaptive_max_pool2d_backward_channels_last", [&]{
        cpu_adaptive_max_pool_backward_channels_last<scalar_t>(grad_input, grad_output, indices);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

} // anonymous namespace

REGISTER_DISPATCH(adaptive_max_pool2d_kernel, &adaptive_max_pool2d_kernel_impl);
REGISTER_DISPATCH(adaptive_max_pool2d_backward_kernel, &adaptive_max_pool2d_backward_kernel_impl);

} // at::native
