#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <vector>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/UpSample.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <c10/util/irange.h>
#include <ATen/cpu/vec/vec.h>

namespace at::native {
namespace {

using scale_t = std::vector<c10::optional<double>>;

template <typename acc_t, typename scalar_t>
void inline nearest_channels_last_acc(acc_t* gin, scalar_t* gout, int64_t size) {
  TORCH_CHECK((std::is_same<acc_t, scalar_t>::value),
              "acc data type of Upsample backward should be same as scalar_t for float or double on CPU.")
  using Vec = vec::Vectorized<acc_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec gin_vec = Vec::loadu(gin + d) + Vec::loadu(gout + d);
    gin_vec.store(gin + d);
  }
  for (; d < size; d++) {
    gin[d] += gout[d];
  }
}

template <>
void inline nearest_channels_last_acc(float* gin, BFloat16* gout, int64_t size) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec gout_bvec = bVec::loadu(gout + d);
    auto [gout_fvec0, gout_fvec1] = convert_bfloat16_float(gout_bvec);
    fVec gin_fvec0 = fVec::loadu(gin + d) + gout_fvec0;
    fVec gin_fvec1 = fVec::loadu(gin + d + fVec::size()) + gout_fvec1;
    gin_fvec0.store(gin + d);
    gin_fvec1.store(gin + d + fVec::size());
  }
  for (; d < size; d++) {
    gin[d] += gout[d];
  }
}

template <typename acc_t, typename scalar_t>
void inline linear_channels_last_acc(acc_t* gin, scalar_t* gout, acc_t w, int64_t size) {
  TORCH_CHECK((std::is_same<acc_t, scalar_t>::value),
              "acc data type of Upsample backward should be same as scalar_t for float or double on CPU.")
  using Vec = vec::Vectorized<acc_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec gin_vec = Vec::loadu(gin + d) + Vec(w) * Vec::loadu(gout + d);
    gin_vec.store(gin + d);
  }
  for (; d < size; d++) {
    gin[d] += w * gout[d];
  }
}

template <>
void inline linear_channels_last_acc(float* gin, BFloat16* gout, float w, int64_t size) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec gout_bvec = bVec::loadu(gout + d);
    auto [gout_fvec0, gout_fvec1] = convert_bfloat16_float(gout_bvec);
    fVec gin_fvec0 = fVec::loadu(gin + d) + fVec(w) * gout_fvec0;
    fVec gin_fvec1 = fVec::loadu(gin + d + fVec::size()) + fVec(w) * gout_fvec1;
    gin_fvec0.store(gin + d);
    gin_fvec1.store(gin + d + fVec::size());
  }
  for (; d < size; d++) {
    gin[d] += w * gout[d];
  }
}

template <typename scalar_t, typename scale_type, nearest_idx_fn_t nearest_idx_fn>
void cpu_upsample_nearest_backward(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    const scale_type& scales) {
  TORCH_CHECK(grad_input_.dtype() == grad_output_.dtype(), "expected dtype ", grad_output_.dtype(),
              " for `grad_input` but got dtype ", grad_input_.dtype());

  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto grad_input_data = grad_input.data_ptr<scalar_t>();
  auto input_sizes = grad_input.sizes().vec();
  auto output_sizes = grad_output.sizes().vec();
  auto ndim = input_sizes.size();

  // treat nbatch and channels as one dimension
  int64_t channels = input_sizes[0] * input_sizes[1];
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];

  int64_t output_slice_size = output_depth * output_height * output_width;
  int64_t input_slice_size = input_depth * input_height * input_width;

  using opmath_t = at::opmath_type<scalar_t>;
  auto loop1d = [&](int64_t begin, int64_t end) {
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
      buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
      acc_data_ptr = buffer_data.get();
      memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }

    for (const auto c : c10::irange(begin, end)) {
      int64_t input_offset = buffer_data.get() == nullptr ? c * input_slice_size : 0;
      for (const auto ow : c10::irange(output_width)) {
        int64_t iw = nearest_idx_fn(ow, input_width, output_width, scales[0]);
        int64_t output_offset = c * output_slice_size + ow;
        acc_data_ptr[input_offset + iw] += grad_output_data[output_offset];
      }
      if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        auto gin = grad_input_data + c * input_slice_size;
        apply_grad_input(acc_data_ptr, gin, input_slice_size);
      }
    }
  };

  auto loop2d = [&](int64_t begin, int64_t end) {
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        acc_data_ptr = buffer_data.get();
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }

    for (const auto c : c10::irange(begin, end)) {
      int64_t input_offset = buffer_data.get() == nullptr ? c * input_slice_size : 0;
      for (const auto oh : c10::irange(output_height)) {
        int64_t ih = nearest_idx_fn(oh, input_height, output_height, scales[0]);
        for (const auto ow : c10::irange(output_width)) {
          int64_t iw = nearest_idx_fn(ow, input_width, output_width, scales[1]);
          int64_t output_offset = c * output_slice_size + oh * output_width + ow;
          acc_data_ptr[input_offset + ih * input_width + iw] += grad_output_data[output_offset];
        }
      }
      if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        auto gin = grad_input_data + c * input_slice_size;
        apply_grad_input(acc_data_ptr, gin, input_slice_size);
      }
    }
  };

  auto loop3d = [&](int64_t begin, int64_t end) {
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        acc_data_ptr = buffer_data.get();
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }

    for (const auto c : c10::irange(begin, end)) {
      int64_t input_offset = buffer_data.get() == nullptr ? c * input_slice_size : 0;
      for (const auto od : c10::irange(output_depth)) {
        int64_t id = nearest_idx_fn(od, input_depth, output_depth, scales[0]);
        for (const auto oh : c10::irange(output_height)) {
          int64_t ih = nearest_idx_fn(oh, input_height, output_height, scales[1]);
          for (const auto ow : c10::irange(output_width)) {
            int64_t iw = nearest_idx_fn(ow, input_width, output_width, scales[2]);
            int64_t output_offset = c * output_slice_size +
                od *  output_height * output_width + oh * output_width + ow;
            acc_data_ptr[input_offset + id * input_height * input_width + ih * input_width + iw] +=
              grad_output_data[output_offset];
          }
        }
      }
      if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        auto gin = grad_input_data + c * input_slice_size;
        apply_grad_input(acc_data_ptr, gin, input_slice_size);
      }
    }
  };

  if (ndim == 3) {
    // upsample nearest 1d
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size, loop1d);
  } else if (ndim == 4) {
    // upsample nearest 2d
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size , loop2d);
  } else {
    // upsample nearest 3d
    TORCH_INTERNAL_ASSERT(ndim == 5);
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size, loop3d);
  }

  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

template <typename scalar_t, typename scale_type, nearest_idx_fn_t nearest_idx_fn>
void cpu_upsample_nearest_backward_channels_last(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    const scale_type& scales) {
  TORCH_CHECK(grad_input_.dtype() == grad_output_.dtype(), "expected dtype ", grad_output_.dtype(),
              " for `grad_input` but got dtype ", grad_input_.dtype());

  auto ndim = grad_output_.ndimension();
  TORCH_CHECK(ndim >=4 && ndim <= 5, "Upsample with NHWC format supports tensors with 4 or 5 dims.")

  auto channels_last_memory_format = ndim == 4 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::ChannelsLast3d;
  auto grad_output = grad_output_.contiguous(channels_last_memory_format);
  auto grad_input = grad_input_.contiguous(channels_last_memory_format);

  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto grad_input_data = grad_input.data_ptr<scalar_t>();

  auto input_sizes = grad_input.sizes().vec();
  auto output_sizes = grad_output.sizes().vec();

  int64_t num_batches =  input_sizes[0];
  int64_t channels =  input_sizes[1];
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];
  int64_t input_slice_size = input_depth * input_height * input_width * channels;

  using opmath_t = at::opmath_type<scalar_t>;
  auto loop2d = [&](int64_t begin, int64_t end) {
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        acc_data_ptr = buffer_data.get();
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }

    for (const auto n : c10::irange(begin, end)) {
      int64_t input_offset = buffer_data.get() == nullptr ? n * input_slice_size : 0;
      for (const auto oh : c10::irange(output_height)) {
        int64_t ih = nearest_idx_fn(oh, input_height, output_height, scales[0]);
        for (const auto ow : c10::irange(output_width)) {
          int64_t iw = nearest_idx_fn(ow, input_width, output_width, scales[1]);
          scalar_t* grad_output_ptr = grad_output_data +
              (n * output_height * output_width + oh * output_width + ow) * channels;
          opmath_t* buffer_ptr = acc_data_ptr + input_offset + (ih * input_width + iw) * channels;
          nearest_channels_last_acc(buffer_ptr, grad_output_ptr, channels);
        }
      }
      if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        auto gin = grad_input_data + n * input_slice_size;
        apply_grad_input(acc_data_ptr, gin, input_slice_size);
      }
    }

  };

  auto loop3d = [&](int64_t begin, int64_t end) {
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        acc_data_ptr = buffer_data.get();
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }

    for (const auto n : c10::irange(begin, end)) {
      int64_t input_offset = buffer_data.get() == nullptr ? n * input_slice_size : 0;
      for (int64_t od = 0; od < output_depth; od++) {
        int64_t id = nearest_idx_fn(od, input_depth, output_depth, scales[0]);
        for (int64_t oh = 0; oh < output_height; oh++) {
          int64_t ih = nearest_idx_fn(oh, input_height, output_height, scales[1]);
          for (int64_t ow = 0; ow < output_width; ow++) {
            int64_t iw = nearest_idx_fn(ow, input_width, output_width, scales[2]);
            scalar_t* grad_output_ptr = grad_output_data +
                (n * output_depth * output_height * output_width +
                od * output_height * output_width + oh * output_width + ow) * channels;

            opmath_t* buffer_ptr = acc_data_ptr + input_offset + (id * input_height * input_width + ih * input_width + iw) * channels;
            nearest_channels_last_acc(buffer_ptr, grad_output_ptr, channels);
          }
        }
      }
      if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        auto gin = grad_input_data + n * input_slice_size;
        apply_grad_input(acc_data_ptr, gin, input_slice_size);
      }
    }

  };

  if (ndim == 4) {
    // upsample nearest 2d
    at::parallel_for(0, num_batches, 0, loop2d);
  } else {
    // upsample nearest 3d
    TORCH_INTERNAL_ASSERT(ndim == 5);
    at::parallel_for(0, num_batches, 0, loop3d);
  }

  if (!grad_input_.is_contiguous(channels_last_memory_format)) {
    grad_input_.copy_(grad_input);
  }
}

void upsample_nearest1d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grad_output.scalar_type(), "upsample_nearest1d_backward", [&] {
    cpu_upsample_nearest_backward<scalar_t, scale_t, nearest_idx>(grad_input, grad_output, {scales_w});
  });
}

void _upsample_nearest_exact1d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grad_output.scalar_type(), "_upsample_nearest_exact1d_backward", [&] {
    cpu_upsample_nearest_backward<scalar_t, scale_t, nearest_exact_idx>(grad_input, grad_output, {scales_w});
  });
}

void upsample_nearest2d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  if (grad_output.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grad_output.scalar_type(), "upsample_nearest2d_backward_cl", [&] {
      cpu_upsample_nearest_backward_channels_last<scalar_t, scale_t, nearest_idx>(grad_input, grad_output, {scales_h, scales_w});
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grad_output.scalar_type(), "upsample_nearest2d_backward", [&] {
      cpu_upsample_nearest_backward<scalar_t, scale_t, nearest_idx>(grad_input, grad_output, {scales_h, scales_w});
    });
  }
}

void _upsample_nearest_exact2d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  if (grad_output.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grad_output.scalar_type(), "_upsample_nearest_exact2d_backward_cl", [&] {
      cpu_upsample_nearest_backward_channels_last<scalar_t, scale_t, nearest_exact_idx>(grad_input, grad_output, {scales_h, scales_w});
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grad_output.scalar_type(), "_upsample_nearest_exact2d_backward", [&] {
      cpu_upsample_nearest_backward<scalar_t, scale_t, nearest_exact_idx>(grad_input, grad_output, {scales_h, scales_w});
    });
  }
}

void upsample_nearest3d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grad_output.scalar_type(), "upsample_nearest3d_backward", [&] {
    cpu_upsample_nearest_backward<scalar_t, scale_t, nearest_idx>(grad_input, grad_output, {scales_d, scales_h, scales_w});
  });
}

void _upsample_nearest_exact3d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grad_output.scalar_type(), "_upsample_nearest_exact3d_backward", [&] {
    cpu_upsample_nearest_backward<scalar_t, scale_t, nearest_exact_idx>(grad_input, grad_output, {scales_d, scales_h, scales_w});
  });
}

template <typename scalar_t, typename scale_type>
void cpu_upsample_linear_backward(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    bool align_corners,
    const scale_type& scales) {
  TORCH_CHECK(grad_input_.dtype() == grad_output_.dtype(), "expected dtype ", grad_output_.dtype(),
              " for `grad_input` but got dtype ", grad_input_.dtype());

  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto grad_input_data = grad_input.data_ptr<scalar_t>();
  auto input_sizes = grad_input.sizes().vec();
  auto output_sizes = grad_output.sizes().vec();
  auto ndim = input_sizes.size();

  // treat nbatch and channels as one dimension
  int64_t channels = input_sizes[0] * input_sizes[1];
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];

  int64_t input_slice_size = input_depth * input_height * input_width;
  int64_t output_slice_size = output_depth * output_height * output_width;
  using opmath_t = at::opmath_type<scalar_t>;
  auto loop1d = [&](int64_t begin, int64_t end) {
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        acc_data_ptr = buffer_data.get();
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }

    const opmath_t width_scale = area_pixel_compute_scale<opmath_t>(
        input_width, output_width, align_corners, scales[0]);

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t iw0, iw1;
    opmath_t w0lambda, w1lambda;
    for (const auto c : c10::irange(begin, end)) {
      int64_t input_offset = buffer_data.get() == nullptr ? c * input_slice_size : 0;
      for (const auto ow : c10::irange(output_width)) {
        compute_source_index_and_lambda(
            iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
        opmath_t grad_output_value = grad_output_data[c * output_slice_size + ow];
        acc_data_ptr[input_offset + iw0] += w0lambda * grad_output_value; /* i0 */
        acc_data_ptr[input_offset + iw1] += w1lambda * grad_output_value; /* i1*/
      }
      if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        auto gin = grad_input_data + c * input_slice_size;
        apply_grad_input(acc_data_ptr, gin, input_slice_size);
      }
    }
  };

  auto loop2d = [&](int64_t begin, int64_t end) {
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        acc_data_ptr = buffer_data.get();
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }

    const opmath_t height_scale = area_pixel_compute_scale<opmath_t>(
        input_height, output_height, align_corners, scales[0]);
    const opmath_t width_scale = area_pixel_compute_scale<opmath_t>(
        input_width, output_width, align_corners, scales[1]);

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t ih0, ih1, iw0, iw1;
    opmath_t h0lambda, h1lambda, w0lambda, w1lambda;
    for (const auto c : c10::irange(begin, end)) {
      int64_t input_offset = buffer_data.get() == nullptr ? c * input_slice_size : 0;
      for (const auto oh : c10::irange(output_height)) {
        compute_source_index_and_lambda(
            ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
        for (const auto ow : c10::irange(output_width)) {
          compute_source_index_and_lambda(
              iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
          opmath_t grad_output_value = grad_output_data[c * output_slice_size + oh * output_width + ow];
          acc_data_ptr[input_offset + ih0 * input_width + iw0] += h0lambda * w0lambda * grad_output_value; /* i00 */
          acc_data_ptr[input_offset + ih0 * input_width + iw1] += h0lambda * w1lambda * grad_output_value; /* i01 */
          acc_data_ptr[input_offset + ih1 * input_width + iw0] += h1lambda * w0lambda * grad_output_value; /* i10 */
          acc_data_ptr[input_offset + ih1 * input_width + iw1] += h1lambda * w1lambda * grad_output_value; /* i11 */
        }
      }
      if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        auto gin = grad_input_data + c * input_slice_size;
        apply_grad_input(acc_data_ptr, gin, input_slice_size);
      }
    }
  };

  auto loop3d = [&](int64_t begin, int64_t end) {
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        acc_data_ptr = buffer_data.get();
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }

    const opmath_t depth_scale = area_pixel_compute_scale<opmath_t>(
        input_depth, output_depth, align_corners, scales[0]);
    const opmath_t height_scale = area_pixel_compute_scale<opmath_t>(
        input_height, output_height, align_corners, scales[1]);
    const opmath_t width_scale = area_pixel_compute_scale<opmath_t>(
        input_width, output_width, align_corners, scales[2]);

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t id0, id1, ih0, ih1, iw0, iw1;
    opmath_t d0lambda, d1lambda, h0lambda, h1lambda, w0lambda, w1lambda;
    for (const auto c : c10::irange(begin, end)) {
      int64_t input_offset = buffer_data.get() == nullptr ? c * input_slice_size : 0;
      for (const auto od : c10::irange(output_depth)) {
        compute_source_index_and_lambda(
            id0, id1, d0lambda, d1lambda, depth_scale, od, input_depth, output_depth, align_corners);
        for (const auto oh : c10::irange(output_height)) {
          compute_source_index_and_lambda(
              ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
          for (const auto ow : c10::irange(output_width)) {
            compute_source_index_and_lambda(
                iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
            opmath_t grad_output_value = grad_output_data[c * output_slice_size +
                od *  output_height * output_width + oh * output_width + ow];
            acc_data_ptr[input_offset + id0 * input_height * input_width + ih0 * input_width + iw0] += d0lambda * h0lambda * w0lambda * grad_output_value; /* i000 */
            acc_data_ptr[input_offset + id0 * input_height * input_width + ih0 * input_width + iw1] += d0lambda * h0lambda * w1lambda * grad_output_value; /* i001 */
            acc_data_ptr[input_offset + id0 * input_height * input_width + ih1 * input_width + iw0] += d0lambda * h1lambda * w0lambda * grad_output_value; /* i010 */
            acc_data_ptr[input_offset + id0 * input_height * input_width + ih1 * input_width + iw1] += d0lambda * h1lambda * w1lambda * grad_output_value; /* i011 */
            acc_data_ptr[input_offset + id1 * input_height * input_width + ih0 * input_width + iw0] += d1lambda * h0lambda * w0lambda * grad_output_value; /* i100 */
            acc_data_ptr[input_offset + id1 * input_height * input_width + ih0 * input_width + iw1] += d1lambda * h0lambda * w1lambda * grad_output_value; /* i101 */
            acc_data_ptr[input_offset + id1 * input_height * input_width + ih1 * input_width + iw0] += d1lambda * h1lambda * w0lambda * grad_output_value; /* i110 */
            acc_data_ptr[input_offset + id1 * input_height * input_width + ih1 * input_width + iw1] += d1lambda * h1lambda * w1lambda * grad_output_value; /* i111 */
          }
        }
      }
      if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        auto gin = grad_input_data + c * input_slice_size;
        apply_grad_input(acc_data_ptr, gin, input_slice_size);
      }
    }
  };

  if (ndim == 3) {
    // upsample linear 1d
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 2, loop1d);
  } else if (ndim == 4) {
    // upsample bilinear 2d
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 4, loop2d);
  } else {
    // upsample trilinear 3d
    TORCH_INTERNAL_ASSERT(ndim == 5);
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 8, loop3d);
  }

  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

template <typename scalar_t, typename scale_type>
void cpu_upsample_linear_backward_channels_last(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    bool align_corners,
    const scale_type& scales) {
  TORCH_CHECK(grad_input_.dtype() == grad_output_.dtype(), "expected dtype ", grad_output_.dtype(),
              " for `grad_input` but got dtype ", grad_input_.dtype());

  auto ndim = grad_output_.ndimension();
  TORCH_CHECK(ndim >=4 && ndim <= 5, "Upsample with NHWC format supports tensors with 4 or 5 dims.")

  auto channels_last_memory_format = ndim == 4 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::ChannelsLast3d;
  auto grad_output = grad_output_.contiguous(channels_last_memory_format);
  auto grad_input = grad_input_.contiguous(channels_last_memory_format);

  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto grad_input_data = grad_input.data_ptr<scalar_t>();

  auto input_sizes = grad_input.sizes().vec();
  auto output_sizes = grad_output.sizes().vec();

  int64_t num_batches =  input_sizes[0];
  int64_t channels =  input_sizes[1];
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];
  int64_t input_slice_size = input_depth * input_height * input_width * channels;
  using opmath_t = at::opmath_type<scalar_t>;

  auto loop2d = [&](int64_t begin, int64_t end) {
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        acc_data_ptr = buffer_data.get();
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }

    const opmath_t height_scale = area_pixel_compute_scale<opmath_t>(
        input_height, output_height, align_corners, scales[0]);
    const opmath_t width_scale = area_pixel_compute_scale<opmath_t>(
        input_width, output_width, align_corners, scales[1]);

    auto input_indexr = [=](int64_t n, int64_t h, int64_t w, int64_t offset){
      return acc_data_ptr + offset + (h * input_width + w) * channels;
    };

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t ih0, ih1, iw0, iw1;
    opmath_t h0lambda, h1lambda, w0lambda, w1lambda;
    for (const auto n : c10::irange(begin, end)) {
      int64_t input_offset = buffer_data.get() == nullptr ? n * input_slice_size : 0;
      for (const auto oh : c10::irange(output_height)) {
        compute_source_index_and_lambda(
            ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
        for (const auto ow : c10::irange(output_width)) {
          compute_source_index_and_lambda(
              iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
          scalar_t* grad_output_ptr = grad_output_data +
              (n * output_height * output_width + oh * output_width + ow) * channels;
          linear_channels_last_acc(input_indexr(n, ih0, iw0, input_offset), grad_output_ptr, h0lambda * w0lambda, channels); /* i00 */
          linear_channels_last_acc(input_indexr(n, ih0, iw1, input_offset), grad_output_ptr, h0lambda * w1lambda, channels); /* i01 */
          linear_channels_last_acc(input_indexr(n, ih1, iw0, input_offset), grad_output_ptr, h1lambda * w0lambda, channels); /* i10 */
          linear_channels_last_acc(input_indexr(n, ih1, iw1, input_offset), grad_output_ptr, h1lambda * w1lambda, channels); /* i11 */
        }
      }
      if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        auto gin = grad_input_data + n * input_slice_size;
        apply_grad_input(acc_data_ptr, gin, input_slice_size);
      }

    }
  };

  auto loop3d = [&](int64_t begin, int64_t end) {
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
        acc_data_ptr = buffer_data.get();
        memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    } else {
      acc_data_ptr = reinterpret_cast<opmath_t*>(grad_input_data);
    }

    const opmath_t depth_scale = area_pixel_compute_scale<opmath_t>(
        input_depth, output_depth, align_corners, scales[0]);
    const opmath_t height_scale = area_pixel_compute_scale<opmath_t>(
        input_height, output_height, align_corners, scales[1]);
    const opmath_t width_scale = area_pixel_compute_scale<opmath_t>(
        input_width, output_width, align_corners, scales[2]);

    auto input_indexr = [=](int64_t n, int64_t d, int64_t h, int64_t w, int64_t offset) {
      return acc_data_ptr + offset + (d * input_height * input_width + h * input_width + w) * channels;
    };

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t id0, id1, ih0, ih1, iw0, iw1;
    opmath_t d0lambda, d1lambda, h0lambda, h1lambda, w0lambda, w1lambda;
    for (const auto n : c10::irange(begin, end)) {
      int64_t input_offset = buffer_data.get() == nullptr ? n * input_slice_size : 0;
      for (const auto od : c10::irange(output_depth)) {
        compute_source_index_and_lambda(
            id0, id1, d0lambda, d1lambda, depth_scale, od, input_depth, output_depth, align_corners);
        for (const auto oh : c10::irange(output_height)) {
          compute_source_index_and_lambda(
              ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
          for (const auto ow : c10::irange(output_width)) {
            compute_source_index_and_lambda(
                iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
            scalar_t* grad_output_ptr = grad_output_data + (n * output_depth * output_height * output_width +
                od *  output_height * output_width + oh * output_width + ow) * channels;
            linear_channels_last_acc(input_indexr(n, id0, ih0, iw0, input_offset), grad_output_ptr, d0lambda * h0lambda * w0lambda, channels); /* i000 */
            linear_channels_last_acc(input_indexr(n, id0, ih0, iw1, input_offset), grad_output_ptr, d0lambda * h0lambda * w1lambda, channels); /* i001 */
            linear_channels_last_acc(input_indexr(n, id0, ih1, iw0, input_offset), grad_output_ptr, d0lambda * h1lambda * w0lambda, channels); /* i010 */
            linear_channels_last_acc(input_indexr(n, id0, ih1, iw1, input_offset), grad_output_ptr, d0lambda * h1lambda * w1lambda, channels); /* i011 */
            linear_channels_last_acc(input_indexr(n, id1, ih0, iw0, input_offset), grad_output_ptr, d1lambda * h0lambda * w0lambda, channels); /* i100 */
            linear_channels_last_acc(input_indexr(n, id1, ih0, iw1, input_offset), grad_output_ptr, d1lambda * h0lambda * w1lambda, channels); /* i101 */
            linear_channels_last_acc(input_indexr(n, id1, ih1, iw0, input_offset), grad_output_ptr, d1lambda * h1lambda * w0lambda, channels); /* i110 */
            linear_channels_last_acc(input_indexr(n, id1, ih1, iw1, input_offset), grad_output_ptr, d1lambda * h1lambda * w1lambda, channels); /* i111 */
          }
        }
      }
      if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
        auto gin = grad_input_data + n * input_slice_size;
        apply_grad_input(acc_data_ptr, gin, input_slice_size);
      }
    }
  };

  if (ndim == 4) {
    // upsample bilinear 2d
    at::parallel_for(0, num_batches, 0, loop2d);
  } else {
    // upsample trilinear 3d
    TORCH_INTERNAL_ASSERT(ndim == 5);
    at::parallel_for(0, num_batches, 0, loop3d);
  }

  if (!grad_input_.is_contiguous(channels_last_memory_format)) {
    grad_input_.copy_(grad_input);
  }
}

void upsample_linear1d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grad_output.scalar_type(), "upsample_linear1d_backward", [&] {
    cpu_upsample_linear_backward<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_w});
  });
}

void upsample_bilinear2d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  if (grad_output.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grad_output.scalar_type(), "upsample_bilinear2d_backward_channels_last", [&] {
      cpu_upsample_linear_backward_channels_last<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_h, scales_w});
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grad_output.scalar_type(), "upsample_bilinear2d_backward", [&] {
      cpu_upsample_linear_backward<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_h, scales_w});
    });
  }
}

void upsample_trilinear3d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  if (grad_output.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grad_output.scalar_type(), "upsample_trilinear3d_backward_channels_last", [&] {
      cpu_upsample_linear_backward_channels_last<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_d, scales_h, scales_w});
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grad_output.scalar_type(), "upsample_trilinear3d_backward", [&] {
      cpu_upsample_linear_backward<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_d, scales_h, scales_w});
    });
  }
}

} // anonymous namespace

REGISTER_DISPATCH(upsample_nearest1d_backward_kernel, &upsample_nearest1d_backward_kernel_impl);
REGISTER_DISPATCH(_upsample_nearest_exact1d_backward_kernel, &_upsample_nearest_exact1d_backward_kernel_impl);
REGISTER_DISPATCH(upsample_nearest2d_backward_kernel, &upsample_nearest2d_backward_kernel_impl);
REGISTER_DISPATCH(_upsample_nearest_exact2d_backward_kernel, &_upsample_nearest_exact2d_backward_kernel_impl);
REGISTER_DISPATCH(upsample_nearest3d_backward_kernel, &upsample_nearest3d_backward_kernel_impl);
REGISTER_DISPATCH(_upsample_nearest_exact3d_backward_kernel, &_upsample_nearest_exact3d_backward_kernel_impl);

REGISTER_DISPATCH(upsample_linear1d_backward_kernel, &upsample_linear1d_backward_kernel_impl);
REGISTER_DISPATCH(upsample_bilinear2d_backward_kernel, &upsample_bilinear2d_backward_kernel_impl);
REGISTER_DISPATCH(upsample_trilinear3d_backward_kernel, &upsample_trilinear3d_backward_kernel_impl);

} // namespace at::native
