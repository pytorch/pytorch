#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>

namespace at {
namespace native {

namespace /* simd */ {

C10_ALWAYS_INLINE auto maximum(__m512 a, __m512 b) noexcept {
  const auto nan_mask = _mm512_cmp_ps_mask(a, a, _CMP_ORD_Q);
  return _mm512_mask_max_ps(a, nan_mask, a, b);
}

C10_ALWAYS_INLINE auto reduce_maximum(__m512 a) noexcept {
  const auto nan_mask = _mm512_cmp_ps_mask(a, a, _CMP_UNORD_Q);
  return _mm512_mask2int(nan_mask) != 0
      ? std::numeric_limits<float>::quiet_NaN()
      : _mm512_reduce_max_ps(a);
}

} // namespace

namespace /* kernel */ {

C10_ALWAYS_INLINE void _maximum_kernel_stride_1(
    float* C10_RESTRICT out,
    const float* C10_RESTRICT in,
    int64_t n) {
  constexpr int N = 16;

  const auto load = [](const float* p) noexcept { return _mm512_loadu_ps(p); };

  const auto store = [](float* p, __m512 v) noexcept {
    return _mm512_storeu_ps(p, v);
  };

  const auto compute = [&](int64_t offset) noexcept {
    store(out + offset, maximum(load(in + offset), load(out + offset)));
  };

  int64_t i = 0;

  // Vectorized loops
  for (; i < n - (n % (N << 2)); i += (N << 2)) {
    compute(i + 0 * N);
    compute(i + 1 * N);
    compute(i + 2 * N);
    compute(i + 3 * N);
  }

  for (; i < n - (n % (N << 1)); i += (N << 1)) {
    compute(i + 0 * N);
    compute(i + 1 * N);
  }

  for (; i < n - (n % (N << 0)); i += (N << 0)) {
    compute(i + 0 * N);
  }

  // Cleanup
  for (; i < n; ++i) {
    out[i] = at::native::max_impl(out[i], in[i]);
  }
}

C10_ALWAYS_INLINE void _maximum_kernel_stride_n(
    float* C10_RESTRICT out,
    int64_t out_stride,
    const float* C10_RESTRICT in,
    int64_t in_stride,
    int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    *out = at::native::max_impl(*out, *in);
    out += out_stride;
    in += in_stride;
  }
}

C10_ALWAYS_INLINE void _max_kernel_stride_1(
    float* C10_RESTRICT out,
    const float* C10_RESTRICT in,
    int64_t n) noexcept {
  constexpr int N = 16;

  const auto load = [](const float* p) noexcept { return _mm512_loadu_ps(p); };

  int64_t i = 0;

  // Vectorized loops
  if (n >= (N << 0)) {
    auto result_0 = load(in + i);
    i += N;

    // Unroll 2
    if (n >= (N << 1)) {
      auto result_1 = load(in + i);
      i += N;

      // Unroll 4
      if (n >= (N << 2)) {
        auto result_2 = load(in + i);
        i += N;
        auto result_3 = load(in + i);
        i += N;

        for (; i < n - (n % (N << 2)); i += (N << 2)) {
          result_0 = maximum(result_0, load(in + i + 0 * N));
          result_1 = maximum(result_1, load(in + i + 1 * N));
          result_2 = maximum(result_2, load(in + i + 2 * N));
          result_3 = maximum(result_3, load(in + i + 3 * N));
        }

        result_0 = maximum(result_0, result_2);
        result_1 = maximum(result_1, result_3);
      }

      for (; i < n - (n % (N << 1)); i += (N << 1)) {
        result_0 = maximum(result_0, load(in + i + 0 * N));
        result_1 = maximum(result_1, load(in + i + 1 * N));
      }

      result_0 = maximum(result_0, result_1);
    }

    for (; i < n - (n % (N << 0)); i += (N << 0)) {
      result_0 = maximum(result_0, load(in + i + 0 * N));
    }

    *out = at::native::max_impl(*out, reduce_maximum(result_0));
  }

  // Cleanup
  for (; i < n; ++i) {
    *out = at::native::max_impl(*out, in[i]);
  }
}

C10_ALWAYS_INLINE void _max_kernel_stride_2(
    float* C10_RESTRICT out,
    const float* C10_RESTRICT in,
    int64_t n) noexcept {
  constexpr int N = 16 << 1;
  n <<= 1;

  const auto load = [](const float* p) noexcept -> __m512 {
    const auto even = _mm512_loadu_ps(p);
    return _mm512_mask_loadu_ps(even, 0xAAAA, p + (N >> 1) - 1);
  };

  int64_t i = 0;

  // Vectorized loops
  if (n >= (N << 0)) {
    auto result_0 = load(in + i);
    i += N;

    // Unroll 2
    if (n >= (N << 1)) {
      auto result_1 = load(in + i);
      i += N;

      // Unroll 4
      if (n >= (N << 2)) {
        auto result_2 = load(in + i);
        i += N;
        auto result_3 = load(in + i);
        i += N;

        for (; i < n - (n % (N << 2)); i += (N << 2)) {
          result_0 = maximum(result_0, load(in + i + 0 * N));
          result_1 = maximum(result_1, load(in + i + 1 * N));
          result_2 = maximum(result_2, load(in + i + 2 * N));
          result_3 = maximum(result_3, load(in + i + 3 * N));
        }

        result_0 = maximum(result_0, result_2);
        result_1 = maximum(result_1, result_3);
      }

      for (; i < n - (n % (N << 1)); i += (N << 1)) {
        result_0 = maximum(result_0, load(in + i + 0 * N));
        result_1 = maximum(result_1, load(in + i + 1 * N));
      }

      result_0 = maximum(result_0, result_1);
    }

    for (; i < n - (n % (N << 0)); i += (N << 0)) {
      result_0 = maximum(result_0, load(in + i + 0 * N));
    }

    *out = at::native::max_impl(*out, reduce_maximum(result_0));
  }

  // Cleanup
  for (; i < n; i += 2) {
    *out = at::native::max_impl(*out, in[i]);
  }
}

C10_ALWAYS_INLINE void _max_kernel_stride_n(
    float* C10_RESTRICT out,
    const float* C10_RESTRICT in,
    int64_t n,
    int stride) noexcept {
  const int N = 16 * stride;
  n *= stride;

  const auto vindex = _mm512_set_epi32(
      stride * 15,
      stride * 14,
      stride * 13,
      stride * 12,
      stride * 11,
      stride * 10,
      stride * 9,
      stride * 8,
      stride * 7,
      stride * 6,
      stride * 5,
      stride * 4,
      stride * 3,
      stride * 2,
      stride * 1,
      stride * 0);

  const auto load = [&vindex](const float* p) noexcept -> __m512 {
    return _mm512_i32gather_ps(vindex, p, sizeof(float));
  };

  int64_t i = 0;

  // Vectorized loops
  if (n >= (N << 0)) {
    auto result_0 = load(in + i);
    i += N;

    // Unroll 2
    if (n >= (N << 1)) {
      auto result_1 = load(in + i);
      i += N;

      // Unroll 4
      if (n >= (N << 2)) {
        auto result_2 = load(in + i);
        i += N;
        auto result_3 = load(in + i);
        i += N;

        for (; i < n - (n % (N << 2)); i += (N << 2)) {
          result_0 = maximum(result_0, load(in + i + 0 * N));
          result_1 = maximum(result_1, load(in + i + 1 * N));
          result_2 = maximum(result_2, load(in + i + 2 * N));
          result_3 = maximum(result_3, load(in + i + 3 * N));
        }

        result_0 = maximum(result_0, result_2);
        result_1 = maximum(result_1, result_3);
      }

      for (; i < n - (n % (N << 1)); i += (N << 1)) {
        result_0 = maximum(result_0, load(in + i + 0 * N));
        result_1 = maximum(result_1, load(in + i + 1 * N));
      }

      result_0 = maximum(result_0, result_1);
    }

    for (; i < n - (n % (N << 0)); i += (N << 0)) {
      result_0 = maximum(result_0, load(in + i + 0 * N));
    }

    *out = at::native::max_impl(*out, reduce_maximum(result_0));
  }

  // Cleanup
  for (; i < n; i += stride) {
    *out = at::native::max_impl(*out, in[i]);
  }
}

} // namespace

namespace /* impl */ {

void _max_impl_parallel(const Tensor& in, Tensor& out) {
  constexpr auto grain_size = int64_t{32768 << 1};

  const auto n = in.numel();
  const auto num_chunks = std::max(n / grain_size, int64_t{1});
  const auto max_threads = int64_t{at::get_num_threads()};
  const auto num_threads = std::min(num_chunks, max_threads);
  const auto chunk_size = at::divup(n, num_threads);

  using scalar_t = float;

  constexpr scalar_t identity = std::numeric_limits<scalar_t>::has_infinity
      ? -std::numeric_limits<scalar_t>::infinity()
      : std::numeric_limits<scalar_t>::lowest();

  const auto in_ptr = in.data_ptr<scalar_t>();
  auto out_ptr = out.data_ptr<scalar_t>();

  auto partials = std::make_unique<scalar_t[]>(num_threads);
  std::fill_n(partials.get(), num_threads, identity);

#pragma omp parallel num_threads(num_threads)
  {
    const auto tid = at::get_thread_num();
    const auto offset = tid * chunk_size;
    const auto work = std::min(chunk_size, n - offset);
    const auto ptr = in_ptr + offset * in.stride(-1);
    if (in.is_contiguous()) {
      _max_kernel_stride_1(&partials[tid], ptr, work);
    } else if (in.stride(0) == 2) {
      _max_kernel_stride_2(&partials[tid], ptr, work);
    } else {
      const int stride = static_cast<int>(in.stride(0));
      _max_kernel_stride_n(&partials[tid], ptr, work, stride);
    }
  }

  *out_ptr = identity;
  _max_kernel_stride_1(out_ptr, partials.get(), num_threads);
}

void _max_impl_serial(const Tensor& in, Tensor& out) {
  constexpr auto identity = std::numeric_limits<float>::has_infinity
      ? -std::numeric_limits<float>::infinity()
      : std::numeric_limits<float>::lowest();

  out.fill_(identity);

  const float* in_ptr = in.data_ptr<float>();
  float* out_ptr = out.data_ptr<float>();

  const auto sizes = in.sizes();
  const auto in_strides = in.strides();
  const auto out_strides = out.broadcast_to(sizes).strides().vec();
  const auto nd = in.ndimension();
  std::vector<int64_t> indices(nd, 0);
  while (indices[0] != 1) {
    if (out_strides.back() == 0) {
      if (in_strides.back() == 1) {
        _max_kernel_stride_1(out_ptr, in_ptr, sizes.back());
      } else if (in_strides.back() == 2) {
        _max_kernel_stride_2(out_ptr, in_ptr, sizes.back());
      } else {
        _max_kernel_stride_n(out_ptr, in_ptr, sizes.back(), in_strides.back());
      }
    } else if (in_strides.back() == 1 && out_strides.back() == 1) {
      _maximum_kernel_stride_1(out_ptr, in_ptr, sizes.back());
    } else {
      _maximum_kernel_stride_n(
          out_ptr, out_strides.back(), in_ptr, in_strides.back(), sizes.back());
    }
    for (int64_t i = nd - 1; i >= 0; --i) {
      ++indices[i];
      if (i > 0) {
        if (indices[i] < sizes[i - 1]) {
          in_ptr += in_strides[i - 1];
          out_ptr += out_strides[i - 1];
          break;
        } else {
          in_ptr -= (sizes[i - 1] - 1) * in_strides[i - 1];
          out_ptr -= (sizes[i - 1] - 1) * out_strides[i - 1];
          indices[i] = 0;
        }
      }
    }
  }
}

} // namespace

} // namespace native
} // namespace at
