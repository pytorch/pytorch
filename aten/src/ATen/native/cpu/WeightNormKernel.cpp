#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/TensorBase.h>

#include <ATen/Dispatch.h>
#include <ATen/EmptyTensor.h>
#include <ATen/Parallel.h>
#include <ATen/OpMathType.h>
#include <ATen/native/cpu/WeightNormKernel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

namespace at::native {

namespace {

template <typename scalar_t, typename accscalar_t>
void weight_norm_first_dim_kernel(
    TensorBase& w,
    TensorBase& norm,
    const TensorBase& v,
    const TensorBase& g,
    int64_t M, int64_t N) {
  const auto v_data = v.data_ptr<scalar_t>();
  const auto g_data = g.data_ptr<scalar_t>();
  auto w_data = w.data_ptr<scalar_t>();
  auto norm_data = norm.data_ptr<accscalar_t>();

  using Vec = vec::Vectorized<accscalar_t>;
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      accscalar_t norm_val = vec::map_reduce_all<scalar_t>(
          [](Vec x) { return x * x; },
          [](Vec x, Vec y) { return x + y; },
          v_data + i * N,
          N);
      norm_val = std::sqrt(norm_val);
      norm_data[i] = norm_val;

      accscalar_t a = g_data[i] / norm_val;
      vec::map(
          [a](Vec x) { return x * Vec(a); },
          w_data + i * N,
          v_data + i * N,
          N);
    }
  });
}

template <typename scalar_t>
inline void sum_norm_per_row(
    scalar_t* out_ptr,
    const scalar_t* v_ptr,
    int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  vec::map2(
      [](Vec out, Vec v) { return out + v * v; },
      out_ptr,
      out_ptr,
      v_ptr,
      size);
}

inline void sum_norm_per_row(
    float* out_ptr,
    const BFloat16* v_ptr,
    int64_t size) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec v_bvec = bVec::loadu(v_ptr + d);
    auto [v_fvec0, v_fvec1] = convert_bfloat16_float(v_bvec);

    fVec out_fvec0 = fVec::loadu(out_ptr + d) + v_fvec0 * v_fvec0;
    fVec out_fvec1 = fVec::loadu(out_ptr + d + fVec::size()) + v_fvec1 * v_fvec1;
    out_fvec0.store(out_ptr + d);
    out_fvec1.store(out_ptr + d + fVec::size());
  }
  for(; d < size; ++d) {
    float v_val = float(v_ptr[d]);
    out_ptr[d] += v_val * v_val;
  }
}

template <typename scalar_t>
inline void apply_norm_per_row(
    scalar_t* w_ptr,
    const scalar_t* v_ptr,
    const scalar_t* a_ptr,
    int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  vec::map2(
      [](Vec v, Vec a) { return v * a; },
      w_ptr,
      v_ptr,
      a_ptr,
      size);
}

inline void apply_norm_per_row(
    BFloat16* w_ptr,
    const BFloat16* v_ptr,
    const float* a_ptr,
    int64_t size) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec v_bvec = bVec::loadu(v_ptr + d);
    auto [v_fvec0, v_fvec1] = convert_bfloat16_float(v_bvec);

    fVec w_fvec0 = fVec::loadu(a_ptr + d) * v_fvec0;
    fVec w_fvec1 = fVec::loadu(a_ptr + d + fVec::size()) * v_fvec1;
    bVec w_bvec = convert_float_bfloat16(w_fvec0, w_fvec1);
    w_bvec.store(w_ptr + d);
  }
  for(; d < size; ++d) {
    w_ptr[d] = float(v_ptr[d]) * a_ptr[d];
  }
}

template <typename scalar_t, typename accscalar_t>
void weight_norm_last_dim_kernel(
    TensorBase& w,
    TensorBase& norm,
    const TensorBase& v,
    const TensorBase& g,
    int64_t M, int64_t N) {
  const auto v_data = v.data_ptr<scalar_t>();
  const auto g_data = g.data_ptr<scalar_t>();
  auto w_data = w.data_ptr<scalar_t>();
  auto norm_data = norm.data_ptr<accscalar_t>();

  int num_threads = at::get_num_threads();
  TensorBase buffer = at::detail::empty_cpu({num_threads, N}, norm.options()).zero_();
  auto buffer_data = buffer.data_ptr<accscalar_t>();

  // vertical parallel reduction
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    auto buffer_ptr = buffer_data + tid * N;
    for (const auto i : c10::irange(begin, end)) {
      sum_norm_per_row(buffer_ptr, v_data + i * N, N);
    }
  });

  for (const auto j : c10::irange(N)) {
    accscalar_t sum = 0;
    for (const auto t : c10::irange(num_threads)) {
      sum += buffer_data[t * N + j];
    }
    norm_data[j] = std::sqrt(sum);
  }

  // reuse the first row of buffer to store g / norm
  vec::convert(g_data, buffer_data, N);
  using Vec = vec::Vectorized<accscalar_t>;
  vec::map2(
      [](Vec g, Vec norm) { return g / norm; },
      buffer_data,
      buffer_data,
      norm_data,
      N);

  // apply w = v * (g/norm)
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      apply_norm_per_row(w_data + i * N, v_data + i * N, buffer_data, N);
    }
  });
}

template <typename scalar_t, typename accscalar_t>
void weight_norm_backward_first_dim_kernel(
    TensorBase& grad_v,
    TensorBase& grad_g,
    const TensorBase& grad_w,
    const TensorBase& saved_v,
    const TensorBase& saved_g,
    const TensorBase& saved_norm,
    int64_t M, int64_t N) {
  const auto grad_w_data = grad_w.data_ptr<scalar_t>();
  const auto saved_v_data = saved_v.data_ptr<scalar_t>();
  const auto saved_g_data = saved_g.data_ptr<scalar_t>();
  const auto saved_norm_data = saved_norm.data_ptr<accscalar_t>();
  auto grad_v_data = grad_v.data_ptr<scalar_t>();
  auto grad_g_data = grad_g.data_ptr<scalar_t>();

  using Vec = vec::Vectorized<accscalar_t>;
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      accscalar_t per_dim_sum_val = vec::map2_reduce_all<scalar_t>(
          [](Vec grad_w, Vec saved_v) { return grad_w * saved_v; },
          [](Vec x, Vec y) { return x + y; },
          grad_w_data + i * N,
          saved_v_data + i * N,
          N);

      accscalar_t saved_norm_val = saved_norm_data[i];
      accscalar_t saved_g_val = accscalar_t(saved_g_data[i]);
      accscalar_t grad_g_val = per_dim_sum_val / saved_norm_val;

      // grad_g = sum / norm
      // grad_v = (g / norm) * (grad_w - v * (sum / norm^2))
      //  let a = g /norm
      //      b = a * grad_g / norm
      // grad_v = a * grad_w - b * v
      grad_g_data[i] = scalar_t(grad_g_val);
      accscalar_t a = saved_g_val / saved_norm_val;
      accscalar_t b = a * grad_g_val / saved_norm_val;

      vec::map2(
          [a, b](Vec grad_w, Vec v) { return Vec(a) * grad_w - Vec(b) * v; },
          grad_v_data + i * N,
          grad_w_data + i * N,
          saved_v_data + i * N,
          N);
    }
  });
}

template <typename scalar_t>
inline void sum_product_per_row(
    scalar_t* out_ptr,
    const scalar_t* grad_w_ptr,
    const scalar_t* v_ptr,
    int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  vec::map3(
      [](Vec out, Vec grad_w, Vec v) { return out + grad_w * v; },
      out_ptr,
      out_ptr,
      grad_w_ptr,
      v_ptr,
      size);
}

inline void sum_product_per_row(
    float* out_ptr,
    const BFloat16* grad_w_ptr,
    const BFloat16* v_ptr,
    int64_t size) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec grad_w_bvec = bVec::loadu(grad_w_ptr + d);
    auto [grad_w_fvec0, grad_w_fvec1] = convert_bfloat16_float(grad_w_bvec);
    bVec v_bvec = bVec::loadu(v_ptr + d);
    auto [v_fvec0, v_fvec1] = convert_bfloat16_float(v_bvec);

    fVec out_fvec0 = fVec::loadu(out_ptr + d) + grad_w_fvec0 * v_fvec0;
    fVec out_fvec1 = fVec::loadu(out_ptr + d + fVec::size()) + grad_w_fvec1 * v_fvec1;
    out_fvec0.store(out_ptr + d);
    out_fvec1.store(out_ptr + d + fVec::size());
  }
  for(; d < size; ++d) {
    float grad_w_val = float(grad_w_ptr[d]);
    float v_val = float(v_ptr[d]);
    out_ptr[d] += grad_w_val * v_val;
  }
}

template <typename scalar_t>
inline void apply_per_row_backward(
    scalar_t* grad_v_ptr,
    const scalar_t* grad_w_ptr,
    const scalar_t* v_ptr,
    const scalar_t* a_ptr,
    const scalar_t* b_ptr,
    int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  vec::map4(
      [](Vec grad_w, Vec v, Vec a, Vec b) { return a * grad_w - b * v; },
      grad_v_ptr,
      grad_w_ptr,
      v_ptr,
      a_ptr,
      b_ptr,
      size);
}

inline void apply_per_row_backward(
    BFloat16* grad_v_ptr,
    const BFloat16* grad_w_ptr,
    const BFloat16* v_ptr,
    const float* a_ptr,
    const float* b_ptr,
    int64_t size) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec grad_w_bvec = bVec::loadu(grad_w_ptr + d);
    auto [grad_w_fvec0, grad_w_fvec1] = convert_bfloat16_float(grad_w_bvec);
    bVec v_bvec = bVec::loadu(v_ptr + d);
    auto [v_fvec0, v_fvec1] = convert_bfloat16_float(v_bvec);

    fVec grad_v_fvec0 = fVec::loadu(a_ptr + d) * grad_w_fvec0 - fVec::loadu(b_ptr + d) * v_fvec0;
    fVec grad_v_fvec1 = fVec::loadu(a_ptr + d + fVec::size()) * grad_w_fvec1
        - fVec::loadu(b_ptr + d + fVec::size()) * v_fvec1;
    bVec grad_v_bvec = convert_float_bfloat16(grad_v_fvec0, grad_v_fvec1);
    grad_v_bvec.store(grad_v_ptr + d);
  }
  for(; d < size; ++d) {
    grad_v_ptr[d] = float(grad_w_ptr[d]) * a_ptr[d] - float(v_ptr[d]) * b_ptr[d];
  }
}

template <typename scalar_t, typename accscalar_t>
void weight_norm_backward_last_dim_kernel(
    TensorBase& grad_v,
    TensorBase& grad_g,
    const TensorBase& grad_w,
    const TensorBase& saved_v,
    const TensorBase& saved_g,
    const TensorBase& saved_norm,
    int64_t M, int64_t N) {
  const auto grad_w_data = grad_w.data_ptr<scalar_t>();
  const auto saved_v_data = saved_v.data_ptr<scalar_t>();
  const auto saved_g_data = saved_g.data_ptr<scalar_t>();
  const auto saved_norm_data = saved_norm.data_ptr<accscalar_t>();
  auto grad_v_data = grad_v.data_ptr<scalar_t>();
  auto grad_g_data = grad_g.data_ptr<scalar_t>();

  // the temp buffer will be used twice:
  // 1. vertical reduction from [M, N] to [T, N]
  // 2. store the intermediate data of `sum`, `a` and `b`,
  //    so need to make sure it has at least 3 rows
  //
  int num_threads = at::get_num_threads();
  int K = std::max(3, num_threads);
  TensorBase buffer = at::detail::empty_cpu({K, N}, saved_norm.options()).zero_();
  auto buffer_data = buffer.data_ptr<accscalar_t>();

  // vertical parallel reduction
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    auto buffer_ptr = buffer_data + tid * N;
    for (const auto i : c10::irange(begin, end)) {
      sum_product_per_row(buffer_ptr, grad_w_data + i * N, saved_v_data + i * N, N);
    }
  });

  // store result on the first row of buffer
  for (const auto j : c10::irange(N)) {
    accscalar_t sum = 0;
    for (const auto t : c10::irange(num_threads)) {
      sum += buffer_data[t * N + j];
    }
    buffer_data[j] = sum;
  }

  // reuse the 1st row of buffer to store the sum
  // 2nd row to store coefficient a
  // 3rd row to store coefficient b
  accscalar_t* per_dim_sum = buffer_data;
  accscalar_t* a = buffer_data + N;
  accscalar_t* b = buffer_data + 2 * N;

  // a = g /norm
  // b = a * grad_g / norm
  for (const auto j : c10::irange(N)) {
    accscalar_t saved_norm_val = saved_norm_data[j];
    accscalar_t saved_g_val = accscalar_t(saved_g_data[j]);
    accscalar_t grad_g_val = per_dim_sum[j] / saved_norm_val;
    grad_g_data[j] = scalar_t(grad_g_val);

    a[j] = saved_g_val / saved_norm_val;
    b[j] = a[j] * grad_g_val / saved_norm_val;
  }

  // apply grad_v = a * grad_w - b * v
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      apply_per_row_backward(
          grad_v_data + i * N,
          grad_w_data + i * N,
          saved_v_data + i * N,
          a,
          b,
          N);
    }
  });
}

void weight_norm_kernel(
    TensorBase& w,
    TensorBase& norm,
    const TensorBase& v,
    const TensorBase& g,
    int64_t dim) {
  TORCH_INTERNAL_ASSERT(dim == 0 || dim == v.dim() - 1,
      "fused kernels can only be applied for first or last dim");
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, v.scalar_type(),
      "weight_norm_kernel", [&]() {
    using accscalar_t = at::opmath_type<scalar_t>;
    if (dim == 0) {
      int64_t M = v.size(0);
      int64_t N = v.numel() / M;
      weight_norm_first_dim_kernel<scalar_t, accscalar_t>(w, norm, v, g, M, N);
    } else {
      int64_t N = v.size(-1);
      int64_t M = v.numel() / N;
      weight_norm_last_dim_kernel<scalar_t, accscalar_t>(w, norm, v, g, M, N);
    }
  });
}

void weight_norm_backward_kernel(
    TensorBase& grad_v,
    TensorBase& grad_g,
    const TensorBase& grad_w,
    const TensorBase& saved_v,
    const TensorBase& saved_g,
    const TensorBase& saved_norm,
    int64_t dim) {
  TORCH_INTERNAL_ASSERT(dim == 0 || dim == saved_v.dim() - 1,
      "fused kernels can only be applied for first or last dim");
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, saved_v.scalar_type(),
      "weight_norm_backward_kernel", [&]() {
    using accscalar_t = at::opmath_type<scalar_t>;
    if (dim == 0) {
      int64_t M = saved_v.size(0);
      int64_t N = saved_v.numel() / M;
      weight_norm_backward_first_dim_kernel<scalar_t, accscalar_t>(grad_v, grad_g, grad_w, saved_v, saved_g, saved_norm, M, N);
    } else {
      int64_t N = saved_v.size(-1);
      int64_t M = saved_v.numel() / N;
      weight_norm_backward_last_dim_kernel<scalar_t, accscalar_t>(grad_v, grad_g, grad_w, saved_v, saved_g, saved_norm, M, N);
    }
  });
}

} // anonymous namespace

REGISTER_DISPATCH(weight_norm_stub, &weight_norm_kernel)
REGISTER_DISPATCH(weight_norm_backward_stub, &weight_norm_backward_kernel)

} // at::native
