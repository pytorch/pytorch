#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/native/cpu/WeightNormKernel.h>

namespace at { namespace native {

namespace {

template <typename scalar_t>
void weight_norm_first_dim_impl(
    Tensor& w,
    Tensor& norms,
    const Tensor& v,
    const Tensor& g,
    int64_t M, int64_t N) {
  // v tensor shape: [M, N]
  // g tensor shape: [M, 1]
  scalar_t* v_data = v.data_ptr<scalar_t>();
  scalar_t* g_data = g.data_ptr<scalar_t>();
  scalar_t* w_data = w.data_ptr<scalar_t>();
  scalar_t* norms_data = norms.data_ptr<scalar_t>();

  using Vec = vec256::Vec256<scalar_t>;
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; m++) {
      // local pointer per row
      scalar_t* v_ptr = v_data + m * N;
      scalar_t* w_ptr = w_data + m * N;
      // calculate p2 norm per row
      scalar_t norm_val = vec256::map_reduce_all<scalar_t>(
          [](Vec x) { return x * x; },
          [](Vec x, Vec y) {return x + y; },
          v_ptr,
          N);
      norm_val = std::sqrt(norm_val);
      norms_data[m] = norm_val;
      // update w per row
      scalar_t a = scalar_t(1) / norm_val * g_data[m];
      vec256::map<scalar_t>(
          [a](Vec x) { return x * Vec(a); },
          w_ptr,
          v_ptr,
          N);
    }
  });
}

template <typename scalar_t>
void weight_norm_last_dim_impl(
    Tensor& w,
    Tensor& norms,
    const Tensor& v,
    const Tensor& g,
    int64_t M, int64_t N) {
  // v tensor shape: [M, N]
  // g tensor shape: [1, N]
  scalar_t* v_data = v.data_ptr<scalar_t>();
  scalar_t* g_data = g.data_ptr<scalar_t>();
  scalar_t* w_data = w.data_ptr<scalar_t>();
  scalar_t* norms_data = norms.data_ptr<scalar_t>();

  // temp buffer for v * v
  int num_threads = at::get_num_threads();
  Tensor buffer = at::empty({num_threads, N}, v.options()).zero_();
  scalar_t* buf_data = buffer.data_ptr<scalar_t>();

  // vertical vectorized reduce.
  using Vec = vec256::Vec256<scalar_t>;
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads,
        "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    scalar_t* buf_ptr = buf_data + tid * N;

    for (int64_t m = begin; m < end; m++) {
      // local pointer per row
      scalar_t* v_ptr = v_data + m * N;
      vec256::map2<scalar_t>(
          [](Vec v, Vec buf) { return buf + v * v; },
          buf_ptr,
          v_ptr,
          buf_ptr,
          N);
    }
  });

  // temp buffer for storing value of { 1.0 / norm * g } for vectorization purpose.
  Tensor a = at::empty({N}, g.options());
  scalar_t* a_data = a.data_ptr<scalar_t>();

  at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
    for (int64_t n = begin; n < end; n++) {
      scalar_t norm_val = scalar_t(0);
      for (int64_t t = 0; t < num_threads; t++) {
        norm_val += buf_data[t * N + n];
      }
      norm_val = std::sqrt(norm_val);
      norms_data[n] = norm_val;
      a_data[n] = scalar_t(1) / norm_val * g_data[n];
    }
  });

  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; m++) {
      scalar_t* v_ptr = v_data + m * N;
      scalar_t* w_ptr = w_data + m * N;
      vec256::map2<scalar_t>(
          [](Vec v, Vec a) { return v * a; },
          w_ptr,
          v_ptr,
          a_data,
          N);
    }
  });
}

template <typename scalar_t>
void weight_norm_backward_first_dim_impl(
    Tensor& grad_v,
    Tensor& grad_g,
    const Tensor& grad_w,
    const Tensor& v,
    const Tensor& g,
    const Tensor& norms,
    int64_t M, int64_t N) {
  // v tensor shape: [M, N]
  // g tensor shape: [M, 1]
  // norms tensor shape: [M, 1]
  // grad_w tensor shape: [M, N]
  scalar_t* grad_w_data = grad_w.data_ptr<scalar_t>();
  scalar_t* v_data = v.data_ptr<scalar_t>();
  scalar_t* g_data = g.data_ptr<scalar_t>();
  scalar_t* norms_data = norms.data_ptr<scalar_t>();
  scalar_t* grad_v_data = grad_v.data_ptr<scalar_t>();
  scalar_t* grad_g_data = grad_g.data_ptr<scalar_t>();

  using Vec = vec256::Vec256<scalar_t>;
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; m++) {
      // local pointer per row
      scalar_t* grad_w_ptr = grad_w_data + m * N;
      scalar_t* grad_v_ptr = grad_v_data + m * N;
      scalar_t* v_ptr = v_data + m * N;

      // accumulate the sum of grad_w * v per row
      scalar_t per_dim_sums = vec256::map2_reduce_all<scalar_t>(
          [](Vec x, Vec y) { return x * y; },
          [](Vec x, Vec y) { return x + y; },
          grad_w_ptr,
          v_ptr,
          N);

      // update grad_v per row:
      //
      //   grad_v = (g / norms) * (grad_w - v * per_dims_sums / (norms * norms))
      //
      // since we are doing vectorization on N, abstract grad_w and v:
      //   grad_v = a * grad_w + b * v
      //       a := g / norms
      //       b := -per_dim_sums * g / (norms^3)
      //
      scalar_t g_val = g_data[m];
      scalar_t norms_val = norms_data[m];
      scalar_t norms3_val = norms_val * norms_val * norms_val;
      scalar_t a = g_val / norms_val;
      scalar_t b = -per_dim_sums * g_val / norms3_val;

      vec256::map2<scalar_t>(
          [a, b](Vec grad_w, Vec v) { return Vec(a) * grad_w + Vec(b) * v; },
          grad_v_ptr,
          grad_w_ptr,
          v_ptr,
          N);

      // update grad_g
      grad_g_data[m] = per_dim_sums / norms_val;
    }
  });
}

template <typename scalar_t>
void weight_norm_backward_last_dim_impl(
    Tensor& grad_v,
    Tensor& grad_g,
    const Tensor& grad_w,
    const Tensor& v,
    const Tensor& g,
    const Tensor& norms,
    int64_t M, int64_t N) {
  // v tensor shape: [M, N]
  // g tensor shape: [1, N]
  // norms tensor shape: [1, N]
  // grad_w tensor shape: [M, N]
  scalar_t* grad_w_data = grad_w.data_ptr<scalar_t>();
  scalar_t* v_data = v.data_ptr<scalar_t>();
  scalar_t* g_data = g.data_ptr<scalar_t>();
  scalar_t* norms_data = norms.data_ptr<scalar_t>();
  scalar_t* grad_v_data = grad_v.data_ptr<scalar_t>();
  scalar_t* grad_g_data = grad_g.data_ptr<scalar_t>();

  // temp buffer for grad_w * v
  int num_threads = at::get_num_threads();
  Tensor buffer = at::empty({num_threads, N}, v.options()).zero_();
  scalar_t* buf_data = buffer.data_ptr<scalar_t>();

  using Vec = vec256::Vec256<scalar_t>;
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads,
        "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    scalar_t* buf_ptr = buf_data + tid * N;

    for (int64_t m = begin; m < end; m++) {
      scalar_t* grad_w_ptr = grad_w_data + m * N;
      scalar_t* v_ptr = v_data + m * N;
      vec256::map3<scalar_t>(
          [](Vec buf, Vec grad_w, Vec v) { return buf + grad_w * v; },
          buf_ptr,
          buf_ptr,
          grad_w_ptr,
          v_ptr,
          N);
    }
  });

  // temp buffer for holding per_dim_sums
  Tensor per_dim_sums = at::empty({N}, g.options());
  scalar_t* per_dim_sums_data = per_dim_sums.data_ptr<scalar_t>();

  // temp buffer for holding a and b, for vectorization purpose
  Tensor buffer1 = at::empty({2 * N}, g.options());
  scalar_t* a_data = buffer1.data_ptr<scalar_t>();
  scalar_t* b_data = a_data + N;

  at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
    for (int64_t n = begin; n < end; n++) {
      scalar_t per_dim_sums_val = scalar_t(0);
      for (int64_t t = 0; t < num_threads; t++) {
        per_dim_sums_val += buf_data[t * N + n];
      }
      per_dim_sums_data[n] = per_dim_sums_val;

      // fusing parallel session is marginally faster.
      // update parameters
      //   a := g / norms
      //   b := -per_dim_sums * g / (norms^3)
      scalar_t g_val = g_data[n];
      scalar_t norms_val = norms_data[n];
      scalar_t norms3_val = norms_val * norms_val * norms_val;
      a_data[n] = g_val / norms_val;
      b_data[n] = -per_dim_sums_val * g_val / norms3_val;

      grad_g_data[n] = per_dim_sums_val / norms_val;
    }
  });
 
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; m++) {
      scalar_t* grad_v_ptr = grad_v_data + m * N;
      scalar_t* grad_w_ptr = grad_w_data + m * N;
      scalar_t* v_ptr = v_data + m * N;

      // grad_v = a * grad_w
      vec256::map2<scalar_t>(
          [](Vec a, Vec grad_w) { return a * grad_w; },
          grad_v_ptr,
          a_data,
          grad_w_ptr,
          N);

      // grad_v = grad_v + b * v
      vec256::map3<scalar_t>(
          [](Vec grad_v, Vec b, Vec v) { return grad_v + b * v; },
          grad_v_ptr,
          grad_v_ptr,
          b_data,
          v_ptr,
          N);
    }
  });
}

void weight_norm_kernel(
    Tensor& w,
    Tensor& norms,
    const Tensor& v,
    const Tensor& g,
    int64_t dim) {
  TORCH_CHECK(dim == 0 || dim == v.dim() - 1,
      "weight_norm_kernel: fused kernels can only be applied for first or last dim");
  TORCH_CHECK(w.is_contiguous(),
      "weight_norm_kernel: w needs to be contiguous");
  TORCH_CHECK(norms.is_contiguous(),
      "weight_norm_kernel: norms needs to be contiguous");
  TORCH_CHECK(v.is_contiguous(),
      "weight_norm_kernel: v needs to be contiguous");
  TORCH_CHECK(g.is_contiguous(),
      "weight_norm_kernel: g needs to be contiguous");

  AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "weight_norm_kernel", [&] {
    // view v in 2d: M is row and N is column
    if (dim == 0) {
      int64_t M = v.size(0);
      int64_t N = v.numel() / M;
      weight_norm_first_dim_impl<scalar_t>(w, norms, v, g, M, N);
    } else {
      int64_t N = v.size(-1);
      int64_t M = v.numel() / N;
      weight_norm_last_dim_impl<scalar_t>(w, norms, v, g, M, N);
    }
  });
}

void weight_norm_backward_kernel(
    Tensor& grad_v,
    Tensor& grad_g,
    const Tensor& grad_w,
    const Tensor& v,
    const Tensor& g,
    const Tensor& norms,
    int64_t dim) {
  TORCH_CHECK(dim == 0 || dim == v.dim() - 1,
      "weight_norm_backward_kernel: fused kernels can only be applied for first or last dim");
  TORCH_CHECK(v.is_contiguous(),
      "weight_norm_backward_kernel: v needs to be contiguous");
  TORCH_CHECK(g.is_contiguous(),
      "weight_norm_backward_kernel: g needs to be contiguous");
  TORCH_CHECK(norms.is_contiguous(),
      "weight_norm_backward_kernel: norms needs to be contiguous");

  AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "weight_norm_backward_kernel", [&] {
    // view v in 2d: M is row and N is column
    if (dim == 0) {
      int64_t M = v.size(0);
      int64_t N = v.numel() / M;
      weight_norm_backward_first_dim_impl<scalar_t>(grad_v, grad_g, grad_w, v, g, norms, M, N);
    } else {
      int64_t N = v.size(-1);
      int64_t M = v.numel() / N;
       weight_norm_backward_last_dim_impl<scalar_t>(grad_v, grad_g, grad_w, v, g, norms, M, N);
    }
  });
}

} // anonymous namespace

REGISTER_DISPATCH(weight_norm_stub, &weight_norm_kernel);
REGISTER_DISPATCH(weight_norm_backward_stub, &weight_norm_backward_kernel);

}} // at::native
