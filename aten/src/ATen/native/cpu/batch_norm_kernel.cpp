#include <ATen/native/batch_norm.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

namespace at { namespace native {
namespace {

using namespace vec;

// helper functions:
template <typename scalar_t>
static inline scalar_t vec_acc(const Vectorized<scalar_t>& vec) {
  constexpr int N = Vectorized<scalar_t>::size();
  scalar_t vec_arr[N];
  vec.store(vec_arr);
  scalar_t acc = scalar_t(0);
  for (int i = 0; i < N; i++) {
    acc += vec_arr[i];
  }
  return acc;
}

static inline std::tuple<Vectorized<float>, Vectorized<float>> load2f(const BFloat16* mean) {
  using bVec = Vectorized<BFloat16>;
  bVec mean_bvec = bVec::loadu(mean);
  return convert_bfloat16_float(mean_bvec);
}

static inline std::tuple<Vectorized<float>, Vectorized<float>> load2f(const float* mean) {
  using fVec = Vectorized<float>;
  fVec mean_fvec0 = fVec::loadu(mean);
  fVec mean_fvec1 = fVec::loadu(mean + fVec::size());
  return std::make_tuple(mean_fvec0, mean_fvec1);
}

template <typename scalar_t>
struct BatchNormImpl {
  using Vec = Vectorized<scalar_t>;

  // kernel0: 'contiguous' memory format when image size != 1
  static inline void kernel0(
      scalar_t* out, const scalar_t* in,
      scalar_t alpha, scalar_t beta,
      int64_t size) {

    const Vec alpha_vec(alpha);
    const Vec beta_vec(beta);
    int64_t d = 0;
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
      Vec data_vec = Vec::loadu(in + d);
      Vec out_vec = data_vec * alpha_vec + beta_vec;
      out_vec.store(out + d);
    }
    if (size - d > 0) {
      Vec data_vec = Vec::loadu(in + d, size - d);
      Vec out_vec = data_vec * alpha_vec + beta_vec;
      out_vec.store(out + d, size - d);
    }
  }

  // kernel1: 'channels last' memory format or
  //   'contiguous' memory format when image_size == 1, aka 'NC11'
  static inline void kernel1(
      scalar_t* out, const scalar_t* in,
      const scalar_t* alpha, const scalar_t* beta,
      int64_t size) {

    int64_t d = 0;
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
      Vec alpha_vec = Vec::loadu(alpha + d);
      Vec beta_vec = Vec::loadu(beta + d);
      Vec data_vec = Vec::loadu(in + d);
      Vec out_vec = data_vec * alpha_vec + beta_vec;
      out_vec.store(out + d);
    }
    if (size - d > 0) {
      Vec alpha_vec = Vec::loadu(alpha + d, size - d);
      Vec beta_vec = Vec::loadu(beta + d, size - d);
      Vec data_vec = Vec::loadu(in + d, size - d);
      Vec out_vec = data_vec * alpha_vec + beta_vec;
      out_vec.store(out + d, size - d);
    }
  }
};

template <>
struct BatchNormImpl<BFloat16> {
  using bVec = Vectorized<BFloat16>;
  using fVec = Vectorized<float>;

  static inline void kernel0(
      BFloat16* out, const BFloat16* in,
      float alpha, float beta,
      int64_t size) {

    const fVec alpha_fvec(alpha);
    const fVec beta_fvec(beta);
    int64_t d = 0;
    for (; d < size - (size % bVec::size()); d += bVec::size()) {
      bVec data_bvec = bVec::loadu(in + d);
      fVec data_fvec0, data_fvec1;
      std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);

      fVec out_fvec0 = data_fvec0 * alpha_fvec + beta_fvec;
      fVec out_fvec1 = data_fvec1 * alpha_fvec + beta_fvec;
      bVec out_bvec = convert_float_bfloat16(out_fvec0, out_fvec1);
      out_bvec.store(out + d);
    }
    for (; d < size; d++) {
      out[d] = BFloat16(float(in[d]) * alpha + beta);
    }
  }

  static inline void kernel1(
      BFloat16* out, const BFloat16* in,
      const float* alpha, const float* beta,
      int64_t size) {

    int64_t d = 0;
    for (; d < size - (size % bVec::size()); d += bVec::size()) {
      fVec alpha_fvec0 = fVec::loadu(alpha + d);
      fVec alpha_fvec1 = fVec::loadu(alpha + d + fVec::size());
      fVec beta_fvec0 = fVec::loadu(beta + d);
      fVec beta_fvec1 = fVec::loadu(beta + d + fVec::size());
      bVec data_bvec = bVec::loadu(in + d);
      fVec data_fvec0, data_fvec1;
      std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);

      fVec out_fvec0 = data_fvec0 * alpha_fvec0 + beta_fvec0;
      fVec out_fvec1 = data_fvec1 * alpha_fvec1 + beta_fvec1;
      bVec out_bvec = convert_float_bfloat16(out_fvec0, out_fvec1);
      out_bvec.store(out + d);
    }
    for (; d < size; d++) {
      out[d] = BFloat16(float(in[d]) * alpha[d] + beta[d]);
    }
  }
};

template <typename scalar_t, typename param_t>
struct BatchNormCollectStatsImpl {
  using accscalar_t = at::acc_type<scalar_t, false>;

  // kernel0: 'contiguous' memory format, compute mean
  static inline void kernel0(
      param_t* mean, const scalar_t* in,
      int64_t n_batch, int64_t image_size,
      int64_t n_channel, int64_t N, int64_t c) {

    accscalar_t sum = 0;
    for (int64_t n = 0; n < n_batch; n++) {
      for (int64_t i = 0; i < image_size; i++) {
        int64_t offset = n * n_channel * image_size + c * image_size + i;
        sum += in[offset];
      }
    }
    accscalar_t mean_val = sum / N;
    mean[c] = param_t(mean_val);
  }

  // kernekl1: 'contiguous' memory format, compute variance
  static inline void kernel1(
      param_t* var_sum, const scalar_t* in, param_t mean,
      int64_t n_batch, int64_t image_size,
      int64_t n_channel, int64_t c) {

    accscalar_t _var_sum = 0;
    for (int64_t n = 0; n < n_batch; n++) {
      for (int64_t i = 0; i < image_size; i++) {
        int64_t offset = n * n_channel * image_size + c * image_size + i;
        scalar_t x = in[offset];
        _var_sum += (x - mean) * (x - mean);
      }
    }
    var_sum[c] = param_t(_var_sum);
  }

  // kernel2: 'channels last' memory format, compute sum per channel
  using Vec = Vectorized<scalar_t>;
  static inline void kernel2(
      scalar_t* buffer, const scalar_t* in,
      int64_t size) {

    vec::map2<scalar_t>(
        [](Vec x, Vec y) { return x + y; },
        buffer, in, buffer, size);
  }

  // kernel3: 'channels last' memory format, compute var per channel
  static inline void kernel3(
      scalar_t* buffer, const scalar_t* in,
      param_t* mean, int64_t size) {

    vec::map3<scalar_t>(
        [](Vec x, Vec y, Vec mean) { return y + (x - mean) * (x - mean); },
        buffer, in, buffer, mean, size);
  }
};

template <typename param_t>
struct BatchNormCollectStatsImpl<BFloat16, param_t> {
  using bVec = Vectorized<BFloat16>;
  using fVec = Vectorized<float>;

  static inline void kernel0(
      param_t* mean, const BFloat16* in,
      int64_t n_batch, int64_t image_size,
      int64_t n_channel, int64_t N, int64_t c) {

    float sum_val = float(0);
    fVec sum_fvec = fVec(float(0));
    for (int64_t n = 0; n < n_batch; n++) {
      int64_t offset = n * n_channel * image_size + c * image_size;
      int64_t d = 0;
      for (; d < image_size - (image_size % bVec::size()); d += bVec::size()) {
        bVec data_bvec = bVec::loadu(in + offset + d);
        fVec data_fvec0, data_fvec1;
        std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
        sum_fvec += data_fvec0;
        sum_fvec += data_fvec1;
      }
      for (; d < image_size; d++) {
        sum_val += in[offset + d];
      }
    }
    sum_val += vec_acc(sum_fvec);
    float mean_val = sum_val / N;
    mean[c] = param_t(mean_val);
  }

  static inline void kernel1(
      param_t* var_sum, const BFloat16* in, param_t mean,
      int64_t n_batch, int64_t image_size,
      int64_t n_channel, int64_t c) {

    float mean_val = float(mean);
    fVec mean_fvec = fVec(mean_val);
    float var_val = float(0);
    fVec var_fvec = fVec(float(0));
    for (int64_t n = 0; n < n_batch; n++) {
      int64_t offset = n * n_channel * image_size + c * image_size;
      int64_t d = 0;
      for (; d < image_size - (image_size % bVec::size()); d += bVec::size()) {
        bVec data_bvec = bVec::loadu(in + offset + d);
        fVec data_fvec0, data_fvec1;
        std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
        var_fvec += (data_fvec0 - mean_fvec) * (data_fvec0 - mean_fvec);
        var_fvec += (data_fvec1 - mean_fvec) * (data_fvec1 - mean_fvec);
      }
      for (; d < image_size; d++) {
        float data_val = in[offset + d];
        var_val += (data_val - mean_val) * (data_val - mean_val);
      }
    }
    var_val += vec_acc(var_fvec);
    var_sum[c] = param_t(var_val);
  }

  static inline void kernel2(
      float* buffer, const BFloat16* in,
      int64_t size) {

    int64_t d = 0;
    for (; d < size - (size % bVec::size()); d += bVec::size()) {
      bVec data_bvec = bVec::loadu(in + d);
      fVec data_fvec0, data_fvec1;
      std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
      fVec sum_fvec0 = fVec::loadu(buffer + d) + data_fvec0;
      fVec sum_fvec1 = fVec::loadu(buffer + d + fVec::size()) + data_fvec1;
      sum_fvec0.store(buffer + d);
      sum_fvec1.store(buffer + d + fVec::size());
    }
    for (; d < size; d++) {
      buffer[d] += in[d];
    }
  }

  static inline void kernel3(
      float* buffer, const BFloat16* in,
      param_t* mean, int64_t size) {

    int64_t d = 0;
    for (; d < size - (size % bVec::size()); d += bVec::size()) {
      bVec data_bvec = bVec::loadu(in + d);
      fVec data_fvec0, data_fvec1;
      std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
      fVec mean_fvec0, mean_fvec1;
      std::tie(mean_fvec0, mean_fvec1) = load2f(mean + d);
      fVec var_fvec0 = fVec::loadu(buffer + d);
      fVec var_fvec1 = fVec::loadu(buffer + d + fVec::size());
      var_fvec0 += (data_fvec0 - mean_fvec0) * (data_fvec0 - mean_fvec0);
      var_fvec1 += (data_fvec1 - mean_fvec1) * (data_fvec1 - mean_fvec1);
      var_fvec0.store(buffer + d);
      var_fvec1.store(buffer + d + fVec::size());
    }
    for (; d < size; d++) {
      float data_val = float(in[d]);
      float mean_val = float(mean[d]);
      buffer[d] += (data_val - mean_val) * (data_val - mean_val);
    }
  }
};

template <typename scalar_t, typename param_t>
struct BatchNormBackwardImpl {
  using param2_t = param_acc_t<scalar_t>;
  using Vec = Vectorized<scalar_t>;
  using accscalar_t = at::acc_type<scalar_t, false>;

  // kernel0: 'contiguous' memory format, compute sum and dotp
  static inline std::tuple<accscalar_t, accscalar_t> kernel0(
      const scalar_t* input, const scalar_t* grad_output, param_t mean,
      int64_t n_batch, int64_t n_channel, int64_t image_size, int64_t c) {

    accscalar_t sum = 0;
    accscalar_t dotp = 0;
    for (int64_t n = 0; n < n_batch; n++) {
      const scalar_t* x = input + n * n_channel * image_size + c * image_size;
      const scalar_t* dy = grad_output + n * n_channel * image_size + c * image_size;

      sum += vec::reduce_all<scalar_t>(
          [](Vec& x, Vec& y) { return x + y; },
          dy, image_size);

      dotp += vec::map2_reduce_all<scalar_t>(
          [mean](Vec x, Vec dy) { return (x - Vec(mean)) * dy; },
          [](Vec x, Vec y) { return x + y; },
          x, dy, image_size);
      }
    return std::make_tuple(sum, dotp);
  }

  // kernel1: 'channels last' memory format, compute sum and dotp per channel
  static inline void kernel1(
      param2_t* sum, param2_t* dotp,
      const scalar_t* x, const scalar_t* dy, const param_t* mean,
      int64_t size) {

    vec::map2<scalar_t>(
        [](Vec sum, Vec dy) { return sum + dy; },
        sum, sum, dy, size);

    vec::map4<scalar_t>(
        [](Vec dotp, Vec x, Vec mean, Vec dy) { return dotp + (x - mean) * dy; },
        dotp, dotp, x, mean, dy, size);
  }

  // kernel2: 'channels last' memory format, training mode, compute dx
  static inline void kernel2(
      scalar_t* dx, const scalar_t* x, const scalar_t* dy,
      const param2_t* sum, const param2_t* dotp,
      const param_t* weight, const param_t* mean, const param_t* invstd,
      int64_t N, int64_t size) {

    int64_t d = 0;
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
      Vec x_vec = Vec::loadu(x + d);
      Vec mean_vec = Vec::loadu(mean + d);
      Vec dotp_vec = Vec::loadu(dotp + d);
      Vec invstd_vec = Vec::loadu(invstd + d);
      Vec k_vec = dotp_vec * invstd_vec * invstd_vec / Vec(N);
      Vec dx_vec = (x_vec - mean_vec) * k_vec;
      Vec dy_vec = Vec::loadu(dy + d);
      Vec grad_mean_vec = Vec::loadu(sum + d) / Vec(N);
      Vec w_vec = Vec::loadu(weight + d);
      dx_vec = (dy_vec - grad_mean_vec - dx_vec) * invstd_vec * w_vec;
      dx_vec.store(dx + d);
    }
    if (size - d > 0) {
      Vec x_vec = Vec::loadu(x + d, size - d);
      Vec mean_vec = Vec::loadu(mean + d, size - d);
      Vec dotp_vec = Vec::loadu(dotp + d, size - d);
      Vec invstd_vec = Vec::loadu(invstd + d, size - d);
      Vec k_vec = dotp_vec * invstd_vec * invstd_vec / Vec(N);
      Vec dx_vec = (x_vec - mean_vec) * k_vec;
      Vec dy_vec = Vec::loadu(dy + d, size - d);
      Vec grad_mean_vec = Vec::loadu(sum + d, size - d) / Vec(N);
      Vec w_vec = Vec::loadu(weight + d, size - d);
      dx_vec = (dy_vec - grad_mean_vec - dx_vec) * invstd_vec * w_vec;
      dx_vec.store(dx + d, size - d);
    }
  }

  // kernel3: 'channels last' memory format, evaluation mode, compute dx
  static inline void kernel3(
      scalar_t* dx, const scalar_t* dy,
      const param_t* weight, const param_t* invstd,
      int64_t size) {

    int64_t d = 0;
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
      Vec dy_vec = Vec::loadu(dy + d);
      Vec invstd_vec = Vec::loadu(invstd + d);
      Vec w_vec = Vec::loadu(weight + d);
      Vec dx_vec = dy_vec * invstd_vec * w_vec;
      dx_vec.store(dx + d);
    }
    if (size - d > 0) {
      Vec dy_vec = Vec::loadu(dy + d, size - d);
      Vec invstd_vec = Vec::loadu(invstd + d, size - d);
      Vec w_vec = Vec::loadu(weight + d, size - d);
      Vec dx_vec = dy_vec * invstd_vec * w_vec;
      dx_vec.store(dx + d, size - d);
    }
  }
};

template <typename param_t>
struct BatchNormBackwardImpl<BFloat16, param_t> {
  using bVec = Vectorized<BFloat16>;
  using fVec = Vectorized<float>;

  static inline std::tuple<float, float> kernel0(
      const BFloat16* input, const BFloat16* grad_output, param_t mean,
      int64_t n_batch, int64_t n_channel, int64_t image_size, int64_t c) {

    float sum_val{0}, dotp_val{0};
    fVec sum_fvec{0}, dotp_fvec{0};
    for (int64_t n = 0; n < n_batch; n++) {
      auto offset = n * n_channel * image_size + c * image_size;
      int64_t d = 0;
      for (; d < image_size - (image_size % bVec::size()); d += bVec::size()) {
        bVec dy_bvec = bVec::loadu(grad_output + offset + d);
        fVec dy_fvec0, dy_fvec1;
        std::tie(dy_fvec0, dy_fvec1) = convert_bfloat16_float(dy_bvec);
        sum_fvec += dy_fvec0;
        sum_fvec += dy_fvec1;

        bVec x_bvec = bVec::loadu(input + offset + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = convert_bfloat16_float(x_bvec);
        dotp_fvec += (x_fvec0 - fVec(mean)) * dy_fvec0;
        dotp_fvec += (x_fvec1 - fVec(mean)) * dy_fvec1;
      }
      for (; d < image_size; d++) {
        float dy = grad_output[offset + d];
        float x = input[offset + d];
        sum_val += dy;
        dotp_val += (x - mean) * dy;
      }
    }
    sum_val += vec_acc(sum_fvec);
    dotp_val += vec_acc(dotp_fvec);
    return std::make_tuple(sum_val, dotp_val);
  }

  static inline void kernel1(
      float* sum, float* dotp,
      const BFloat16* x, const BFloat16* dy, const param_t* mean,
      int64_t size) {

    int64_t d = 0;
    for(; d < size - (size % bVec::size()); d += bVec::size()) {
      bVec dy_bvec = bVec::loadu(dy + d);
      fVec dy_fvec0, dy_fvec1;
      std::tie(dy_fvec0, dy_fvec1) = convert_bfloat16_float(dy_bvec);
      fVec sum_fvec0 = dy_fvec0 + fVec::loadu(sum + d);
      fVec sum_fvec1 = dy_fvec1 + fVec::loadu(sum + d + fVec::size());
      sum_fvec0.store(sum + d);
      sum_fvec1.store(sum + d + fVec::size());

      bVec x_bvec = bVec::loadu(x + d);
      fVec x_fvec0, x_fvec1;
      std::tie(x_fvec0, x_fvec1) = convert_bfloat16_float(x_bvec);
      fVec mean_fvec0, mean_fvec1;
      std::tie(mean_fvec0, mean_fvec1) = load2f(mean + d);
      fVec dotp_fvec0 = fVec::loadu(dotp + d);
      fVec dotp_fvec1 = fVec::loadu(dotp + d + fVec::size());
      dotp_fvec0 += (x_fvec0 - mean_fvec0) * dy_fvec0;
      dotp_fvec1 += (x_fvec1 - mean_fvec1) * dy_fvec1;
      dotp_fvec0.store(dotp + d);
      dotp_fvec1.store(dotp + d + fVec::size());
    }
    for (; d < size; d++) {
      float dy_val = dy[d];
      float x_val = x[d];
      float mean_val = mean[d];
      sum[d] += dy_val;
      dotp[d] += (x_val - mean_val) * dy_val;
    }
  }

  static inline void kernel2(
      BFloat16* dx, const BFloat16* x, const BFloat16* dy,
      const float* sum, const float* dotp,
      const param_t* weight, const param_t* mean, const param_t* invstd,
      int64_t N, int64_t size) {

    int64_t d = 0;
    for (; d < size - (size % bVec::size()); d += bVec::size()) {
      bVec x_bvec = bVec::loadu(x + d);
      fVec x_fvec0, x_fvec1;
      std::tie(x_fvec0, x_fvec1) = convert_bfloat16_float(x_bvec);
      fVec mean_fvec0, mean_fvec1;
      std::tie(mean_fvec0, mean_fvec1) = load2f(mean + d);
      fVec dotp_fvec0 = fVec::loadu(dotp + d);
      fVec dotp_fvec1 = fVec::loadu(dotp + d + fVec::size());
      fVec invstd_fvec0, invstd_fvec1;
      std::tie(invstd_fvec0, invstd_fvec1) = load2f(invstd + d);
      fVec k_fvec0 = dotp_fvec0 * invstd_fvec0 * invstd_fvec0 / fVec(N);
      fVec k_fvec1 = dotp_fvec1 * invstd_fvec1 * invstd_fvec1 / fVec(N);
      fVec dx_fvec0 = (x_fvec0 - mean_fvec0) * k_fvec0;
      fVec dx_fvec1 = (x_fvec1 - mean_fvec1) * k_fvec1;
      bVec dy_bvec = bVec::loadu(dy + d);
      fVec dy_fvec0, dy_fvec1;
      std::tie(dy_fvec0, dy_fvec1) = convert_bfloat16_float(dy_bvec);
      fVec grad_mean_fvec0 = fVec::loadu(sum + d) / fVec(N);
      fVec grad_mean_fvec1 = fVec::loadu(sum + d + fVec::size()) / fVec(N);
      fVec w_fvec0, w_fvec1;
      std::tie(w_fvec0, w_fvec1) = load2f(weight + d);
      dx_fvec0 = (dy_fvec0 - grad_mean_fvec0 - dx_fvec0) * invstd_fvec0 * w_fvec0;
      dx_fvec1 = (dy_fvec1 - grad_mean_fvec1 - dx_fvec1) * invstd_fvec1 * w_fvec1;
      bVec dx_bvec = convert_float_bfloat16(dx_fvec0, dx_fvec1);
      dx_bvec.store(dx + d);
    }
    for (; d < size; d++) {
      float x_val = x[d];
      float mean_val = mean[d];
      float dotp_val = dotp[d];
      float invstd_val = invstd[d];
      float k_val = dotp_val * invstd_val * invstd_val / N;
      float dx_val = (x_val - mean_val) * k_val;
      float dy_val = dy[d];
      float grad_mean_val = sum[d] / N;
      float w_val = weight[d];
      dx_val = (dy_val - grad_mean_val - dx_val) * invstd_val * w_val;
      dx[d] = BFloat16(dx_val);
    }
  }

  static inline void kernel3(
      BFloat16* dx, const BFloat16* dy,
      const param_t* weight, const param_t* invstd,
      int64_t size) {

    int64_t d = 0;
    for (; d < size - (size % bVec::size()); d += bVec::size()) {
      bVec dy_bvec = bVec::loadu(dy + d);
      fVec dy_fvec0, dy_fvec1;
      std::tie(dy_fvec0, dy_fvec1) = convert_bfloat16_float(dy_bvec);
      fVec invstd_fvec0, invstd_fvec1;
      std::tie(invstd_fvec0, invstd_fvec1) = load2f(invstd + d);
      fVec w_fvec0, w_fvec1;
      std::tie(w_fvec0, w_fvec1) = load2f(weight + d);
      fVec dx_fvec0 = dy_fvec0 * invstd_fvec0 * w_fvec0;
      fVec dx_fvec1 = dy_fvec1 * invstd_fvec1 * w_fvec1;
      bVec dx_bvec = convert_float_bfloat16(dx_fvec0, dx_fvec1);
      dx_bvec.store(dx + d);
    }
    for (; d < size; d++) {
      float dy_val = dy[d];
      float invstd_val = invstd[d];
      float w_val = weight[d];
      float dx_val = dy_val * invstd_val * w_val;
      dx[d] = BFloat16(dx_val);
    }
  }
};

template<typename param_t, typename param2_t>
void batch_norm_cpu_collect_linear_and_constant_terms(
    param2_t* alpha, param2_t* beta, int64_t n_channel,
    const Tensor& weight /* optional */, const Tensor& bias /* optional */,
    const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {

  const param_t* weight_data = weight.defined() ? weight.data_ptr<param_t>() : nullptr;
  const param_t* bias_data = bias.defined() ? bias.data_ptr<param_t>() : nullptr;

  auto save_mean_a = conditional_accessor_1d<param_t>(save_mean);
  auto save_invstd_a = conditional_accessor_1d<param_t>(save_invstd);
  auto running_mean_a = conditional_accessor_1d<param_t>(running_mean);
  auto running_var_a = conditional_accessor_1d<param_t>(running_var);

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
  for (const auto c : c10::irange(n_channel)) {
    param2_t mean, invstd;
    if (train) {
      mean = param2_t(save_mean_a[c]);
      invstd = param2_t(save_invstd_a[c]);
    } else {
      mean = param2_t(running_mean_a[c]);
      invstd = 1 / std::sqrt(running_var_a[c] + static_cast<param2_t>(eps));
    }
    param2_t weight_v = weight_data ? param2_t(weight_data[c]) : param2_t(1);
    param2_t bias_v = bias_data ? param2_t(bias_data[c]) : param2_t(0);
    alpha[c] = invstd * weight_v;
    beta[c] = bias_v - mean * alpha[c];
  }
}

/// A fast path for CPU inference and training forward when all tensors are contiguous.
template<typename scalar_t, typename param_t>
void batch_norm_cpu_contiguous_impl(Tensor& output, const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {

  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;

  // alpha/beta will be float when input is bfloat16
  using param2_t = param_acc_t<scalar_t>;
  Tensor alpha = at::empty({n_channel}, param_options(input));
  Tensor beta = at::empty({n_channel}, param_options(input));
  param2_t* alpha_data = alpha.data_ptr<param2_t>();
  param2_t* beta_data = beta.data_ptr<param2_t>();

  batch_norm_cpu_collect_linear_and_constant_terms<param_t, param2_t>(
     alpha_data, beta_data, n_channel, weight, bias,
     save_mean, save_invstd, running_mean, running_var, train, eps);

  scalar_t* output_data = output.data_ptr<scalar_t>();
  const scalar_t* input_data = input.data_ptr<scalar_t>();

  // Apply the linear terms to the input,
  // output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)
  if (image_size != 1) {
    at::parallel_for(0, n_batch * n_channel, 1, [&](int64_t begin, int64_t end) {
      int64_t n = 0;
      int64_t c = 0;
      data_index_init(begin, n, n_batch, c, n_channel);

      for (const auto i : c10::irange(begin, end)) {
        int64_t offset = i * image_size;
        BatchNormImpl<scalar_t>::kernel0(
            output_data + offset, input_data + offset,
            alpha_data[c], beta_data[c], image_size);

        // move on to next index
        data_index_step(n, n_batch, c, n_channel);
      }
    });
  } else {
    // image_size == 1
    at::parallel_for(0, n_batch, 1, [&](int64_t begin, int64_t end) {
      for (const auto n : c10::irange(begin, end)) {
        int64_t offset = n * n_channel;
        BatchNormImpl<scalar_t>::kernel1(
            output_data + offset, input_data + offset,
            alpha_data, beta_data, n_channel);
      }
    });
  }
}

template <typename scalar_t, typename param_t>
void batch_norm_cpu_channels_last_impl(Tensor& output, const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& runnning_var, bool train, double eps) {

  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;

  // alpha/beta will be float when input is bfloat16
  using param2_t = param_acc_t<scalar_t>;
  Tensor alpha = at::empty({n_channel}, param_options(input));
  Tensor beta = at::empty({n_channel}, param_options(input));
  param2_t* alpha_data = alpha.data_ptr<param2_t>();
  param2_t* beta_data = beta.data_ptr<param2_t>();

  batch_norm_cpu_collect_linear_and_constant_terms<param_t, param2_t>(
      alpha_data, beta_data, n_channel, weight, bias,
      save_mean, save_invstd, running_mean, runnning_var, train, eps);

  scalar_t* output_data = output.data_ptr<scalar_t>();
  const scalar_t* input_data = input.data_ptr<scalar_t>();

  // Apply the linear terms to the input,
  // output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)
  at::parallel_for(0, n_batch * image_size, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      int64_t offset = i * n_channel;
      BatchNormImpl<scalar_t>::kernel1(
          output_data + offset, input_data + offset,
          alpha_data, beta_data, n_channel);
    }
  });
}

template <typename scalar_t, typename param_t>
void batch_norm_cpu_collect_stats_contiguous_impl(
    Tensor& mean, Tensor& var_sum, const Tensor& input) {

  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;
  int64_t N = input.numel() / n_channel;

  const scalar_t* input_data = input.data_ptr<scalar_t>();
  param_t* mean_data = mean.data_ptr<param_t>();
  param_t* var_sum_data = var_sum.data_ptr<param_t>();

  // parallel dim reduce on 'channel'
  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      // compute mean per input
      BatchNormCollectStatsImpl<scalar_t, param_t>::kernel0(
          mean_data, input_data, n_batch, image_size, n_channel, N, c);

      // compute variance per input
      BatchNormCollectStatsImpl<scalar_t, param_t>::kernel1(
          var_sum_data, input_data, mean_data[c], n_batch, image_size, n_channel, c);
    }
  });
}

template <typename scalar_t, typename param_t>
void batch_norm_cpu_collect_stats_channels_last_impl(
    Tensor& mean, Tensor& var_sum, const Tensor& input) {

  using accscalar_t = at::acc_type<scalar_t, false>;
  int64_t n_channel = input.size(1);
  int64_t N = input.numel() / n_channel;

  const scalar_t* input_data = input.data_ptr<scalar_t>();
  param_t* mean_data = mean.data_ptr<param_t>();
  param_t* var_sum_data = var_sum.data_ptr<param_t>();

  // Typical vertical reduce from shape of {NHW, C} to {C}.
  // Apply two path parallel reduction:
  // First path: allocate an immediate buffer of size {max_threads, C}, parallel along dim0,
  //    {NHW, C} => {max_threads, C}
  //
  // Second path: parallel along dim1 of the immediate buffer,
  //    {max_threads, C} => {C}
  //
  // Normal size of C should fit in L1, otherwise consider blocking on C.
  //
  // Use float immediate buffer when input is bfloat16
  //
  using param2_t = param_acc_t<scalar_t>;
  int num_threads = at::get_num_threads();
  Tensor buffer = at::empty({num_threads, n_channel}, param_options(input)).zero_();
  param2_t* buffer_data = buffer.data_ptr<param2_t>();

  // compute mean per input
  at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads,
                "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    param2_t* buffer_ptr = buffer_data + tid * n_channel;
    for (const auto i : c10::irange(begin, end)) {
      const scalar_t* x_ptr = input_data + i * n_channel;
      BatchNormCollectStatsImpl<scalar_t, param_t>::kernel2(
          buffer_ptr, x_ptr, n_channel);
    }
  });

  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      accscalar_t sum = 0;
      for (const auto t : c10::irange(num_threads)) {
        sum += buffer_data[t * n_channel + c];
      }
      accscalar_t mean = sum / N;
      mean_data[c] = param_t(mean);
    }
  });

  // compute variance per input, reuse the immediate buffer
  buffer.zero_();
  at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    param2_t* buffer_ptr = buffer_data + tid * n_channel;
    for (const auto i : c10::irange(begin, end)) {
      const scalar_t* x_ptr = input_data + i * n_channel;
      BatchNormCollectStatsImpl<scalar_t, param_t>::kernel3(
          buffer_ptr, x_ptr, mean_data, n_channel);
    }
  });

  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      accscalar_t _var_sum = 0;
      for (const auto t : c10::irange(num_threads)) {
        _var_sum += buffer_data[t * n_channel + c];
      }
      var_sum_data[c] = param_t(_var_sum);
    }
  });
}

template <typename scalar_t, typename param_t>
void batch_norm_cpu_backward_contiguous_impl(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps) {

  using Vec = Vectorized<vec_scalar_t<scalar_t>>;
  using accscalar_t = at::acc_type<scalar_t, false>;
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;
  int64_t N = input.numel() / n_channel;

  const scalar_t* grad_output_data = grad_output.data_ptr<scalar_t>();
  const scalar_t* input_data = input.data_ptr<scalar_t>();

  scalar_t* grad_input_data = grad_input.defined() ? grad_input.data_ptr<scalar_t>() : nullptr;
  param_t* grad_weight_data = grad_weight.defined() ? grad_weight.data_ptr<param_t>() : nullptr;
  param_t* grad_bias_data = grad_bias.defined() ? grad_bias.data_ptr<param_t>() : nullptr;
  const bool grad_input_null = grad_input_data == nullptr;
  const bool grad_weight_null = grad_weight_data == nullptr;
  const bool grad_bias_null = grad_bias_data == nullptr;

  auto weight_a = conditional_accessor_1d<param_t>(weight);
  auto save_mean_a = conditional_accessor_1d<param_t>(save_mean);
  auto save_invstd_a = conditional_accessor_1d<param_t>(save_invstd);
  auto running_mean_a = conditional_accessor_1d<param_t>(running_mean);
  auto running_var_a = conditional_accessor_1d<param_t>(running_var);

  // parallel dim reduce on 'channel'
  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      param_t w = weight.defined() ? weight_a[c] : param_t(1);

      param_t mean, invstd;
      if (train) {
        mean = save_mean_a[c];
        invstd = save_invstd_a[c];
      } else {
        mean = running_mean_a[c];
        invstd = 1 / std::sqrt(running_var_a[c] + eps);
      }

      // reduce over grad_output in feature plane
      // compute 1) sum; 2) dot product of Q(X) and dY.
      // fuse into a single loop to reuse dY
      //
      accscalar_t sum, dotp;
      std::tie(sum, dotp) = BatchNormBackwardImpl<scalar_t, param_t>::kernel0(
          input_data, grad_output_data, mean,
          n_batch, n_channel, image_size, c);

      if (!grad_input_null) {
        if (train) {
          accscalar_t k = (accscalar_t) dotp * invstd * invstd / N;
          accscalar_t grad_mean = sum / N;

          for (const auto n : c10::irange(n_batch)) {
            const scalar_t* x_ptr = input_data + n * n_channel * image_size + c * image_size;
            scalar_t* dx_ptr = grad_input_data + n * n_channel * image_size + c * image_size;
            const scalar_t* dy_ptr = grad_output_data + n * n_channel * image_size + c * image_size;

            // Scalar math:
            // for (const auto j : c10::irange(image_size)) {
            //   scalar_t dx = (x_ptr[j] - mean) * k;
            //   dx_ptr[j] = (dy_ptr[j] - grad_mean - dx) * invstd * w;
            // }
            vec::map2<scalar_t>(
                [=](Vec x, Vec dy) {
                  Vec dx = (x - Vec(mean)) * Vec(k);
                  return (dy - Vec(grad_mean) - dx) * Vec(invstd) * Vec(w);
                },
                dx_ptr,
                x_ptr,
                dy_ptr,
                image_size);
          }
        } else { // evaluation mode
          for (const auto n : c10::irange(n_batch)) {
            scalar_t* dx_ptr = grad_input_data + n * n_channel * image_size + c * image_size;
            const scalar_t* dy_ptr = grad_output_data + n * n_channel * image_size + c * image_size;

            // Scalar math:
            // for (const auto j : c10::irange(image_size)) {
            //   dx_ptr[j] = dy_ptr[j] * invstd * w;
            // }
            vec::map<scalar_t>(
                [=](Vec dy) { return dy * Vec(invstd) * Vec(w); },
                dx_ptr,
                dy_ptr,
                image_size);
          }
        }
      }

      if (!grad_weight_null) {
        grad_weight_data[c] = dotp * invstd;
      }

      if (!grad_bias_null) {
        grad_bias_data[c] = sum;
      }
    }
  });
}

template <typename scalar_t, typename param_t>
void batch_norm_cpu_backward_channels_last_impl(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps) {

  using Vec = Vectorized<scalar_t>;
  using accscalar_t = at::acc_type<scalar_t, false>;
  int64_t n_channel = input.size(1);
  int64_t N = input.numel() / n_channel;

  const scalar_t* grad_output_data = grad_output.data_ptr<scalar_t>();
  const scalar_t* input_data = input.data_ptr<scalar_t>();

  scalar_t* grad_input_data = grad_input.defined() ? grad_input.data_ptr<scalar_t>() : nullptr;
  param_t* grad_weight_data = grad_weight.defined() ? grad_weight.data_ptr<param_t>() : nullptr;
  param_t* grad_bias_data = grad_bias.defined() ? grad_bias.data_ptr<param_t>() : nullptr;

  param_t* save_mean_data = conditional_data_ptr<param_t>(save_mean);
  param_t* save_invstd_data = conditional_data_ptr<param_t>(save_invstd);
  param_t* running_mean_data = conditional_data_ptr<param_t>(running_mean);
  param_t* running_var_data = conditional_data_ptr<param_t>(running_var);

  const bool mixed_type = !std::is_same<scalar_t, param_t>::value;
  const auto dtype = mixed_type ? kFloat : input.scalar_type();
  Tensor weight_ = weight.defined() ? weight : at::ones({n_channel}, input.options().dtype(dtype));
  const param_t* weight_data = weight_.data_ptr<param_t>();

  param_t* mean_ptr = nullptr;
  param_t* invstd_ptr = nullptr;
  Tensor invstd = at::empty({0}, input.options().dtype(dtype));
  if (train) {
    mean_ptr = save_mean_data;
    invstd_ptr = save_invstd_data;
  } else {
    mean_ptr = running_mean_data;

    invstd.resize_({n_channel});
    invstd_ptr = invstd.data_ptr<param_t>();
    for (const auto c : c10::irange(n_channel)) {
      invstd_ptr[c] = 1 / std::sqrt(running_var_data[c] + eps);
    }
  }

  // Typical vertical reduce from shape of {NHW, C} to {C}.
  // Apply two path parallel reduction:
  // First path: allocate an immediate buffer of size {2, max_threads, C}, parallel along dim0,
  //    sum = buffer[0], dotp = buffer[2]
  //
  // Second path: parallel along dim1 of the immediate buffer.
  //
  using param2_t = param_acc_t<scalar_t>;
  int num_threads = at::get_num_threads();
  Tensor buffer = at::empty({2, num_threads, n_channel}, param_options(input)).zero_();
  param2_t* sum_data = buffer.data_ptr<param2_t>();
  param2_t* dotp_data = sum_data + num_threads * n_channel;

  // compute sum and dotp per feature plain,
  // fuse into a single loop to reuse grad_output in L1.
  at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    param2_t* sum_ptr = sum_data + tid * n_channel;
    param2_t* dotp_ptr = dotp_data + tid * n_channel;
    for (const auto i : c10::irange(begin, end)) {
      const scalar_t* x_ptr = input_data + i * n_channel;
      const scalar_t* dy_ptr = grad_output_data + i * n_channel;

      BatchNormBackwardImpl<scalar_t, param_t>::kernel1(
          sum_ptr, dotp_ptr, x_ptr, dy_ptr, mean_ptr, n_channel);
    }
  });

  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      // store the final result of sum and dotp in the 1st lane of immediate buffer,
      // so that we won't need to allocate anther buffer to store the temp values.
      accscalar_t _sum = 0;
      for (const auto t : c10::irange(num_threads)) {
        _sum += sum_data[t * n_channel + c];
      }
      sum_data[/* 0 * n_channel + */c] = _sum;

      accscalar_t _dotp = 0;
      for (const auto t : c10::irange(num_threads)) {
        _dotp += dotp_data[t * n_channel + c];
      }
      dotp_data[/* 0 * n_channel + */c] = _dotp;
    }
  });

  // compute grad_input
  if (grad_input.defined()) {
    at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        scalar_t* dx_ptr = grad_input_data + i * n_channel;
        const scalar_t* x_ptr = input_data + i * n_channel;
        const scalar_t* dy_ptr = grad_output_data + i * n_channel;
        if (train) {
          BatchNormBackwardImpl<scalar_t, param_t>::kernel2(
            dx_ptr, x_ptr, dy_ptr, sum_data, dotp_data,
            weight_data, mean_ptr, invstd_ptr, N, n_channel);
        } else { // evaluation mode
          BatchNormBackwardImpl<scalar_t, param_t>::kernel3(
            dx_ptr, dy_ptr, weight_data, invstd_ptr, n_channel);
        }
      }
    });
  }

  if (grad_weight.defined()) {
    // grad_weight = dotp * invstd
    for (int64_t d = 0; d < n_channel; d++) {
      grad_weight_data[d] = param_t(dotp_data[d] * invstd_ptr[d]);
    }
  }

  // grad_bias = sum
  if (grad_bias.defined()) {
      for (int64_t d = 0; d < n_channel; d++) {
      grad_bias_data[d] = param_t(sum_data[d]);
    }
  }
}

void batch_norm_cpu_kernel(Tensor& output, const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& save_mean,  const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {
  const bool mixed_type = is_mixed_type(input, weight, bias, save_mean, save_invstd, running_mean, running_var);
  if (input.is_contiguous()) {
    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, input.scalar_type(), "batch_norm_cpu_contiguous", [&] {
      if (mixed_type) {
        batch_norm_cpu_contiguous_impl<BFloat16, float>(output, input, weight, bias,
            save_mean, save_invstd, running_mean, running_var, train, eps);
      } else {
        batch_norm_cpu_contiguous_impl<scalar_t, scalar_t>(output, input, weight, bias,
            save_mean, save_invstd, running_mean, running_var, train, eps);
      }
    });
  } else if (input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, input.scalar_type(), "batch_norm_cpu_channels_last", [&] {
      if (mixed_type) {
        batch_norm_cpu_channels_last_impl<BFloat16, float>(output, input, weight, bias,
            save_mean, save_invstd, running_mean, running_var, train, eps);
      } else {
        batch_norm_cpu_channels_last_impl<scalar_t, scalar_t>(output, input, weight, bias,
            save_mean, save_invstd, running_mean, running_var, train, eps);
      }
    });
  } else {
    TORCH_CHECK(false, "batch_norm_cpu_kernel: expecting input to be contiguous.");
  }
}

void batch_norm_cpu_collect_stats_kernel(
    Tensor& mean, Tensor& var_sum, const Tensor& input) {
  const bool mixed_type = is_mixed_type(input, mean, var_sum);
  int64_t image_size = input.numel() / input.size(0) / input.size(1);
  if (input.is_contiguous()) {
    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, input.scalar_type(), "batch_norm_cpu_collect_stats_contiguous", [&] {
      if (mixed_type) {
        if (image_size == 1) { // NC11 is also channels last
          batch_norm_cpu_collect_stats_channels_last_impl<BFloat16, float>(mean, var_sum, input);
        } else {
          batch_norm_cpu_collect_stats_contiguous_impl<BFloat16, float>(mean, var_sum, input);
        }
      } else {
        if (image_size == 1) { // NC11 is also channels last
          batch_norm_cpu_collect_stats_channels_last_impl<scalar_t, scalar_t>(mean, var_sum, input);
        } else {
          batch_norm_cpu_collect_stats_contiguous_impl<scalar_t, scalar_t>(mean, var_sum, input);
        }
      }
    });
  } else if (input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, input.scalar_type(), "batch_norm_cpu_collect_stats_channels_last", [&] {
      if (mixed_type) {
        batch_norm_cpu_collect_stats_channels_last_impl<BFloat16, float>(mean, var_sum, input);
      } else {
        batch_norm_cpu_collect_stats_channels_last_impl<scalar_t, scalar_t>(mean, var_sum, input);
      }
    });
  } else {
    TORCH_CHECK(false, "batch_norm_cpu_collect_stats_kernel: expecting input to be contiguous.");
  }
}

void batch_norm_cpu_backward_kernel(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps) {
  const bool mixed_type = is_mixed_type(input, weight, running_mean, running_var, save_mean, save_invstd);
  int64_t image_size = input.numel() / input.size(0) / input.size(1);
  if (input.is_contiguous()) {
    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, input.scalar_type(), "batch_norm_cpu_backward_contiguous", [&] {
      if (mixed_type) {
        if (image_size == 1) { // NC11 is also channels last
          batch_norm_cpu_backward_channels_last_impl<BFloat16, float>(grad_input, grad_weight, grad_bias,
              grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
        } else {
          batch_norm_cpu_backward_contiguous_impl<BFloat16, float>(grad_input, grad_weight, grad_bias,
              grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
        }
      } else {
        if (image_size == 1) { // NC11 is also channels last
          batch_norm_cpu_backward_channels_last_impl<scalar_t, scalar_t>(grad_input, grad_weight, grad_bias,
              grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
        } else {
          batch_norm_cpu_backward_contiguous_impl<scalar_t, scalar_t>(grad_input, grad_weight, grad_bias,
              grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
        }
      }
    });
  } else if (input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, input.scalar_type(), "batch_norm_cpu_backward_channels_last", [&] {
      if (mixed_type) {
        batch_norm_cpu_backward_channels_last_impl<BFloat16, float>(grad_input, grad_weight, grad_bias,
            grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
      } else {
        batch_norm_cpu_backward_channels_last_impl<scalar_t, scalar_t>(grad_input, grad_weight, grad_bias,
            grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
      }
    });
  } else {
    TORCH_CHECK(false, "batch_norm_cpu_backward_kernel: expecting input to be contiguous.");
  }
}

}// anonymous namespace

REGISTER_DISPATCH(batch_norm_cpu_stub, &batch_norm_cpu_kernel);
REGISTER_DISPATCH(batch_norm_cpu_collect_stats_stub, &batch_norm_cpu_collect_stats_kernel);
REGISTER_DISPATCH(batch_norm_cpu_backward_stub, &batch_norm_cpu_backward_kernel);

}} // namespace at::native
