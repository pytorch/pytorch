#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/SortingUtils.h>

#include <cmath>
#ifdef USE_FBGEMM
#include "fbgemm/QuantUtils.h"
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace at {
namespace native {
namespace {

// ****************** HEY YOU! YES YOU! Read this! ********************
//
// Please read the README.md in this directory before editing this file

template <bool ReLUFused = false>
Tensor qcat_nhwc_kernel(
    const c10::List<Tensor>& qxs,
    int64_t dim,
    double scale,
    int64_t zero_point) {
  const at::Tensor& qx0 = qxs[0];
  int64_t C_out = 0;
  std::vector<int64_t> Cs_in;
  // Prefix sum of input channels for fast indexing
  std::vector<int64_t> Cs_sum;
  std::vector<double> scales;
  std::vector<int64_t> zero_pts;
  std::vector<void*> data_ptrs;

  for (const at::Tensor& qx : qxs) {
    TORCH_CHECK(
        qx.dim() == qx0.dim(),
        "Tensors must have the same number of dimensions: got ",
        qx.dim(),
        " and ",
        qx0.dim());
#define CHECK_DIM(d)                                            \
  TORCH_CHECK(                                                  \
      qx.size(d) == qx0.size(d),                                \
      "Sizes of tensors must match expect in dimension 1. Got", \
      qx.size(d),                                               \
      " and ",                                                  \
      qx0.size(d));
    CHECK_DIM(0);
    CHECK_DIM(2);
    CHECK_DIM(3);
    TORCH_CHECK(
        qx.scalar_type() == qx0.scalar_type(),
        "Expected object of scalar type ",
        toString(qx0.scalar_type()),
        " but got scalar type ",
        toString(qx.scalar_type()));
    Cs_in.push_back(qx.size(1));
    Cs_sum.push_back(C_out);
    C_out += qx.size(1);
    scales.push_back(qx.q_scale());
    zero_pts.push_back(qx.q_zero_point());
    data_ptrs.push_back(qx.data_ptr());
  }

  const int64_t N = qx0.size(0);
  const int64_t H = qx0.size(2);
  const int64_t W = qx0.size(3);
  float inv_scale = 1.0 / scale;

  auto output = at::_empty_affine_quantized(
      {N, C_out, H, W},
      qx0.options().memory_format(MemoryFormat::ChannelsLast),
      scale,
      zero_point);

  // N, H, and W are explicitly captured here because there's a bug in GCC5
  // which causes an internal compiler error if they're not
  AT_DISPATCH_QINT_TYPES(output.scalar_type(), "qcat_nhwc", [&, N, H, W]() {
    using Vec = Vec256<scalar_t>;
    for (int64_t batch = 0; batch < N; ++batch) {
      for (int64_t row = 0; row < H; ++row) {
        for (int64_t col = 0; col < W; ++col) {
          // loop over input tensors
          for (int64_t tidx = 0; tidx < Cs_in.size(); ++tidx) {
            scalar_t::underlying* optr =
                reinterpret_cast<scalar_t::underlying*>(output.data_ptr()) +
                batch * H * W * C_out + row * W * C_out + col * C_out +
                Cs_sum[tidx];

            auto curr_C = Cs_in[tidx];
            float curr_scale = scales[tidx];
            int64_t curr_zero_pt = zero_pts[tidx];

            scalar_t::underlying* iptr =
                reinterpret_cast<scalar_t::underlying*>(data_ptrs[tidx]) +
                batch * H * W * curr_C + row * W * curr_C + col * curr_C;

            constexpr int64_t VLEN = Vec::size();
            int64_t c = 0;

            // Vectorized loop
            if (c + VLEN <= curr_C) {
              auto curr_scale_vec = Vec256<float>(curr_scale);
              auto curr_zero_pt_vec = Vec256<float>((float)curr_zero_pt);
              auto scale_neg_zp_premul = curr_scale_vec * curr_zero_pt_vec.neg();
              for (; c + VLEN <= curr_C; c += VLEN) {
                auto inp_vec = Vec::loadu(iptr + c);
                auto float_values = inp_vec.dequantize(
                    curr_scale_vec, curr_zero_pt_vec, scale_neg_zp_premul);
                Vec::float_vec_return_type retvals;
                for (int i = 0; i < Vec::float_num_vecs(); ++i) {
                  if (ReLUFused) {
                    retvals[i] =
                        vec256::maximum(float_values[i], Vec256<float>(0.0f));
                  } else {
                    retvals[i] = float_values[i];
                  }
                }
                auto quantized =
                    Vec::quantize(retvals, scale, zero_point, inv_scale);
                quantized.store(optr + c);
              }
            }

            // Scalar loop
            for (; c < curr_C; ++c) {
              auto float_val = at::dequantize_val(
                  curr_scale,
                  curr_zero_pt,
                  reinterpret_cast<scalar_t*>(iptr)[c]);
              if (ReLUFused) {
                float_val = std::max(0.0f, float_val);
              }
              optr[c] =
                  at::quantize_val<scalar_t>(scale, zero_point, float_val).val_;
            } // for c

          } // for tidx
        } // for col
      } // for row
    } // for b
  });

  return output;
}

void qrelu_kernel(const Tensor& qx, Tensor& qy) {
  const auto zero_point = qx.q_zero_point();
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu", [&]() {
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point());
    using Vec = Vec256<scalar_t>;
    auto zero_point_vec = Vec(scalar_t(zero_point));
    auto iter = TensorIterator::unary_op(qy, qx);
    cpu_kernel_vec(
        iter,
        [&](scalar_t value) -> scalar_t {
          return scalar_t(std::max<underlying_t>(value.val_, zero_point));
        },
        [&](Vec value) -> Vec { return value.relu(zero_point_vec); });
  });
}

void qrelu6_kernel(const Tensor& qx, Tensor& qy) {
  const auto zero_point = qx.q_zero_point();
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu6", [&]() {
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point());
    using Vec = Vec256<scalar_t>;
    auto iter = TensorIterator::unary_op(qy, qx);
    scalar_t six =
        at::quantize_val<scalar_t>(qx.q_scale(), qx.q_zero_point(), 6.0);
    auto zero_point_vec = Vec(scalar_t(zero_point));
    auto six_vec = Vec(six);
    cpu_kernel_vec(
        iter,
        [&](scalar_t value) -> scalar_t {
          underlying_t relu_val =
              std::max<underlying_t>(value.val_, zero_point);
          return scalar_t(std::min<underlying_t>(relu_val, six.val_));
        },
        [&](Vec val) -> Vec { return val.relu6(zero_point_vec, six_vec); });
  });
}

static void leaky_qrelu_out_kernel(Tensor& out, const Tensor& qx,
                                   Scalar negval_) {
  int64_t i_zp = qx.q_zero_point();
  float i_scale = qx.q_scale();

  int64_t o_zp = out.q_zero_point();
  float o_scale = out.q_scale();
  float o_inv_scale = 1.0f / o_scale;

  float negval = negval_.to<float>();

  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "leaky_qrelu", [&] {
    using Vec = Vec256<float>;  // Naïve implementation uses dequant/quant loop.
    using qVec = Vec256<scalar_t>;
    Vec zero_vec = Vec(0.0f);
    Vec one_vec = Vec(1.0f);

    Vec i_scale_vec = Vec((float)i_scale);
    Vec i_zp_vec = Vec((float)i_zp);
    Vec i_scale_zp_neg_premul_vec = i_scale_vec * i_zp_vec.neg();

    Vec negval_vec = Vec(negval);

    auto iter = TensorIterator::unary_op(out, qx);

    cpu_kernel_vec(
        iter,
        [&](scalar_t value_qx) -> scalar_t {
          auto value_dx = at::dequantize_val(i_scale, i_zp, value_qx);
          auto value_dy = value_dx > 0 ? value_dx : value_dx * negval;
          return at::quantize_val<scalar_t>(o_scale, o_zp, value_dy);
        },
        [&](qVec qx_vec) -> qVec {
          /* Vectorized implementation creates a multiplicand vector, which has
           * "alpha" for all negative dx values and ones-vector for all
           * positive values of dx. The multiplicand then is multiplied by the
           * input.
           */
          auto dx_vec_vec = qx_vec.dequantize(i_scale_vec, i_zp_vec,
                                              i_scale_zp_neg_premul_vec);
          for (int idx = 0; idx < dx_vec_vec.size(); ++idx) {
            const auto dx_vec = dx_vec_vec[idx];
            const auto multiplicand = Vec::blendv(negval_vec, one_vec,
                                                  dx_vec > zero_vec);
            dx_vec_vec[idx] = dx_vec * multiplicand;
          }
          return qVec::quantize(dx_vec_vec, o_scale, o_zp, o_inv_scale);
        });
  });
}

void qsigmoid_kernel(const Tensor& qx, Tensor& qy) {
  int64_t zero_point = qx.q_zero_point();
  float scale = qx.q_scale();
  auto scale_vec = Vec256<float>(scale);
  auto zero_point_vec = Vec256<float>((float)zero_point);
  auto scale_neg_zp_premul_vec = scale_vec * zero_point_vec.neg();

  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qsigmoid", [&]() {
    // Naive implemenentation: uses dequantize/execute/quantize routine
    // - Output scale is set to 1.0 / 2^(BIT_NUM)
    // - For signed types output zero point is set to 0
    // - For unsigned types output zero point is set to (qmax + qmin) / 2.0
    // See https://stackoverflow.com/a/34448562/3606192 for potential
    // optimizations
    float output_scale = 0.00390625;  // 1.0 / 2^8
    int64_t output_zero_point = 0;
    if (SCALAR_TYPE == at::kQInt32) {
      output_scale = 2.3283064365386963e-10;  // 1.0 / 2^32
    } else if (SCALAR_TYPE == at::kQInt8) {
      output_zero_point = -128;
    }
    float inv_output_scale = 1.0 / output_scale;

    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        output_scale,
        output_zero_point);
    auto iter = TensorIterator::unary_op(qy, qx);

    using Vec = Vec256<scalar_t>;
    cpu_kernel_vec(
      iter,
      [&](scalar_t value_qx) -> scalar_t {
        const auto value_dx = at::dequantize_val(scale, zero_point, value_qx);
        const auto value_dy = 1.0f / (1.0 + std::exp((-value_dx)));
        return at::quantize_val<scalar_t>(output_scale, output_zero_point,
                                          value_dy);
      },
      [&](Vec value_qx) -> Vec {
        auto value_dx = value_qx.dequantize(scale_vec, zero_point_vec,
                                            scale_neg_zp_premul_vec);
        for (int idx = 0; idx < value_dx.size(); ++idx) {
          value_dx[idx] = value_dx[idx].neg();
          value_dx[idx] = value_dx[idx].exp();
          value_dx[idx] = Vec256<float>(1.0f) + value_dx[idx];
          value_dx[idx] = value_dx[idx].reciprocal();
        }
        return Vec::quantize(value_dx, output_scale, output_zero_point,
                             inv_output_scale);
      }
    );
  });
}

void qclamp_kernel(
    const Tensor& qx,
    Scalar min_scalar,
    Scalar max_scalar,
    Tensor& qy) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qclamp", [&]() {
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point());
    using Vec = Vec256<scalar_t>;
    auto iter = TensorIterator::unary_op(qy, qx);
    auto min = min_scalar.to<float>();
    auto max = max_scalar.to<float>();
    scalar_t min_q =
        at::quantize_val<scalar_t>(qx.q_scale(), qx.q_zero_point(), min);
    scalar_t max_q =
        at::quantize_val<scalar_t>(qx.q_scale(), qx.q_zero_point(), max);
    auto min_vec = Vec(min_q);
    auto max_vec = Vec(max_q);
    cpu_kernel_vec(
        iter,
        [&](scalar_t value) -> scalar_t {
          underlying_t min_clamped =
              std::max<underlying_t>(value.val_, min_q.val_);
          return scalar_t(std::min<underlying_t>(min_clamped, max_q.val_));
        },
        [&](Vec val) -> Vec {
          auto min_clamped = val.maximum(min_vec);
          return min_clamped.minimum(max_vec);
        });
  });
}


void qtanh_kernel(const Tensor& qx, Tensor& qy) {
  int64_t zero_point = qx.q_zero_point();
  float scale = qx.q_scale();
  auto scale_vec = Vec256<float>(scale);
  auto zero_point_vec = Vec256<float>((float)zero_point);
  auto scale_neg_zp_premul_vec = scale_vec * zero_point_vec.neg();

  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qtanh", [&]() {
    // Naive implemenentation: uses dequantize/execute/quantize routine
    // - Output scale is set to 2.0 / 2^(BIT_NUM)
    // - For signed types output zero point is set to 0
    // - For unsigned types output zero point is set to (qmax + qmin) / 2.0
    float output_scale = 0.0078125;  // 2.0 / 512
    int64_t output_zero_point = 0;
    if (SCALAR_TYPE == at::kQInt32) {
      output_scale = 4.656612873077393e-10;  // 2.0 / 2^32
    } else if (SCALAR_TYPE == at::kQUInt8) {
      output_zero_point = 128;
    }
    float inv_output_scale = 1.0 / output_scale;

    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        output_scale,
        output_zero_point);
    auto iter = TensorIterator::unary_op(qy, qx);

    using Vec = Vec256<scalar_t>;
    cpu_kernel_vec(
      iter,
      [&](scalar_t value_qx) -> scalar_t {
        const auto value_dx = at::dequantize_val(scale, zero_point, value_qx);
        return at::quantize_val<scalar_t>(output_scale, output_zero_point,
                                          std::tanh(value_dx));
      },
      [&](Vec value_qx) -> Vec {
        const auto value_dx = value_qx.dequantize(scale_vec, zero_point_vec,
                                                  scale_neg_zp_premul_vec);
        Vec::float_vec_return_type retvals;
        for (int idx = 0; idx < Vec::float_num_vecs(); ++idx) {
          retvals[idx] = value_dx[idx].tanh();
        }
        return Vec::quantize(retvals, output_scale, output_zero_point,
                             inv_output_scale);
      }
    );
  });
}

// Note: out is assumed to be the same size as self and other.
// Note: Addition is only supported when self, other, out are of the same dtype.
template <bool ReLUFused = false>
void qadd_kernel(Tensor& out, const Tensor& self, const Tensor& other) {
  int64_t zero_point = out.q_zero_point();
  float scale = out.q_scale();
  float inv_scale = 1.0f / scale;
  int64_t self_zero_point = self.q_zero_point();
  float self_scale = self.q_scale();
  int64_t other_zero_point = other.q_zero_point();
  float other_scale = other.q_scale();

  // Broadcast out the parameters here to amortize out that cost across
  // loop iterations.
  // TODO: we can optimize dequantization by doing a premultiplication
  // of the zero point by scale and doing FMA on scale*x_q - (scale*zero_point)
  auto self_zero_point_vec = Vec256<float>((float)self_zero_point);
  auto self_scale_vec = Vec256<float>(self_scale);
  auto other_zero_point_vec = Vec256<float>((float)other_zero_point);
  auto other_scale_vec = Vec256<float>(other_scale);

  auto self_scale_neg_zp_premul_vec = self_scale_vec * self_zero_point_vec.neg();
  auto other_scale_zp_premul_vec = other_scale_vec * other_zero_point_vec.neg();

  auto iter = TensorIterator::binary_op(out, self, other);

  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qadd", [&]() {
    using Vec = Vec256<scalar_t>;
    cpu_kernel_vec(
        iter,
        [&](scalar_t a, scalar_t b) -> scalar_t {
          const auto da = at::dequantize_val(self_scale, self_zero_point, a);
          const auto db = at::dequantize_val(other_scale, other_zero_point, b);
          float c = da + db;
          if (ReLUFused) {
            c = std::max<float>(c, 0.0);
          }
          return at::quantize_val<scalar_t>(scale, zero_point, c);
        },
        [&](Vec a, Vec b) -> Vec {
          const auto da = a.dequantize(
              self_scale_vec, self_zero_point_vec, self_scale_neg_zp_premul_vec);
          const auto db = b.dequantize(
              other_scale_vec, other_zero_point_vec, other_scale_zp_premul_vec);
          Vec::float_vec_return_type retvals;
          for (int i = 0; i < Vec::float_num_vecs(); ++i) {
            auto c = da[i] + db[i];
            if (ReLUFused) {
              c = vec256::maximum(c, Vec256<float>(0.0f));
            }
            retvals[i] = c;
          }
          // TODO: fbgemm::Quantize doesn't support taking in the
          // pre-broadcasted parameters. We might be able to save some cycles by
          // enabling that in the API.
          // TODO: specialize fbgemm::Quantize for a single vector and make it
          // inlineable. This could help with interleaving as suggested by the
          // TensorIterator implementations
          auto rv = Vec::quantize(retvals, scale, zero_point, inv_scale);
          return rv;
        });
  });
}

void qmaxpool_2d_nhwc_kernel(
    const Tensor& qx,
    int64_t iC, // input/output channels
    int64_t iH,
    int64_t iW, // input sizes
    int64_t oH,
    int64_t oW, // output sizes
    int64_t kH,
    int64_t kW, // kernel size
    int64_t sH,
    int64_t sW, // strides
    int64_t pH,
    int64_t pW, // padding
    int64_t dH,
    int64_t dW, // dilation
    Tensor& qy) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "max_pool2d_nhwc", [&]() {
    scalar_t* idata = static_cast<scalar_t*>(qx.data_ptr());
    scalar_t* odata = static_cast<scalar_t*>(qy.data_ptr());

    // Loop over N
    for (int64_t b = 0; b < qx.size(0); ++b) {
      // Loop over H
      auto* i_p =
          reinterpret_cast<scalar_t::underlying*>(idata + b * iW * iH * iC);
      for (int64_t row = 0; row < oH; ++row) {
        // Loop over W
        for (int64_t col = 0; col < oW; ++col) {
          // Pointer to output data for this specific N,H,W position
          auto* o_p = reinterpret_cast<scalar_t::underlying*>(
              odata + b * oH * oW * iC + row * oW * iC + col * iC);

          // Loop over reduction block
          int64_t h_start = row * sH - pH;
          int64_t w_start = col * sW - pW;
          int64_t h_end = std::min(h_start + (kH - 1) * dH + 1, iH);
          int64_t w_end = std::min(w_start + (kW - 1) * dW + 1, iW);
          while (h_start < 0)
            h_start += dH;
          while (w_start < 0)
            w_start += dW;

          int64_t c = 0;

          // Interleaved vector loop 4x
          constexpr auto vec_width = Vec256<scalar_t>::size();
          for (; c + 4 * vec_width <= iC; c += 4 * vec_width) {
            Vec256<scalar_t> acc{
                scalar_t(std::numeric_limits<scalar_t::underlying>::lowest())};
            Vec256<scalar_t> accs[4] = {acc, acc, acc, acc};
            int64_t tcntr = 0;
            int64_t x, y;
            for (y = h_start; y < h_end; y += dH) {
              for (x = w_start; x < w_end; x += dW) {
                for (int i = 0; i < 4; ++i) {
                  tcntr = y * iW + x;
                  auto vals = Vec256<scalar_t>::loadu(
                      i_p + tcntr * iC + c + Vec256<scalar_t>::size() * i);
                  accs[i] = vec256::maximum(accs[i], vals);
                }
              } // for x
            } // for y
            for (int i = 0; i < 4; ++i) {
              accs[i].store(o_p + c + Vec256<scalar_t>::size() * i);
            }
          } // for c

          // Vector loop
          for (; c + vec_width <= iC; c += vec_width) {
            Vec256<scalar_t> acc{
                scalar_t(std::numeric_limits<scalar_t::underlying>::lowest())};
            int64_t tcntr = 0;
            int64_t x, y;
            for (y = h_start; y < h_end; y += dH) {
              for (x = w_start; x < w_end; x += dW) {
                tcntr = y * iW + x;
                auto vals = Vec256<scalar_t>::loadu(i_p + tcntr * iC + c);
                acc = vec256::maximum(acc, vals);
              } // for x
            } // for y
            acc.store(o_p + c);
          } // for c

          for (; c < iC; ++c) {
            auto max_val = std::numeric_limits<scalar_t::underlying>::lowest();
            int64_t tcntr = 0;
            int64_t x, y;
            for (y = h_start; y < h_end; y += dH) {
              for (x = w_start; x < w_end; x += dW) {
                tcntr = y * iW + x;
                auto val = *(i_p + tcntr * iC + c);
                max_val = std::max(max_val, val);
              } // for x
            } // for y

            o_p[c] = max_val;
          } // for c
        } // for col
      } // for row
    } // for b
  });
}

template <typename T>
void do_avg_pool_on_AVX2(
    typename T::underlying* i_p,
    typename T::underlying* o_p,
    int64_t& c,
    int64_t channel_size,
    int64_t channel_multiplier,
    int32_t input_zero_point_m_size,
    int32_t output_zero_point,
    float multiplier,
    int64_t hstart,
    int64_t hend,
    int64_t wstart,
    int64_t wend,
    int64_t stride_D,
    int64_t stride_H,
    int64_t stride_W) {
#if defined(__AVX2__) && !defined(_MSC_VER)
  constexpr auto vec_width = Vec256<T>::size() / 4;
  if (vec_width == 8) {
    for (; c + vec_width <= channel_size; c += vec_width) {
      int64_t tcntr = 0;

      Vec256<int32_t> acc(input_zero_point_m_size);
      for (int64_t ih = hstart; ih < hend; ih++) {
        for (int64_t iw = wstart; iw < wend; iw++) {
          tcntr = ih * stride_H + iw * stride_W;
          auto vals = vec256::convert_to_int32<typename T::underlying>(
              i_p + tcntr * channel_multiplier + c * stride_D);
          acc = acc + vals;
        }
      }
      int32_t acc_int[vec_width];
      float acc_fp[vec_width];
      acc.store(acc_int);
      vec256::convert(acc_int, acc_fp, vec_width);
      at::quantize_vec<T>(
          1.0f / multiplier,
          output_zero_point,
          acc_fp,
          reinterpret_cast<T*>(o_p + c),
          vec_width);
    }
  }
#endif
}

void qadaptive_avg_pool2d_nhwc_kernel(
    const Tensor& qx,
    Tensor& qy,
    int64_t b,
    int64_t sizeD,
    int64_t isizeH,
    int64_t isizeW,
    int64_t osizeH,
    int64_t osizeW,
    int64_t istrideB,
    int64_t istrideD,
    int64_t istrideH,
    int64_t istrideW) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "adaptive_avg_pool2d_nhwc", [&]() {
    scalar_t* idata = static_cast<scalar_t*>(qx.data_ptr());
    scalar_t* odata = static_cast<scalar_t*>(qy.data_ptr());
    auto minimum = std::numeric_limits<scalar_t::underlying>::lowest();
    auto maximum = std::numeric_limits<scalar_t::underlying>::max();
    auto* i_p =
        reinterpret_cast<typename scalar_t::underlying*>(idata + b * istrideB);
    for (int64_t oh = 0; oh < osizeH; oh++) {
      int istartH = (int)std::floor((float)(oh * isizeH) / osizeH);
      int iendH = (int)std::ceil((float)((oh + 1) * isizeH) / osizeH);
      int kH = iendH - istartH;
      for (int64_t ow = 0; ow < osizeW; ow++) {
        auto* o_p = reinterpret_cast<typename scalar_t::underlying*>(
            odata + b * osizeH * osizeW * sizeD + (oh * osizeW + ow) * sizeD);
        int istartW = (int)std::floor((float)(ow * isizeW) / osizeW);
        int iendW = (int)std::ceil((float)((ow + 1) * isizeW) / osizeW);
        int kW = iendW - istartW;
        int size = kH * kW;
        float multiplier = qx.q_scale() / qy.q_scale() / size;
        int64_t c = 0;
        // For int8 or uint8quantization, we implicitly use int32 as
        // accumulation Or else, it will go to the slow path
        // TODO: support 16bit, 32bit, and etc.
        auto* internal_i_p = i_p + istartH * istrideH + istartW * istrideW;

        // TODO: more vectorization with loop interleaving
        do_avg_pool_on_AVX2<scalar_t>(
            internal_i_p,
            o_p,
            c,
            sizeD,
            1,
            -qx.q_zero_point() * size,
            qy.q_zero_point(),
            multiplier,
            0,
            kH,
            0,
            kW,
            istrideD,
            istrideH,
            istrideW);
        // 1) The following loop handles the remaining channels
        // 2) It also handles the Non-AVX2 path
        for (; c < sizeD; ++c) {
          int32_t acc_int32 = -qx.q_zero_point() * size;
          int64_t tcntr = 0;
          for (int64_t ih = 0; ih < kH; ih++) {
            for (int64_t iw = 0; iw < kW; iw++) {
              tcntr = ih * istrideH + iw * istrideW;
              auto val = *(internal_i_p + tcntr + c * istrideD);
              acc_int32 += val;
            }
          }
          // clamp
          o_p[c] = at::quantize_val<scalar_t>(
                       1.0f / multiplier, qy.q_zero_point(), acc_int32)
                       .val_;
        } // c
      } // oh
    } // ow
  });
}

void qavg_pool2d_nhwc_kernel(
    const Tensor& qx,
    Tensor& qy,
    int64_t b,
    int64_t nInputPlane,
    int64_t inputWidth,
    int64_t inputHeight,
    int64_t outputWidth,
    int64_t outputHeight,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "avg_pool2d_nhwc", [&]() {
    scalar_t* idata = static_cast<scalar_t*>(qx.data_ptr());
    scalar_t* odata = static_cast<scalar_t*>(qy.data_ptr());
    auto minimum = std::numeric_limits<scalar_t::underlying>::lowest();
    auto maximum = std::numeric_limits<scalar_t::underlying>::max();
    int64_t batch_size = nInputPlane * inputWidth * inputHeight;
    auto* i_p = reinterpret_cast<typename scalar_t::underlying*>(
        idata + b * batch_size);

    for (int64_t oh = 0; oh < outputHeight; oh++) {
      for (int64_t ow = 0; ow < outputWidth; ow++) {
        auto* o_p = reinterpret_cast<typename scalar_t::underlying*>(
            odata + b * nInputPlane * outputWidth * outputHeight +
            (oh * outputWidth + ow) * nInputPlane);
        int64_t hstart = oh * dH - padH;
        int64_t wstart = ow * dW - padW;
        int64_t hend = std::min(hstart + kH, inputHeight + padH);
        int64_t wend = std::min(wstart + kW, inputWidth + padW);
        int64_t pool_size = (hend - hstart) * (wend - wstart);
        hstart = std::max(hstart, (int64_t)0);
        wstart = std::max(wstart, (int64_t)0);
        hend = std::min(hend, inputHeight);
        wend = std::min(wend, inputWidth);

        int64_t size = (hend - hstart) * (wend - wstart);
        int64_t divide_factor;
        if (divisor_override.has_value()) {
          divide_factor = divisor_override.value();
        } else {
          if (count_include_pad) {
            divide_factor = pool_size;
          } else {
            divide_factor = (hend - hstart) * (wend - wstart);
          }
        }

        int64_t c = 0;
        // For int8 quantization, we implicitly use int32 as accumulation
        // Or else, it will go to the slow path
        // TODO: support 16bit, 32bit, and etc.
        float multiplier = qx.q_scale() / qy.q_scale() / divide_factor;
        do_avg_pool_on_AVX2<scalar_t>(
            i_p,
            o_p,
            c,
            nInputPlane,
            nInputPlane,
            -qx.q_zero_point() * size,
            qy.q_zero_point(),
            multiplier,
            hstart,
            hend,
            wstart,
            wend,
            1,
            inputWidth,
            1);
        // 1) The following loop handles the remaining channels
        // 2) It also handles the Non-AVX2 path
        for (; c < nInputPlane; ++c) {
          int32_t acc_int32 = -qx.q_zero_point() * size;
          int64_t tcntr = 0;
          for (int64_t ih = hstart; ih < hend; ih++) {
            for (int64_t iw = wstart; iw < wend; iw++) {
              tcntr = ih * inputWidth + iw;
              auto val = *(i_p + tcntr * nInputPlane + c);
              acc_int32 += val;
            }
          }
          double acc_fp = acc_int32 * 1.0;
          // clamp
          o_p[c] = at::quantize_val<scalar_t>(
                       1.0f / multiplier, qy.q_zero_point(), acc_fp)
                       .val_;
        } // c
      } // ow
    } // oh
  });
}

template <typename T>
int64_t do_quantized_bilinear_on_AVX2(
    const typename T::underlying*& pos1,
    typename T::underlying*& pos2,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t channels,
    int32_t output_zero_point,
    int32_t input_zero_point,
    float inverse_scale,
    const float h0lambda,
    const float h1lambda,
    const float w0lambda,
    const float w1lambda,
    const int64_t h1p,
    const int64_t w1p) {
  int64_t c = 0;
#if defined(__AVX2__) && !defined(_MSC_VER)
  constexpr auto vec_width = Vec256<T>::size() / 4;
  if (vec_width == 8) {
    for (; c + vec_width <= channels; c += vec_width) {
      Vec256<float> pos1_fp_v[4];
      Vec256<int32_t> pos1_int_v[4];
      pos1_int_v[0] = vec256::convert_to_int32<typename T::underlying>(pos1);
      pos1_int_v[1] = vec256::convert_to_int32<typename T::underlying>(
          pos1 + w1p * channels);
      pos1_int_v[2] = vec256::convert_to_int32<typename T::underlying>(
          pos1 + h1p * input_width * channels);
      pos1_int_v[3] = vec256::convert_to_int32<typename T::underlying>(
          pos1 + (h1p * input_width + w1p) * channels);
      for (int i = 0; i < 4; i++) {
        int32_t pos1_int[vec_width];
        float pos1_fp[vec_width];
        pos1_int_v[i].store(pos1_int);
        vec256::convert(pos1_int, pos1_fp, vec_width);
        pos1_fp_v[i] = Vec256<float>::loadu(pos1_fp, 8);
      }
      Vec256<float> h0lambda_v(h0lambda);
      Vec256<float> h1lambda_v(h1lambda);
      Vec256<float> w0lambda_v(w0lambda);
      Vec256<float> w1lambda_v(w1lambda);
      Vec256<float> input_zero_point_v(input_zero_point);
      Vec256<float> result =
          h0lambda_v * (w0lambda_v * pos1_fp_v[0] + w1lambda_v * pos1_fp_v[1]) +
          h1lambda_v * (w0lambda_v * pos1_fp_v[2] + w1lambda_v * pos1_fp_v[3]) -
          input_zero_point_v;
      float result_fp[vec_width];
      result.store(result_fp);
      at::quantize_vec<T>(
          inverse_scale,
          output_zero_point,
          result_fp,
          reinterpret_cast<T*>(pos2),
          vec_width);
      pos1 += vec_width;
      pos2 += vec_width;
    }
  }
#endif
  return c;
}

void qupsample_bilinear2d_nhwc_kernel(
    Tensor& output,
    const Tensor& input,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_DISPATCH_QINT_TYPES(
      input.scalar_type(), "upsample_bilinear2d_nhwc", [&]() {
        auto* idata = static_cast<scalar_t*>(input.data_ptr());
        auto* odata = static_cast<scalar_t*>(output.data_ptr());
        float inverse_scale = output.q_scale() / input.q_scale();
        const auto rheight = area_pixel_compute_scale<float>(
            input_height, output_height, align_corners, scales_h);
        const auto rwidth = area_pixel_compute_scale<float>(
            input_width, output_width, align_corners, scales_w);

        for (int64_t b = 0; b < nbatch; ++b) {
          auto* i_p = reinterpret_cast<typename scalar_t::underlying*>(
              idata + b * input_height * input_width * channels);
          auto* o_p = reinterpret_cast<typename scalar_t::underlying*>(
              odata + b * output_height * output_width * channels);

          for (int64_t h2 = 0; h2 < output_height; ++h2) {
            const auto h1r = area_pixel_compute_source_index<float>(
                rheight, h2, align_corners, /*cubic=*/false);

            const int64_t h1 = h1r;
            const int64_t h1p = (h1 < input_height - 1) ? 1 : 0;
            const float h1lambda = h1r - h1;
            const float h0lambda = static_cast<float>(1.) - h1lambda;

            for (int64_t w2 = 0; w2 < output_width; ++w2) {
              const auto w1r = area_pixel_compute_source_index<float>(
                  rwidth, w2, align_corners, /*cubic=*/false);
              const int64_t w1 = w1r;
              const int64_t w1p = (w1 < input_width - 1) ? 1 : 0;

              const float w1lambda = w1r - w1;
              const float w0lambda = static_cast<float>(1.) - w1lambda;

              int64_t c = 0;
              // We use float32 to do the computation
              const typename scalar_t::underlying* pos1 =
                  i_p + (h1 * input_width + w1) * channels;
              typename scalar_t::underlying* pos2 =
                  o_p + (h2 * output_width + w2) * channels;
              // We have to isolate this function out because the VS does not
              // expand the macro correctly.
              c = do_quantized_bilinear_on_AVX2<scalar_t>(
                  pos1,
                  pos2,
                  input_height,
                  input_width,
                  output_height,
                  output_width,
                  channels,
                  output.q_zero_point(),
                  input.q_zero_point(),
                  inverse_scale,
                  h0lambda,
                  h1lambda,
                  w0lambda,
                  w1lambda,
                  h1p,
                  w1p);
              // 1) The following loop handles the remaining channels
              // 2) It also handles the Non-AVX2 path
              for (; c < channels; ++c) {
                float result = h0lambda *
                        (w0lambda * pos1[0] + w1lambda * pos1[w1p * channels]) +
                    h1lambda *
                        (w0lambda * pos1[h1p * input_width * channels] +
                         w1lambda * pos1[(h1p * input_width + w1p) * channels]);
                pos2[0] = at::quantize_val<scalar_t>(
                              inverse_scale,
                              output.q_zero_point(),
                              result - input.q_zero_point())
                              .val_;
                pos1 += 1;
                pos2 += 1;
              } // c
            } // w2
          } // h2
        } // b
      });
}

void qtopk_kernel(Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "qtopk_cpu", [&] {
    dim_apply(
        {self, values, indices},
        dim,
        [&](int64_t i, TensorList tl) {
          auto tmp_values = tl[0].accessor<scalar_t, 1>();
          auto mode_values = tl[1].accessor<scalar_t, 1>();
          auto mode_indices = tl[2].accessor<int64_t, 1>();

          auto n = tmp_values.size(0);
          auto use_partial_sort = k * 64 <= n;

          using elem_t = std::pair<typename scalar_t::underlying, int64_t>;
          std::vector<elem_t> queue(n);
          for (int64_t j = 0; j < n; j++) {
            queue[j].first = tmp_values[j].val_;
            queue[j].second = j;
          }

          // we want NaN to be sorted as top for numpy compatibility
          if (use_partial_sort) {
            if (largest) {
              std::partial_sort(queue.begin(), queue.begin() + k, queue.end(),
                [](const elem_t& x, const elem_t& y) -> bool {
                  return x.first > y.first;
                });
            } else {
              std::partial_sort(queue.begin(), queue.begin() + k, queue.end(),
                [](const elem_t& x, const elem_t& y) -> bool {
                  return x.first < y.first;
                });
            }
          } else {
            if (largest) {
              std::nth_element(queue.begin(), queue.begin() + k - 1, queue.end(),
                [](const elem_t& x, const elem_t& y) -> bool {
                  return x.first > y.first;
                });
              if (sorted) {
                std::sort(queue.begin(), queue.begin() + k - 1,
                  [](const elem_t& x, const elem_t& y) -> bool {
                    return x.first > y.first;
                  });
              }
            } else {
              std::nth_element(queue.begin(), queue.begin() + k -1, queue.end(),
                [](const elem_t& x, const elem_t& y) -> bool {
                  return x.first < y.first;
                });
              if (sorted) {
                std::sort(queue.begin(), queue.begin() + k -1,
                  [](const elem_t& x, const elem_t& y) -> bool {
                    return x.first < y.first;
                  });
              }
            }
          }

          for (int64_t j = 0; j < k; j++) {
            mode_values[j] = scalar_t(queue[j].first);
            mode_indices[j] = queue[j].second;
          }
        });
  });
}

template <bool ReluFused>
void q_batch_norm_kernel(
    int64_t N,
    int64_t C,
    int64_t HxW,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const Tensor& input,
    const Tensor& a,
    const Tensor& b,
    Tensor& output) {

  AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qbatch_norm", [&]() {
    float* alpha = a.data_ptr<float>();
    float* beta = b.data_ptr<float>();
    scalar_t::underlying* X =
        reinterpret_cast<scalar_t::underlying*>(input.data_ptr());
    scalar_t::underlying* Y = reinterpret_cast<scalar_t::underlying*>(output.data_ptr());

    constexpr int kVLen = 8;
    const int64_t outer_size = N * HxW;
    using Vec = Vec256<scalar_t>;
    // Hoisted variables
    auto in_zp_vec = Vec256<float>(static_cast<float>(in_zero_point));
    auto fake_scale = Vec256<float>(1.0f);
    auto scale_neg_zp_premul = fake_scale * in_zp_vec.neg();
    auto out_zero_point_v = Vec(scalar_t(out_zero_point));

    // TODO replace with TensorIterator implementation once #33166 is fixed.
    for (int64_t i = 0; i < outer_size; ++i) {
      int64_t n = C / (Vec::float_num_vecs() * kVLen) * (Vec::float_num_vecs() * kVLen);
      int64_t r = C % (Vec::float_num_vecs() * kVLen);
      auto* X_ptr = reinterpret_cast<typename scalar_t::underlying*>(X + i * C);
      auto* Y_ptr = reinterpret_cast<typename scalar_t::underlying*>(Y + i * C);

      for (int64_t j = 0; j < n; j += Vec::float_num_vecs() * kVLen) {
        auto vals_q = Vec::loadu(X_ptr + j);
        // Fake scale of 1.0 here, should not affect performance (FMA in place of sub)
        auto vals_dq = vals_q.dequantize(fake_scale, in_zp_vec, scale_neg_zp_premul);
        for (size_t idx = 0; idx < vals_dq.size(); ++idx) {
          auto alpha_v = Vec256<float>::loadu(alpha + j + idx * kVLen);
          auto beta_v = Vec256<float>::loadu(beta + j + idx * kVLen);
          vals_dq[idx] = vec256::fmadd(alpha_v, vals_dq[idx], beta_v);
        }
        // Fake scale again
        auto outputs_q = Vec::quantize(vals_dq, /*output_scale=*/1.0f, out_zero_point, /*inv_output_scale=*/1.0f);
        if (ReluFused) {
          outputs_q = outputs_q.relu(out_zero_point_v);
        }
        outputs_q.store(Y_ptr + j);
      }

      for (int64_t j = 0; j < r; ++j) {
        long quantized_down = out_zero_point +
            lrintf(alpha[n + j] * (X_ptr[n + j] - in_zero_point) +
                        beta[n + j]);
        if (ReluFused) { // static if
          quantized_down = std::max<long>(quantized_down, out_zero_point);
        }
        Y_ptr[n + j] = std::min<long>(
            std::max<long>(quantized_down, std::numeric_limits<scalar_t::underlying>::min()),
            std::numeric_limits<scalar_t::underlying>::max());
      }
    }
});

}

void fake_quantize_tensor_kernel(
    Tensor& output,
    const Tensor& input,
    float sc,
    int64_t z_point,
    int64_t quant_min,
    int64_t quant_max) {
  float inv_scale = 1.0f / sc;
  auto iter = TensorIterator::unary_op(output, input);
  cpu_kernel(iter, [&](float self) -> float {
    return (std::fmin(
                std::fmax(
                    static_cast<int64_t>(
                        std::nearbyint(self * inv_scale + z_point)),
                    quant_min),
                quant_max) -
            z_point) *
        sc;
  });
}

void fake_quantize_grad_tensor_kernel(
    Tensor& input_grad,
    const Tensor& input,
    const Tensor& output_grad,
    float sc,
    int64_t z_point,
    int64_t quant_min,
    int64_t quant_max) {
  float inv_scale = 1.0f / sc;
  auto iter = TensorIterator::binary_op(input_grad, input, output_grad);
  cpu_kernel(iter, [&](float x, float dy) -> float {
    int64_t xq = static_cast<int64_t>(std::nearbyint(x * inv_scale + z_point));
    return dy * (xq >= quant_min && xq <= quant_max);
  });
}

void fake_quant_per_channel_cpu(TensorIterator &iter, int64_t quant_min, int64_t quant_max) {
  cpu_kernel(iter,
    [=](float self, float scale, int64_t zero_point) -> float {
      float inv_scale = 1.0f / scale;
      return (std::fmin(
                std::fmax(
                    static_cast<int64_t>(
                        std::nearbyint(self * inv_scale + zero_point)),
                    quant_min),
                quant_max) -
            zero_point) *
        scale;
    });
}

void fake_quant_grad_per_channel_cpu(TensorIterator &iter, int64_t quant_min, int64_t quant_max) {
  cpu_kernel(iter,
    [=](float x, float dy, float scale, int64_t zero_point) -> float {
      float inv_scale = 1.0f / scale;
      int64_t xq = static_cast<int64_t>(std::nearbyint(x * inv_scale + zero_point));
      return dy * (xq >= quant_min && xq <= quant_max);
    });
}

} // namespace

REGISTER_DISPATCH(qrelu_stub, &qrelu_kernel);
REGISTER_DISPATCH(qrelu6_stub, &qrelu6_kernel);
REGISTER_DISPATCH(qrelu_leaky_stub, &leaky_qrelu_out_kernel);
REGISTER_DISPATCH(qsigmoid_stub, &qsigmoid_kernel);
REGISTER_DISPATCH(qclamp_stub, &qclamp_kernel);
REGISTER_DISPATCH(qtanh_stub, &qtanh_kernel);
REGISTER_DISPATCH(qadd_relu_stub, &qadd_kernel<true>);
REGISTER_DISPATCH(qadd_stub, &qadd_kernel<false>);
REGISTER_DISPATCH(qmaxpool_2d_nhwc_stub, &qmaxpool_2d_nhwc_kernel);
REGISTER_DISPATCH(
    qadaptive_avg_pool2d_nhwc_stub,
    &qadaptive_avg_pool2d_nhwc_kernel);
REGISTER_DISPATCH(qavg_pool2d_nhwc_stub, &qavg_pool2d_nhwc_kernel);
REGISTER_DISPATCH(
    qupsample_bilinear2d_nhwc_stub,
    &qupsample_bilinear2d_nhwc_kernel);
REGISTER_DISPATCH(qcat_nhwc_stub, &qcat_nhwc_kernel<false>);
REGISTER_DISPATCH(qcat_relu_nhwc_stub, &qcat_nhwc_kernel<true>);
REGISTER_DISPATCH(qtopk_stub, &qtopk_kernel);
REGISTER_DISPATCH(qbatch_norm_stub, &q_batch_norm_kernel<false>);
REGISTER_DISPATCH(fake_quant_tensor_stub, &fake_quantize_tensor_kernel);
REGISTER_DISPATCH(fake_quant_grad_tensor_stub, &fake_quantize_grad_tensor_kernel);
REGISTER_DISPATCH(fake_quant_per_channel_stub, &fake_quant_per_channel_cpu);
REGISTER_DISPATCH(fake_quant_grad_per_channel_stub, &fake_quant_grad_per_channel_cpu);

} // namespace native
} // namespace at
