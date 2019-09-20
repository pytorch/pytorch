#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>

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

  auto output = at::_empty_affine_quantized(
      {N, C_out, H, W},
      qx0.options(),
      scale,
      zero_point,
      MemoryFormat::ChannelsLast);

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
              float inv_scale = 1.0f / scale;
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
        at::device(kCPU).dtype(SCALAR_TYPE),
        qx.q_scale(),
        qx.q_zero_point(),
        qx.suggest_memory_format());
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
        at::device(kCPU).dtype(SCALAR_TYPE),
        qx.q_scale(),
        qx.q_zero_point(),
        qx.suggest_memory_format());
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

void qmaxpool_2d_nhwc_kernel(const Tensor &qx,
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
                             Tensor &qy) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "max_pool2d_nhwc", [&]() {
    scalar_t *idata = static_cast<scalar_t*>(qx.data_ptr());
    scalar_t *odata = static_cast<scalar_t*>(qy.data_ptr());

    // Loop over N
    for (int64_t b = 0; b < qx.size(0); ++b) {
      // Loop over H
      auto *i_p = reinterpret_cast<scalar_t::underlying*>(idata + b * iW * iH * iC);
      for (int64_t row = 0; row < oH; ++row) {
        // Loop over W
        for (int64_t col = 0; col < oW; ++col) {
          // Pointer to output data for this specific N,H,W position
          auto *o_p = reinterpret_cast<scalar_t::underlying*>(odata + b * oH * oW * iC + row * oW * iC + col * iC);

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
          for (; c + 4 * vec_width <= iC; c+= 4 * vec_width) {
            Vec256<scalar_t> acc{scalar_t(std::numeric_limits<scalar_t::underlying>::lowest())};
            Vec256<scalar_t> accs[4] = {acc, acc, acc, acc};
            int64_t tcntr = 0;
            int64_t x, y;
            for (y = h_start; y < h_end; y += dH) {
              for (x = w_start; x < w_end; x += dW) {
                for (int i = 0; i < 4; ++i) {
                  tcntr = y * iW + x;
                  auto vals = Vec256<scalar_t>::loadu(i_p + tcntr * iC + c + Vec256<scalar_t>::size() * i);
                  accs[i] = vec256::maximum(accs[i], vals);
                }
              }  // for x
            }  // for y 
            for (int i = 0; i < 4; ++i) {
              accs[i].store(o_p + c + Vec256<scalar_t>::size() * i);
            }
          }  // for c

          // Vector loop
          for (; c + vec_width <= iC; c+= vec_width) {
            Vec256<scalar_t> acc{scalar_t(std::numeric_limits<scalar_t::underlying>::lowest())};
            int64_t tcntr = 0;
            int64_t x, y;
            for (y = h_start; y < h_end; y += dH) {
              for (x = w_start; x < w_end; x += dW) {
                tcntr = y * iW + x;
                auto vals = Vec256<scalar_t>::loadu(i_p + tcntr * iC + c);
                acc = vec256::maximum(acc, vals);
              }  // for x
            }  // for y 
            acc.store(o_p + c);
          }  // for c

          for (; c < iC; ++c) {
            auto max_val = std::numeric_limits<scalar_t::underlying>::lowest();
            int64_t tcntr = 0;
            int64_t x, y;
            for (y = h_start; y < h_end; y += dH) {
              for (x = w_start; x < w_end; x += dW) {
                tcntr = y * iW + x;
                auto val = *(i_p + tcntr * iC + c);
                max_val = std::max(max_val, val);
              }  // for x
            }  // for y

            o_p[c] = max_val;
          }  // for c
        }  // for col
      }  // for row
    }  // for b
  });
} 

} // namespace

REGISTER_DISPATCH(qrelu_stub, &qrelu_kernel);
REGISTER_DISPATCH(qrelu6_stub, &qrelu6_kernel);
REGISTER_DISPATCH(qadd_relu_stub, &qadd_kernel<true>);
REGISTER_DISPATCH(qadd_stub, &qadd_kernel<false>);
REGISTER_DISPATCH(qmaxpool_2d_nhwc_stub, &qmaxpool_2d_nhwc_kernel);
REGISTER_DISPATCH(qcat_nhwc_stub, &qcat_nhwc_kernel<false>);
REGISTER_DISPATCH(qcat_relu_nhwc_stub, &qcat_nhwc_kernel<true>);

} // namespace native
} // namespace at
