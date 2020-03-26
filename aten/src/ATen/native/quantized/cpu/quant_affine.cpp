#include <ATen/Parallel.h>
#include <ATen/Dispatch.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/quant_affine.h>

#ifdef USE_FBGEMM
#include <fbgemm/QuantUtils.h>
#endif
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

namespace at {
namespace native {
namespace {

#ifdef USE_FBGEMM
void quantize_tensor_affine_cpu(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point) {
  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), "quantize_tensor_affine_cpu", [&]() {  
    const float* rd = rtensor.data_ptr<float>();
    auto qd = reinterpret_cast<underlying_t*>(qtensor.data_ptr<scalar_t>());
    fbgemm::TensorQuantizationParams qparams;
    qparams.scale = scale;
    qparams.zero_point = zero_point;
    qparams.precision = CHAR_BIT * sizeof(underlying_t);
    int num_tasks = at::get_num_threads();
    at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
      for (int task_id = begin; task_id < end; ++task_id) {
        fbgemm::Quantize<underlying_t>(
            rd, /*src=*/
            qd, /*dst=*/
            rtensor.numel(), /*len*/
            qparams, /*qparams=*/
            task_id, /*thread_id*/
            num_tasks /*num_threads*/);
      }
    });
  });
}

void dequantize_tensor_affine_cpu(Tensor qtensor, Tensor rtensor, double scale, int64_t zero_point) {
  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), "dequantize_tensor_affine_cpu", [&]() {  
    const auto* qd = reinterpret_cast<const underlying_t*>(qtensor.data_ptr<scalar_t>());
    fbgemm::TensorQuantizationParams qparams;
    qparams.scale = scale;
    qparams.zero_point = zero_point;
    qparams.precision = CHAR_BIT * sizeof(underlying_t);
    float* rd = rtensor.data_ptr<float>();
    int num_tasks = at::get_num_threads();
    at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
      for (int task_id = begin; task_id < end; ++task_id) {
        fbgemm::Dequantize<underlying_t>(
            qd, /*src=*/
            rd, /*dst=*/
            qtensor.numel(), /*len=*/
            qparams, /*qparams=*/
            task_id, /*thread_id*/
            num_tasks /*num_threads*/);
      }
    });
  });
}
#else  // USE_FBGEMM

#ifdef __ARM_NEON__
#include <ATen/quantized/Quantize.h>
// Generic template defaults to naive quantize implementation
template <typename T>
void quantize_tensor_arm(
    const float* in,
    Tensor qtensor,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  auto out = qtensor.data_ptr<T>();
  for (int i = 0; i < N; ++i) {
    out[i] = quantize_val<T>(scale, zero_point, in[i]);
  }
}

// Specialized implementation from caffe2::Int8Quantize.
// There may be slight accuracy difference between this and implementation of quantize_val
// TODO Update quantize_tensor_arm implementation to follow quantize_val,
// i.e. f = Round(value/scale + zero_point)
// TODO Make quantize_tensor_arm work for other datatypes too (int8, int32).
template <>
void quantize_tensor_arm<c10::quint8>(
    const float* in,
    Tensor qtensor,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  const float inv_scale = 1.0f / scale;
  uint32_t i = 0;
  auto out = (uint8_t*)qtensor.data_ptr<c10::quint8>();
  const float32x4_t vinv_scale = vdupq_n_f32(inv_scale);
  // magic float and magic int to take care of rounding
  // int magic_round(float f): interpret_int32(f + 12582912.0f) - 0x4B400000
  // Some detail:
  // 12582912.0f is 2**23 + 2**22. The trick is based on the fact that when you
  // add a small number to a large number, the result rounds to the precision of
  // the least significant bit of the large number. For IEEE-754
  // single-precision number mantissa has 23 bits, and adding 2**23 would cause
  // rounding to the nearest even integer. The we cast to int and subtract the
  // same number (0x4B400000 is the integer representation of 12582912.0f) to
  // get only the mantissa. This works if -2**22 < x < 2**22, but preserves the
  // sign for negative numbers.
  const int32x4_t voffset = vdupq_n_s32(zero_point - 0x4B400000);
  const float32x4_t vmagic_float = vdupq_n_f32(12582912.0f);
  for (i = 0; i + 8 < N; i += 8) {
    const float32x4_t vin0123 = vld1q_f32(in);
    in += 4;
    const float32x4_t vin4567 = vld1q_f32(in);
    in += 4;
    const int32x4_t vraw0123 = vaddq_s32(
        voffset,
        vreinterpretq_s32_f32(
            vaddq_f32(vmagic_float, vmulq_f32(vin0123, vinv_scale))));
    const int32x4_t vraw4567 = vaddq_s32(
        voffset,
        vreinterpretq_s32_f32(
            vaddq_f32(vmagic_float, vmulq_f32(vin4567, vinv_scale))));
    const int16x8_t vraw01234567 =
        vcombine_s16(vqmovn_s32(vraw0123), vqmovn_s32(vraw4567));
    const uint8x8_t vout01234567 = vqmovun_s16(vraw01234567);
    vst1_u8(out, vout01234567);
    out += 8;
  }
  for (; i < N; ++i) {
    (*out++) = quantize_val_arm(scale, zero_point, (*in++));
  }
}
#endif // __ARM_NEON__

void quantize_tensor_affine_cpu(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point) {
  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), "quantize_tensor_affine_cpu", [&]() {
    TORCH_CHECK(rtensor.is_contiguous(), "Float tensor should be contiguous");
    const float* const rdata = rtensor.data_ptr<float>();
    // If QEngine is set to QNNPACK, use caffe2 specialized Int8Quantize implementation on ARM
    #if defined(__ARM_NEON__)
      if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
        quantize_tensor_arm<scalar_t>(rdata, qtensor, rtensor.numel(), scale, zero_point);
        return qtensor;
      }
    #endif
    auto qdata = qtensor.data_ptr<scalar_t>();
    auto numel = rtensor.numel();
    for (int i = 0; i < numel; ++i) {
      qdata[i] = quantize_val<scalar_t>(scale, zero_point, rdata[i]);
    }
  });
}

void dequantize_tensor_affine_cpu(Tensor qtensor, Tensor rtensor, double scale, int64_t zero_point) {
  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), "dequantize_tensor_affine_cpu", [&]() {
    const auto* qd = qtensor.data_ptr<scalar_t>();
    float* rd = rtensor.data_ptr<float>();
    auto numel = qtensor.numel();
    for (auto i = 0; i < numel; ++i) {
      rd[i] = dequantize_val<scalar_t>(scale, zero_point, qd[i]);
    }
  });
}
#endif  // USE_FBGEMM

// TODO: add fbgemm for per channel
void quantize_tensor_per_channel_affine_cpu(Tensor rtensor,
                                          Tensor qtensor,
                                          Tensor scales,
                                          Tensor zero_points,
                                          int64_t axis) {
  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), "quantize_tensor_per_channel_affine_cpu", [&]() {
    int64_t batches = size_to_dim_(axis, rtensor.sizes());
    int64_t elements_per_channel = size_from_dim_(axis + 1, rtensor.sizes());
    int64_t channel = rtensor.size(axis);
    auto scales_data = scales.data_ptr<double>();
    auto zero_points_data = zero_points.data_ptr<int64_t>();
    const float* rdata = rtensor.data_ptr<float>();
    auto qdata = qtensor.data_ptr<scalar_t>();
    for (auto b = 0; b < batches; ++b) {
      for (auto c = 0; c < channel; ++c) {
        for (auto e = 0; e < elements_per_channel; ++e) {
          auto i = b * channel * elements_per_channel + c * elements_per_channel + e;
          qdata[i] = quantize_val<scalar_t>(scales_data[c], zero_points_data[c], rdata[i]);
        }
      }
    }
  });
}

void dequantize_tensor_per_channel_affine_cpu(Tensor qtensor,
                                            Tensor rtensor,
                                            Tensor scales,
                                            Tensor zero_points,
                                            int64_t axis) {
  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), "dequantize_tensor_per_channel_affine_cpu", [&]() {
    int64_t batches = size_to_dim_(axis, rtensor.sizes());
    int64_t elements_per_channel = size_from_dim_(axis + 1, rtensor.sizes());
    int64_t channel = rtensor.size(axis);
    auto scales_data = scales.data_ptr<double>();
    auto zero_points_data = zero_points.data_ptr<int64_t>();
    const auto* qd = qtensor.data_ptr<scalar_t>();
    float* rd = rtensor.data_ptr<float>();
    for (auto b = 0; b < batches; ++b) {
      for (auto c = 0; c < channel; ++c) {
        for (auto e = 0; e < elements_per_channel; ++e) {
          auto i = b * channel * elements_per_channel + c * elements_per_channel + e;
          // We need to convert the qint8 value to float to ensure the subtraction
          // subexpression returns a float
          rd[i] = (static_cast<float>(qd[i].val_) - zero_points_data[c]) * scales_data[c];
        }
      }
    }
  });
}
} // anonymous namespace

// Note: currently, quantize* functions are naively dispatched to a single realization. Move everything to the QuantizedOpKernels, once optimized versions are available.
REGISTER_ARCH_DISPATCH(quantize_tensor_affine_stub, DEFAULT,  &quantize_tensor_affine_cpu);
REGISTER_AVX_DISPATCH(quantize_tensor_affine_stub,  &quantize_tensor_affine_cpu);
REGISTER_AVX2_DISPATCH(quantize_tensor_affine_stub,  &quantize_tensor_affine_cpu);

REGISTER_ARCH_DISPATCH(quantize_tensor_per_channel_affine_stub, DEFAULT, &quantize_tensor_per_channel_affine_cpu);
REGISTER_AVX_DISPATCH(quantize_tensor_per_channel_affine_stub, &quantize_tensor_per_channel_affine_cpu);
REGISTER_AVX2_DISPATCH(quantize_tensor_per_channel_affine_stub, &quantize_tensor_per_channel_affine_cpu);

REGISTER_ARCH_DISPATCH(dequantize_tensor_affine_stub, DEFAULT, &dequantize_tensor_affine_cpu);
REGISTER_AVX_DISPATCH(dequantize_tensor_affine_stub,  &dequantize_tensor_affine_cpu);
REGISTER_AVX2_DISPATCH(dequantize_tensor_affine_stub,  &dequantize_tensor_affine_cpu);

REGISTER_ARCH_DISPATCH(dequantize_tensor_per_channel_affine_stub, DEFAULT, &dequantize_tensor_per_channel_affine_cpu);
REGISTER_AVX_DISPATCH(dequantize_tensor_per_channel_affine_stub,  &dequantize_tensor_per_channel_affine_cpu);
REGISTER_AVX2_DISPATCH(dequantize_tensor_per_channel_affine_stub,  &dequantize_tensor_per_channel_affine_cpu);

} // namespace native
} // namespace at
