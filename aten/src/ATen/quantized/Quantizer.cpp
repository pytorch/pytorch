#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/utils/Allocator.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/core/Tensor.h>
#include <typeinfo>

#ifdef USE_FBGEMM
#include <fbgemm/QuantUtils.h>
#endif
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

namespace at {

// Note: this is not a native function as Quantizer is not exposed to python yet
QuantizerPtr Tensor::quantizer() const {
  // This is a terrible hack to emulate what VariableType is doing
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  return get_qtensorimpl(*this)->quantizer();
}

void checkFloatCPUTensor(std::string fn_name, Tensor t) {
  TORCH_CHECK(
      t.scalar_type() == kFloat,
      fn_name,
      " expects a Float Tensor.");
  TORCH_CHECK(
      t.device() == kCPU,
      fn_name,
      " expects a CPU Tensor.");
}

template <typename T>
void checkQuantizedCPUTensor(std::string fn_name, Tensor t) {
  TORCH_CHECK(t.is_quantized(),
           fn_name,
           " expects a quantized Tensor.");
  TORCH_CHECK(t.scalar_type() == caffe2::TypeMeta::Make<T>(),
           fn_name,
           " expects a ",
           caffe2::TypeMeta::Make<T>(),
           " Tensor");
  TORCH_CHECK(t.device() == kCPU,
           fn_name,
           " expects a CPU quantized Tensor");
}

template <typename T>
void checkZeroPoint(std::string fn_name, int64_t zero_point) {
  TORCH_CHECK(zero_point <= std::numeric_limits<T>::max(),
              fn_name,
              " zero_point ",
              zero_point,
              " is out of range.");
  TORCH_CHECK(zero_point >= std::numeric_limits<T>::min(),
              fn_name,
              " zero_point ",
              zero_point,
              " is out of range.");
}

template <typename T>
void checkZeroPoints(std::string fn_name, Tensor zero_points) {
  auto zero_points_data = zero_points.data_ptr<int64_t>();
  for (size_t i = 0; i < zero_points.numel(); ++i) {
    TORCH_CHECK(zero_points_data[i] <= std::numeric_limits<T>::max(),
                fn_name,
                "zero_point",
                i,
                "is out of range.");
    TORCH_CHECK(zero_points_data[i] >= std::numeric_limits<T>::min(),
                fn_name,
                "zero_point",
                i,
                "is out of range.");
  }
}

#ifdef USE_FBGEMM
// Note: quantize_val is only explicitly used in test outside of this file
template <typename T>
T quantize_val(double scale, int64_t zero_point, float value) {
  // Internally, fbgemm::Quantize uses std::nearbyint.
  // std::nearbyint results in nearest integer value according to the current
  // rounding mode and the default rounding mode is rounds to even in half-way
  // cases in most popular processor architectures like x86 and ARM. This is
  // typically faster than an alternatives like std::round that rounds half-way
  // cases away from zero, and can be consistent with SIMD implementations for
  // example in x86 using _mm512_cvtps_epi32 or mm512_round_ps with
  // _MM_FROUND_CUR_DIRECTION option that also follow the current rounding mode.
  int32_t qvalue;
  qvalue = fbgemm::Quantize<typename T::underlying>(
      value,
      static_cast<int32_t>(zero_point),
      static_cast<double>(scale),
      /*result_precision=*/CHAR_BIT * sizeof(typename T::underlying));
  return static_cast<T>(qvalue);
}

template <typename T, int precision>
void quantize_vec(double scale, int64_t zero_point, const float *src, T *dst, size_t count) {
  fbgemm::Quantize<typename T::underlying>(
    src,
    (typename T::underlying*)dst,
    count,
    fbgemm::TensorQuantizationParams{(float)scale, (int32_t)zero_point, precision}
  );
}

template <typename T>
Tensor quantize_tensor(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point) {
  auto fn_name = "quantize_tensor";
  checkFloatCPUTensor(fn_name, rtensor);
  checkQuantizedCPUTensor<T>(fn_name, qtensor);
  checkZeroPoint<typename T::underlying>(fn_name, zero_point);
  const float* rd = rtensor.data_ptr<float>();
  auto qd = reinterpret_cast<typename T::underlying*>(qtensor.data_ptr<T>());
  fbgemm::TensorQuantizationParams qparams;
  qparams.scale = scale;
  qparams.zero_point = zero_point;
  qparams.precision = CHAR_BIT * sizeof(typename T::underlying);
  int num_tasks = at::get_num_threads();
  at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
    for (int task_id = begin; task_id < end; ++task_id) {
      fbgemm::Quantize<typename T::underlying>(
          rd, /*src=*/
          qd, /*dst=*/
          rtensor.numel(), /*len*/
          qparams, /*qparams=*/
          task_id, /*thread_id*/
          num_tasks /*num_threads*/);
    }
  });
  return qtensor;
}

template <typename T>
inline float dequantize_val(double scale, int64_t zero_point, T value) {
  fbgemm::TensorQuantizationParams qparams;
  qparams.scale = static_cast<float>(scale);
  qparams.zero_point = static_cast<int32_t>(zero_point);
  return fbgemm::Dequantize<typename T::underlying>(value.val_, qparams);
}

template <typename T>
Tensor dequantize_tensor(Tensor qtensor, Tensor rtensor, double scale, int64_t zero_point) {
  auto fn_name = "dequantize_tensor";
  checkFloatCPUTensor(fn_name, rtensor);
  checkQuantizedCPUTensor<T>(fn_name, qtensor);
  checkZeroPoint<typename T::underlying>(fn_name, zero_point);
  const auto* qd = reinterpret_cast<const typename T::underlying*>(qtensor.data_ptr<T>());
  fbgemm::TensorQuantizationParams qparams;
  qparams.scale = scale;
  qparams.zero_point = zero_point;
  qparams.precision = CHAR_BIT * sizeof(typename T::underlying);
  float* rd = rtensor.data_ptr<float>();
  int num_tasks = at::get_num_threads();
  at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
    for (int task_id = begin; task_id < end; ++task_id) {
      fbgemm::Dequantize<typename T::underlying>(
          qd, /*src=*/
          rd, /*dst=*/
          qtensor.numel(), /*len=*/
          qparams, /*qparams=*/
          task_id, /*thread_id*/
          num_tasks /*num_threads*/);
    }
  });
  return rtensor;
}
#else  // USE_FBGEMM

#if defined(__ANDROID__) && !defined(__NDK_MAJOR__)
template <class T>
inline float Round(const float x) {
  return ::nearbyintf(x);
}
inline double Round(const double x) {
  return ::nearbyint(x);
}
#else
template <class T>
inline T Round(const T x) {
  return std::nearbyint(x);
}
#endif

template <typename T>
T quantize_val(double scale, int64_t zero_point, float value) {
  // std::nearbyint results in nearest integer value according to the current
  // rounding mode and the default rounding mode is rounds to even in half-way
  // cases in most popular processor architectures like x86 and ARM. This is
  // typically faster than an alternatives like std::round that rounds half-way
  // cases away from zero, and can be consistent with SIMD implementations for
  // example in x86 using _mm512_cvtps_epi32 or mm512_round_ps with
  // _MM_FROUND_CUR_DIRECTION option that also follow the current rounding mode.
  int64_t qvalue;
  constexpr int64_t qmin = std::numeric_limits<typename T::underlying>::min();
  constexpr int64_t qmax = std::numeric_limits<typename T::underlying>::max();
  qvalue = static_cast<int64_t>(Round(value / scale + zero_point));
  qvalue = std::max<int64_t>(qvalue, qmin);
  qvalue = std::min<int64_t>(qvalue, qmax);
  return static_cast<T>(qvalue);
}

template <typename T, int precision>
void quantize_vec(double scale, int64_t zero_point, const float *src, T *dst, size_t count) {
  checkZeroPoint<typename T::underlying>("quantize_val", zero_point);
  for (int64_t i = 0; i < count; ++i) {
    dst[i] = quantize_val<T>(scale, zero_point, src[i]);
  }
}

// TODO combine this with quantize_val once the numerics for ARM are aligned with it
inline uint8_t quantize_val_arm(const float scale, const int32_t zero_point, const float value) {
  const int32_t qmin = std::numeric_limits<uint8_t>::min();
  const int32_t qmax = std::numeric_limits<uint8_t>::max();
  auto r = zero_point + static_cast<int32_t>(Round(value / scale));
  r = std::max(r, qmin);
  r = std::min(r, qmax);
  return static_cast<uint8_t>(r);
}

#ifdef __ARM_NEON__
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

template <typename T>
Tensor quantize_tensor(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point) {
  auto fn_name = "quantize_tensor";
  checkFloatCPUTensor(fn_name, rtensor);
  checkQuantizedCPUTensor<T>(fn_name, qtensor);
  checkZeroPoint<typename T::underlying>(fn_name, zero_point);
  TORCH_CHECK(rtensor.is_contiguous(), "Float tensor should be contiguous");
  const float* const rdata = rtensor.data_ptr<float>();
  // If QEngine is set to QNNPACK, use caffe2 specialized Int8Quantize implementation on ARM
#if defined(__ARM_NEON__)
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
    quantize_tensor_arm<T>(rdata, qtensor, rtensor.numel(), scale, zero_point);
    return qtensor;
  }
#endif
  auto qdata = qtensor.data_ptr<T>();
  auto numel = rtensor.numel();
  for (int i = 0; i < numel; ++i) {
    qdata[i] = quantize_val<T>(scale, zero_point, rdata[i]);
  }
  return qtensor;
}

template <typename T>
CAFFE2_API float dequantize_val(double scale, int64_t zero_point, T value) {
  // We need to convert the qint8 value to float to ensure the subtraction
  // subexpression returns a float
  return (static_cast<float>(value.val_) - zero_point) * scale;
}

template <typename T>
Tensor dequantize_tensor(Tensor qtensor, Tensor rtensor, double scale, int64_t zero_point) {
  auto fn_name = "dequantize_tensor";
  checkFloatCPUTensor(fn_name, rtensor);
  checkQuantizedCPUTensor<T>(fn_name, qtensor);
  checkZeroPoint<typename T::underlying>(fn_name, zero_point);
  const auto* qd = qtensor.data_ptr<T>();
  float* rd = rtensor.data_ptr<float>();
  auto numel = qtensor.numel();
  for (auto i = 0; i < numel; ++i) {
    rd[i] = dequantize_val<T>(scale, zero_point, qd[i]);
  }
  return rtensor;
}
#endif  // USE_FBGEMM

template <typename SRC_T, typename DST_T>
DST_T requantize_val(double src_scale, int64_t src_zero_point,
                     double dst_scale, int64_t dst_zero_point,
                     SRC_T src) {
  const auto dq = dequantize_val<SRC_T>(src_scale, src_zero_point, src);
  return quantize_val<DST_T>(dst_scale, dst_zero_point, dq);
}

template CAFFE2_API qint8 quantize_val<qint8>(double scale, int64_t zero_point, float value);
template CAFFE2_API quint8 quantize_val<quint8>(double scale, int64_t zero_point, float value);
template CAFFE2_API qint32 quantize_val<qint32>(double scale, int64_t zero_point, float value);
template CAFFE2_API void quantize_vec<c10::qint8>(double scale, int64_t zero_point, const float *src, c10::qint8 *dst, size_t count);
template CAFFE2_API void quantize_vec<c10::quint8>(double scale, int64_t zero_point, const float *src, c10::quint8 *dst, size_t count);
template CAFFE2_API void quantize_vec<c10::qint32, 32>(double scale, int64_t zero_point, const float *src, c10::qint32 *dst, size_t count);
template CAFFE2_API Tensor quantize_tensor<qint8>(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point);
template CAFFE2_API Tensor quantize_tensor<quint8>(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point);
template CAFFE2_API Tensor quantize_tensor<qint32>(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point);

template CAFFE2_API float dequantize_val<qint8>(double scale, int64_t zero_point, qint8 value);
template CAFFE2_API float dequantize_val<quint8>(double scale, int64_t zero_point, quint8 value);
template CAFFE2_API float dequantize_val<qint32>(double scale, int64_t zero_point, qint32 value);
template CAFFE2_API Tensor dequantize_tensor<qint8>(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point);
template CAFFE2_API Tensor dequantize_tensor<quint8>(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point);
template CAFFE2_API Tensor dequantize_tensor<qint32>(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point);

template CAFFE2_API qint8 requantize_val<qint8, qint8>(double, int64_t, double, int64_t, qint8);
template CAFFE2_API quint8 requantize_val<qint8, quint8>(double, int64_t, double, int64_t, qint8);
template CAFFE2_API qint32 requantize_val<qint8, qint32>(double, int64_t, double, int64_t, qint8);
template CAFFE2_API qint8 requantize_val<quint8, qint8>(double, int64_t, double, int64_t, quint8);
template CAFFE2_API quint8 requantize_val<quint8, quint8>(double, int64_t, double, int64_t, quint8);
template CAFFE2_API qint32 requantize_val<quint8, qint32>(double, int64_t, double, int64_t, quint8);
template CAFFE2_API qint8 requantize_val<qint32, qint8>(double, int64_t, double, int64_t, qint32);
template CAFFE2_API quint8 requantize_val<qint32, quint8>(double, int64_t, double, int64_t, qint32);
template CAFFE2_API qint32 requantize_val<qint32, qint32>(double, int64_t, double, int64_t, qint32);

// TODO: add fbgemm for per channel
template <typename T>
Tensor quantize_tensor_per_channel_affine(Tensor rtensor,
                                          Tensor qtensor,
                                          Tensor scales,
                                          Tensor zero_points,
                                          int64_t axis) {
  auto fn_name = "quantize_tensor_per_channel_affine";
  checkFloatCPUTensor(fn_name, rtensor);
  checkQuantizedCPUTensor<T>(fn_name, qtensor);
  checkZeroPoints<typename T::underlying>(fn_name, zero_points);
  TORCH_CHECK(0 <= axis && axis < rtensor.dim(), "Channel axis out of range in per channel affine quantization.");
  int64_t batches = size_to_dim_(axis, rtensor.sizes());
  int64_t elements_per_channel = size_from_dim_(axis + 1, rtensor.sizes());
  int64_t channel = rtensor.size(axis);
  auto scales_data = scales.data_ptr<double>();
  auto zero_points_data = zero_points.data_ptr<int64_t>();
  TORCH_CHECK(channel == int64_t(scales.numel()),
              "length of scales must equal to channel");
  TORCH_CHECK(channel == int64_t(zero_points.numel()),
              "length of zero_points must equal to channel");
  const float* rdata = rtensor.data_ptr<float>();
  auto qdata = qtensor.data_ptr<T>();
  for (auto b = 0; b < batches; ++b) {
    for (auto c = 0; c < channel; ++c) {
      for (auto e = 0; e < elements_per_channel; ++e) {
        auto i = b * channel * elements_per_channel + c * elements_per_channel + e;
        qdata[i] = quantize_val<T>(scales_data[c], zero_points_data[c], rdata[i]);
      }
    }
  }
  return qtensor;
}

template <typename T>
Tensor dequantize_tensor_per_channel_affine(Tensor qtensor,
                                            Tensor rtensor,
                                            Tensor scales,
                                            Tensor zero_points,
                                            int64_t axis) {
  auto fn_name = "dequantize_tensor_per_channel_affine";
  checkFloatCPUTensor(fn_name, rtensor);
  checkQuantizedCPUTensor<T>(fn_name, qtensor);
  checkZeroPoints<typename T::underlying>(fn_name, zero_points);
  TORCH_CHECK(0 <= axis && axis < qtensor.dim(),
              "Channel axis out of range in per channel affine dequantization.");
  int64_t batches = size_to_dim_(axis, rtensor.sizes());
  int64_t elements_per_channel = size_from_dim_(axis + 1, rtensor.sizes());
  int64_t channel = rtensor.size(axis);
  auto scales_data = scales.data_ptr<double>();
  auto zero_points_data = zero_points.data_ptr<int64_t>();
  TORCH_CHECK(channel == int64_t(scales.numel()),
              "length of scales must equal to channel");
  TORCH_CHECK(channel == int64_t(zero_points.numel()),
              "length of zero_points must equal to channel");
  const auto* qd = qtensor.data_ptr<T>();
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
  return rtensor;
}

QuantizerPtr make_per_tensor_affine_quantizer(
    double scale,
    int64_t zero_point,
    ScalarType scalar_type) {
  return c10::make_intrusive<PerTensorAffineQuantizer>(scalar_type,
      scale, zero_point);
}

QuantizerPtr make_per_channel_affine_quantizer(
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType scalar_type) {
  TORCH_CHECK(scales.dim() == 1, "scale tensor must have dimension 1");
  TORCH_CHECK(
      zero_points.dim() == 1, "zero_points tensor must have dimension 1");
  TORCH_CHECK(
      scales.numel() == zero_points.numel(),
      "number of elements in scales and zero_points must match");
  TORCH_CHECK(
      isFloatingType(scales.scalar_type()),
      "scale tensor must be floating point");
  TORCH_CHECK(
      isIntegralType(zero_points.scalar_type(), false /*includeBool*/),
      "zero_points tensor must have integral type");
  Tensor scales_double = scales.to(kDouble).contiguous();
  Tensor zero_points_int64 = zero_points.to(kLong).contiguous();
  return c10::make_intrusive<PerChannelAffineQuantizer>(scalar_type,
                                                        scales_double, zero_points_int64,
                                                        axis);
}

QTensorImpl* get_qtensorimpl(const Tensor& self) {
  TORCH_CHECK(
      !self.requires_grad(),
      "quantized tensors do not support autograd");
  TORCH_INTERNAL_ASSERT(self.is_quantized(), "get_qtensorimpl: not a quantized tensor");
  return static_cast<QTensorImpl*>(self.unsafeGetTensorImpl());
}

#ifdef USE_PYTORCH_QNNPACK

// QNNPACK can access up to 8 bytes beyond the beginning of the tensor's storage
// boundary which does trigger ASAN, and can result in a segfault if the memory falls
// on a different page out of the process's address space.
// Here we define a custom allocator that allocates the extra storage required to keep
// this behavior safe.  This same allocator can be used for FBGEMM as well.

using QAllocator = native::GuardingAllocator<8u, 0u>;

#endif

inline Tensor new_qtensor_cpu(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer) {
  AT_ASSERT(options.device().is_cpu());

  auto memory_format = options.memory_format_opt().value_or(MemoryFormat::Contiguous);

  at::Allocator* allocator = at::getCPUAllocator();

#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
    static QAllocator qallocator;
    allocator = &qallocator;
  }
#endif

  native::check_size_nonnegative(sizes);
  int64_t nelements = at::prod_intlist(sizes);
  auto dtype = options.dtype();
  TORCH_CHECK(isQIntType(typeMetaToScalarType(dtype)),
           "ScalarType is not supported in new_qtensor_cpu.");
  auto storage = c10::make_intrusive<StorageImpl>(
      dtype,
      nelements,
      allocator->allocate(nelements * dtype.itemsize()),
      allocator,
      /*resizable=*/true);
  auto tensor = detail::make_tensor<QTensorImpl>(
      storage, at::DispatchKeySet(at::DispatchKey::QuantizedCPUTensorId), quantizer);
  get_qtensorimpl(tensor)->set_sizes_contiguous(sizes);
  get_qtensorimpl(tensor)->empty_tensor_restride(memory_format);
  return tensor;
}

Tensor PerTensorAffineQuantizer::quantize(Tensor rtensor) {
  TORCH_CHECK(
      rtensor.scalar_type() == kFloat,
      "quantize only works on Float Tensor.");
  TORCH_CHECK(
      rtensor.device() == kCPU,
      "quantize only works for CPU backend right now.");
  // Here we need a std::intrusive_ptr<Quantizer>.. but actually "this" is the
  // quantizer that can be reused, so I'm using intrusive_from_this here
  Tensor qtensor = new_qtensor_cpu(
      rtensor.sizes(),
      rtensor.options().dtype(scalar_type_),
      intrusive_from_this());

  rtensor = rtensor.contiguous();
  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), "quantize_tensor", [&]() {
    qtensor = quantize_tensor<scalar_t>(rtensor, qtensor, scale_, zero_point_);
  });
  return qtensor;
}

Tensor PerTensorAffineQuantizer::dequantize(Tensor qtensor) {
  TORCH_CHECK(qtensor.is_quantized(),
           "dequantize is only supported in quantized Tensor.");
  TORCH_CHECK(
      qtensor.device() == kCPU,
      "dequantize only works for CPU backend right now.");
  Tensor rtensor = at::empty(qtensor.sizes(), qtensor.options().dtype(at::kFloat));
  qtensor = qtensor.contiguous();

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), "dequantize_tensor", [&]() {
    rtensor = dequantize_tensor<scalar_t>(qtensor, rtensor, scale_, zero_point_);
  });

  return rtensor;
}

Tensor PerChannelAffineQuantizer::quantize(Tensor rtensor) {
  TORCH_CHECK(
      rtensor.scalar_type() == kFloat,
      "quantize only works on Float Tensor.");
  TORCH_CHECK(
      rtensor.device() == kCPU,
      "quantize only works for CPU backend right now.");
  // Here we need a std::intrusive_ptr<Quantizer>.. but actually "this" is the
  // quantizer that can be reused, so I'm using intrusive_from_this here
  Tensor qtensor = new_qtensor_cpu(
      rtensor.sizes(),
      rtensor.options().dtype(scalar_type_),
      intrusive_from_this());

  rtensor = rtensor.contiguous();
  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(),
                         "quantize_tensor_per_channel_affine",
                         [&]() {
    qtensor = quantize_tensor_per_channel_affine<scalar_t>(
        rtensor, qtensor, scales_, zero_points_, axis_);
  });
  return qtensor;
}

Tensor PerChannelAffineQuantizer::dequantize(Tensor qtensor) {
  TORCH_CHECK(qtensor.is_quantized(),
           "dequantize is only supported in quantized Tensor.");
  TORCH_CHECK(
      qtensor.device() == kCPU,
      "dequantize only works for CPU backend right now.");
  Tensor rtensor = at::empty(qtensor.sizes(), qtensor.options().dtype(at::kFloat));
  qtensor = qtensor.contiguous();

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(),
                         "dequantize_tensor_per_channel_affine",
                         [&]() {
    rtensor = dequantize_tensor_per_channel_affine<scalar_t>(
        qtensor, rtensor, scales_, zero_points_, axis_);
  });

  return rtensor;
}

Quantizer::~Quantizer() {}

} // namespace at
