#include <ATen/ATen.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/Allocator.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/core/Tensor.h>

#include <typeinfo>

#ifdef USE_FBGEMM
#include <fbgemm/QuantUtils.h>
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
void checkZeroPoints(std::string fn_name, std::vector<int64_t> zero_points) {
  for (size_t i = 0; i < zero_points.size(); ++i) {
    TORCH_CHECK(zero_points[i] <= std::numeric_limits<T>::max(),
                fn_name,
                "zero_point",
                i,
                "is out of range.");
    TORCH_CHECK(zero_points[i] >= std::numeric_limits<T>::min(),
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

// TODO: dequantize_val?

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
  fbgemm::Quantize<typename T::underlying>(/*src=*/rd,
                             /*dst=*/qd,
                             /*len=*/rtensor.numel(),
                             /*qparams=*/qparams);
  return qtensor;
}

template <typename T>
inline float dequantize_val(double scale, int64_t zero_point, T value) {
  fbgemm::TensorQuantizationParams qparams = {
    .scale = static_cast<float>(scale),
    .zero_point = static_cast<int32_t>(zero_point)
  };
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
  fbgemm::Dequantize<typename T::underlying>(/*src=*/qd,
                              /*dst=*/rd,
                              /*len=*/qtensor.numel(),
                              /*qparams=*/qparams);
  return rtensor;
}
#else  // USE_FBGEMM

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
  checkZeroPoint<typename T::underlying>("quantize_val", zero_point);
  qvalue = static_cast<int64_t>(std::nearbyint(value / scale + zero_point));
  qvalue = std::max<int64_t>(qvalue, qmin);
  qvalue = std::min<int64_t>(qvalue, qmax);
  return static_cast<T>(qvalue);
}

template <typename T, int precision>
void quantize_vec(double scale, int64_t zero_point, const float *src, T *dst, size_t count) {
  for (int64_t i = 0; i < count; ++i) {
    dst[i] = quantize_val<T>(scale, zero_point, src[i]);
  }
}

template <typename T>
Tensor quantize_tensor(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point) {
  auto fn_name = "quantize_tensor";
  checkFloatCPUTensor(fn_name, rtensor);
  checkQuantizedCPUTensor<T>(fn_name, qtensor);
  checkZeroPoint<typename T::underlying>(fn_name, zero_point);
  const float* rdata = rtensor.data_ptr<float>();
  auto qdata = qtensor.data_ptr<T>();
  for (int i = 0; i < rtensor.numel(); ++i) {
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
  for (auto i = 0; i < qtensor.numel(); ++i) {
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
                                          const std::vector<double>& scales,
                                          const std::vector<int64_t>& zero_points,
                                          int64_t axis) {
  auto fn_name = "quantize_tensor_per_channel_affine";
  checkFloatCPUTensor(fn_name, rtensor);
  checkQuantizedCPUTensor<T>(fn_name, qtensor);
  checkZeroPoints<typename T::underlying>(fn_name, zero_points);
  TORCH_CHECK(0 <= axis && axis < rtensor.dim(), "Channel axis out of range in per channel affine quantization.");
  int64_t batches = size_to_dim_(axis, rtensor.sizes());
  int64_t elements_per_channel = size_from_dim_(axis + 1, rtensor.sizes());
  int64_t channel = rtensor.size(axis);
  TORCH_CHECK(channel == int64_t(scales.size()),
              "length of scales must equal to channel");
  TORCH_CHECK(channel == int64_t(zero_points.size()),
              "length of zero_points must equal to channel");
  const float* rdata = rtensor.data_ptr<float>();
  auto qdata = qtensor.data_ptr<T>();
  for (auto b = 0; b < batches; ++b) {
    for (auto c = 0; c < channel; ++c) {
      for (auto e = 0; e < elements_per_channel; ++e) {
        auto i = b * channel * elements_per_channel + c * elements_per_channel + e;
        qdata[i] = quantize_val<T>(scales[c], zero_points[c], rdata[i]);
      }
    }
  }
  return qtensor;
}

template <typename T>
Tensor dequantize_tensor_per_channel_affine(Tensor qtensor,
                                            Tensor rtensor,
                                            const std::vector<double>& scales,
                                            const std::vector<int64_t>& zero_points,
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
  TORCH_CHECK(channel == int64_t(scales.size()),
              "length of scales must equal to channel");
  TORCH_CHECK(channel == int64_t(zero_points.size()),
              "length of zero_points must equal to channel");
  const auto* qd = qtensor.data_ptr<T>();
  float* rd = rtensor.data_ptr<float>();
  for (auto b = 0; b < batches; ++b) {
    for (auto c = 0; c < channel; ++c) {
      for (auto e = 0; e < elements_per_channel; ++e) {
        auto i = b * channel * elements_per_channel + c * elements_per_channel + e;
        // We need to convert the qint8 value to float to ensure the subtraction
        // subexpression returns a float
        rd[i] = (static_cast<float>(qd[i].val_) - zero_points[c]) * scales[c];
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
    const std::vector<double>& scales,
    const std::vector<int64_t>& zero_points,
    int64_t axis,
    ScalarType scalar_type) {
  return c10::make_intrusive<PerChannelAffineQuantizer>(scalar_type,
                                                        scales, zero_points, axis);
}

QTensorImpl* get_qtensorimpl(const Tensor& self) {
  // TODO: remove this when Variable and Tensor are merged
  TORCH_INTERNAL_ASSERT(
      !self.is_variable(),
      "_internal_get_QTensorImpl: should not be a variable");
  TORCH_INTERNAL_ASSERT(self.is_quantized(), "get_qtensorimpl: not a quantized tensor");
  return static_cast<QTensorImpl*>(self.unsafeGetTensorImpl());
}

inline Tensor new_qtensor_cpu(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer,
    MemoryFormat memory_format=MemoryFormat::Contiguous) {
  AT_ASSERT(options.device().is_cpu());

  native::check_size_nonnegative(sizes);
  auto* allocator = at::getCPUAllocator();
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
      storage, at::TensorTypeSet(at::TensorTypeId::QuantizedCPUTensorId), quantizer);
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
