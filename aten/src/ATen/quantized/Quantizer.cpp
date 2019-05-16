#include <ATen/quantized/Quantizer.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Type.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/quantized/QTensorImpl.h>

#ifdef USE_FBGEMM
#include <fbgemm/QuantUtils.h>
#endif

namespace at {

void checkFloatCPUTensor(std::string fn_name, Tensor t) {
  TORCH_CHECK(
      t.scalar_type() == kFloat,
      fn_name,
      "expects a Float Tensor.");
  TORCH_CHECK(
      t.device() == kCPU,
      fn_name,
      "expects a CPU Tensor.");
}

template <typename T>
void checkQuantizedCPUTensor(std::string fn_name, Tensor t) {
  TORCH_CHECK(t.is_quantized(),
           fn_name,
           "expects a quantized Tensor.");
  TORCH_CHECK(t.scalar_type() == caffe2::TypeMeta::Make<T>(),
           fn_name,
           "expects a ",
           caffe2::TypeMeta::Make<T>(),
           "Tensor");
  TORCH_CHECK(t.device() == kCPU,
           fn_name,
           "expects a CPU quantized Tensor");
}

template <typename T>
void checkZeroPoint(std::string fn_name, int32_t zero_point) {
  TORCH_CHECK(zero_point <= std::numeric_limits<T>::max(),
           fn_name,
           "zero_point is out of range.");
  TORCH_CHECK(zero_point >= std::numeric_limits<T>::min(),
           fn_name,
           "zero_point is out of range.");
}

#ifdef USE_FBGEMM
template <typename T>
T quantize_val(float scale, int32_t zero_point, float value) {
  // Internally, fbgemm::Quantize uses std::nearbyint.
  // std::nearbyint results in nearest integer value according to the current
  // rounding mode and the default rounding mode is rounds to even in half-way
  // cases in most popular processor architectures like x86 and ARM. This is
  // typically faster than an alternatives like std::round that rounds half-way
  // cases away from zero, and can be consistent with SIMD implementations for
  // example in x86 using _mm512_cvtps_epi32 or mm512_round_ps with
  // _MM_FROUND_CUR_DIRECTION option that also follow the current rounding mode.
  int32_t qvalue;
  qvalue = fbgemm::Quantize<typename T::underlying>(value, zero_point, scale,
                                                    /*result_precision=*/std::numeric_limits<typename T::underlying>::digits);
  return static_cast<T>(qvalue);
}

template <typename T>
Tensor quantize_tensor(Tensor rtensor, Tensor qtensor, float scale, int32_t zero_point) {
  auto fn_name = "quantize_tensor";
  checkFloatCPUTensor(fn_name, rtensor);
  checkQuantizedCPUTensor<T>(fn_name, qtensor);
  checkZeroPoint<typename T::underlying>(fn_name, zero_point);
  const float* rd = rtensor.data<float>();
  auto qd = reinterpret_cast<typename T::underlying*>(qtensor.data<T>());
  fbgemm::TensorQuantizationParams qparams;
  qparams.scale = scale;
  qparams.zero_point = zero_point;
  qparams.precision = std::numeric_limits<typename T::underlying>::digits;
  fbgemm::Quantize<typename T::underlying>(/*src=*/rd,
                             /*dst=*/qd,
                             /*len=*/rtensor.numel(),
                             /*qparams=*/qparams);
  return qtensor;
}

template <typename T>
Tensor dequantize_tensor(Tensor qtensor, Tensor rtensor, float scale, int32_t zero_point) {
  auto fn_name = "dequantize_tensor";
  checkFloatCPUTensor(fn_name, rtensor);
  checkQuantizedCPUTensor<T>(fn_name, qtensor);
  checkZeroPoint<typename T::underlying>(fn_name, zero_point);
  const auto* qd = reinterpret_cast<const typename T::underlying*>(qtensor.data<T>());
  fbgemm::TensorQuantizationParams qparams;
  qparams.scale = scale;
  qparams.zero_point = zero_point;
  qparams.precision = std::numeric_limits<typename T::underlying>::digits;
  float* rd = rtensor.data<float>();
  fbgemm::Dequantize<typename T::underlying>(/*src=*/qd,
                              /*dst=*/rd,
                              /*len=*/qtensor.numel(),
                              /*qparams=*/qparams);
  return rtensor;
}
#else

template <typename T>
T quantize_val(float scale, int32_t zero_point, float value) {
  // std::nearbyint results in nearest integer value according to the current
  // rounding mode and the default rounding mode is rounds to even in half-way
  // cases in most popular processor architectures like x86 and ARM. This is
  // typically faster than an alternatives like std::round that rounds half-way
  // cases away from zero, and can be consistent with SIMD implementations for
  // example in x86 using _mm512_cvtps_epi32 or mm512_round_ps with
  // _MM_FROUND_CUR_DIRECTION option that also follow the current rounding mode.
  int32_t qvalue;
  constexpr int32_t qmin = std::numeric_limits<typename T::underlying>::min();
  constexpr int32_t qmax = std::numeric_limits<typename T::underlying>::max();
  checkZeroPoint<typename T::underlying>("quantize_val", zero_point);
  qvalue = static_cast<int32_t>(std::nearbyint(value / scale + zero_point));
  qvalue = std::max(qvalue, qmin);
  qvalue = std::min(qvalue, qmax);
  return static_cast<T>(qvalue);
}

template <typename T>
Tensor quantize_tensor(Tensor rtensor, Tensor qtensor, float scale, int32_t zero_point) {
  auto fn_name = "quantize_tensor";
  checkFloatCPUTensor(fn_name, rtensor);
  checkQuantizedCPUTensor<T>(fn_name, qtensor);
  checkZeroPoint<typename T::underlying>(fn_name, zero_point);
  const float* rdata = rtensor.data<float>();
  auto qdata = qtensor.data<T>();
  for (int i = 0; i < rtensor.numel(); ++i) {
    qdata[i] = quantize_val<T>(scale, zero_point, rdata[i]);
  }
  return qtensor;
}

template <typename T>
Tensor dequantize_tensor(Tensor qtensor, Tensor rtensor, float scale, int32_t zero_point) {
  auto fn_name = "dequantize_tensor";
  checkFloatCPUTensor(fn_name, rtensor);
  checkQuantizedCPUTensor<T>(fn_name, qtensor);
  checkZeroPoint<typename T::underlying>(fn_name, zero_point);
  const auto* qd = qtensor.data<T>();
  float* rd = rtensor.data<float>();
  for (auto i = 0; i < qtensor.numel(); ++i) {
    // We need to convert the qint8 value to float to ensure the subtraction
    // subexpression returns a float
    rd[i] = (static_cast<float>(qd[i].val_) - zero_point) * scale;
  }
  return rtensor;
}
#endif

template CAFFE2_API qint8 quantize_val<qint8>(float scale, int32_t zero_point, float value);
template CAFFE2_API qint32 quantize_val<qint32>(float scale, int32_t zero_point, float value);
template CAFFE2_API Tensor quantize_tensor<qint8>(Tensor rtensor, Tensor qtensor, float scale, int32_t zero_point);
template CAFFE2_API Tensor quantize_tensor<qint32>(Tensor rtensor, Tensor qtensor, float scale, int32_t zero_point);
template CAFFE2_API Tensor dequantize_tensor<qint8>(Tensor rtensor, Tensor qtensor, float scale, int32_t zero_point);
template CAFFE2_API Tensor dequantize_tensor<qint32>(Tensor rtensor, Tensor qtensor, float scale, int32_t zero_point);

QuantizerPtr make_per_tensor_affine_quantizer(
    double scale,
    int64_t zero_point,
    ScalarType scalar_type) {
  return c10::make_intrusive<PerTensorAffineQuantizer>(scalar_type,
      static_cast<float>(scale), static_cast<int32_t>(zero_point));
}

QTensorImpl* get_qtensorimpl(const Tensor& self) {
  // TODO: remove this when Variable and Tensor are merged
  AT_ASSERTM(
      !self.is_variable(),
      "_internal_get_QTensorImpl: should not be a variable");
  // TODO: uncomment after is_quantized() is implemented
  // AT_ASSERTM(self.is_quantized(), "_internal_get_QTensorImpl: not a quantized
  // tensor");
  return static_cast<QTensorImpl*>(self.unsafeGetTensorImpl());
}

inline Tensor new_qtensor_cpu(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer) {
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
      storage, at::QuantizedCPUTensorId(), quantizer);
  get_qtensorimpl(tensor)->set_sizes_contiguous(sizes);
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

Quantizer::~Quantizer() {}

} // namespace at
