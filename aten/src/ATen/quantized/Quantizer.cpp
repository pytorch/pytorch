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

QuantizerPtr make_per_tensor_affine_quantizer(
    double scale,
    int64_t zero_point) {
  return c10::make_intrusive<PerTensorAffineQuantizer>(
      static_cast<float>(scale), static_cast<uint8_t>(zero_point));
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

qint8 quantize_uint8(float scale, uint8_t zero_point, float value) {
  // Internally, fbgemm::Quantize uses std::nearbyint.
  // std::nearbyint results in nearest integer value according to the current
  // rounding mode and the default rounding mode is rounds to even in half-way
  // cases in most popular processor architectures like x86 and ARM. This is
  // typically faster than an alternatives like std::round that rounds half-way
  // cases away from zero, and can be consistent with SIMD implementations for
  // example in x86 using _mm512_cvtps_epi32 or mm512_round_ps with
  // _MM_FROUND_CUR_DIRECTION option that also follow the current rounding mode.
  int32_t qvalue;
#ifdef USE_FBGEMM
  qvalue = fbgemm::Quantize<uint8_t>(value, zero_point, scale,
                                     /*result_precision=*/8);
#else
  constexpr int32_t qmin = std::numeric_limits<uint8_t>::min();
  constexpr int32_t qmax = std::numeric_limits<uint8_t>::max();
  qvalue = static_cast<int32_t>(std::nearbyint(value / scale + zero_point));
  qvalue = std::max(qvalue, qmin);
  qvalue = std::min(qvalue, qmax);
#endif
  return static_cast<qint8>(qvalue);
}

Tensor PerTensorAffineQuantizer::quantize(Tensor tensor) {
  IntArrayRef sizes = tensor.sizes();
  // Here we need a std::intrusive_ptr<Quantizer>.. but actually "this" is the
  // quantizer that can be reused, so I'm using intrusive_from_this here
  TORCH_CHECK(
      tensor.options().device() == kCPU,
      "quantize only works for CPU backend right now.");
  Tensor qv = new_qtensor_cpu(
      sizes,
      tensor.options().dtype(at::kQInt8),
      intrusive_from_this());

  tensor = tensor.contiguous();
  const float* svd = tensor.data<float>();

#ifdef USE_FBGEMM
  auto qvd = reinterpret_cast<uint8_t*>(qv.data<qint8>());
  fbgemm::TensorQuantizationParams qparams;
  qparams.scale = scale_;
  qparams.zero_point = zero_point_;
  qparams.precision = 8;
  fbgemm::Quantize<uint8_t>(/*src=*/svd,
                            /*dst=*/qvd,
                            /*len=*/tensor.numel(),
                            /*qparams=*/qparams);
#else
  auto qvd = qv.data<qint8>();
  for (int i = 0; i < tensor.numel(); ++i) {
    qvd[i] = quantize_uint8(scale_, zero_point_, svd[i]);
  }
#endif
  return qv;
}

Tensor PerTensorAffineQuantizer::dequantize(Tensor tensor) {
  std::vector<int64_t> sizes = tensor.sizes().vec();
  at::TensorOptions options = tensor.options().dtype(at::kFloat);

  Tensor rv = at::empty(sizes, options);
  float* rvd = rv.data<float>();
  tensor = tensor.contiguous();

#ifdef USE_FBGEMM
  const auto* qvd = reinterpret_cast<const uint8_t*>(tensor.data<qint8>());
  fbgemm::TensorQuantizationParams qparams;
  qparams.scale = scale_;
  qparams.zero_point = zero_point_;
  qparams.precision = 8;
  fbgemm::Dequantize<uint8_t>(/*src=*/qvd,
                              /*dst=*/rvd,
                              /*len=*/tensor.numel(),
                              /*qparams=*/qparams);
#else
  const auto* qvd = tensor.data<qint8>();
  for (auto i = 0; i < tensor.numel(); ++i) {
    // We need to convert the qint8 value to float to ensure the subtraction
    // subexpression returns a float
    rvd[i] = (static_cast<float>(qvd[i].val_) - zero_point_) * scale_;
  }
#endif

  return rv;
}

Quantizer::~Quantizer() {}

} // namespace at
