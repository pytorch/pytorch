#include <ATen/quantized/Quantizer.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Type.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/quantized/QTensorImpl.h>

namespace at {

QuantizerPtr make_per_tensor_affine_quantizer(
    double scale,
    int64_t zero_point) {
  return c10::make_intrusive<PerTensorAffineQuantizer>(
      static_cast<float>(scale), static_cast<uint8_t>(zero_point));
}

QTensorImpl* get_qtensorimpl(const QTensor& self) {
  // TODO: remove this when Variable and Tensor are merged
  AT_ASSERTM(
      !self.is_variable(),
      "_internal_get_QTensorImpl: should not be a variable");
  // TODO: uncomment after is_quantized() is implemented
  // AT_ASSERTM(self.is_quantized(), "_internal_get_QTensorImpl: not a quantized
  // tensor");
  return static_cast<QTensorImpl*>(self.unsafeGetTensorImpl());
}

inline QTensor new_qtensor_cpu(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer) {
  AT_ASSERT(options.device().is_cpu());

  native::check_size_nonnegative(sizes);
  auto* allocator = at::getCPUAllocator();
  int64_t nelements = at::prod_intlist(sizes);
  auto dtype = options.dtype();
  AT_CHECK(isQIntType(typeMetaToScalarType(dtype)), "ScalarType not supported for QTensor in new_qtensor_cpu.");
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
  const int32_t qmin = std::numeric_limits<uint8_t>::min();
  const int32_t qmax = std::numeric_limits<uint8_t>::max();

  // std::nearbyint results in nearest integer value according to the current
  // rounding mode and the default rounding mode is rounds to even in half-way
  // cases in most popular processor architectures like x86 and ARM. This is
  // typically faster than an alternatives like std::round that rounds half-way
  // cases away from zero, and can be consistent with SIMD implementations for
  // example in x86 using _mm512_cvtps_epi32 or mm512_round_ps with
  // _MM_FROUND_CUR_DIRECTION option that also follow the current rounding mode.
  int32_t r = zero_point + static_cast<int32_t>(std::nearbyint(value / scale));
  r = std::max(r, qmin);
  r = std::min(r, qmax);
  return static_cast<qint8>(r);
}

QTensor PerTensorAffineQuantizer::quantize(RealTensor tensor) {
  IntArrayRef sizes = tensor.sizes();
  // Here we need a std::intrusive_ptr<Quantizer>.. but actually "this" is the
  // quantizer that can be reused, so I'm using intrusive_from_this here
  AT_CHECK(
      tensor.options().device() == kCPU,
      "quantize only works for CPU backend right now.");
  QTensor qv = new_qtensor_cpu(
      sizes,
      tensor.options().dtype(at::kQInt8),
      intrusive_from_this());
  auto qvd = qv.data<qint8>();
  tensor.contiguous();
  const float* svd = tensor.data<float>();
  for (int i = 0; i < tensor.numel(); ++i) {
    qvd[i] = quantize_uint8(scale_, zero_point_, svd[i]);
  }
  return qv;
}

RealTensor PerTensorAffineQuantizer::dequantize(QTensor tensor) {
  std::vector<int64_t> sizes = tensor.sizes().vec();
  at::TensorOptions real_options = tensor.options().dtype(at::kFloat);

  RealTensor rv = at::empty(sizes, real_options);
  tensor.contiguous();
  const auto* qvd = tensor.data<qint8>();
  float* rvd = rv.data<float>();
  for (auto i = 0; i < tensor.numel(); ++i) {
    // We need to convert the qint8 value to float to ensure the subtraction
    // subexpression returns a float
    rvd[i] = (static_cast<float>(qvd[i].val_) - zero_point_) * scale_;
  }
  return rv;
}

Quantizer::~Quantizer() {}

} // namespace at
