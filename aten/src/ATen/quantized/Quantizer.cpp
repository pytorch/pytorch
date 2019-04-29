#include <ATen/quantized/Quantizer.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Type.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/quantized/QTensorImpl.h>

namespace at {

#define AT_DISPATCH_ALL_QINT_TYPES(TYPE, NAME, ...)

QuantizerPtr make_per_tensor_affine_quantizer(
    double scale,
    int64_t zero_point,
    optional<ScalarType> scalar_type_opt = c10::nullopt) {
  auto scalar_type = scalar_type_opt;
  if (scalar_type == c10::nullopt) {
    scalar_type = c10::kQInt8;
  }
  return c10::make_intrusive<PerTensorAffineQuantizer>(scalar_type.value(),
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
  AT_CHECK(isQIntType(typeMetaToScalarType(dtype)),
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
  IntArrayRef sizes = rtensor.sizes();
  // Here we need a std::intrusive_ptr<Quantizer>.. but actually "this" is the
  // quantizer that can be reused, so I'm using intrusive_from_this here
  AT_CHECK(
      rtensor.options().device() == kCPU,
      "quantize only works for CPU backend right now.");
  Tensor qtensor = new_qtensor_cpu(
      sizes,
      rtensor.options().dtype(scalar_type_),
      intrusive_from_this());

  rtensor = rtensor.contiguous();
#ifdef USE_FBGEMM
  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), "quantize_fbgemm", [&]() {
    qtensor = quantize_fbgemm<scalar_t>(rtensor, qtensor, scale_, zero_point_);
  });
#else
  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), "quantize_naive", [&]() {
    qtensor = quantize_naive<scalar_t>(qtensor, qtensor, scale_, zero_point_);
  });
#endif
  return qtensor;
}

Tensor PerTensorAffineQuantizer::dequantize(Tensor qtensor) {
  std::vector<int64_t> sizes = qtensor.sizes().vec();
  at::TensorOptions options = qtensor.options().dtype(at::kFloat);

  Tensor rtensor = at::empty(sizes, options);
  qtensor = qtensor.contiguous();

#ifdef USE_FBGEMM
  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), "dequantize_fbgemm", [&]() {
    rtensor = dequantize_fbgemm<scalar_t>(qtensor, rtensor, scale_, zero_point_);
  });
#else
  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), "dequantize_naive", [&]() {
    rtensor = dequantize_naive<scalar_t>(qtensor, rtensor, scale_, zero_point_);
  });
#endif

  return rtensor;
}

Quantizer::~Quantizer() {}

} // namespace at
