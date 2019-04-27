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

Tensor PerTensorAffineQuantizer::quantize(Tensor tensor) {
  IntArrayRef sizes = tensor.sizes();
  // Here we need a std::intrusive_ptr<Quantizer>.. but actually "this" is the
  // quantizer that can be reused, so I'm using intrusive_from_this here
  AT_CHECK(
      tensor.options().device() == kCPU,
      "quantize only works for CPU backend right now.");
  Tensor qv = new_qtensor_cpu(
      sizes,
      tensor.options().dtype(scalar_type_),
      intrusive_from_this());

  tensor = tensor.contiguous();
#ifdef USE_FBGEMM
  AT_DISPATCH_QINT_TYPES(qv.scalar_type(), "quantize_fbgemm", [&]() {
    qv = quantize_fbgemm<scalar_t>(tensor, qv, scale_, zero_point_);
  });
#else
  AT_DISPATCH_QINT_TYPES(qv.scalar_type(), "quantize_naive", [&]() {
    qv = quantize_naive<scalar_t>(tensor, qv, scale_, zero_point_);
  });
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
