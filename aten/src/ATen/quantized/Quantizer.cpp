#include <ATen/quantized/Quantizer.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/quantized/affine_quantizer.h>
#include <ATen/quantized/QTensorImpl.h>
#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <cmath>
#include <typeinfo>

namespace at {

// Note: this is not a native function as Quantizer is not exposed to python yet
QuantizerPtr Tensor::quantizer() const {
  // This is a terrible hack to emulate what VariableType is doing
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  return get_qtensorimpl(*this)->quantizer();
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

inline Tensor new_qtensor(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer) {
  auto memory_format = options.memory_format_opt().value_or(MemoryFormat::Contiguous);
  at::Allocator* allocator = GetAllocator(options.device().type());

#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
    allocator = c10::GetDefaultMobileCPUAllocator();
  }
#endif

  at::DispatchKey tensorDispatchKey = options.computeDispatchKey();
  native::check_size_nonnegative(sizes);
  int64_t nelements = at::prod_intlist(sizes);
  auto dtype = options.dtype();
  TORCH_CHECK(
      isQIntType(typeMetaToScalarType(dtype)),
      "ScalarType is not supported in new_qtensor.");
  auto storage = c10::make_intrusive<StorageImpl>(
      dtype,
      nelements,
      allocator->allocate(nelements * dtype.itemsize()),
      allocator,
      /*resizable=*/true);
  auto tensor = detail::make_tensor<QTensorImpl>(
      storage, at::DispatchKeySet(tensorDispatchKey), quantizer);
  get_qtensorimpl(tensor)->set_sizes_contiguous(sizes);
  get_qtensorimpl(tensor)->empty_tensor_restride(memory_format);
  return tensor;
}

Tensor PerTensorAffineQuantizer::quantize(Tensor rtensor) {
  TORCH_CHECK(
      rtensor.scalar_type() == kFloat, "quantize only works on Float Tensor.");
  // Here we need a std::intrusive_ptr<Quantizer>.. but actually "this" is the
  // quantizer that can be reused, so I'm using intrusive_from_this here
  Tensor qtensor = new_qtensor(
      rtensor.sizes(),
      rtensor.options().dtype(scalar_type_),
      intrusive_from_this());

  rtensor = rtensor.contiguous();
  native::quantize_tensor_per_tensor_affine(
      rtensor, qtensor, scale_, zero_point_);
  return qtensor;
}

Tensor PerTensorAffineQuantizer::dequantize(Tensor qtensor) {
  Tensor rtensor = at::empty(qtensor.sizes(), qtensor.options().dtype(at::kFloat));
  qtensor = qtensor.contiguous();
  native::dequantize_tensor_per_tensor_affine(
      qtensor, rtensor, scale_, zero_point_);
  return rtensor;
}

Tensor PerChannelAffineQuantizer::quantize(Tensor rtensor) {
  // Here we need a std::intrusive_ptr<Quantizer>.. but actually "this" is the
  // quantizer that can be reused, so I'm using intrusive_from_this here
  Tensor qtensor = new_qtensor(
      rtensor.sizes(),
      rtensor.options().dtype(scalar_type_),
      intrusive_from_this());
  rtensor = rtensor.contiguous();
  native::quantize_tensor_per_channel_affine(
      rtensor, qtensor, scales_, zero_points_, axis_);
  return qtensor;
}

Tensor PerChannelAffineQuantizer::dequantize(Tensor qtensor) {
  Tensor rtensor = at::empty(qtensor.sizes(), qtensor.options().dtype(at::kFloat));
  qtensor = qtensor.contiguous();
  native::dequantize_tensor_per_channel_affine(
      qtensor, rtensor, scales_, zero_points_, axis_);
  return rtensor;
}

Quantizer::~Quantizer() {}

} // namespace at
