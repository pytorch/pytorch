#include <ATen/ArrayRef.h>
#include <ATen/ATen.h>
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/Dispatch.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/CPUAllocator.h>
#include <c10/util/accumulate.h>

#include <cmath>
#include <typeinfo>

namespace at {

namespace {

  void checkPerChannelParamDims(const Tensor& scales, const Tensor& zero_points) {
    TORCH_CHECK(scales.dim() == 1, "scale tensor must have dimension 1");
    TORCH_CHECK(
        zero_points.dim() == 1, "zero_points tensor must have dimension 1");
    TORCH_CHECK(
        scales.numel() == zero_points.numel(),
        "number of elements in scales and zero_points must match");
  }

} // anonymous namespace

// Note: this is not a native function as Quantizer is not exposed to python yet
QuantizerPtr TensorBase::quantizer() const {
  // This is a terrible hack to emulate what VariableType is doing
  at::AutoDispatchBelowAutograd mode;
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
  checkPerChannelParamDims(scales, zero_points);
  TORCH_CHECK(
      isFloatingType(scales.scalar_type()),
      "scale tensor must be floating point");

  if (isFloatingType(zero_points.scalar_type())) {
    Tensor scales_float = scales.to(kFloat).contiguous();
    Tensor zero_points_float = zero_points.to(kFloat).contiguous();
    return c10::make_intrusive<PerChannelAffineFloatQParamsQuantizer>(scalar_type,
                                                                      scales_float,
                                                                      zero_points_float,
                                                                      axis);
  }
  else {
    Tensor scales_double = scales.to(kDouble).contiguous();
    Tensor zero_points_int64 = zero_points.to(kLong).contiguous();
    return c10::make_intrusive<PerChannelAffineQuantizer>(scalar_type,
                                                          scales_double,
                                                          zero_points_int64,
                                                          axis);
  }
}

QTensorImpl* get_qtensorimpl(const TensorBase& self) {
  TORCH_CHECK(
      !self.requires_grad(),
      "quantized tensors do not support autograd");
  TORCH_INTERNAL_ASSERT(self.is_quantized(), "get_qtensorimpl: not a quantized tensor");
  return static_cast<QTensorImpl*>(self.unsafeGetTensorImpl());
}

int64_t get_sub_byte_tensor_size(IntArrayRef sizes, size_t dtype_itemsize, at::ScalarType t) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t element_per_byte;
  switch(t) {
    case at::ScalarType::QUInt4x2:
      element_per_byte = 2;
      break;
    case at::ScalarType::QUInt2x4:
      element_per_byte = 4;
      break;
    default:
      element_per_byte = 1;
  }
  // zero dim tensor
  if (sizes.size() == 0) {
    return c10::multiply_integers(sizes) * dtype_itemsize;
  }
  // Consider most inner dim as cols
  int64_t cols = sizes.at(sizes.size()-1);
  int64_t bytes_per_row = cols * dtype_itemsize;
  // align qtensor most inner dim, compute ceil (bytes_per_row / element_per_byte)
  return c10::multiply_integers(IntArrayRef(sizes.data(), sizes.size() - 1)) * at::ceil_div(bytes_per_row, element_per_byte);
}

inline Tensor new_qtensor(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer) {
  auto memory_format = options.memory_format_opt().value_or(MemoryFormat::Contiguous);
  auto device = options.device();
  at::Allocator* allocator = nullptr;
  // TODO: why isn't this just using GetAllocator
  if (device.is_cuda()) {
    allocator = at::detail::getCUDAHooks().getCUDADeviceAllocator();
  } else if (device.is_cpu()) {
    allocator = at::getCPUAllocator();
  } else if (device.is_meta()) {
    allocator = GetAllocator(kMeta);
  } else {
    TORCH_INTERNAL_ASSERT(0, "unrecognized device for new_qtensor: ", device);
  }

#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
    TORCH_CHECK(!device.is_cuda(), "It looks like you are trying to quantize a CUDA tensor ",
                "while QNNPACK backend is enabled. Although not expected to happen in ",
                "practice, you might have done it for testing purposes. ",
                "Please, either change the quantization engine or move the tensor to a CPU.");
    allocator = c10::GetDefaultMobileCPUAllocator();
  }
#endif

  at::DispatchKey tensorDispatchKey = options.computeDispatchKey();
  native::check_size_nonnegative(sizes);
  auto dtype = options.dtype();
  TORCH_CHECK(
      isQIntType(typeMetaToScalarType(dtype)),
      "ScalarType ",
      typeMetaToScalarType(dtype),
      " is not supported in new_qtensor.");
  auto scalar_type = typeMetaToScalarType(dtype);
  int64_t size_bytes = get_sub_byte_tensor_size(sizes, dtype.itemsize(), scalar_type);

  auto storage = c10::make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizable=*/true);
  auto tensor = detail::make_tensor<QTensorImpl>(
      storage, at::DispatchKeySet(tensorDispatchKey), dtype, quantizer);
  get_qtensorimpl(tensor)->set_sizes_contiguous(sizes);
  get_qtensorimpl(tensor)->empty_tensor_restride(memory_format);
  return tensor;
}

Tensor PerTensorAffineQuantizer::quantize(const Tensor& rtensor) {
  TORCH_CHECK(
      rtensor.scalar_type() == kFloat,
      "Quantize only works on Float Tensor, got ", rtensor.scalar_type());
  // Here we need a std::intrusive_ptr<Quantizer>.. but actually "this" is the
  // quantizer that can be reused, so I'm using intrusive_from_this here
  Tensor qtensor = new_qtensor(
      rtensor.sizes(),
      rtensor.options()
          .dtype(scalar_type_)
          .memory_format(rtensor.suggest_memory_format()),
      intrusive_from_this());

  auto rtensor_contig = rtensor.expect_contiguous(rtensor.suggest_memory_format());
  native::quantize_tensor_per_tensor_affine(
      *rtensor_contig, qtensor, scale_, zero_point_);
  return qtensor;
}

void per_tensor_affine_dequantize_impl(
    Tensor& rtensor,
    const Tensor& qtensor,
    const double scale,
    const int64_t zero_point) {
  const auto qtensor_contig =
    qtensor.expect_contiguous(qtensor.suggest_memory_format());
  native::dequantize_tensor_per_tensor_affine(
      *qtensor_contig, rtensor, scale, zero_point);
}

Tensor& PerTensorAffineQuantizer::dequantize_out(
    Tensor& rtensor, const Tensor& qtensor) {
  rtensor.resize_(qtensor.sizes());
  TORCH_CHECK(
      rtensor.is_contiguous(qtensor.suggest_memory_format()) &&
      rtensor.scalar_type() == kFloat,
      "Dequantize out should be a contiguous Float Tensor; instead got type ",
      rtensor.scalar_type(),
      ", and is_contiguous ",
      rtensor.is_contiguous(qtensor.suggest_memory_format()));
  per_tensor_affine_dequantize_impl(rtensor, qtensor, scale_, zero_point_);
  return rtensor;
}

Tensor PerTensorAffineQuantizer::dequantize(const Tensor& qtensor) {
  Tensor rtensor = at::empty(
      qtensor.sizes(),
      qtensor.options()
          .dtype(at::kFloat)
          .memory_format(qtensor.suggest_memory_format()));
  per_tensor_affine_dequantize_impl(rtensor, qtensor, scale_, zero_point_);
  return rtensor;
}

Tensor PerChannelAffineQuantizer::quantize(const Tensor& rtensor) {
  // Here we need a std::intrusive_ptr<Quantizer>.. but actually "this" is the
  // quantizer that can be reused, so I'm using intrusive_from_this here
  Tensor qtensor = new_qtensor(
      rtensor.sizes(),
      rtensor.options()
          .dtype(scalar_type_)
          .memory_format(rtensor.suggest_memory_format()),
      intrusive_from_this());
  auto rtensor_contig = rtensor.expect_contiguous(rtensor.suggest_memory_format());
  native::quantize_tensor_per_channel_affine(
      *rtensor_contig, qtensor, scales_, zero_points_, axis_);
  return qtensor;
}

void per_channel_affine_dequantize_impl(
    Tensor& rtensor,
    const Tensor& qtensor,
    const Tensor& scale,
    const Tensor& zero_point,
    const int64_t axis) {
  const auto qtensor_contig =
    qtensor.expect_contiguous(qtensor.suggest_memory_format());
  native::dequantize_tensor_per_channel_affine(
      *qtensor_contig, rtensor, scale, zero_point, axis);
}

Tensor PerChannelAffineQuantizer::dequantize(const Tensor& qtensor) {
  Tensor rtensor = at::empty(
      qtensor.sizes(),
      qtensor.options()
          .dtype(at::kFloat)
          .memory_format(qtensor.suggest_memory_format()));
  per_channel_affine_dequantize_impl(rtensor, qtensor, scales_, zero_points_, axis_);
  return rtensor;
}

Tensor& PerChannelAffineQuantizer::dequantize_out(
    Tensor& rtensor, const Tensor& qtensor) {
  rtensor.resize_(qtensor.sizes());
  TORCH_CHECK(
      rtensor.is_contiguous(qtensor.suggest_memory_format()) &&
      rtensor.scalar_type() == kFloat,
      "Dequantize out should be a contiguous Float Tensor; instead got type ",
      rtensor.scalar_type(),
      ", and is_contiguous ",
      rtensor.is_contiguous(qtensor.suggest_memory_format()));
  per_channel_affine_dequantize_impl(rtensor, qtensor, scales_, zero_points_, axis_);
  return rtensor;
}

Tensor PerChannelAffineFloatQParamsQuantizer::quantize(const Tensor& rtensor) {
 TORCH_CHECK(
      rtensor.scalar_type() == kFloat,
      "Quantize only works on Float Tensor, got ", rtensor.scalar_type());
 Tensor qtensor = new_qtensor(
      rtensor.sizes(),
      rtensor.options().dtype(scalar_type_),
      intrusive_from_this());
 auto rtensor_contig = rtensor.expect_contiguous();
 native::quantize_tensor_per_channel_float_qparams(
   *rtensor_contig, qtensor, scales_, zero_points_, axis_);
  return qtensor;
}

void per_channel_affine_float_q_params_dequantize_impl(
    Tensor& rtensor,
    const Tensor& qtensor,
    const Tensor& scale,
    const Tensor& zero_point,
    const int64_t axis) {
  const auto qtensor_contig =
    qtensor.expect_contiguous(qtensor.suggest_memory_format());
  native::dequantize_tensor_per_channel_float_qparams(
      *qtensor_contig, rtensor, scale, zero_point, axis);
}

Tensor PerChannelAffineFloatQParamsQuantizer::dequantize(const Tensor& qtensor) {
  Tensor rtensor = at::empty(qtensor.sizes(), qtensor.options().dtype(at::kFloat));
  per_channel_affine_float_q_params_dequantize_impl(
      rtensor, qtensor, scales_, zero_points_, axis_);
  return rtensor;
}

Tensor& PerChannelAffineFloatQParamsQuantizer::dequantize_out(
    Tensor& rtensor, const Tensor& qtensor) {
  rtensor.resize_(qtensor.sizes());
  TORCH_CHECK(
      rtensor.is_contiguous(qtensor.suggest_memory_format()) &&
      rtensor.scalar_type() == kFloat,
      "Dequantize out should be a contiguous Float Tensor; instead got type ",
      rtensor.scalar_type(),
      ", and is_contiguous ",
      rtensor.is_contiguous(qtensor.suggest_memory_format()));
  per_channel_affine_float_q_params_dequantize_impl(
      rtensor, qtensor, scales_, zero_points_, axis_);
  return rtensor;
}

Quantizer::~Quantizer() = default;

C10_EXPORT void set_quantizer_(const Tensor& self, ConstQuantizerPtr quantizer) {
  get_qtensorimpl(self)->set_quantizer_(quantizer);
}

Tensor from_blob_quantized_per_tensor_affine(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    std::function<void(void*)> deleter,
    const float scale,
    const int64_t zeroPoint,
    const TensorOptions& options) {
  auto dtype = typeMetaToScalarType(options.dtype());
  TORCH_CHECK(
      isQIntType(dtype),
      "from_blob_quantized_per_tensor_affine expects QInt dtypes, got ", dtype);

  const std::size_t itemsize = options.dtype().itemsize();
  std::size_t size = 1;
  for (std::int64_t s : sizes) {
    size *= static_cast<std::size_t>(s);
  }
  const std::size_t datasize = size * itemsize;

  DataPtr data_ptr = InefficientStdFunctionContext::makeDataPtr(
      data, deleter, options.device());

  Storage storage{Storage::use_byte_size_t{}, datasize, std::move(data_ptr)};

  QuantizerPtr quantizer =
      make_per_tensor_affine_quantizer(scale, zeroPoint, dtype);

  Tensor qtensor = at::detail::make_tensor<QTensorImpl>(
      std::move(storage),
      at::DispatchKeySet(options.computeDispatchKey()),
      options.dtype(),
      quantizer);
  get_qtensorimpl(qtensor)->set_sizes_and_strides(sizes, strides);
  return qtensor;
}

Tensor from_blob_quantized_per_tensor_affine(
    void* data,
    IntArrayRef sizes,
    std::function<void(void*)> deleter,
    const float scale,
    const int64_t zeroPoint,
    const TensorOptions& options) {
  std::vector<int64_t> strides;
  const auto ndim = sizes.size();
  if (ndim > 0) {
    strides.resize(ndim);
    // NOLINTNEXTLINE
    int32_t i = ndim - 1;
    // NOLINTNEXTLINE
    strides[i] = 1;
    while (--i >= 0) {
      strides[i] = sizes[i + 1] * strides[i + 1];
    }
  }
  return from_blob_quantized_per_tensor_affine(
      data,
      sizes,
      strides,
      deleter,
      scale,
      zeroPoint,
      options);
}

Tensor from_blob_quantized_per_channel_affine(
    void* data,
    IntArrayRef sizes,
    std::function<void(void*)> deleter,
    const Tensor& scales,
    const Tensor& zero_points,
    const int64_t axis,
    const TensorOptions& options) {
  checkPerChannelParamDims(scales, zero_points);
  int64_t channel = sizes[axis];
  TORCH_CHECK(
      channel == int64_t(scales.numel()),
      "length of scales must equal to channel, expected ", channel, " got, ", scales.numel());
  TORCH_CHECK(
      channel == int64_t(zero_points.numel()),
      "length of zero_points must equal to channel, expected ", channel, " got, ", zero_points.numel());

  auto dtype = typeMetaToScalarType(options.dtype());
  TORCH_CHECK(
      isQIntType(dtype),
      "from_blob_quantized_per_channel_affine expects QInt dtypes, got ", dtype);

  const std::size_t itemsize = options.dtype().itemsize();
  std::size_t size = 1;
  for (std::int64_t s : sizes) {
    size *= static_cast<std::size_t>(s);
  }
  const std::size_t datasize = size * itemsize;

  DataPtr data_ptr = InefficientStdFunctionContext::makeDataPtr(
      data, deleter, options.device());

  Storage storage{Storage::use_byte_size_t{}, datasize, std::move(data_ptr)};

  QuantizerPtr quantizer =
      make_per_channel_affine_quantizer(scales, zero_points, axis, dtype);

  Tensor qtensor = at::detail::make_tensor<QTensorImpl>(
      std::move(storage),
      at::DispatchKeySet(options.computeDispatchKey()),
      options.dtype(),
      quantizer);
  get_qtensorimpl(qtensor)->set_sizes_contiguous(sizes);

  return qtensor;
}

Tensor UnknownQuantizer::quantize(const Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(false, "cannot call quantize on UnknownQuantizer");
}
Tensor UnknownQuantizer::dequantize(const Tensor& qtensor) {
  TORCH_INTERNAL_ASSERT(false, "cannot call dequantize on UnknownQuantizer");
}
Tensor& UnknownQuantizer::dequantize_out(Tensor& rtensor, const Tensor& qtensor) {
  TORCH_INTERNAL_ASSERT(false, "cannot call dequantize_out on UnknownQuantizer");
}
QScheme UnknownQuantizer::qscheme() const {
  TORCH_INTERNAL_ASSERT(false, "cannot call qscheme on UnknownQuantizer");
}
bool UnknownQuantizer::equalTo(QuantizerPtr other) const{
  TORCH_INTERNAL_ASSERT(false, "cannot call equalTo on UnknownQuantizer");
}
QuantizerPtr make_unknown_quantizer(ScalarType scalar_type) {
  return c10::make_intrusive<UnknownQuantizer>(scalar_type);
}

} // namespace at
