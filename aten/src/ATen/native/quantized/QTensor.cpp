#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/cpu/quant_utils.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>

namespace at {
namespace native {

Tensor quantize_per_tensor(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    ScalarType dtype) {
  auto quantizer = make_per_tensor_affine_quantizer(scale, zero_point, dtype);
  return quantizer->quantize(self);
}

std::vector<Tensor> quantize_per_tensor_list_cpu(
    TensorList tensors,
    const Tensor& scales,
    const Tensor& zero_points,
    ScalarType dtype) {
  std::vector<Tensor> quantized_tensors;
  for (auto i = 0; i < tensors.size(); ++i) {
    quantized_tensors.push_back(at::quantize_per_tensor(
        tensors[i],
        scales[i].item<double>(),
        zero_points[i].item<int64_t>(),
        dtype));
  }
  return quantized_tensors;
}

Tensor quantize_per_channel_cpu(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType dtype) {
  auto quantizer = make_per_channel_affine_quantizer(scales, zero_points, axis, dtype);
  return quantizer->quantize(self);
}

Tensor dequantize_quant(const Tensor& self) {
  return get_qtensorimpl(self)->quantizer()->dequantize(self);
}

std::vector<Tensor> dequantize_tensors_quantized_cpu(TensorList tensors) {
  std::vector<Tensor> dequantized_tensors;
  for (auto i = 0; i < tensors.size(); ++i) {
    dequantized_tensors.push_back(tensors[i].dequantize());
  }
  return dequantized_tensors;
}

double q_scale_quant(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerTensorAffine);
  return static_cast<PerTensorAffineQuantizer*>(quantizer.get())->scale();
}

int64_t q_zero_point_quant(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerTensorAffine);
  return static_cast<PerTensorAffineQuantizer*>(quantizer.get())->zero_point();
}

Tensor q_per_channel_scales(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerChannelAffine || quantizer->qscheme() == kPerChannelAffineFloatQParams);
  return static_cast<PerChannelAffineQuantizer*>(quantizer.get())->scales();
}

Tensor q_per_channel_zero_points(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerChannelAffine || quantizer->qscheme() == kPerChannelAffineFloatQParams);
  return static_cast<PerChannelAffineQuantizer*>(quantizer.get())->zero_points();
}

int64_t q_per_channel_axis(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerChannelAffine || quantizer->qscheme() == kPerChannelAffineFloatQParams);
  return static_cast<PerChannelAffineQuantizer*>(quantizer.get())->axis();
}

Tensor make_per_channel_quantized_tensor_cpu(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  Tensor dst = at::_empty_per_channel_affine_quantized(
      self.sizes(),
      scales,
      zero_points,
      axis,
      self.options().dtype(toQIntType(self.scalar_type())));
  Tensor self_contig = self.contiguous();
  AT_DISPATCH_QINT_TYPES(
      dst.scalar_type(), "per_channel_affine_qtensor", [&]() {
        underlying_t* self_data = self_contig.data_ptr<underlying_t>();
        underlying_t* dst_data =
            reinterpret_cast<underlying_t*>(dst.data_ptr<scalar_t>());
        if (self.numel() > 0) {
          memcpy(dst_data, self_data, self.nbytes());
        }
      });
  return dst;
}

Tensor& set_storage_quantized_(
    Tensor& self,
    Storage storage,
    int64_t storage_offset,
    IntArrayRef sizes,
    IntArrayRef strides) {
  auto* self_ = self.unsafeGetTensorImpl();
  self_->set_storage_keep_dtype(storage);
  self_->set_storage_offset(storage_offset);
  self_->set_sizes_and_strides(sizes, strides);
  return self;
}

QScheme qscheme_quant(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  return quantizer->qscheme();
}

Tensor& set_quantizer_(Tensor& self, ConstQuantizerPtr quantizer) {
  get_qtensorimpl(self)->set_quantizer_(quantizer);
  return self;
}

Tensor quantized_clone(
    const Tensor& self,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  // TODO: add per channel support
  TORCH_INTERNAL_ASSERT(
      self.qscheme() == at::kPerTensorAffine,
      "clone for quantized Tensor only works for PerTensorAffine scheme right now");

  auto memory_format =
      optional_memory_format.value_or(MemoryFormat::Contiguous);

  // TODO: To support all features of MemoryFormat::Preserve we need to add
  // _empty_affine_quantized_strided function and use it similarly to
  // Tensor clone(const Tensor& src, c10::optional<c10::MemoryFormat>
  // optional_memory_format) if (self.is_non_overlapping_and_dense()) ->
  // _empty_affine_quantized_strided
  if (memory_format == MemoryFormat::Preserve) {
    memory_format = self.suggest_memory_format();
  }

  Tensor dst = at::_empty_affine_quantized(
      self.sizes(),
      self.options().memory_format(memory_format),
      self.q_scale(),
      self.q_zero_point(),
      c10::nullopt);

  at::native::copy_(dst, self, false);

  return dst;
}

bool equal_quantized_cpu(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      self.device().type() == kCPU && other.device().type() == kCPU,
      "quantized_equal is implemented only for the QuantizedCPU backend");
  if (!other.is_quantized()) {
    return false;
  }

  // Delegate to virtual equalTo method. This will ensure different concrete
  // Quantizers can have specific logic for comparison
  auto self_quantizer = get_qtensorimpl(self)->quantizer();
  auto other_quantizer = get_qtensorimpl(other)->quantizer();
  if (!self_quantizer->equalTo(other_quantizer)) {
    return false;
  }

  // Sizes and element types must be the same
  if (self.sizes() != other.sizes()) {
    return false;
  }
  if (self.element_size() != other.element_size()) {
    return false;
  }

  // Data must be the same
  auto self_contig = self.contiguous();
  auto other_contig = other.contiguous();

  void* self_data = self_contig.data_ptr();
  void* other_data = other_contig.data_ptr();
  return 0 == memcmp(self_data, other_data, self.numel() * self.element_size());
}

/* Calculate the quantization params for the activation tensor */
std::tuple<double, int64_t> _choose_qparams_per_tensor(
    const Tensor& self,
    bool reduce_range) {
  at::Tensor a;
  auto input_contig = self.contiguous();
  float x_min = input_contig.min().item<float>();
  float x_max = input_contig.max().item<float>();

  if (reduce_range && at::globalContext().qEngine() == at::QEngine::QNNPACK) {
    reduce_range = false;
  }

  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/0,
      /*qmax=*/255,
      /*preserve_sparsity=*/false,
      /*force_scale_power_of_two=*/false,
      /*reduce_range=*/reduce_range);

  return std::make_tuple(q_params.scale, q_params.zero_point);
}

} // namespace native
} // namespace at
