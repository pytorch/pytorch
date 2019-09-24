#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>

namespace at {
namespace native {

Tensor quantize_per_tensor_cpu(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    ScalarType dtype) {
  auto quantizer = make_per_tensor_affine_quantizer(scale, zero_point, dtype);
  return quantizer->quantize(self);
}

Tensor quantize_per_channel_cpu(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    IntArrayRef axis,
    ScalarType dtype) {
  TORCH_CHECK(scales.dim() == 1, "scale tensor must have dimension 1");
  TORCH_CHECK(
      zero_points.dim() == 1, "zero_points tensor must have dimension 1");
  TORCH_CHECK(
      scales.numel() == zero_points.numel(),
      "number of elements in scales and zero_points must match");
  TORCH_CHECK(axis.size() == 1, "only axis of size 1 is supported right now");
  double* scales_data = scales.data_ptr<double>();
  int64_t* zero_points_data = zero_points.data_ptr<int64_t>();
  std::vector<double> scale_vals(scales_data, scales_data + scales.numel());
  std::vector<int64_t> zero_point_vals(
      zero_points_data, zero_points_data + zero_points.numel());
  auto quantizer = make_per_channel_affine_quantizer(
      scale_vals, zero_point_vals, axis, dtype);
  return quantizer->quantize(self);
}

Tensor dequantize_quant(const Tensor& self) {
  return get_qtensorimpl(self)->quantizer()->dequantize(self);
}

Tensor dequantize_per_tensor_cpu(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    ScalarType dtype) {
  TORCH_CHECK(
      isQIntType(toQIntType(self.scalar_type())),
      "Scalar type for quantized Tensor must have same underlying type as input.");
  TORCH_CHECK(
      dtype == toQIntType(self.scalar_type()),
      "ScalarType argument must match the corresponding quantized scalar type of input integer Tensor");
  // scalar type of output Tensor is hard-coded as float
  Tensor f = at::empty(self.sizes(), self.options().dtype(at::kFloat));
  AT_DISPATCH_QINT_TYPES(
      toQIntType(self.scalar_type()), "dequantize_linear_cpu", [&]() {
        underlying_t* qdata = self.data_ptr<underlying_t>();
        auto* fdata = f.data_ptr<float>();
        for (int i = 0; i < self.numel(); ++i) {
          fdata[i] = (static_cast<float>(qdata[i]) - zero_point) * scale;
        }
      });
  return f;
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

Tensor q_per_channel_scales_quant(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerChannelAffine);
  return at::tensor(
      static_cast<PerChannelAffineQuantizer*>(quantizer.get())->scales(),
      self.options().dtype(at::kDouble));
}

Tensor q_per_channel_zero_points_quant(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerChannelAffine);
  return at::tensor(
      static_cast<PerChannelAffineQuantizer*>(quantizer.get())->zero_points(),
      self.options().dtype(at::kLong));
}

IntArrayRef q_per_channel_axis_quant(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerChannelAffine);
  return static_cast<PerChannelAffineQuantizer*>(quantizer.get())->axis();
}

// When input Tensor is non-dense, i.e. the allocated memory
// is larger than the memory used by all the elements, we'll
// convert it to dense tensor, otherwise we'll keep the memory
// format of the output the same as input
Tensor int_repr_quant(const Tensor& self) {
  Tensor dst;
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "int_repr", [&]() {
    dst = at::empty(
        self.sizes(),
        self.options().dtype(UNDERLYING_TYPE),
        self.suggest_memory_format());
    auto iter = TensorIterator();
    iter.add_output(dst);
    iter.add_input(self);
    iter.dont_compute_common_dtype();
    iter.build();
    cpu_kernel(iter, [](scalar_t value) -> underlying_t { return value.val_; });
  });
  return dst;
}

Tensor per_tensor_affine_qtensor_cpu(
    const Tensor& self,
    double scale,
    int64_t zero_point) {
  Tensor dst = at::_empty_affine_quantized(
      self.sizes(),
      self.options().dtype(toQIntType(self.scalar_type())),
      scale,
      zero_point);
  Tensor self_contig = self.contiguous();
  AT_DISPATCH_QINT_TYPES(dst.scalar_type(), "per_tensor_affine_qtensor", [&]() {
    underlying_t* self_data = self_contig.data_ptr<underlying_t>();
    underlying_t* dst_data =
        reinterpret_cast<underlying_t*>(dst.data_ptr<scalar_t>());
    if (self.numel() > 0) {
      memcpy(dst_data, self_data, self.nbytes());
    }
  });
  return dst;
}

Tensor per_channel_affine_qtensor_cpu(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    IntArrayRef axis) {
  Tensor dst = at::_empty_per_channel_affine_quantized_like(
      scales,
      zero_points,
      self.sizes(),
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

Tensor& set_storage(
    Tensor& self,
    Storage storage,
    int64_t storage_offset,
    IntArrayRef sizes,
    IntArrayRef strides) {
  auto* self_ = self.unsafeGetTensorImpl();
  self_->set_storage(storage);
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

Tensor quantized_clone(const Tensor& self) {
  // TODO: add per channel support
  TORCH_INTERNAL_ASSERT(
      self.qscheme() == at::kPerTensorAffine,
      "clone for quantized Tensor only works for PerTensorAffine scheme right now");
  Tensor dst = at::_empty_affine_quantized(
      self.sizes(), self.options(), self.q_scale(), self.q_zero_point());

  at::native::copy_(dst, self, false);

  return dst;
}

bool quantized_equal(const Tensor& self, const Tensor& other) {
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

} // namespace native
} // namespace at
