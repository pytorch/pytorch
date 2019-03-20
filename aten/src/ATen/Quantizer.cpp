#include <ATen/ATen.h>
#include <ATen/Quantizer.h>
#include <ATen/QTensorImpl.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorFactories.h>

namespace at {

std::shared_ptr<Quantizer> make_per_layer_affine_quantizer(double scale, int64_t zero_point) {
  return std::make_shared<PerLayerAffineQuantizer>(static_cast<float>(scale), static_cast<int32_t>(zero_point));
}

// std::unique_ptr<Quantizer> create_quantizer(QScheme qscheme) {
//   switch (qscheme.qtype_) {
//     case QType.AFFINE:
//       if (is_per_layer) {
//         return make_shared<PerLayerAffineQuantizer>();
//       } else {
//         return make_shared<PerChannelAffineQuantizer>();
//       }
//     case QType.Symmetric:
//       if (is_per_layer) {
//         return make_shared<PerLayerSymmetricQuantizer>();
//       } else {
//         return make_shared<PerChannelSymmetricQuantizer>();
//       }
//   }
//   AT_ERROR("Unrecoginized QScheme in create_quantizer.");
// }

// This is an internal utility function for getting at the QTensorImpl,
// You should only use this for writing low level
// setters/getters for QTensorImpl fields; otherwise, you should use
// the low level setters/getters that were implemented using this.
// This may be called repeatedly, so make sure it's pretty cheap.
inline QTensorImpl* get_qtensorimpl(const QTensor& self) {
  // TODO: remove this when Variable and Tensor are merged
  AT_ASSERTM(!self.is_variable(), "_internal_get_QTensorImpl: should not be a variable");
  // TODO: uncomment
  // AT_ASSERTM(self.is_quantized(), "_internal_get_QTensorImpl: not a quantized tensor");
  return static_cast<QTensorImpl*>(self.unsafeGetTensorImpl());
}

inline QTensor new_qtensor(
    IntList sizes, const TensorOptions& options, float scale, int32_t zero_point) {
  AT_ASSERT(options.device().is_cpu());

  native::check_size_nonnegative(sizes);
//  auto* allocator = at::GetAllocator(DeviceType::QUANTIZED);
  auto* allocator = at::getCPUAllocator();
  int64_t nelements = at::prod_intlist(sizes);
  // TODO get from options
  auto dtype = at::dtype(at::kQInt8).dtype();
  std::cout << "nele: " << nelements * dtype.itemsize() << std::endl;
  std::cout << "alloc: " << allocator->allocate(nelements * dtype.itemsize()) << std::endl;
  auto storage_impl = c10::make_intrusive<StorageImpl>(
    dtype,
    nelements,
    allocator->allocate(nelements * dtype.itemsize()),
    allocator,
    /*resizeable=*/true);
  std::cout << "storage: " << static_cast<qint8*>(storage_impl->data())[0].val_ << std::endl;
  auto quantizer = make_per_layer_affine_quantizer(scale, zero_point);
  auto tensor = detail::make_tensor<QTensorImpl>(
      storage_impl, at::CPUTensorId(), false, quantizer);
  // Default TensorImpl has size [0]
  if (sizes.size() != 1 || sizes[0] != 0) {
    get_qtensorimpl(tensor)->set_sizes_contiguous(sizes);
  }
  return tensor;
}

void ChooseParams(RealTensor tensor, float* r_scale, int* r_zero_point) {
  // TODO: dispatch according to tensor.dtype()

  RealTensor real_max_t_ = tensor.max();
  RealTensor real_min_t_ = tensor.min();
  float real_max = real_max_t_.data<float>()[0];
  float real_min = real_min_t_.data<float>()[0];
  // We extend the [min, max] interval to ensure that it contains 0.
  // Otherwise, we would not meet the requirement that 0 be an exactly
  // representable value.
  auto min = std::min(real_min, 0.f);
  auto max = std::max(real_max, 0.f);

  // the min and max quantized values, as floating-point values
  const float qmin = 0;
  const float qmax = 255;

  // First determine the scale.
  const double scale = (max - min) / (qmax - qmin);

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // Let's use the first one here.
  const double initial_zero_point = qmin - min / scale;

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with SAME
  // padding).
  std::uint8_t nudged_zero_point = 0;
  if (initial_zero_point < qmin) {
    nudged_zero_point = qmin;
  } else if (initial_zero_point > qmax) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point =
      static_cast<std::uint8_t>(std::round(initial_zero_point));
  }
  *r_scale = scale;
  *r_zero_point = int(nudged_zero_point);
}

inline qint8 QuantizeUint8(float scale, int32_t zero_point, float value) {
  const int32_t qmin = std::numeric_limits<uint8_t>::min();
  const int32_t qmax = std::numeric_limits<uint8_t>::max();

  auto r = zero_point + static_cast<int32_t>(Round(value / scale));
  r = std::max(r, qmin);
  r = std::min(r, qmax);
  return static_cast<qint8>(r);
}

QTensor PerLayerAffineQuantizer::quantize(RealTensor tensor) {
  //ChooseParams(tensor, &scale_, &zero_point_);

  IntList sizes = tensor.sizes();
  QTensor qv = new_qtensor(sizes, tensor.options(), scale_, zero_point_);
  std::cout << qv.sizes() << std::endl;
  std::cout << qv.options() << std::endl;
  std::cout << qv.device() << std::endl;
  std::cout << qv.data_ptr() << std::endl;
  //auto qvd = qv.data<qint8>();
  auto qvd = qv.data<qint8>();
  const float* svd = tensor.data<float>();
  for (int i = 0; i < tensor.numel(); ++i) {
    qvd[i] = QuantizeUint8(scale_, zero_point_, svd[i]);
  }
  return qv;
}

RealTensor PerLayerAffineQuantizer::dequantize(QTensor tensor) {
  std::vector<int64_t> sizes = tensor.sizes().vec();
  at::TensorOptions real_options = tensor.options().dtype(at::kFloat);

  RealTensor rv = at::empty(sizes, real_options);
  const auto* qvd = tensor.data<qint8>();
  float* rvd = rv.data<float>();
  for (auto i = 0; i < tensor.numel(); ++i) {
    rvd[i] = (static_cast<uint32_t>(qvd[i].val_) - zero_point_) * scale_;
  }
  return rv;
}

} // namespace at
