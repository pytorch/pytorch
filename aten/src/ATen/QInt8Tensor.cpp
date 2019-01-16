// Basic functions on sparse tensors

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/Layout.h>
#include <ATen/NativeFunctions.h>
#include <ATen/QInt8TensorImpl.h>
#include <ATen/Utils.h>

#include <cmath>
#include <iostream>

#include <TH/THBlasUtils.h>

namespace at { namespace native {


using QInt8Tensor = at::Tensor;
using RealTensor = at::Tensor;

namespace {
// TODO: unify this with C2 one

template <class T>
inline T Round(const T x) {
  return std::nearbyint(x);
}

inline uint8_t QuantizeUint8(float scale, int32_t zero_point, float value) {
  const int32_t qmin = std::numeric_limits<uint8_t>::min();
  const int32_t qmax = std::numeric_limits<uint8_t>::max();

  auto r = zero_point + static_cast<int32_t>(Round(value / scale));
  r = std::max(r, qmin);
  r = std::min(r, qmax);
  return static_cast<uint8_t>(r);
}

// A structure to hold quantization parameters 'scale' and 'zero_point'
// as discussed in doc/quantization.md. As explained there, the meaning
// of these values is as the constants in the quantization equation
//
//   real_value = scale * (quantized_value - zero_point)
//
// In other words, 'zero_point' is the quantized value that corresponds
// to the real value 0, and 'scale' is the difference of real values
// corresponding to consecutive quantized values.
struct QuantizationParams {
  float scale;
  std::int32_t zero_point;
};

// Given the min and max values of a float array, return
// reasonable quantization parameters to use for this array.
QuantizationParams ChooseQuantizationParams(float min, float max) {
  // We extend the [min, max] interval to ensure that it contains 0.
  // Otherwise, we would not meet the requirement that 0 be an exactly
  // representable value.
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);

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

  QuantizationParams result;
  result.scale = scale;
  result.zero_point = int(nudged_zero_point);
  return result;
}

inline void Int8Dequantize(
    const uint8_t* in,
    float* out,
    const int64_t N,
    const float X_scale,
    const int32_t X_offset) {
  for (auto i = 0; i < N; ++i) {
    out[i] = (static_cast<int32_t>(in[i]) - X_offset) * X_scale;
  }
}

}

// This is an internal utility function for getting at the QInt8TensorImpl,
// so that we can write sparse tensor specific accessors for special fields
// in SparseTensor.  You should only use this for writing low level
// setters/getters for SparseTensorImpl fields; otherwise, you should use
// the low level setters/getters that were implemented using this.
//
// This may be called repeatedly, so make sure it's pretty cheap.
inline QInt8TensorImpl* get_qint8_impl(const QInt8Tensor& self) {
  return static_cast<QInt8TensorImpl*>(self.unsafeGetTensorImpl());
}

inline void* get_qint8_impl_raw(const QInt8Tensor& self) {
  return static_cast<void*>(self.unsafeGetTensorImpl());
}

/******************************************************************************
 * creation methods
 ******************************************************************************/

/*** Helper methods ***/

inline QInt8Tensor new_qtensor(
  float scale, int32_t zero_point, IntList sizes, const TensorOptions& options) {
  AT_ASSERT(options.device().is_cpu());
  TensorTypeId type_id = QCPUTensorId();
  auto* allocator = at::getCPUAllocator();
  int64_t nelements = at::prod_intlist(sizes);
  auto dtype = at::dtype(at::kByte).dtype();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
    dtype,
    nelements,
    allocator->allocate(nelements * dtype.itemsize()),
    allocator,
    /*resizeable=*/true);
  auto tensor = detail::make_tensor<QInt8TensorImpl>(
      storage_impl, type_id, scale, zero_point);
  // Default TensorImpl has size [0]
  if (sizes.size() != 1 || sizes[0] != 0) {
    get_qint8_impl(tensor)->set_sizes_contiguous(sizes);
  }
  return tensor;
}

/** Actual dispatched creation methods ***/

QInt8Tensor new_with_scale_zero_point(
  float scale, int32_t zero_point, IntList sizes, const TensorOptions& options) {
  AT_CHECK(sizes.size() != 0,
    "cannot construct quantize tensor with 0 dimensions and no values; you must specify at least 1 dimension if you want to create a quantize tensor with no elements, \
or you must provide a single-element `values` tensor (e.g. x = torch.quantize_tensor(torch.zeros(0, 1), 12.3, [])) if you want to create a scalar quantize tensor");
  QInt8Tensor self = new_qtensor(scale, zero_point, sizes, options);
  return self;
}

/** Public creation API that dispatch to methods above **/

/** Empty init **/
QInt8Tensor empty_quantized(IntList size, const TensorOptions& options) {
  return new_with_scale_zero_point(1.0, 0, size, options);
}

QInt8Tensor real_to_quantize(const RealTensor& self){
  IntList sizes = self.sizes();
  // support more dtype here
  RealTensor real_max_t_ = self.max();
  RealTensor real_min_t_ = self.min();
  float real_max = real_max_t_.data<float>()[0];
  float real_min = real_min_t_.data<float>()[0];
  QuantizationParams qparam = ChooseQuantizationParams(real_min, real_max);

  QInt8Tensor qv = new_with_scale_zero_point(qparam.scale, qparam.zero_point, sizes, self.options());
  uint8_t* qvd = qv.data<uint8_t>();
  const float* svd = self.data<float>();
  for (int i = 0; i < self.numel(); ++i) {
    qvd[i] = QuantizeUint8(qparam.scale, qparam.zero_point, svd[i]);
  }
  // std::cout << get_qint8_impl(qv)->scale() << std::endl;
  std::cout << "r2q imple addr: " << (get_qint8_impl(qv)) << std::endl;
  std::cout << "r2q imple addr raw: " << get_qint8_impl_raw(qv) << std::endl;
  return qv;
}

RealTensor quantize_to_real(const QInt8Tensor& self){
  std::vector<int64_t> sizes = self.sizes().vec();
  std::cout << "q2r imple addr: " << get_qint8_impl(self) << std::endl;
  std::cout << "q2r imple addr raw: " << get_qint8_impl_raw(self) << std::endl;
  std::cout << "Is it a QTensor? " << get_qint8_impl(self)->is_quantize() << std::endl;
  // support more dtype here
  at::TensorOptions real_options = self.options().dtype(at::kFloat);
  float scale = get_qint8_impl(self)->scale();
  int32_t zero_point = get_qint8_impl(self)->zero_point();
  std::cout << scale << std::endl;
  std::cout << zero_point << std::endl;
  std::cout << "====" << std::endl;

  RealTensor rv = at::empty(sizes, real_options);
  const uint8_t* qvd = self.data<uint8_t>();
  float* rvd = rv.data<float>();
  Int8Dequantize(qvd, rvd, self.numel(), scale, zero_point);
  std::cout << int(qvd[1]) << std::endl;
  std::cout << rvd[0] << std::endl;
  return rv;
}

Scalar get_qtensor_scale(const QInt8Tensor& self) {
  float scale = get_qint8_impl(self)->scale();
  return Scalar(scale);
}

Scalar get_qtensor_zero_point(const QInt8Tensor& self) {
  int32_t zero_point = get_qint8_impl(self)->zero_point();
  return Scalar(zero_point);
}

}} // namespace at::native
