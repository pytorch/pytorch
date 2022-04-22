#include <ATen/OpaqueTensorImpl.h>
#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <c10/core/Allocator.h>

#if AT_ZENDNN_ENABLED()

#include "ATen/native/zendnn/AbstractTypes.hpp"
#include "ATen/native/zendnn/Tensor.hpp"

namespace at {
namespace native {

/**
 * `IntrusivePtrTargetWrapper` wraps a custom storage handle  of a tensor
 *  (as template param) and inherits `c10::intrusive_ptr_target` so that it
 *  can be used with `c10::intrusive_ptr`.
 *
 * It currently only supports wrapping the custom handle by:
 * - Constructing with an existing custom handle by copy/move constructor.
 *
 * See `OpaqueTensorImpl::opaque_handle_`.
 *
 * NOTE: if this is generally useful we may want to move this to its own header.
 */
template <typename T>
struct TORCH_API IntrusivePtrTargetWrapper : c10::intrusive_ptr_target {
 private:
  T target_;

 public:
  IntrusivePtrTargetWrapper() = delete;
  IntrusivePtrTargetWrapper(const T& target) : target_(target) {}
  IntrusivePtrTargetWrapper(T&& target) : target_(std::move(target)) {}

  T& get_target() {
    return target_;
  }
};

using IDeepTensorWrapper = IntrusivePtrTargetWrapper<zendnn::tensor>;
using IDeepTensorWrapperPtr = c10::intrusive_ptr<IDeepTensorWrapper>;
using ZENDNNTensorImpl = OpaqueTensorImpl<IDeepTensorWrapperPtr>;
using ZENDNNTensor = Tensor;

// conversion from pytorch type to zendnn type
zendnn::tensor::data_type get_zendnn_dtype(ScalarType type) {
  switch (type) {
    case ScalarType::Float:
      return zendnn::tensor::data_type::f32;
    case ScalarType::QInt32:
      return zendnn::tensor::data_type::s32;
    case ScalarType::QInt8:
      return zendnn::tensor::data_type::s8;
    case ScalarType::QUInt8:
    case ScalarType::Byte:
      return zendnn::tensor::data_type::u8;
    case ScalarType::BFloat16:
      return zendnn::tensor::data_type::bf16;
    case ScalarType::Char:
      return zendnn::tensor::data_type::s8;
    case ScalarType::Int:
      return zendnn::tensor::data_type::s32;
    default:
      TORCH_CHECK(false, "get_zendnn_dtype: unsupported data type");
  }
}

// conversion from zendnn type to pytorch type
ScalarType get_tensor_dtype(zendnn::tensor::data_type type) {
  switch (type) {
    case zendnn::tensor::data_type::f32:
      return ScalarType::Float;
    case zendnn::tensor::data_type::s32:
      return ScalarType::Int;
    case zendnn::tensor::data_type::u8:
      return ScalarType::Byte;
    case zendnn::tensor::data_type::bf16:
      return ScalarType::BFloat16;
    case zendnn::tensor::data_type::s8:
      return ScalarType::Char;
    default:
      TORCH_CHECK(false, "get_tensor_dtype: unsupported data type");
  }
}

Tensor new_with_itensor_zendnn(
    zendnn::tensor&& it,
    c10::optional<ScalarType> dtype,
    c10::optional<Device> device) {
  // NOTE: int32_t dims from zendnn::tensor but sizes needs int64_t
  // TODO: support int64_t dims in zendnn::tensor to avoid extra conversion
  auto dims = it.get_dims();
  IDeepTensorWrapperPtr handle =
      c10::make_intrusive<IDeepTensorWrapper>(std::move(it));
  caffe2::TypeMeta dtype_ = scalarTypeToTypeMeta(dtype_or_default(dtype));
  Device device_ = device_or_default(device);
  return detail::make_tensor<ZENDNNTensorImpl>(
      DispatchKeySet(DispatchKey::ZendnnCPU),
      dtype_,
      device_,
      handle,
      std::vector<int64_t>(dims.begin(), dims.end()));
}

zendnn::tensor& itensor_from_zendnn(const ZENDNNTensor& zendnn_tensor) {
  TORCH_CHECK(
      zendnn_tensor.is_zendnn(),
      "itensor_from_zendnn expects ZENDNN tensor input");
  TORCH_INTERNAL_ASSERT(at::impl::variable_excluded_from_dispatch());
  ZENDNNTensorImpl* zendnnimpl =
      static_cast<ZENDNNTensorImpl*>(zendnn_tensor.unsafeGetTensorImpl());
  return zendnnimpl->unsafe_opaque_handle()->get_target();
}

zendnn::tensor itensor_view_from_dense(const Tensor& tensor) {
  TORCH_CHECK(
      tensor.device().is_cpu(),
      "itensor_view_from_dense expects CPU tensor input");
  TORCH_CHECK(
      tensor.layout() == Layout::Strided,
      "itensor_view_from_dense expects dense tensor input");
  TORCH_CHECK(
      tensor.scalar_type() == ScalarType::Float,
      "itensor_view_from_dense expects float tensor input");
  TORCH_INTERNAL_ASSERT(at::impl::variable_excluded_from_dispatch());
  return {
      {{tensor.sizes().cbegin(), tensor.sizes().cend()},
       zendnn::tensor::data_type::f32},
      tensor.template data_ptr<float>()};
}

// Helper function for getting an zendnn tensor out of an aten Tensor.
// Note in case the aten Tensor is a dense tensor, the returned zendnn
// tensor is just a view of the storage of the aten dense tensor, so
// caller needs to make sure the aten dense tensor's lifetime is
// longer than the zendnn tensor.
zendnn::tensor itensor_from_tensor(const Tensor& tensor) {
  if (tensor.is_zendnn()) {
    return itensor_from_zendnn(tensor);
  } else {
    return itensor_view_from_dense(tensor);
  }
}

// functions to get a zendnn tensor from a Tensor and vice versa.
zendnn::tensor zendnn_tensor_view_from_dense(const Tensor& ttensor) {
  // sanity check on input
  TORCH_CHECK(
      ttensor.device().type() == DeviceType::CPU,
      "zendnn_tensor_view_from_dense expects CPU tensor input");
  TORCH_CHECK(
      ttensor.layout() == Layout::Strided,
      "zendnn_tensor_view_from_dense expects dense tensor input");

  // check if tensor type is supported in zendnn
  auto ttensor_arg = TensorArg(ttensor, "ttensor", 1);
  checkScalarTypes(
      "zendnn_tensor_view_from_dense",
      ttensor_arg,
      {kByte, kChar, kInt, kLong, kFloat});

  // get c++ type corresponding to ScalarType
  // TODO : remove switch statment from here and make a function/macro
  // for type conversion between torch and zendnn tensor.
  auto dtype = ttensor.scalar_type();

  switch (dtype) {
    case kByte: {
      auto atype = zendnn::tensor::data_type::u8;
      using cpptype = decltype(c10::impl::ScalarTypeToCPPType<kByte>::t);
      return zendnn::tensor{
          {ttensor.sizes().cbegin(), ttensor.sizes().cend()},
          atype,
          ttensor.template data_ptr<cpptype>()};
    }
    case kChar: {
      auto atype = zendnn::tensor::data_type::s8;
      using cpptype = decltype(c10::impl::ScalarTypeToCPPType<kChar>::t);
      return zendnn::tensor{
          {ttensor.sizes().cbegin(), ttensor.sizes().cend()},
          atype,
          ttensor.template data_ptr<cpptype>()};
    }
    case kLong:
    case kInt: {
      auto atype = zendnn::tensor::data_type::s32;
      using cpptype = decltype(c10::impl::ScalarTypeToCPPType<kInt>::t);
      return zendnn::tensor{
          {ttensor.sizes().cbegin(), ttensor.sizes().cend()},
          atype,
          ttensor.template data_ptr<cpptype>()};
    }
    case kFloat: {
      auto atype = zendnn::tensor::data_type::f32;
      using cpptype = decltype(c10::impl::ScalarTypeToCPPType<kFloat>::t);
      return zendnn::tensor{
          {ttensor.sizes().cbegin(), ttensor.sizes().cend()},
          atype,
          ttensor.template data_ptr<cpptype>()};
    }
    default:
      TORCH_CHECK(
          false, "zendnn_tensor_view_from_dense: unsupported data type");
  }

  // default is always float
  auto atype = zendnn::tensor::data_type::f32;
  using cpptype = float;

  return zendnn::tensor{
      {ttensor.sizes().cbegin(), ttensor.sizes().cend()},
      atype,
      ttensor.template data_ptr<cpptype>()};
}

Tensor new_dense_from_zendnn(
    const zendnn::tensor& zendnn_tensor,
    const TensorOptions& options) {
  // get data_type of zendnn_tensor and figure out appropriate
  // pytorch ScalarType
  using zendnn_type = zendnn::tensor::data_type;

  auto zendnn_tensor_type = zendnn_tensor.get_data_type();
  ScalarType tensor_type = get_tensor_dtype(zendnn_tensor_type);

  // allocate empty tensor
  auto dims = zendnn_tensor.get_dims();

  // NOTE: int32_t dims from zendnn::tensor but sizes needs int64_t
  Tensor cpu_tensor = at::empty(
      std::vector<int64_t>(dims.begin(), dims.end()),
      options.layout(c10::kStrided).dtype(tensor_type));

  // if input zendnn tensor is empty, return empty tensor
  if (zendnn_tensor.is_empty())
    return cpu_tensor;

  auto pub_tensor =
      zendnn_tensor.to_public(cpu_tensor.data_ptr(), zendnn_tensor_type);
  cpu_tensor.as_strided_(dims, pub_tensor.get_strides());
  return cpu_tensor;
}

} // namespace native
} // namespace at

#endif // AT_ZENDNN_ENABLED()
