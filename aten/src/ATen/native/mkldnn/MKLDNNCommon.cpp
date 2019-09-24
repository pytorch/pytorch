#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/OpaqueTensorImpl.h>
#include <c10/core/Allocator.h>

#if AT_MKLDNN_ENABLED()

#include <ideep.hpp>

namespace at { namespace native {

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
struct CAFFE2_API IntrusivePtrTargetWrapper : c10::intrusive_ptr_target {
private:
  T target_;

public:
  IntrusivePtrTargetWrapper() = delete;
  IntrusivePtrTargetWrapper(const T& target): target_(target) {}
  IntrusivePtrTargetWrapper(T&& target): target_(std::move(target)) {}

  T& get_target() {
    return target_;
  }
};

using IDeepTensorWrapper = IntrusivePtrTargetWrapper<ideep::tensor>;
using IDeepTensorWrapperPtr = c10::intrusive_ptr<IDeepTensorWrapper>;
using MKLDNNTensorImpl = OpaqueTensorImpl<IDeepTensorWrapperPtr>;
using MKLDNNTensor = Tensor;

Tensor new_with_itensor_mkldnn(ideep::tensor&& it, const TensorOptions& options) {
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  // TODO: support int64_t dims in ideep::tensor to avoid extra conversion
  auto dims = it.get_dims();
  IDeepTensorWrapperPtr handle = c10::make_intrusive<IDeepTensorWrapper>(std::move(it));
  return detail::make_tensor<MKLDNNTensorImpl>(
    TensorTypeSet(TensorTypeId::MkldnnCPUTensorId),
    options.dtype(), options.device(), handle,
    std::vector<int64_t>(dims.begin(), dims.end()));
}

ideep::tensor& itensor_from_mkldnn(const MKLDNNTensor& mkldnn_tensor) {
  AT_ASSERTM(mkldnn_tensor.is_mkldnn(),
             "mkldnn_to_dense expects MKL-DNN tensor input");
  AT_ASSERTM(!mkldnn_tensor.is_variable(), "_internal_get_MKLDNNImpl: should not be a variable");
  MKLDNNTensorImpl *mklimpl = static_cast<MKLDNNTensorImpl *>(mkldnn_tensor.unsafeGetTensorImpl());
  return mklimpl->unsafe_opaque_handle()->get_target();
}

ideep::tensor itensor_view_from_dense(const Tensor& tensor) {
  AT_ASSERTM(
      tensor.device().type() == DeviceType::CPU,
      "itensor_view_from_dense expects CPU tensor input");
  AT_ASSERTM(
      tensor.layout() == Layout::Strided,
      "itensor_view_from_dense expects dense tensor input");
  AT_ASSERTM(tensor.scalar_type() == ScalarType::Float,
             "itensor_view_from_dense expects float tensor input");
  AT_ASSERTM(
      !tensor.is_variable(),
      "itensor_view_from_dense: should not be a variable");
  return {{{tensor.sizes().cbegin(), tensor.sizes().cend()},
           ideep::tensor::data_type::f32},
          tensor.template data_ptr<float>()};
}

ideep::tensor::data_type get_mkldnn_dtype(ScalarType type) {
  switch(type) {
    case ScalarType::Float:
      return ideep::tensor::data_type::f32;
    case ScalarType::QInt32:
      return ideep::tensor::data_type::s32;
    case ScalarType::QInt8:
      return ideep::tensor::data_type::s8;
    case ScalarType::QUInt8:
      return ideep::tensor::data_type::u8;
    default:
      AT_ASSERTM(false, "get_mkldnn_dtype: unsupported data type");
  }
}

ideep::scale_t ConvertScales(const std::vector<double> &scales_z) {
  ideep::scale_t scales(scales_z.size());
  for (int i = 0; i < scales_z.size(); i++) {
    scales[i] = 1.0f / scales_z[i];
  }
  return scales;
}
}}

#endif // AT_MKLDNN_ENABLED()
