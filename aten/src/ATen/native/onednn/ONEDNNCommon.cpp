#include <ATen/native/onednn/ONEDNNCommon.h>
#include <ATen/OpaqueTensorImpl.h>
#include <c10/core/Allocator.h>
#include <torch/library.h>

#if AT_ONEDNN_ENABLED()

#include <ideep.hpp>

namespace at::native {

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

ideep::tensor::data_type get_mkldnn_dtype(ScalarType type) {
  switch (type) {
    case ScalarType::Float:
      return ideep::tensor::data_type::f32;
    case ScalarType::QInt32:
      return ideep::tensor::data_type::s32;
    case ScalarType::QInt8:
    case ScalarType::Char:
      return ideep::tensor::data_type::s8;
    case ScalarType::QUInt8:
    case ScalarType::Byte:
      return ideep::tensor::data_type::u8;
    case ScalarType::BFloat16:
      return ideep::tensor::data_type::bf16;
    case ScalarType::Half:
      return ideep::tensor::data_type::f16;
    default:
      TORCH_CHECK(false, "get_mkldnn_dtype: unsupported data type");
  }
}

int64_t data_ptr_from_mkldnn(const Tensor& mkldnn_tensor) {
  MKLDNNTensorImpl *mklimpl = static_cast<MKLDNNTensorImpl *>(mkldnn_tensor.unsafeGetTensorImpl());
  void* data_ptr = mklimpl->unsafe_opaque_handle()->get_target().get_data_handle();
  return reinterpret_cast<int64_t>(data_ptr);
}

at::Tensor mkldnn_tensor_from_data_ptr(
    void* data_ptr,
    at::IntArrayRef dims,
    at::ScalarType dtype,
    at::Device device,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
  std::vector<uint8_t> vector_serialized_md{
      opaque_metadata, opaque_metadata + opaque_metadata_size};
  ideep::tensor::desc deserialized_ideep_desc;
#if IDEEP_PREREQ(3, 4, 1, 2)
  // groups is needed for grouped conv
  deserialized_ideep_desc = ideep::tensor::desc(vector_serialized_md);
#else
  TORCH_CHECK(false, "Unexpected IDeep version to do weight deserialization.");
#endif

  auto a = ideep::tensor(deserialized_ideep_desc, data_ptr);
  return at::native::new_with_itensor_mkldnn(std::move(a), dtype, device);
}

Tensor new_with_itensor_mkldnn(ideep::tensor&& it, std::optional<ScalarType> dtype, std::optional<Device> device) {
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  // TODO: support int64_t dims in ideep::tensor to avoid extra conversion
  auto dims = it.get_dims();
  IDeepTensorWrapperPtr handle = c10::make_intrusive<IDeepTensorWrapper>(std::move(it));
  caffe2::TypeMeta dtype_ = scalarTypeToTypeMeta(dtype_or_default(dtype));
  Device device_ = device_or_default(device);
  return detail::make_tensor<MKLDNNTensorImpl>(
    DispatchKeySet(DispatchKey::MkldnnCPU),
    dtype_, device_, handle,
    std::vector<int64_t>(dims.begin(), dims.end()));
}

ideep::tensor& itensor_from_mkldnn(const MKLDNNTensor& mkldnn_tensor) {
  TORCH_CHECK(mkldnn_tensor.is_mkldnn(),
             "itensor_from_mkldnn expects MKL-DNN tensor input");
  MKLDNNTensorImpl *mklimpl = static_cast<MKLDNNTensorImpl *>(mkldnn_tensor.unsafeGetTensorImpl());
  return mklimpl->unsafe_opaque_handle()->get_target();
}

int64_t nbytes_from_mkldnn(const Tensor& mkldnn_tensor) {
  ideep::tensor t = itensor_from_mkldnn(mkldnn_tensor);
  return t.get_desc().get_size();
}

ideep::tensor itensor_view_from_dense(const Tensor& tensor, bool from_const_data_ptr) {
  TORCH_CHECK(
      tensor.device().is_cpu(),
      "itensor_view_from_dense expects CPU tensor input");
  TORCH_CHECK(
      tensor.layout() == Layout::Strided,
      "itensor_view_from_dense expects dense tensor input");
  if (tensor.scalar_type() == ScalarType::Float) {
    return {{tensor.sizes().vec(),
            ideep::tensor::data_type::f32,
            tensor.strides().vec()},
            from_const_data_ptr ?
              const_cast<float*>(tensor.template const_data_ptr<float>()) :
              tensor.template data_ptr<float>()};
  }
  else if (tensor.scalar_type() == ScalarType::BFloat16) {
    return {{tensor.sizes().vec(),
            ideep::tensor::data_type::bf16,
            tensor.strides().vec()},
            from_const_data_ptr ?
              const_cast<BFloat16*>(tensor.template const_data_ptr<BFloat16>()) :
              tensor.template data_ptr<BFloat16>()};
  }
  else if (tensor.scalar_type() == ScalarType::Half) {
    return {{tensor.sizes().vec(),
            ideep::tensor::data_type::f16,
            tensor.strides().vec()},
            from_const_data_ptr ?
              const_cast<Half*>(tensor.template const_data_ptr<Half>()) :
              tensor.template data_ptr<Half>()};
  }
  else if (tensor.scalar_type() == ScalarType::Byte) {
    return {{tensor.sizes().vec(),
            ideep::tensor::data_type::u8,
            tensor.strides().vec()},
            from_const_data_ptr ?
              const_cast<void*>(tensor.const_data_ptr()) :
              tensor.data_ptr()};
  }
  else if (tensor.scalar_type() == ScalarType::Char) {
    return {{tensor.sizes().vec(),
            ideep::tensor::data_type::s8,
            tensor.strides().vec()},
            from_const_data_ptr ?
              const_cast<void*>(tensor.const_data_ptr()) :
              tensor.data_ptr()};
  }
  else {
    TORCH_CHECK(false, "itensor_view_from_dense expects float/bfloat16/half/int8 tensor input");
  }
}

ideep::tensor itensor_view_from_dense(
    const at::Tensor& tensor,
    const ideep::tensor::desc& desc) {
  TORCH_CHECK(
      tensor.device().is_cpu(),
      "itensor_view_from_dense expects CPU tensor input");
  TORCH_CHECK(
      tensor.layout() == at::Layout::Strided,
      "itensor_view_from_dense expects dense tensor input");
  TORCH_CHECK(
      tensor.scalar_type() == at::ScalarType::Float ||
          tensor.scalar_type() == at::ScalarType::BFloat16 ||
          tensor.scalar_type() == at::ScalarType::Half,
      "itensor_view_from_dense expects float, bfloat16 or half tensor input");
  return {desc, tensor.data_ptr()};
}

// Helper function for getting an ideep tensor out of an aten Tensor.
// Note in case the aten Tensor is a dense tensor, the returned ideep
// tensor is just a view of the storage of the aten dense tensor, so
// caller needs to make sure the aten dense tensor's lifetime is
// longer than the ideep tensor.
ideep::tensor itensor_from_tensor(const Tensor& tensor, bool from_const_data_ptr) {
  if (tensor.is_mkldnn()) {
    return itensor_from_mkldnn(tensor);
  } else {
    return itensor_view_from_dense(tensor, from_const_data_ptr);
  }
}

int set_verbose(int level) {
    return ideep::utils::set_verbose(level);
}

TORCH_LIBRARY_IMPL(mkldnn, MkldnnCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::data_ptr"),
      TORCH_FN(data_ptr_from_mkldnn));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_nbytes"),
      TORCH_FN(nbytes_from_mkldnn));
}

}

#endif // AT_ONEDNN_ENABLED()
