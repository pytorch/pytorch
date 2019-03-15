#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <c10/core/OpaqueHandle.h>

#if AT_MKLDNN_ENABLED()
#include <ideep.hpp>
#include <caffe2/core/macros.h>
#ifndef CAFFE2_USE_MKLDNN
// Define IDEEP singletons here when IDEEP is not built with CAFFE2
#include <ideep_pin_singletons.hpp>
#endif

#endif

namespace at { namespace native {

#if AT_MKLDNN_ENABLED()

// TODO: move this helper function to a header file if other source files need it.
Tensor new_mkldnn_with_itensor(ideep::tensor&& ideep_tensor, const TensorOptions& options) {
  auto dims = ideep_tensor.get_dims();
  // NOTE: int32_t dims from ideep::tensor but sizes_ needs int64_t
  return detail::make_tensor<TensorImpl>(
    MkldnnCPUTensorId(), options.dtype(), false,
    c10::make_intrusive<c10::OpaqueHandle<ideep::tensor> >(std::move(ideep_tensor)),
    std::vector<int64_t>(dims.begin(), dims.end()));
}

// TODO: move this helper function to a header file if other source files need it.
ideep::tensor& itensor_from_mkldnn(const Tensor& mkldnn_tensor) {
  AT_ASSERTM(mkldnn_tensor.type_id() == MkldnnCPUTensorId(),
             "mkldnn_to_dense expects MKL-DNN tensor input");
  auto it_handle =
    (OpaqueHandle<ideep::tensor>*)mkldnn_tensor.unsafeGetTensorImpl()->unsafe_opaque_handle();
  return it_handle->get_handle();
}

Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor) {
  // TODO: share buffer without copy when MKLDNN tensor is in plain format
  ideep::tensor& stensor = itensor_from_mkldnn(mkldnn_tensor);
  auto dims = stensor.get_dims();
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  Tensor cpu_tensor = at::zeros(
    std::vector<int64_t>(dims.begin(), dims.end()),
    mkldnn_tensor.options().layout(c10::kStrided));
  stensor.reorder_to(cpu_tensor.template data<float>());
  return cpu_tensor;
}

Tensor dense_to_mkldnn(const Tensor& cpu_tensor) {
  // TODO: share CPU storage without explicit buffer copy here
  AT_ASSERTM(cpu_tensor.type_id() == CPUTensorId(),
             "dense_to_mkldnn expects dense CPU tensor input");
  AT_ASSERTM(cpu_tensor.scalar_type() == ScalarType::Float,
             "dense_to_mkldnn expects float tensor input");
  auto src_dims = cpu_tensor.sizes();
  ideep::tensor::dims dst_dims (src_dims.begin(), src_dims.end());
  ideep::tensor dtensor;
  dtensor.resize(dst_dims, ideep::tensor::data_type::f32);
  dtensor.reorder_from(dst_dims, ideep::tensor::data_type::f32,
                       (void*)(cpu_tensor.template data<float>()));
  return new_mkldnn_with_itensor(std::move(dtensor), cpu_tensor.options());
}

#else

Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor) {
  AT_ERROR("MKL-DNN build is disabled");
}

Tensor dense_to_mkldnn(const Tensor& cpu_tensor) {
  AT_ERROR("MKL-DNN build is disabled");
}

#endif // AT_MKLDNN_ENABLED()

}}
