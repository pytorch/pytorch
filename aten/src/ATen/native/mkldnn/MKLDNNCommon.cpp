#include <c10/core/OpaqueHandle.h>
#include "MKLDNNCommon.h"

#if AT_MKLDNN_ENABLED()

#include <ideep.hpp>
#include <caffe2/core/macros.h>
#ifndef CAFFE2_USE_MKLDNN
// Define IDEEP singletons here when IDEEP is not built with CAFFE2
#include <ideep_pin_singletons.hpp>
#endif

namespace at { namespace native {

Tensor new_mkldnn_with_itensor(ideep::tensor&& ideep_tensor, const TensorOptions& options) {
  auto dims = ideep_tensor.get_dims();
  // NOTE: int32_t dims from ideep::tensor but sizes_ needs int64_t
  return detail::make_tensor<TensorImpl>(
    MkldnnCPUTensorId(), options.dtype(), false,
    c10::make_intrusive<c10::OpaqueHandle<ideep::tensor> >(std::move(ideep_tensor)),
    std::vector<int64_t>(dims.begin(), dims.end()));
}

ideep::tensor& itensor_from_mkldnn(const Tensor& mkldnn_tensor) {
  AT_ASSERTM(mkldnn_tensor.type_id() == MkldnnCPUTensorId(),
             "mkldnn_to_dense expects MKL-DNN tensor input");
  auto it_handle =
    (OpaqueHandle<ideep::tensor>*)mkldnn_tensor.unsafeGetTensorImpl()->unsafe_opaque_handle();
  return it_handle->get_handle();
}

}}

#endif // AT_MKLDNN_ENABLED()
