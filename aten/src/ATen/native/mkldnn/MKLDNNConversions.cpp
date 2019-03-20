#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include "MKLDNNCommon.h"

namespace at { namespace native {

#if AT_MKLDNN_ENABLED()

Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor) {
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
