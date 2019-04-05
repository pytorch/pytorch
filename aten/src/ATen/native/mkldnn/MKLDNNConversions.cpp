#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at { namespace native {

#if AT_MKLDNN_ENABLED()

Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor) {
  ideep::tensor& stensor = itensor_from_mkldnn(mkldnn_tensor);
  auto dims = stensor.get_dims();
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  Tensor cpu_tensor = at::empty(
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
  // TODO: consider to convert non-contiguous tensor to `ideep::tensor` directly.
  auto cpu_tensor_cont = cpu_tensor.contiguous();
  Tensor mkldnn_tensor = new_with_sizes_mkldnn(cpu_tensor_cont.sizes(), cpu_tensor_cont.options());
  ideep::tensor& dtensor = itensor_from_mkldnn(mkldnn_tensor);
  dtensor.reorder_from(dtensor.get_dims(),
                       ideep::tensor::data_type::f32,
                       (cpu_tensor_cont.template data<float>()));
  return mkldnn_tensor;
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
