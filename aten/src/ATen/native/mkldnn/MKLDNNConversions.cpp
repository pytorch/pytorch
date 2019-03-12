#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#include "MKLDNNHandle.h"

namespace at { namespace native {

#if AT_MKLDNN_ENABLED()

ideep::tensor& IdeepTensorFromMKLDNNTensor(const Tensor& self) {
  return ((MKLDNNHandle*)self.unsafeGetTensorImpl()->unsafe_opaque_handle())->get_ideep_tensor();
}

Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor) {
  // TODO: share buffer without copy when MKLDNN tensor is in plain format
  AT_ASSERT(mkldnn_tensor.type_id() == MkldnnCPUTensorId());
  ideep::tensor& stensor = IdeepTensorFromMKLDNNTensor(mkldnn_tensor);
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
  AT_ASSERT(cpu_tensor.type_id() == CPUTensorId());
  AT_ASSERT(cpu_tensor.scalar_type() == ScalarType::Float);
  c10::intrusive_ptr<MKLDNNHandle> mkldnn_handle = c10::make_intrusive<MKLDNNHandle>();
  ideep::tensor& dtensor = mkldnn_handle->get_ideep_tensor();
  auto src_dims = cpu_tensor.sizes();
  ideep::tensor::dims dst_dims (src_dims.begin(), src_dims.end());
  dtensor.resize(dst_dims, ideep::tensor::data_type::f32);
  dtensor.reorder_from(dst_dims, ideep::tensor::data_type::f32,
                       (void*)(cpu_tensor.template data<float>()));
  return MKLDNNHandle::tensor_from_handle(mkldnn_handle, cpu_tensor.options().dtype());
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
