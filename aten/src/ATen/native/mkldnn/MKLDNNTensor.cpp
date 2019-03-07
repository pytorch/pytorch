#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#include "MKLDNNTensorImpl.h"

namespace at { namespace native {

using c10::mkldnn::MKLDNNTensorImpl;

ideep::tensor& IdeepTensorFromMKLDNNTensor(const Tensor& self) {
  return ((MKLDNNTensorImpl*)self.unsafeGetTensorImpl())->get_ideep_tensor();
}

Tensor mkldnn_to_plainfmt(const Tensor& mkldnn_tensor) {
  // TODO: share buffer without copy when MKLDNN tensor is in plain format
  AT_ASSERT(mkldnn_tensor.type_id() == MkldnnCPUTensorId());
  Tensor cpu_tensor = at::zeros(mkldnn_tensor.sizes(), mkldnn_tensor.options().layout(c10::kStrided));
  ideep::tensor& stensor = IdeepTensorFromMKLDNNTensor(mkldnn_tensor);
  stensor.reorder_to(cpu_tensor.template data<float>());
  return cpu_tensor;
}

Tensor plainfmt_to_mkldnn(const Tensor& cpu_tensor) {
  // TODO: share CPU storage without explicit buffer copy here
  AT_ASSERT(cpu_tensor.type_id() == CPUTensorId());
  AT_ASSERT(cpu_tensor.scalar_type() == ScalarType::Float);
  Tensor mkldnn_tensor = detail::make_tensor<MKLDNNTensorImpl>(
    MkldnnCPUTensorId(), cpu_tensor.options().dtype());
  ideep::tensor& dtensor = IdeepTensorFromMKLDNNTensor(mkldnn_tensor);
  auto src_dims = cpu_tensor.sizes().vec();
  ideep::tensor::dims dst_dims (src_dims.begin(), src_dims.end());
  dtensor.resize(dst_dims, ideep::tensor::data_type::f32);
  dtensor.reorder_from(dst_dims, ideep::tensor::data_type::f32,
                       (void*)(cpu_tensor.template data<float>()));
  // TODO: sync dims with sizes_ in TensorImpl class since ideep::tensor
  // does not hold the same dims needed by IntArrayRef returned from sizes().
  // Need to avoid this explicit sync in the future.
  ((MKLDNNTensorImpl*)mkldnn_tensor.unsafeGetTensorImpl())->sync_sizes();
  return mkldnn_tensor;
}

}}
