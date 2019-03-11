#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()

#include <ideep.hpp>

#include "c10/util/intrusive_ptr.h"

namespace at { namespace native {

struct CAFFE2_API MKLDNNHandle : c10::intrusive_ptr_target {
private:
  ideep::tensor it_;

public:
  static Tensor tensor_from_handle(c10::intrusive_ptr<MKLDNNHandle> handle,
                                   const caffe2::TypeMeta& dtype) {
    auto dims = handle->get_ideep_tensor().get_dims();
    // TODO: Is it correct? Setting is_variable to false?
    // NOTE: int32_t dims from ideep::tensor but sizes_ needs int64_t
    return detail::make_tensor<TensorImpl>(
      MkldnnCPUTensorId(), dtype, false,
      c10::intrusive_ptr<c10::intrusive_ptr_target>(handle),
      std::vector<int64_t>(dims.begin(), dims.end()));
  }

  ideep::tensor& get_ideep_tensor() {
    return it_;
  }
};

}}

#endif // AT_MKLDNN_ENABLED()
