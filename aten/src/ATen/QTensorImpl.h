#pragma once

#include <c10/core/TensorImpl.h>
#include <ATen/Quantizer.h>
#include <c10/util/Exception.h>

namespace at {

struct CAFFE2_API QTensorImpl : public c10::TensorImpl {
public:
  QTensorImpl(Storage&& storage, TensorTypeId type_id, bool is_variable, std::shared_ptr<at::Quantizer> quantizer);

  at::Quantizer* quantizer() {
    return quantizer_.get();
  }

private:
  std::shared_ptr<at::Quantizer> quantizer_;
};

} // namespace at
