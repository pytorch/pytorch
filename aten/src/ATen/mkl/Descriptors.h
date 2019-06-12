#pragma once

#include "Exceptions.h"
#include <mkl_dfti.h>
#include <ATen/Tensor.h>

namespace at { namespace native {

struct DftiDescriptorDeleter {
  void operator()(DFTI_DESCRIPTOR* desc) {
    if (desc != nullptr) {
      MKL_DFTI_CHECK(DftiFreeDescriptor(&desc));
    }
  }
};

class DftiDescriptor {
public:
  void init(DFTI_CONFIG_VALUE precision, DFTI_CONFIG_VALUE signal_type, MKL_LONG signal_ndim, MKL_LONG* sizes) {
    if (desc_ != nullptr) {
      throw std::runtime_error("DFTI DESCRIPTOR can only be initialized once");
    }
    DFTI_DESCRIPTOR *raw_desc;
    if (signal_ndim == 1) {
      MKL_DFTI_CHECK(DftiCreateDescriptor(&raw_desc, precision, signal_type, 1, sizes[0]));
    } else {
      MKL_DFTI_CHECK(DftiCreateDescriptor(&raw_desc, precision, signal_type, signal_ndim, sizes));
    }
    desc_.reset(raw_desc);
  }

  DFTI_DESCRIPTOR *get() const {
    if (desc_ == nullptr) {
      throw std::runtime_error("DFTI DESCRIPTOR has not been initialized");
    }
    return desc_.get();
  }

private:
  std::unique_ptr<DFTI_DESCRIPTOR, DftiDescriptorDeleter> desc_;
};


}}  // at::native
