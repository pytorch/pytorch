#pragma once

#include "Exceptions.h"
#include <mkl_dfti.h>
#include <ATen/Tensor.h>

namespace at { namespace native {

class DftiDescriptor {
public:
  explicit DftiDescriptor(DFTI_CONFIG_VALUE precision, DFTI_CONFIG_VALUE signal_type,
                          MKL_LONG signal_ndim, MKL_LONG* sizes) {
    if (signal_ndim == 1) {
      MKL_DFTI_CHECK(DftiCreateDescriptor(&raw_desc, precision, signal_type, 1, sizes[0]));
    } else {
      MKL_DFTI_CHECK(DftiCreateDescriptor(&raw_desc, precision, signal_type, signal_ndim, sizes));
    }
  }

  const DFTI_DESCRIPTOR_HANDLE &get() const { return raw_desc; }

  ~DftiDescriptor() {
    MKL_DFTI_CHECK(DftiFreeDescriptor(&raw_desc));
  }
private:
  DFTI_DESCRIPTOR_HANDLE raw_desc;
};


}}  // at::native
