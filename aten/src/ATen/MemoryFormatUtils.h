#pragma once

#include <ATen/ATen.h>

namespace at {

Tensor try_clone_as_contiguous(const Tensor& self) {
  if (self.is_sparse()) {
    return self.clone();
  } else if (self.is_mkldnn()) {
    return self.clone();
  } else {
    return self.clone(at::MemoryFormat::Contiguous);
  }
}

}
