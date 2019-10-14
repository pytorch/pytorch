#pragma once

#include <ATen/ATen.h>

namespace at {

static Tensor clone_if_possible_with_memory_format(const Tensor& self) {
  if (self.is_sparse()) {
    return self.clone();
  } else if (self.is_mkldnn()) {
    return self.clone();
  } else {
    return self.clone(at::MemoryFormat::Contiguous);
  }
}

}

