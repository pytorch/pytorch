#pragma once

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

static Tensor clone_if_possible_with_memory_format(const Tensor& src) {
  if (self.is_sparse()) {
    return self.clone();
  } else if (input.is_mkldnn()) {
    return self.clone();
  } else {
    return self.clone(at::MemoryFormat::Contiguous);
  }
}
