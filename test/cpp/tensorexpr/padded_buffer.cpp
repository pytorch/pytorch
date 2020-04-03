#include "test/cpp/tensorexpr/padded_buffer.h"

#include <sstream>
#include <c10/util/Logging.h>

namespace torch {
namespace jit {
namespace tensorexpr {

int PaddedBufferBase::Index(const std::vector<int>& indices) const {
  DCHECK_EQ(dims_.size(), indices.size());
  int total_index = 0;
  for (int i = 0; i < dims_.size(); i++) {
    total_index += indices[i] * strides_[i];
  }
  return total_index;
}

PaddedBufferBase::PaddedBufferBase(
    const std::vector<int>& dims,
    const std::string& name)
    : dims_(dims), name_(name), strides_(dims.size()) {
  // stride[0] = 1
  // stride[i] = stride[i-1]*dims[i-1], i > 0
  size_t ndim = dims.size();
  if (!ndim) {
    total_size_ = 0;
    return;
  }
  strides_[0] = 1;
  for (size_t i = 1; i < ndim; ++i) {
    strides_[i] = strides_[i - 1] * dims[i - 1];
  }
  total_size_ = strides_[ndim - 1] * dims[ndim - 1];
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
