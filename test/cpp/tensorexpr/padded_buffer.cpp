#include "test/cpp/tensorexpr/padded_buffer.h"

#include <c10/util/Logging.h>
#include <sstream>

namespace torch {
namespace jit {
namespace tensorexpr {

int PaddedBufferBase::Index(const std::vector<int>& indices) const {
  DCHECK_EQ(dims_.size(), indices.size());
  int total_index = 0;
  for (size_t i = 0; i < dims_.size(); i++) {
    total_index += indices[i] * strides_[i];
  }
  return total_index;
}

PaddedBufferBase::PaddedBufferBase(
    const std::vector<int>& dims,
    // NOLINTNEXTLINE(modernize-pass-by-value)
    const std::string& name)
    : dims_(dims), name_(name), strides_(dims.size()) {
  for (int i = (int)dims.size() - 1; i >= 0; --i) {
    if (i == (int)dims.size() - 1) {
      strides_[i] = 1;
    } else {
      strides_[i] = strides_[i + 1] * dims[i + 1];
    }
  }
  total_size_ = strides_[0] * dims[0];
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
