#pragma once

#include "../DataChannel.hpp"

#include <string>
#include <stdexcept>

namespace thd {

inline void assertTensorEqual(const Tensor& tensor1, const Tensor& tensor2,
                              std::string prefix = std::string()) {
  bool equal = tensor1.elementSize() == tensor2.elementSize() &&
               tensor1.numel() == tensor2.numel() &&
               tensor1.type() == tensor2.type();

  if (!prefix.empty())
    prefix = prefix + ": ";

  if (!equal)
    throw std::logic_error(prefix + "tensors are not equal in size or data type");
}

} // namespace thd
