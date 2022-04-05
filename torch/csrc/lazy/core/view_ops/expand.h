#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <vector>

namespace torch {
namespace lazy {

// Expand can be constructed as a regular view op, or a scalar expand.
// For the latter, it loses the is_alias_of check capability provided by view ops.
// The above might not hold true after functionalization (@bdhirsh).
class TORCH_API Expand : public TsNode {
 public:
  Expand(const Value& input, std::vector<int64_t> size, bool is_scalar_expand);

  std::string ToString() const override;

  const std::vector<int64_t>& size() const {
    return size_;
  }

  bool is_scalar_expand() const {
    return is_scalar_expand_;
  }

 private:
  std::vector<int64_t> size_;
  // True iff the input was a scalar and this was generated internally by a
  // lowering and not by user action. For some backends, this difference can be
  // material (for example setting strides according to eager semantics).
  bool is_scalar_expand_;
};

} // namespace lazy
} // namespace torch
