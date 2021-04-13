#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/types.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class AsStridedViewUpdate : public Node {
 public:
  AsStridedViewUpdate(const Value& target, const Value& input,
                      std::vector<lazy_tensors::int64> size,
                      std::vector<lazy_tensors::int64> stride,
                      lazy_tensors::int64 storage_offset);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const std::vector<lazy_tensors::int64>& size() const { return size_; }

  const std::vector<lazy_tensors::int64>& stride() const { return stride_; }

  lazy_tensors::int64 storage_offset() const { return storage_offset_; }

 private:
  std::vector<lazy_tensors::int64> size_;
  std::vector<lazy_tensors::int64> stride_;
  lazy_tensors::int64 storage_offset_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
