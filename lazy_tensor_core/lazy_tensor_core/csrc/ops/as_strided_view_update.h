#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <vector>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class AsStridedViewUpdate : public torch::lazy::TsNode {
 public:
  AsStridedViewUpdate(const torch::lazy::Value& target,
                      const torch::lazy::Value& input,
                      std::vector<int64_t> size, std::vector<int64_t> stride,
                      int64_t storage_offset);

  std::string ToString() const override;

  const std::vector<int64_t>& size() const { return size_; }

  const std::vector<int64_t>& stride() const { return stride_; }

  int64_t storage_offset() const { return storage_offset_; }

 private:
  std::vector<int64_t> size_;
  std::vector<int64_t> stride_;
  int64_t storage_offset_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
