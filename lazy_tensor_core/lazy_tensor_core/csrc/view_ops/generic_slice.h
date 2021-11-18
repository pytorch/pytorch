#pragma once

#include "lazy_tensor_core/csrc/view_ops/opcode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class GenericSlice : public BaseNode {
 public:
  GenericSlice(const torch::lazy::Value& input,
               c10::ArrayRef<int64_t> base_indices,
               c10::ArrayRef<int64_t> sizes);

  std::string ToString() const override;

  const std::vector<int64_t>& base_indices() const { return base_indices_; }

  const std::vector<int64_t>& sizes() const { return sizes_; }

 private:
  std::vector<int64_t> base_indices_;
  std::vector<int64_t> sizes_;
};

class GenericSliceReverse : public BaseNode {
 public:
  GenericSliceReverse(const torch::lazy::Value& input,
                      const torch::lazy::Value& source,
                      c10::ArrayRef<int64_t> base_indices);

  std::string ToString() const override;

  const std::vector<int64_t>& base_indices() const { return base_indices_; }

 private:
  std::vector<int64_t> base_indices_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
