#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <vector>
#include "c10/util/ArrayRef.h"

namespace torch {
namespace lazy {

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

class TORCH_API ExpandView : public TsNode {
 public:

  ExpandView(const Value& input, c10::IntArrayRef size): TsNode(
          OpKind(c10::Symbol::prim("expand_view")),
          {input},
          /*num_outputs=*/1,
          MHash(size)),
      size_(size.begin(), size.end())
      {
  SetShapeDeferred(
      [&]() { return Shape(input.shape().scalar_type(), size_); });
}

  std::string ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", size=(" << c10::Join(", ", size_)
     << ")";
  return ss.str();
};

  const std::vector<int64_t>& size() const {
    return size_;
  }

 private:
  std::vector<int64_t> size_;

};

class TORCH_API ExpandViewUpdate : public TsNode {
 public:

  ExpandViewUpdate(const Value& dest, const Value& result): TsNode(
          OpKind(c10::Symbol::prim("expand_view_update")),
          {dest, result},
          /*num_outputs=*/1,
          kNullOpt)
   {
  SetShapeDeferred(
      [&]() {
        // TODO: we should just set shape in a c-tor 
        return dest.shape(); 
      });
}

  std::string ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", size=(" << c10::Join(", ", size_)
     << ")";
  return ss.str();
};

  const std::vector<int64_t>& size() const {
    return size_;
  }

  bool is_scalar_expand() const {
    return is_scalar_expand_;
  }

 private:
  std::vector<int64_t> size_;
  std::vector<int64_t> source_size_;
  // True iff the input was a scalar and this was generated internally by a
  // lowering and not by user action. For some backends, this difference can be
  // material (for example setting strides according to eager semantics).
  bool is_scalar_expand_;
};


} // namespace lazy
} // namespace torch
