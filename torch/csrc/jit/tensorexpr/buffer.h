#pragma once

#include "torch/csrc/jit/tensorexpr/ir.h"

namespace torch {
namespace jit {
namespace tensorexpr {

class Buffer {
 public:
  Buffer(const VarHandle& data, const Dtype& dtype, const std::vector<ExprHandle>& dims)
      : data_(data), dtype_(dtype), dims_(dims), strides_(dims.size()) {
    CHECK_EQ(data.dtype(), kHandle);
    for (int i = ndim() - 1; i >= 0; i--) {
      if (i == ndim() - 1) {
        strides_[i] = 1;
      } else {
        strides_[i] = strides_[i + 1] * dim(i + 1);
      }
    }
  }
  Buffer(
      const std::string& name,
      const Dtype& dtype,
      const std::vector<ExprHandle>& dims)
      : Buffer(VarHandle(name, kHandle), dtype, dims) {}

  const VarHandle& data() const {
    return data_;
  }
  const Dtype& dtype() const {
    return dtype_;
  }
  int ndim() const {
    return dims_.size();
  }
  const ExprHandle& dim(int index) const {
    return dims_[index];
  }

  // TODO: consider defer the storage flatten to a later stage.
  template <typename... Args>
  ExprHandle operator()(Args... args) const {
    ExprHandle index = Index(std::forward<Args>(args)...);
    return LoadValue(index);
  }

  template <typename T>
  ExprHandle call(const std::vector<T>& args) const {
    std::vector<ExprHandle> params(args.begin(), args.end());
    ExprHandle index = Index(params);
    return LoadValue(index);
  }

 private:
  ExprHandle Index(const ExprHandle& x) const {
    CHECK(ndim() == 1);
    return x;
  }
  ExprHandle Index(const ExprHandle& x, const ExprHandle& y) const {
    CHECK(ndim() == 2);
    return x * strides_[0] + y;
  }
  ExprHandle Index(const ExprHandle& x, const ExprHandle& y, const ExprHandle& z) const {
    CHECK(ndim() == 3);
    return x * strides_[0] + y * strides_[1] + z;
  }
  ExprHandle Index(const ExprHandle& x, const ExprHandle& y, const ExprHandle& z, const ExprHandle& w) const {
    CHECK(ndim() == 4);
    return x * strides_[0] + y * strides_[1] + z * strides_[2] + w;
  }
  ExprHandle Index(const std::vector<ExprHandle>& indices) const {
    CHECK(ndim() == (int)indices.size());
    ExprHandle total_index;
    for (size_t i = 0; i < indices.size(); i++) {
      ExprHandle index;
      if (i == indices.size() - 1) {
        index = indices[i];
      } else {
        index = indices[i] * strides_[i];
      }
      if (i == 0) {
        total_index = index;
      } else {
        total_index = total_index + index;
      }
    }
    return total_index;
  }

  ExprHandle LoadValue(const ExprHandle& index) const;

  VarHandle data_;
  Dtype dtype_;
  std::vector<ExprHandle> dims_;
  std::vector<ExprHandle> strides_;
  // TODO: add strides
};

inline ExprHandle Buffer::LoadValue(const ExprHandle& index) const {
  return Load::make(*this, index, ExprHandle(1));
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
