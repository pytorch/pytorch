#pragma once

#include "torch/csrc/jit/tensorexpr/ir.h"

namespace torch {
namespace jit {
namespace tensorexpr {

class Buffer {
 public:
  Buffer(const VarHandle& data, const Dtype& dtype, const std::vector<ExprHandle>& dims)
      : data_(data.node()), dtype_(dtype), dims_(ExprHandleVectorToExprVector(dims)) {
    CHECK_EQ(data.dtype(), kHandle);
    std::vector<ExprHandle> stride_handles(dims.size());
    for (int i = ndim() - 1; i >= 0; i--) {
      if (i == ndim() - 1) {
        stride_handles[i] = 1;
      } else {
        stride_handles[i] = stride_handles[i + 1] * ExprHandle(dim(i + 1));
      }
    }
    strides_ = ExprHandleVectorToExprVector(stride_handles);
  }
  Buffer(
      const std::string& name,
      const Dtype& dtype,
      const std::vector<ExprHandle>& dims)
      : Buffer(VarHandle(name, kHandle), dtype, dims) {}

  const Var* data() const {
    return data_;
  }
  const Dtype& dtype() const {
    return dtype_;
  }
  int ndim() const {
    return dims_.size();
  }
  const Expr* dim(int index) const {
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
    return x * ExprHandle(strides_[0]) + y;
  }
  ExprHandle Index(const ExprHandle& x, const ExprHandle& y, const ExprHandle& z) const {
    CHECK(ndim() == 3);
    return x * ExprHandle(strides_[0]) + y * ExprHandle(strides_[1]) + z;
  }
  ExprHandle Index(const ExprHandle& x, const ExprHandle& y, const ExprHandle& z, const ExprHandle& w) const {
    CHECK(ndim() == 4);
    return x * ExprHandle(strides_[0]) + y * ExprHandle(strides_[1]) + z * ExprHandle(strides_[2]) + w;
  }
  ExprHandle Index(const std::vector<ExprHandle>& indices) const {
    CHECK(ndim() == (int)indices.size());
    ExprHandle total_index;
    for (size_t i = 0; i < indices.size(); i++) {
      ExprHandle index;
      if (i == indices.size() - 1) {
        index = indices[i];
      } else {
        index = indices[i] * ExprHandle(strides_[i]);
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

  const Var* data_;
  Dtype dtype_;
  std::vector<const Expr*> dims_;
  std::vector<const Expr*> strides_;
  // TODO: add strides
};

inline ExprHandle Buffer::LoadValue(const ExprHandle& index) const {
  return Load::make(*this, index, ExprHandle(1));
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
