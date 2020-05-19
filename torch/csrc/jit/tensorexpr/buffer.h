#pragma once

#include <torch/csrc/jit/tensorexpr/ir.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// TODO: Merge this class with 'BufHandle'
class Buffer {
 public:
  Buffer(const BufHandle& data) : data_(data.node()) {
    if (data_->base_handle()->dtype() != kHandle) {
      throw malformed_input("Buffer dtype must be Handle");
    }

    std::vector<ExprHandle> stride_handles(ndim());
    for (int i = (int)ndim() - 1; i >= 0; i--) {
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
      : Buffer(BufHandle(name, dims, dtype)) {}

  const Buf* data() const {
    return data_;
  }
  Dtype dtype() const {
    return data_->dtype();
  }
  int ndim() const {
    return data_->ndim();
  }
  const Expr* dim(int index) const {
    return data_->dim(index);
  }
  std::vector<const Expr*> dims() const {
    return data_->dims();
  }

  // TODO: consider defer the storage flatten to a later stage.
  template <typename... Args>
  ExprHandle operator()(Args... args) const {
    return LoadValue(std::forward<Args>(args)...);
  }

  ExprHandle LoadValue(
      const ExprHandle& x,
      const ExprHandle& y,
      const ExprHandle& z) const {
    return Load::make(*this, {x, y, z}, ExprHandle(1));
  }
  ExprHandle LoadValue(const ExprHandle& x, const ExprHandle& y) const {
    return Load::make(*this, {x, y}, ExprHandle(1));
  }
  ExprHandle LoadValue(const ExprHandle& x) const {
    return Load::make(*this, {x}, ExprHandle(1));
  }

  template <typename T>
  ExprHandle call(const std::vector<T>& args) const {
    std::vector<ExprHandle> params(args.begin(), args.end());
    return LoadValue(params);
  }

 private:
  ExprHandle LoadValue(const std::vector<ExprHandle>& indices) const;

  const Buf* data_;
  std::vector<const Expr*> strides_;
};

inline ExprHandle Buffer::LoadValue(
    const std::vector<ExprHandle>& indices) const {
  return Load::make(*this, indices, ExprHandle(1));
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
