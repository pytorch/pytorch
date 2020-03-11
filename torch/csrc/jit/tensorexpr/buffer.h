#pragma once

#include "torch/csrc/jit/tensorexpr/ir.h"

namespace torch {
namespace jit {
namespace tensorexpr {

class Buffer {
 public:
  Buffer(const Var& data, const Dtype& dtype, const std::vector<Expr>& dims)
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
      const std::vector<Expr>& dims)
      : Buffer(Var(name, kHandle), dtype, dims) {}

  const Var& data() const {
    return data_;
  }
  const Dtype& dtype() const {
    return dtype_;
  }
  int ndim() const {
    return dims_.size();
  }
  const Expr& dim(int index) const {
    return dims_[index];
  }

  // TODO: consider defer the storage flatten to a later stage.
  template <typename... Args>
  Expr operator()(Args... args) const {
    Expr index = Index(std::forward<Args>(args)...);
    return LoadValue(index);
  }

  template <typename T>
  Expr call(const std::vector<T>& args) const {
    std::vector<Expr> params(args.begin(), args.end());
    Expr index = Index(params);
    return LoadValue(index);
  }

 private:
  Expr Index(const Expr& x) const {
    CHECK(ndim() == 1);
    return x;
  }
  Expr Index(const Expr& x, const Expr& y) const {
    CHECK(ndim() == 2);
    return x * strides_[0] + y;
  }
  Expr Index(const Expr& x, const Expr& y, const Expr& z) const {
    CHECK(ndim() == 3);
    return x * strides_[0] + y * strides_[1] + z;
  }
  Expr Index(const Expr& x, const Expr& y, const Expr& z, const Expr& w) const {
    CHECK(ndim() == 4);
    return x * strides_[0] + y * strides_[1] + z * strides_[2] + w;
  }
  Expr Index(const std::vector<Expr>& indices) const {
    CHECK(ndim() == (int)indices.size());
    Expr total_index;
    for (size_t i = 0; i < indices.size(); i++) {
      Expr index;
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

  Expr LoadValue(const Expr& index) const;

  Var data_;
  Dtype dtype_;
  std::vector<Expr> dims_;
  std::vector<Expr> strides_;
  // TODO: add strides
};

inline Expr Buffer::LoadValue(const Expr& index) const {
  return Load::make(*this, index, Expr(1));
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
