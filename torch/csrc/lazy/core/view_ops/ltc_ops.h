#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <mutex>
#include <string>

namespace torch {
namespace lazy {

class OpKindWrapper {
 public:
  OpKindWrapper(const char* name) : name_(name) {}

  const OpKind& operator*() const {
    return get();
  }

  operator OpKind() const {
    return get();
  }

 private:
  const OpKind& get() const {
    std::call_once(once_, [this]() { op_kind_ = OpKind::Get(name_); });
    return op_kind_;
  }

  const char* name_;
  mutable OpKind op_kind_;
  mutable std::once_flag once_;
};

extern const OpKindWrapper ltc_as_strided_view_update;
extern const OpKindWrapper ltc_diagonal_view_update;
extern const OpKindWrapper ltc_narrow_view_update;
extern const OpKindWrapper ltc_select_view_update;

} // namespace lazy
} // namespace torch
