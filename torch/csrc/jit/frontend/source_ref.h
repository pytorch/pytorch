#pragma once

#include <functional>
#include <memory>

#include <ATen/core/ivalue.h>
#include <c10/macros/Export.h>
#include <torch/csrc/jit/frontend/source_range.h>

namespace torch {
namespace jit {

/**
 * SourceRef does two things:
 *   1. Owns a Source object.
 *   2. Serves as lookup key to the owned Source in associative containers, for
 *      runtime data aggregation.
 * We don't want to use std::shared_ptr<Source> directly because we want to
 * support heteogeneous lookup, and also shared_ptr is an implementation detail
 * which should be encapsulated.
 */
class TORCH_API SourceRef : public CustomClassHolder {
 public:
  explicit SourceRef(std::shared_ptr<Source> source_view)
      : source_view_(std::move(source_view)) {}
  bool operator==(const SourceRef& other) const {
    return source_view_ == other.source_view_;
  }
  bool operator<(const Source& other) const {
    return source_view_.get() < &other;
  }
  friend bool operator<(const Source& other, const SourceRef& self) {
    return &other < self.source_view_.get();
  }
  bool operator<(const SourceRef& other) const {
    return *this < *other.source_view_.get();
  }
  const Source* operator->() const {
    return source_view_.get();
  }

 private:
  std::shared_ptr<Source> source_view_;
};

} // namespace jit
} // namespace torch
