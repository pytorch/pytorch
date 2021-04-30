#pragma once

#include <functional>
#include <memory>

#include <c10/macros/Export.h>
#include <torch/csrc/jit/frontend/source_range.h>

namespace torch {
namespace jit {

class TORCH_API SourceRef {
 public:
  explicit SourceRef(std::shared_ptr<Source> source)
      : source_(std::move(source)) {}
  bool operator==(const SourceRef& other) const {
    return source_ == other.source_;
  }
  bool operator<(const Source& other) const {
    return source_.get() < &other;
  }
  friend bool operator<(const Source& other, const SourceRef& self) {
    return &other < self.source_.get();
  }
  bool operator<(const SourceRef& other) const {
    return *this < *other.source_.get();
  }
  const Source* operator->() const {
    return source_.get();
  }

 private:
  friend class std::hash<SourceRef>;

  std::shared_ptr<Source> source_;
};

} // namespace jit
} // namespace torch

namespace std {

template <>
struct hash<torch::jit::SourceRef> {
  size_t operator()(const torch::jit::SourceRef& sr) const noexcept {
    return std::hash<decltype(sr.source_)>{}(sr.source_);
  }
};

} // namespace std
