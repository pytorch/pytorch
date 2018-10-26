#pragma once
#include "torch/csrc/jit/source_location.h"
#include "torch/csrc/jit/assertions.h"


namespace torch {
namespace jit {

// a range of a shared string 'file_' with functions to help debug by highlight
// that
// range.
struct SourceRange : public SourceLocation {
  SourceRange(
      std::shared_ptr<std::string> file_,
      size_t start_,
      size_t end_)
      : file_(std::move(file_)), start_(start_), end_(end_) {}
  const std::string text() const {
    return file().substr(start(), end() - start());
  }
  size_t size() const {
    return end() - start();
  }

  static const size_t CONTEXT = 10;
  void highlight(std::ostream& out) const;
  const std::string& file() const {
    return *file_;
  }
  const std::shared_ptr<std::string>& file_ptr() const {
    return file_;
  }
  size_t start() const {
    return start_;
  }
  size_t end() const {
    return end_;
  }

 private:
  std::shared_ptr<std::string> file_;
  size_t start_;
  size_t end_;
};

} // namespace jit
} // namespace torch
