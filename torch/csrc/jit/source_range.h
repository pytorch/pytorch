#pragma once
#include <c10/util/Exception.h>
#include <torch/csrc/jit/source_location.h>

#include <algorithm>
#include <memory>

namespace torch {
namespace jit {

// a range of a shared string 'file_' with functions to help debug by highlight
// that
// range.
struct SourceRange : public SourceLocation {
  SourceRange(std::shared_ptr<std::string> file_, size_t start_, size_t end_)
      : file_(std::move(file_)), start_(start_), end_(end_) {}
  const std::string text() const {
    return file().substr(start(), end() - start());
  }
  size_t size() const {
    return end() - start();
  }

  static const size_t CONTEXT = 10;
  void highlight(std::ostream& out) const override {
    const std::string& str = file();
    size_t begin_line = start(); // beginning of line to highlight
    size_t end_line = start(); // end of line to highlight
    while (begin_line > 0 && str[begin_line - 1] != '\n')
      --begin_line;
    while (end_line < str.size() && str[end_line] != '\n')
      ++end_line;
    AT_ASSERT(begin_line == 0 || str[begin_line - 1] == '\n');
    AT_ASSERT(end_line == str.size() || str[end_line] == '\n');

    size_t begin_highlight = begin_line; // beginning of context, CONTEXT lines
                                         // before the highlight line
    for (size_t i = 0; begin_highlight > 0; --begin_highlight) {
      if (str[begin_highlight - 1] == '\n')
        ++i;
      if (i >= CONTEXT)
        break;
    }
    AT_ASSERT(begin_highlight == 0 || str[begin_highlight - 1] == '\n');

    size_t end_highlight =
        end_line; // end of context, CONTEXT lines after the highlight line
    for (size_t i = 0; end_highlight < str.size(); ++end_highlight) {
      if (str[end_highlight] == '\n')
        ++i;
      if (i >= CONTEXT)
        break;
    }
    AT_ASSERT(end_highlight == str.size() || str[end_highlight] == '\n');

    out << str.substr(begin_highlight, end_line - begin_highlight) << "\n";
    out << std::string(start() - begin_line, ' ');
    size_t len = std::min(size(), end_line - start());
    out << std::string(len, '~')
        << (len < size() ? "...  <--- HERE" : " <--- HERE");
    out << str.substr(end_line, end_highlight - end_line);
    if (!str.empty() && str.back() != '\n')
      out << "\n";
  }
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
