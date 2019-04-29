#pragma once
#include <c10/util/Exception.h>
#include <c10/util/Color.h>
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
    c10::color::colorize(out);
    const std::string& str = file();

    // Find the start of the line to highlight
    size_t highlighted_line_start = start();
    while (highlighted_line_start > 0 &&
           str[highlighted_line_start - 1] != '\n') {
      --highlighted_line_start;
    }
    AT_ASSERT(
        highlighted_line_start == 0 || str[highlighted_line_start - 1] == '\n');

    // Find the end of the line to highlight
    size_t highlighted_line_end = start();
    while (highlighted_line_end < str.size() &&
           str[highlighted_line_end] != '\n') {
      ++highlighted_line_end;
    }
    AT_ASSERT(
        highlighted_line_end == str.size() ||
        str[highlighted_line_end] == '\n');

    // Find where in the string the context should start
    size_t context_start = highlighted_line_start;
    for (size_t num_lines_seen = 0; context_start > 0; --context_start) {
      if (str[context_start - 1] == '\n') {
        ++num_lines_seen;
      }
      if (num_lines_seen >= CONTEXT) {
        break;
      }
    }
    AT_ASSERT(context_start == 0 || str[context_start - 1] == '\n');

    // Find where in the string the context should end
    size_t context_end = highlighted_line_end;
    for (size_t num_lines_seen = 0; context_end < str.size(); ++context_end) {
      if (str[context_end] == '\n') {
        ++num_lines_seen;
      }
      if (num_lines_seen >= CONTEXT) {
        break;
      }
    }
    AT_ASSERT(context_end == str.size() || str[context_end] == '\n');

    // Beginning context + start of highlighted line
    out << str.substr(context_start, start() - context_start);

    // Highlighted section
    out << c10::color::red;
    out << str.substr(start(), end() - start());
    out << c10::color::reset;

    // Finish highlighted line
    out << str.substr(end(), highlighted_line_end - end());
    out << "\n";


    // Indent to highlighted section
    out << std::string(start() - highlighted_line_start, ' ');
    size_t len = std::min(size(), highlighted_line_end - start());
    out << c10::color::red << std::string(len, '~') << c10::color::reset
        << (len < size() ? "...  <--- HERE" : " <--- HERE");

    // Ending context
    out << str.substr(highlighted_line_end, context_end - highlighted_line_end);

    if (!str.empty() && str.back() != '\n') {
      out << "\n";
    }
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
