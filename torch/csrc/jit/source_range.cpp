#include <torch/csrc/jit/source_range.h>

namespace torch {
namespace jit {

// a range of a shared string 'file_' with
C10_EXPORT void SourceRange::highlight(std::ostream& out) const {
  if (size() == file_->size()) {
    // this is just the entire file, not a subset, so print it out.
    // primarily used to print out python stack traces
    out << *file_;
    return;
  }

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

} // namespace jit
} // namespace torch
