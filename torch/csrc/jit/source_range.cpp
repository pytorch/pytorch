#include <torch/csrc/jit/source_range.h>
#include <torch/csrc/jit/source_range_serialization.h>

namespace torch {
namespace jit {

c10::optional<SourceRange> Source::findSourceRangeThatGenerated(
    const SourceRange& range) {
  if (!gen_ranges_) {
    return c10::nullopt;
  }
  return gen_ranges_->findSourceRangeThatGenerated(range);
}

// a range of a shared string 'file_' with
C10_EXPORT void SourceRange::highlight(std::ostream& out) const {
  // This is an empty SourceRange, used as a sentinel value.
  if (!source_) {
    return;
  }
  const std::string& str = source_->text();
  if (size() == str.size()) {
    // this is just the entire file, not a subset, so print it out.
    // primarily used to print out python stack traces
    out << str;
    return;
  }

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

  if (auto flc = file_line_col()) {
    std::string filename;
    size_t line, col;
    std::tie(filename, line, col) = *flc;
    out << "at " << filename << ":" << line << ":" << col << "\n";
  }
  out << str.substr(begin_highlight, end_line - begin_highlight) << "\n";
  out << std::string(start() - begin_line, ' ');
  size_t len = std::min(size(), end_line - start());
  out << std::string(len, '~')
      << (len < size() ? "...  <--- HERE" : " <--- HERE");
  auto line_substr = str.substr(end_line, end_highlight - end_line);
  out << line_substr;
  if (!line_substr.empty() && line_substr.back() != '\n')
    out << "\n";
  // Retrieve original SourceRange, if present.
  if (auto orig_source_range = findSourceRangeThatGenerated()) {
    out << "Compiled from code ";
    orig_source_range->highlight(out);
  }
}

} // namespace jit
} // namespace torch
