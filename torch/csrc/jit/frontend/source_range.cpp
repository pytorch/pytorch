#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>

namespace torch {
namespace jit {

c10::optional<SourceRange> Source::findSourceRangeThatGenerated(
    const SourceRange& range) {
  if (!gen_ranges_) {
    return c10::nullopt;
  }
  return gen_ranges_->findSourceRangeThatGenerated(range);
}

C10_EXPORT void SourceRange::highlight(std::ostream& out) const {
  // Retrieve original SourceRange, if present.
  if (auto orig_source_range = findSourceRangeThatGenerated()) {
    orig_source_range->highlight(out);
    out << "Serialized ";
  }
  print_with_context(out, CONTEXT, true, "");
}

C10_EXPORT void format_stack_trace(std::ostream& out, const std::vector<StackEntry>& entries) {
  bool has_orig_ranges = false;
  std::vector<SourceRange> orig_ranges;
  // gather original ranges. if we have a situation where we do not have orig
  // ranges for some frames, we still want to report them for the frames we do
  // have,
  //  so substitute the current range for that frame
  for (const StackEntry& entry : entries) {
    if (auto orig_source_range = entry.range.findSourceRangeThatGenerated()) {
      orig_ranges.emplace_back(std::move(orig_source_range.value()));
      has_orig_ranges = true;
    } else {
      orig_ranges.emplace_back(entry.range);
    }
  }
  out << "Traceback of TorchScript";
  if (has_orig_ranges) {
    out << ", serialized code";
  }
  out << " (most recent call last):\n";
  for (const StackEntry& entry : entries) {
    entry.range.print_with_context(
        out, SourceRange::CONTEXT, true, entry.filename);
  }
  if (has_orig_ranges) {
    out << "\nTraceback of TorchScript, original code (most recent call last):\n";
    auto it = entries.begin();
    for (const SourceRange& range : orig_ranges) {
      range.print_with_context(
          out, SourceRange::CONTEXT, true, (*it++).filename);
    }
  }
}

C10_EXPORT void SourceRange::print_with_context(
    std::ostream& out,
    size_t context,
    bool highlight,
    const std::string& funcname) const {
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
    if (i >= context) {
      break;
    }
  }
  AT_ASSERT(begin_highlight == 0 || str[begin_highlight - 1] == '\n');

  size_t end_highlight =
      end_line; // end of context, CONTEXT lines after the highlight line
  for (size_t i = 0; end_highlight < str.size(); ++end_highlight) {
    if (str[end_highlight] == '\n')
      ++i;
    if (i >= context) {
      break;
    }
  }
  AT_ASSERT(end_highlight == str.size() || str[end_highlight] == '\n');

  if (auto flc = file_line_col()) {
    std::string filename;
    size_t line, col;
    std::tie(filename, line, col) = *flc;
    out << "  File \"" << filename << "\", line " << line;
    if (funcname != "") {
      out << ", in " << funcname;
    }
    out << "\n";
  }
  out << str.substr(begin_highlight, end_line - begin_highlight) << "\n";
  if (highlight) {
    out << std::string(start() - begin_line, ' ');
    size_t len = std::min(size(), end_line - start());
    out << std::string(len, '~')
        << (len < size() ? "...  <--- HERE" : " <--- HERE");
  }
  auto line_substr = str.substr(end_line, end_highlight - end_line);
  out << line_substr;
  if (!line_substr.empty() && line_substr.back() != '\n')
    out << "\n";
}

} // namespace jit
} // namespace torch
