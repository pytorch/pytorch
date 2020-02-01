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

C10_EXPORT void SourceRange::highlight(
    std::ostream& out,
    const std::string& context_name) const {
  print_with_context(out, CONTEXT, true, context_name);
}

C10_EXPORT void SourceRange::print_with_context(
    std::ostream& out,
    size_t context,
    bool highlight,
    const std::string& context_name) const {
  // This is an empty SourceRange, used as a sentinel value.
  if (!source_) {
    return;
  }
  // Retrieve original SourceRange, if present.
  if (auto orig_source_range = findSourceRangeThatGenerated()) {
    orig_source_range->highlight(out);
    out << "Serialized ";
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
  TORCH_INTERNAL_ASSERT(begin_line == 0 || str[begin_line - 1] == '\n');
  TORCH_INTERNAL_ASSERT(end_line == str.size() || str[end_line] == '\n');

  size_t begin_context = begin_line; // beginning of context, CONTEXT lines
                                     // before the highlight line
  size_t num_lines_before = 0;
  for (; begin_context > 0; --begin_context) {
    if (str[begin_context - 1] == '\n') {
      ++num_lines_before;
    }

    if (num_lines_before >= context) {
      break;
    }
  }
  TORCH_INTERNAL_ASSERT(begin_context == 0 || str[begin_context - 1] == '\n');

  size_t num_lines_after = 0;
  size_t end_context =
      end_line; // end of context, CONTEXT lines after the highlight line
  for (; end_context < str.size(); ++end_context) {
    if (str[end_context] == '\n') {
      ++num_lines_after;
    }

    if (num_lines_after >= context) {
      break;
    }
  }
  TORCH_INTERNAL_ASSERT(end_context == str.size() || str[end_context] == '\n');

  if (auto flc = file_line_col()) {
    std::string filename;
    size_t line, col;
    std::tie(filename, line, col) = *flc;
    out << "  File \"" << filename << "\", line " << line;
    if (context_name != "") {
      out << ", in " << context_name;
    }
    out << ":\n";
  }

  const auto full_context = str.substr(begin_context, end_context - begin_context);
  const auto lines = c10::split(full_context, '\n', /*ignoreEmpty=*/false);
  const size_t important_line = num_lines_before;

  out << str.substr(begin_context, end_line - begin_context) << "\n";
  if (highlight) {
    out << std::string(start() - begin_line, ' ');
    size_t len = std::min(size(), end_line - start());

    // Add a visual pointer to hard-to-understand squigglies
    std::string here_hint;
    if (len < size()) { // happens when the range crosses two lines
      here_hint = "...  <--- HERE";
    } else if (len < 3) { // small ranges may be hard to spot on their own
      here_hint = "  <--- HERE";
    }
    out << std::string(len, '^') << here_hint;
  }
  auto line_substr = str.substr(end_line, end_context - end_line);
  out << line_substr;
  if (!line_substr.empty() && line_substr.back() != '\n')
    out << "\n";
}

} // namespace jit
} // namespace torch
