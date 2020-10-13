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

C10_EXPORT void format_stack_trace(
    std::ostream& out,
    const std::vector<StackEntry>& entries) {
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

  size_t range_end =
      (str.size() < end()
           ? str.size()
           : end()); // use instead of 'end()' because some ranges extend past
                     // the length of the source

  // determine CONTEXT line range
  size_t begin_line = start(); // beginning of lines to highlight
  size_t end_line = range_end;
  while (begin_line > 0 && str[begin_line - 1] != '\n')
    --begin_line;
  while (end_line < str.size() && str[end_line] != '\n')
    ++end_line;
  AT_ASSERT(begin_line == 0 || str[begin_line - 1] == '\n');
  AT_ASSERT(end_line == str.size() || str[end_line] == '\n');

  size_t begin_context = begin_line; // beginning of context, CONTEXT lines
                                     // before the highlight lines
  for (size_t i = 0; begin_context > 0; --begin_context) {
    if (str[begin_context - 1] == '\n') {
      ++i;
    }
    if (i >= context) {
      break;
    }
  }
  AT_ASSERT(begin_context == 0 || str[begin_context - 1] == '\n');

  size_t end_context =
      end_line; // end of context, CONTEXT lines after the highlight lines
  for (size_t i = 0; end_context < str.size(); ++end_context) {
    if (str[end_context] == '\n') {
      ++i;
    }
    if (i >= context) {
      break;
    }
  }
  AT_ASSERT(end_context == str.size() || str[end_context] == '\n');

  // print out location information
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
  // print out inital context
  out << str.substr(begin_context, start() - begin_context);
  size_t line_start = start();
  size_t line_end = range_end;
  if (highlight) {
    line_end = start();
    while (line_start < range_end) {
      // move line_end to end of line
      while (str[line_end] != '\n' && line_end < str.size()) {
        ++line_end;
      }
      // print line of code
      auto actual_line = str.substr(line_start, (line_end - line_start) + 1);
      out << actual_line;
      if (actual_line.back() != '\n') {
        out << "\n";
      }

      size_t empty_space = 0;
      size_t highlight_space = 0;
      size_t hightlight_begin = line_start;
      size_t highlight_end = line_start;
      // determine length of line which is being highlighted
      while (hightlight_begin > 0 && str[hightlight_begin - 1] != '\n') {
        --hightlight_begin;
      }
      while (highlight_end < range_end && str[highlight_end] != '\n') {
        ++highlight_end;
      }
      AT_ASSERT(hightlight_begin == 0 || str[hightlight_begin - 1] == '\n');
      AT_ASSERT(highlight_end == range_end || str[highlight_end] == '\n');
      // determine amount of empty space vs highlighted space
      for (size_t i = hightlight_begin; i < highlight_end; i++) {
        if (str[i] == ' ' || i < start()) {
          empty_space++;
        } else {
          break;
        }
      }
      highlight_space = highlight_end - hightlight_begin - empty_space;
      if (highlight_space > 0) {
        // some ranges are off and include empty white space on new lines which
        // don't need to be printed
        bool more_lines = false;
        for (size_t i = line_end; i <= range_end; i++) {
          if (str[i] != '\n' && str[i] != ' ') {
            more_lines = true;
          }
        }
        out << std::string(empty_space, ' ');
        out << std::string(highlight_space, '~');
        out << (more_lines && line_end != range_end ? "\n" : " <--- HERE\n");
      }
      ++line_end;
      line_start = line_end;
    }
  } else {
    // print out code with no highlight
    out << str.substr(start(), range_end - start());
  }
  // print out ending context
  if (line_end <= str.size()) {
    auto line_substr = str.substr(line_end, end_context - line_end);
    out << line_substr;
    if (!line_substr.empty() && line_substr.back() != '\n') {
      out << "\n";
    }
  }
}

} // namespace jit
} // namespace torch
