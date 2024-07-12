#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>
#include <iostream>

namespace torch::jit {

// A stringlike class backed by a vector of string_view
// the string represented are logically the concatenation of  the string_views
// This has advantage of not needing continues memory.
StringCordView::StringCordView() {
  accumulated_sizes_.push_back(0);
}

StringCordView::StringCordView(
    std::vector<std::string_view> inputs,
    std::vector<std::shared_ptr<std::string>> ownerships)
    : pieces_(std::move(inputs)), owned_strings_(std::move(ownerships)) {
  accumulated_sizes_.push_back(0);
  size_t running_sum = 0;
  for (auto& s : pieces_) {
    if (!s.empty()) {
      running_sum += s.size();
      accumulated_sizes_.push_back(running_sum);
    }
  }
}

size_t StringCordView::find(const std::string& tok, size_t start) const {
  if (tok.empty()) {
    return 0;
  }

  if ((size() - start) < tok.size()) {
    return std::string::npos;
  }

  Iterator begin = iter_for_pos(start);
  Iterator end_iter = end();
  size_t offset = start;
  for (; begin != end_iter; ++begin, ++offset) {
    if (*begin == tok[0]) {
      auto mis = std::mismatch(begin, end_iter, tok.begin(), tok.end());
      if (mis.second == tok.end()) {
        // no mismatch, and second string (tok) is exhausted.
        return offset;
      }
      if (mis.first == end_iter) {
        // this str is exhausted but tok is not
        return std::string::npos;
      }
    }
  }
  return std::string::npos;
}

size_t StringCordView::find_regex(const std::string& tok, size_t start) const {
  if (tok.empty()) {
    return 0;
  }

  const std::string& target = this->substr(start, this->size()).str();
  std::smatch sm;
  const std::regex re(tok);

  auto regex_found = std::regex_search(target, sm, re);

  return regex_found ? sm.position(0) : std::string::npos;
}

StringCordView StringCordView::substr(size_t start, size_t size) const {
  std::vector<std::string_view> pieces;
  std::vector<std::shared_ptr<std::string>> ownerships;
  if (start >= this->size()) {
    // out of bounds
    return StringCordView();
  }
  if (start + size >= this->size()) {
    size = this->size() - start;
  }
  Iterator begin = iter_for_pos(start);
  Iterator end = iter_for_pos(start + size);

  if (begin.line_ == end.line_) {
    // same line
    pieces.push_back(pieces_[begin.line_].substr(begin.pos_, size));
  } else {
    pieces.push_back(pieces_[begin.line_].substr(begin.pos_));

    size_t last_line = pieces_.size();
    if (end != this->end() && end.line_ < last_line) {
      // end is within the string
      last_line = end.line_;
    }
    for (size_t i = begin.line_ + 1; i < last_line; i++) {
      pieces.push_back(pieces_[i]);
    }
    if (end != this->end()) {
      pieces.push_back(pieces_[end.line_].substr(0, end.pos_));
    }
  }

  // share ownership
  std::copy(
      owned_strings_.begin(),
      owned_strings_.end(),
      std::back_inserter(ownerships));

  return StringCordView(std::move(pieces), std::move(ownerships));
}

bool StringCordView::operator==(const std::string& rhs) const {
  if (size() != rhs.size()) {
    return false;
  }
  auto res = std::mismatch(begin(), end(), rhs.begin(), rhs.end());
  // both need to exhaust
  return res.first == end() && res.second == rhs.end();
}

bool StringCordView::operator==(const StringCordView& rhs) const {
  if (size() != rhs.size()) {
    return false;
  }
  auto res = std::mismatch(begin(), end(), rhs.begin(), rhs.end());
  // both need to exhaust
  return res.first == end() && res.second == rhs.end();
}

StringCordView::Iterator StringCordView::iter_for_pos(size_t pos) const {
  if (pos == 0) {
    return begin();
  }
  if (pos >= size()) {
    return end();
  }
  auto upper = std::upper_bound(
      accumulated_sizes_.begin(), accumulated_sizes_.end(), pos);
  if (upper == accumulated_sizes_.end()) {
    return end();
  }
  size_t line = upper - accumulated_sizes_.begin() - 1;
  assert(accumulated_sizes_[line] <= pos);
  assert(accumulated_sizes_[line + 1] > pos);
  return Iterator(this, line, pos - accumulated_sizes_[line], size() - pos);
}

size_t SourceRangeHasher::operator()(const torch::jit::SourceRange& key) const {
  return (
      std::hash<uintptr_t>()(reinterpret_cast<uintptr_t>(key.source().get())) ^
      std::hash<size_t>()(key.start()) ^ std::hash<size_t>()(key.end()));
}

std::optional<SourceRange> Source::findSourceRangeThatGenerated(
    const SourceRange& range) {
  if (!gen_ranges_) {
    return std::nullopt;
  }
  return gen_ranges_->findSourceRangeThatGenerated(range);
}

void SourceRange::highlight(std::ostream& out) const {
  // Retrieve original SourceRange, if present.
  if (auto orig_source_range = findSourceRangeThatGenerated()) {
    orig_source_range->highlight(out);
    out << "Serialized ";
  }
  print_with_context(out, CONTEXT, true, "");
}

void format_stack_trace(
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

void SourceRange::print_with_context(
    std::ostream& out,
    size_t context,
    bool highlight,
    const std::string& funcname) const {
  // This is an empty SourceRange, used as a sentinel value.
  if (!source_view_) {
    return;
  }

  auto str = source_view_->text_str().str();
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
  if (begin_line > str.size()) {
    return;
  }
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
    auto [filename, line, col] = *flc;
    out << "  File \"" << filename << "\", line " << line;
    if (!funcname.empty()) {
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
      while (line_end < str.size() && str[line_end] != '\n') {
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
      for (const auto i : c10::irange(hightlight_begin, highlight_end)) {
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

} // namespace torch::jit
