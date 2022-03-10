#pragma once
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <unordered_map>
namespace torch {
namespace jit {

class SourceRangeUnpickler;
struct SourceRange;

// SourceView represents a code segment. It keeps track of:
//  - text_view : the view into text of the code segment
//  - filename (optional) : if present, represents the name of the file from
//                          which the code segment originated.
//  - starting_line_no : represents the line in the original file where the
//                       code segment started.
struct SourceView {
  explicit SourceView(
      c10::string_view text_view,
      std::shared_ptr<SourceRangeUnpickler> gen_ranges = nullptr)
      : text_view_(text_view),
        filename_(c10::nullopt),
        starting_line_no_(0),
        gen_ranges_(std::move(gen_ranges)) {
    calc_line_start_offsets();
  }

  SourceView(
      c10::string_view text_view,
      c10::optional<std::string> filename,
      size_t starting_line_no,
      std::shared_ptr<SourceRangeUnpickler> gen_ranges = nullptr)
      : text_view_(text_view),
        filename_(std::move(filename)),
        starting_line_no_(starting_line_no),
        gen_ranges_(std::move(gen_ranges)) {
    calc_line_start_offsets();
  }

  // Given a line number (within source_), return the byte offset of the
  // beginning of that line.
  size_t offset_for_line(size_t line) const {
    return line_starting_offsets_.at(line);
  }

  // Returns number of lines present.
  size_t num_lines() const {
    return line_starting_offsets_.size();
  }

  // Calculate the line (within the code segment) on which `offset` resides.
  size_t lineno_for_offset(size_t offset) const {
    return std::upper_bound(
               line_starting_offsets_.begin(),
               line_starting_offsets_.end(),
               offset) -
        line_starting_offsets_.begin() - 1;
  }

  // Calculate the line (within the original source file, if present) on which
  // `lineno` resides.
  size_t lineno_to_source_lineno(size_t lineno) const {
    if (filename_) {
      return lineno + starting_line_no_;
    } else {
      return lineno;
    }
  }

  const c10::string_view text() const {
    return text_view_;
  }

  const c10::optional<std::string>& filename() const {
    return filename_;
  }

  size_t starting_line_no() const {
    return starting_line_no_;
  }

  c10::optional<SourceRange> findSourceRangeThatGenerated(
      const SourceRange& range);

 protected:
  c10::string_view text_view_;

 private:
  void calc_line_start_offsets() {
    line_starting_offsets_.push_back(0);
    size_t pos = 0;
    while ((pos = text().find('\n', pos)) != std::string::npos) {
      line_starting_offsets_.push_back(++pos);
    }
  }

  c10::optional<std::string> filename_;
  // If filename_ is not present, starting_line_no_ is don't care
  size_t starting_line_no_;
  // Starting offsets for lines into the source. e.g. line 0 starts at
  // line_starting_offsets_[0], etc.
  std::vector<size_t> line_starting_offsets_;

  std::shared_ptr<SourceRangeUnpickler> gen_ranges_;
};

// Source represents a code segment like SourceView, but the former owns a copy
// of source text while the latter doesn't.
struct Source : public SourceView {
  explicit Source(
      std::string text,
      std::shared_ptr<SourceRangeUnpickler> gen_ranges = nullptr)
      : SourceView(text, gen_ranges), text_(std::move(text)) {
    text_view_ = text_;
  }

  explicit Source(
      c10::string_view text_view,
      std::shared_ptr<SourceRangeUnpickler> gen_ranges = nullptr)
      : SourceView(text_view, gen_ranges),
        text_(text_view.begin(), text_view.end()) {
    text_view_ = text_;
  }

  explicit Source(
      std::string text,
      c10::optional<std::string> filename,
      size_t starting_line_no,
      std::shared_ptr<SourceRangeUnpickler> gen_ranges = nullptr)
      : SourceView(text, filename, starting_line_no, gen_ranges),
        text_(std::move(text)) {
    text_view_ = text_;
  }

  explicit Source(
      c10::string_view text_view,
      c10::optional<std::string> filename,
      size_t starting_line_no,
      std::shared_ptr<SourceRangeUnpickler> gen_ranges = nullptr)
      : SourceView(text_view, filename, starting_line_no, gen_ranges),
        text_(text_view.begin(), text_view.end()) {
    text_view_ = text_;
  }

  // Constructor that deepcopies and owns source text referenced in
  // `source_view`.
  explicit Source(const SourceView& source_view) : SourceView(source_view) {
    text_ = std::string(text_view_.begin(), text_view_.end());
    text_view_ = text_;
  }

  std::string text_;
};

// A SourceRange is a reference to subset of a Source, specified by `start` and
// `end` byte offsets into the source text.
struct TORCH_API SourceRange {
  SourceRange(
      std::shared_ptr<SourceView> source_view_,
      size_t start_,
      size_t end_)
      : source_view_(std::move(source_view_)), start_(start_), end_(end_) {}
  SourceRange() : source_view_(nullptr), start_(0), end_(0) {}

  const std::string text() const {
    auto text_view = source_view_->text().substr(start(), end() - start());
    return std::string(text_view.begin(), text_view.end());
  }
  size_t size() const {
    return end() - start();
  }
  static const size_t CONTEXT = 3;
  void highlight(std::ostream& out) const;

  // Customizable version of 'highlight' method.
  void print_with_context(
      std::ostream& out,
      size_t context,
      bool highlight,
      const std::string& funcname) const;

  const std::shared_ptr<SourceView>& source() const {
    return source_view_;
  }
  size_t start() const {
    return start_;
  }
  size_t end() const {
    return end_;
  }
  std::string str() const {
    std::stringstream ss;
    highlight(ss);
    return ss.str();
  }

  c10::optional<std::tuple<std::string, size_t, size_t>> file_line_col() const {
    if (!source_view_ || !source()->filename()) {
      return c10::nullopt;
    }

    auto lineno = source_view_->lineno_for_offset(start_);
    auto col_offset = (int)start_ - (int)source_view_->offset_for_line(lineno);
    // TODO: c10::optional<>::value returns an rvalue ref so can't use it here??
    return std::make_tuple<std::string, size_t, size_t>(
        source_view_->filename().value_or(""),
        source_view_->lineno_to_source_lineno(lineno),
        (size_t)col_offset);
  }

  bool operator==(const SourceRange& rhs) const {
    return start() == rhs.start() && end() == rhs.end() &&
        source() == rhs.source();
  }

  bool operator!=(const SourceRange& rhs) const {
    return !(*this == rhs);
  }

  c10::optional<SourceRange> findSourceRangeThatGenerated() const {
    if (!source_view_) {
      return c10::nullopt;
    }
    return source_view_->findSourceRangeThatGenerated(*this);
  }

 protected:
  std::shared_ptr<SourceView> source_view_;

 private:
  size_t start_;
  size_t end_;
};

// OwnedSourceRange is just like a SourceRange except that it owns a `Source`
// instead of `SourceView`. Thus OwnedSourceRange owns a copy of source text.
struct OwnedSourceRange : public SourceRange {
  OwnedSourceRange(const SourceRange& source_range)
      : SourceRange(source_range) {
    const auto& source = source_range.source();
    if (source) {
      source_view_ = std::make_shared<Source>(*source);
    }
  }
};

struct TORCH_API SourceRangeHasher {
 public:
  size_t operator()(const torch::jit::SourceRange& key) const;
};

struct StackEntry {
  std::string filename;
  SourceRange range;
};

C10_EXPORT void format_stack_trace(
    std::ostream& out,
    const std::vector<StackEntry>& entries);

inline std::ostream& operator<<(std::ostream& out, const SourceRange& range) {
  range.highlight(out);
  return out;
}

// A pair of (byte offset, SourceRange) describing a specific segment
// of the output stream
struct TaggedRange {
  TaggedRange(size_t bytes, SourceRange range)
      : bytes(bytes), range(std::move(range)) {}
  size_t bytes;
  SourceRange range;
};
using SourceRangeRecords = std::vector<TaggedRange>;
using SourceRangeTagMap =
    std::unordered_map<SourceRange, int64_t, SourceRangeHasher>;

} // namespace jit
} // namespace torch
