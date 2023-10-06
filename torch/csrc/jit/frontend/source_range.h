#pragma once
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <unordered_map>

namespace torch {
namespace jit {

class SourceRangeUnpickler;
struct SourceRange;

// A stringlike class backed by a vector of string_view
// the string represented are logically the concatenation of  the string_views
// This has advantage of not needing continues memory.
struct TORCH_API StringCordView {
  StringCordView();
  StringCordView(const StringCordView&) = default;
  StringCordView(
      std::vector<c10::string_view> inputs,
      std::vector<std::shared_ptr<std::string>> ownerships);

  StringCordView& operator=(const StringCordView&) = default;

  size_t size() const {
    return accumulated_sizes_.back();
  }

  size_t find(const std::string& tok, size_t start) const;
  StringCordView substr(size_t start, size_t size) const;

  char at(size_t index) const {
    return *iter_for_pos(index);
  }
  char operator[](size_t index) const {
    return at(index);
  }

  std::string str() const {
    std::stringstream ss;
    for (auto s : pieces_) {
      ss << std::string(s);
    }
    return ss.str();
  }

  bool operator==(const std::string& rhs) const;

  bool operator==(const StringCordView& rhs) const;

  c10::string_view piece(size_t index) const {
    return pieces_[index];
  }

  struct Iterator {
    Iterator(
        const StringCordView* str,
        size_t start_line,
        size_t start_pos,
        size_t size)
        : line_(start_line), pos_(start_pos), str_(str), size_(size) {}
    explicit Iterator(const StringCordView* str)
        : Iterator(str, 0, 0, str->size()) {}

    Iterator() : Iterator(nullptr, 0, 0, 0) {}

    Iterator(const Iterator&) = default;
    Iterator(Iterator&&) = default;
    Iterator& operator=(const Iterator&) = default;
    Iterator& operator=(Iterator&&) = default;

    Iterator operator++() {
      if (size_ == 0) {
        return *this;
      }
      if ((pos_ + 1) < str_->pieces_[line_].size()) {
        pos_++;
      } else {
        line_++;
        pos_ = 0;
      }
      return *this;
    }

    Iterator operator++(int) {
      Iterator prev(*this);
      ++(*this);
      return prev;
    }

    Iterator next_iter() const {
      Iterator next(*this);
      ++next;
      return next;
    }

    Iterator& operator+=(size_t num) {
      if (!has_next()) {
        return *this;
      }
      size_t target_pos = pos_ + num;
      if (target_pos >= str_->accumulated_sizes_[line_] &&
          (line_ + 1) < str_->accumulated_sizes_.size() &&
          target_pos < str_->accumulated_sizes_[line_ + 1]) {
        pos_ = target_pos;
        return *this;
      }

      size_t target_abs_pos = pos() + num;
      *this = str_->iter_for_pos(target_abs_pos);
      return *this;
    }

    bool operator==(const Iterator& rhs) const {
      if (!has_next() && !rhs.has_next()) {
        return true;
      }
      return (str_ == rhs.str_) && (line_ == rhs.line_) && (pos_ == rhs.pos_);
    }
    bool operator!=(const Iterator& rhs) {
      return !((*this) == rhs);
    }
    bool has_next() const {
      return size_ > 0 && (line_ < str_->pieces_.size());
    }

    char operator*() const {
      TORCH_INTERNAL_ASSERT(line_ < str_->pieces_.size());
      TORCH_INTERNAL_ASSERT(pos_ < str_->pieces_[line_].size());
      return str_->pieces_[line_].at(pos_);
    }

    // returns rest of the line of the current iterator
    c10::string_view rest_line() const {
      if (line_ >= str_->pieces_.size()) {
        return "";
      }

      c10::string_view cur_line = str_->pieces_[line_];
      return cur_line.substr(pos_, std::string::npos);
    }

    size_t pos() const {
      if (size_ == 0) {
        return 0;
      }
      return str_->accumulated_sizes_[line_] + pos_;
    }

   private:
    size_t line_;
    size_t pos_;
    const StringCordView* str_;
    size_t size_;
    friend struct StringCordView;
  };

  Iterator begin() const {
    return Iterator(this, 0, 0, size());
  }
  Iterator end() const {
    return Iterator(this, pieces_.size(), 0, 0);
  }
  Iterator iter_for_pos(size_t pos) const;

 private:
  std::vector<c10::string_view> pieces_;
  std::vector<size_t> accumulated_sizes_;
  std::vector<std::shared_ptr<std::string>> owned_strings_;
};

// Source represents a code segment. It keeps track of:
//  - text_view : the view into text of the code segment
//  - filename (optional) : if present, represents the name of the file from
//                          which the code segment originated.
//  - starting_line_no : represents the line in the original file where the
//                       code segment started.
struct TORCH_API Source {
  // Whether or not Source should copy the string passed in the constructor.
  enum CopiesString { COPIES_STRING, DONT_COPY };

  explicit Source(
      c10::string_view text_view,
      c10::optional<std::string> filename = c10::nullopt,
      size_t starting_line_no = 0,
      std::shared_ptr<SourceRangeUnpickler> gen_ranges = nullptr,
      CopiesString copies_str = COPIES_STRING)
      : filename_(std::move(filename)),
        starting_line_no_(starting_line_no),
        gen_ranges_(std::move(gen_ranges)) {
    if (copies_str == COPIES_STRING) {
      std::shared_ptr<std::string> allocated_str =
          std::make_shared<std::string>(text_view.data(), text_view.size());
      text_view_ = StringCordView({*allocated_str}, {allocated_str});
    } else {
      text_view_ = StringCordView({text_view}, {});
    }

    calc_line_start_offsets();
  }

  explicit Source(
      StringCordView str,
      c10::optional<std::string> filename = c10::nullopt,
      size_t starting_line_no = 0,
      std::shared_ptr<SourceRangeUnpickler> gen_ranges = nullptr)
      : text_view_(str),
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
    auto iter = std::upper_bound(
        line_starting_offsets_.begin(), line_starting_offsets_.end(), offset);
    return iter - line_starting_offsets_.begin() - 1;
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

  StringCordView get_line(size_t lineno) const {
    auto start = offset_for_line(lineno);
    auto size = (lineno + 1) < num_lines() ? offset_for_line(lineno + 1) - start
                                           : text_view_.size() - start;
    return text_view_.substr(start, size);
  }

  const StringCordView& text_str() const {
    return text_view_;
  }

  char char_at(size_t index) const {
    return text_view_.at(index);
  }

  size_t size() const {
    return text_view_.size();
  }

  c10::optional<std::string>& filename() {
    return filename_;
  }

  size_t starting_line_no() const {
    return starting_line_no_;
  }

  c10::optional<SourceRange> findSourceRangeThatGenerated(
      const SourceRange& range);

  ~Source() = default;

 private:
  void calc_line_start_offsets() {
    line_starting_offsets_.clear();
    line_starting_offsets_.push_back(0);
    size_t pos = 0;
    while ((pos = text_view_.find("\n", pos)) != std::string::npos) {
      line_starting_offsets_.push_back(++pos);
    }
  }

  StringCordView text_view_;

  c10::optional<std::string> filename_;
  // If filename_ is not present, starting_line_no_ is don't care
  size_t starting_line_no_;
  // Starting offsets for lines into the source. e.g. line 0 starts at
  // line_starting_offsets_[0], etc.
  std::vector<size_t> line_starting_offsets_;

  std::shared_ptr<SourceRangeUnpickler> gen_ranges_;
};

// A SourceRange is a reference to subset of a Source, specified by `start` and
// `end` byte offsets into the source text.
struct TORCH_API SourceRange {
  SourceRange(std::shared_ptr<Source> source_view, size_t start_, size_t end_)
      : source_view_(std::move(source_view)), start_(start_), end_(end_) {
    if (source_view_) {
      start_iter_ = source_view_->text_str().iter_for_pos(start_);
    }
  }

  SourceRange() : source_view_(nullptr), start_(0), end_(0) {}

  SourceRange(
      std::shared_ptr<Source> source_view_,
      StringCordView::Iterator start_iter,
      size_t end_)
      : source_view_(std::move(source_view_)),
        start_(start_iter.pos()),
        end_(end_),
        start_iter_(start_iter) {}

  const c10::string_view token_text() const {
    size_t size = end() - start();
    return start_iter_.rest_line().substr(0, size);
  }

  const StringCordView text() const {
    return source_view_->text_str().substr(start(), end() - start());
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

  const std::shared_ptr<Source>& source() const {
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
  std::shared_ptr<Source> source_view_;

 private:
  size_t start_;
  size_t end_;
  StringCordView::Iterator start_iter_;
};

// OwnedSourceRange is just like a SourceRange except that it owns a `Source`
// instead of `Source`. Thus OwnedSourceRange owns a copy of source text.
struct OwnedSourceRange : public SourceRange {
  explicit OwnedSourceRange(const SourceRange& source_range)
      : SourceRange(source_range) {
    const auto& source = source_range.source();
    if (source) {
      source_view_ = std::make_shared<Source>(
          source->text_str().str(),
          source->filename(),
          source->starting_line_no());
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

TORCH_API void format_stack_trace(
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

namespace std {
template <>
struct iterator_traits<torch::jit::StringCordView::Iterator> {
  using value_type = char;
  using difference_type = ptrdiff_t;
  using pointer = char*;
  using reference = char&;
  using iterator_category = std::forward_iterator_tag;
};
} // namespace std
