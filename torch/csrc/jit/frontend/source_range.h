#pragma once
#include <c10/util/Exception.h>
#include <optional>

#include <algorithm>
#include <iterator>
#include <memory>
#include <ostream>
#include <sstream>
#include <unordered_map>

namespace torch::jit {

class SourceRangeUnpickler;
struct SourceRange;

// A stringlike class backed by a vector of string_view
// the string represented are logically the concatenation of  the string_views
// This has advantage of not needing continues memory.
struct TORCH_API StringCordView {
  StringCordView();
  StringCordView(const StringCordView&) = default;
  StringCordView(StringCordView&&) noexcept = default;
  StringCordView(
      std::vector<std::string_view> inputs,
      std::vector<std::shared_ptr<std::string>> ownerships);

  StringCordView& operator=(const StringCordView&) = default;
  StringCordView& operator=(StringCordView&&) noexcept = default;

  size_t size() const {
    return accumulated_sizes_.back();
  }

  size_t find(const std::string& tok, size_t start) const;
  size_t find_regex(const std::string& tok, size_t start) const;
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

  std::string_view piece(size_t index) const {
    return pieces_[index];
  }

  // General-case iterator implementation.
  struct IteratorImpl {
    IteratorImpl(
        const StringCordView* str,
        size_t start_line,
        size_t start_pos,
        size_t size)
        : line_(start_line), pos_(start_pos), str_(str), size_(size) {}
    explicit IteratorImpl(const StringCordView* str)
        : IteratorImpl(str, 0, 0, str->size()) {}

    IteratorImpl() : IteratorImpl(nullptr, 0, 0, 0) {}

    IteratorImpl(const IteratorImpl&) = default;
    IteratorImpl(IteratorImpl&&) = default;
    IteratorImpl& operator=(const IteratorImpl&) = default;
    IteratorImpl& operator=(IteratorImpl&&) = default;

    IteratorImpl& operator++() {
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

    IteratorImpl operator++(int) {
      IteratorImpl prev(*this);
      ++(*this);
      return prev;
    }

    IteratorImpl next_iter() const {
      IteratorImpl next(*this);
      ++next;
      return next;
    }

    IteratorImpl& operator+=(size_t num);

    IteratorImpl operator+(size_t num) const {
      IteratorImpl it(*this);
      it += num;
      return it;
    }

    bool operator==(const IteratorImpl& rhs) const {
      if (!has_next() && !rhs.has_next()) {
        return true;
      }
      return (str_ == rhs.str_) && (line_ == rhs.line_) && (pos_ == rhs.pos_);
    }

    bool operator!=(const IteratorImpl& rhs) const {
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
    std::string_view rest_line() const {
      if (line_ >= str_->pieces_.size()) {
        return "";
      }

      std::string_view cur_line = str_->pieces_[line_];
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

  // Either an IteratorImpl, or a simple std::string_view::iterator
  // (which is faster) if possible.
  struct Iterator {
    Iterator() = default;

    Iterator(
        const StringCordView* str,
        size_t start_line,
        size_t start_pos,
        size_t size)
        : repr_(
              str->pieces_.size() == 1
                  ? repr_type(FastRepr(
                        start_line ? str->pieces_[0].end()
                                   : str->pieces_[0].begin() + start_pos,
                        str))
                  : repr_type(IteratorImpl(str, start_line, start_pos, size))) {
    }

    Iterator(const StringCordView* str) : Iterator(str, 0, 0, str->size()) {}

    Iterator& operator++() {
      if (auto* pit = std::get_if<IteratorImpl>(&repr_)) {
        ++(*pit);
      } else {
        ++fast_repr().it;
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
      if (auto* pit = std::get_if<IteratorImpl>(&repr_)) {
        *pit += num;
      } else {
        fast_repr().it += num;
      }
      return *this;
    }

    Iterator operator+(size_t num) const {
      Iterator it(*this);
      it += num;
      return it;
    }

    bool operator==(const Iterator& rhs) const {
      return repr_ == rhs.repr_;
    }

    bool operator!=(const Iterator& rhs) const {
      return repr_ != rhs.repr_;
    }

    bool has_next() const {
      if (const auto* pit = std::get_if<IteratorImpl>(&repr_)) {
        return pit->has_next();
      } else {
        return fast_repr().it != fast_repr().str->pieces_[0].end();
      }
    }

    char operator*() const {
      if (const auto* pit = std::get_if<IteratorImpl>(&repr_)) {
        return **pit;
      } else {
        return *fast_repr().it;
      }
    }

    std::string_view rest_line() const {
      if (const auto* pit = std::get_if<IteratorImpl>(&repr_)) {
        return pit->rest_line();
      } else {
        // NOTE: std::string_view(it, end) ctor wasn't added until C++20.
        const auto fast_repr_end = fast_repr().str->pieces_[0].end();
        if (fast_repr().it != fast_repr_end) {
          return std::string_view(
              &*fast_repr().it, fast_repr_end - fast_repr().it);
        }
        return std::string_view();
      }
    }

    size_t pos() const {
      if (const auto* pit = std::get_if<IteratorImpl>(&repr_)) {
        return pit->pos();
      } else {
        return fast_repr().it - fast_repr().str->pieces_[0].begin();
      }
    }

   private:
    // When we have only one entry in pieces_ (importantly, such as
    // when called from torch::Library::def during startup), we can
    // skip extra complexity and just use string_view::iterator
    // directly.
    struct FastRepr {
      std::string_view::iterator it;
      const StringCordView* str;

      FastRepr() : str(nullptr) {}

      explicit FastRepr(
          std::string_view::iterator it_,
          const StringCordView* str_)
          : it(it_), str(str_) {}

      bool operator==(const FastRepr& rhs) const {
        return it == rhs.it && str == rhs.str;
      }

      bool operator!=(const FastRepr& rhs) const {
        return !operator==(rhs);
      }
    };
    using repr_type = std::variant<FastRepr, IteratorImpl>;
    repr_type repr_;

    FastRepr& fast_repr() {
      // -Oz refuses to inline std::get.
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(std::holds_alternative<FastRepr>(repr_));
      return *std::get_if<FastRepr>(&repr_);
    }

    const FastRepr& fast_repr() const {
      // -Oz refuses to inline std::get.
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(std::holds_alternative<FastRepr>(repr_));
      return *std::get_if<FastRepr>(&repr_);
    }
  };

  Iterator begin() const {
    return Iterator(this, 0, 0, size());
  }
  Iterator end() const {
    return Iterator(this, pieces_.size(), 0, 0);
  }
  Iterator iter_for_pos(size_t pos) const;

 private:
  IteratorImpl begin_impl() const {
    return IteratorImpl(this, 0, 0, size());
  }
  IteratorImpl end_impl() const {
    return IteratorImpl(this, pieces_.size(), 0, 0);
  }
  IteratorImpl iter_impl_for_pos(size_t pos) const;
  std::vector<std::string_view> pieces_;
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
      std::string_view text_view,
      std::optional<std::string> filename = std::nullopt,
      size_t starting_line_no = 0,
      std::shared_ptr<SourceRangeUnpickler> gen_ranges = nullptr,
      CopiesString copies_str = COPIES_STRING)
      : text_view_(create_text_view(copies_str, text_view)),
        filename_(std::move(filename)),
        starting_line_no_(starting_line_no),
        gen_ranges_(std::move(gen_ranges)) {
    calc_line_start_offsets();
  }

  explicit Source(
      StringCordView str,
      std::optional<std::string> filename = std::nullopt,
      size_t starting_line_no = 0,
      std::shared_ptr<SourceRangeUnpickler> gen_ranges = nullptr)
      : text_view_(std::move(str)),
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

  std::optional<std::string>& filename() {
    return filename_;
  }

  size_t starting_line_no() const {
    return starting_line_no_;
  }

  std::optional<SourceRange> findSourceRangeThatGenerated(
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

  static StringCordView create_text_view(
      CopiesString copies_str,
      std::string_view text_view) {
    if (copies_str == COPIES_STRING) {
      auto allocated_str =
          std::make_shared<std::string>(text_view.data(), text_view.size());
      return StringCordView({*allocated_str}, {allocated_str});
    } else {
      return StringCordView({text_view}, {});
    }
  }

  StringCordView text_view_;

  std::optional<std::string> filename_;
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

  const std::string_view token_text() const {
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

  std::optional<std::tuple<std::string, size_t, size_t>> file_line_col() const {
    if (!source_view_ || !source()->filename()) {
      return std::nullopt;
    }

    auto lineno = source_view_->lineno_for_offset(start_);
    auto col_offset = (int)start_ - (int)source_view_->offset_for_line(lineno);
    // TODO: std::optional<>::value returns an rvalue ref so can't use it here??
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

  std::optional<SourceRange> findSourceRangeThatGenerated() const {
    if (!source_view_) {
      return std::nullopt;
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

} // namespace torch::jit

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
