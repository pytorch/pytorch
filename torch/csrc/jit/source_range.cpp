#include <torch/csrc/jit/source_range.h>

#include <torch/csrc/jit/pickler.h>

namespace torch {
namespace jit {

C10_EXPORT std::shared_ptr<SourceRange> DebugInfo::query(const SourceRange& q) {
  deserialize();

  auto query = SourceRangeRecord{q.start(), nullptr};
  auto entry = std::lower_bound(
      deserialized_records_->begin(),
      deserialized_records_->end(),
      query,
      [](const SourceRangeRecord& a, const SourceRangeRecord& b) -> bool {
        return std::get<0>(a) < std::get<0>(b);
      });

  if (entry != deserialized_records_->end()) {
    return std::get<1>(*entry);
  }

  return nullptr;
}

void DebugInfo::deserialize() {
  if (deserialized_records_) {
    return;
  }

  Unpickler up(std::get<0>(info_).get(), std::get<1>(info_), nullptr);
  auto ivalues = up.parse_ivalue_list();

  deserialized_records_ = SourceRangeRecords();
  for (auto& val : ivalues) {
    auto tup_elems = val.toTuple()->elements();
    int64_t offset = tup_elems[0].toInt();
    std::shared_ptr<SourceRange> sr = SourceRange::__setstate__(tup_elems[1]);
    deserialized_records_->emplace_back(offset, std::move(sr));
  }
}

// a range of a shared string 'file_' with
C10_EXPORT void SourceRange::highlight(std::ostream& out) const {
  const std::string& str = source_->text();
  if (size() == str.size()) {
    // this is just the entire file, not a subset, so print it out.
    // primarily used to print out python stack traces
    out << str;
    return;
  }

  int64_t begin_line = start(); // beginning of line to highlight
  int64_t end_line = start(); // end of line to highlight
  while (begin_line > 0 && str[begin_line - 1] != '\n')
    --begin_line;
  while (end_line < str.size() && str[end_line] != '\n')
    ++end_line;
  AT_ASSERT(begin_line == 0 || str[begin_line - 1] == '\n');
  AT_ASSERT(end_line == str.size() || str[end_line] == '\n');

  int64_t begin_highlight = begin_line; // beginning of context, CONTEXT lines
                                        // before the highlight line
  for (int64_t i = 0; begin_highlight > 0; --begin_highlight) {
    if (str[begin_highlight - 1] == '\n')
      ++i;
    if (i >= CONTEXT)
      break;
  }
  AT_ASSERT(begin_highlight == 0 || str[begin_highlight - 1] == '\n');

  int64_t end_highlight =
      end_line; // end of context, CONTEXT lines after the highlight line
  for (int64_t i = 0; end_highlight < str.size(); ++end_highlight) {
    if (str[end_highlight] == '\n')
      ++i;
    if (i >= CONTEXT)
      break;
  }
  AT_ASSERT(end_highlight == str.size() || str[end_highlight] == '\n');

  if (auto flc = file_line_col()) {
    std::string filename;
    int64_t line, col;
    std::tie(filename, line, col) = *flc;
    out << "at " << filename << ":" << line << ":" << col << "\n";
  }
  out << str.substr(begin_highlight, end_line - begin_highlight) << "\n";
  out << std::string(start() - begin_line, ' ');
  int64_t len = std::min(size(), end_line - start());
  out << std::string(len, '~')
      << (len < size() ? "...  <--- HERE" : " <--- HERE");
  out << str.substr(end_line, end_highlight - end_line);
  if (!str.empty() && str.back() != '\n')
    out << "\n";
  // Retrieve original SourceRange, if present.
  if (source_) {
    if (auto orig_source_range = orig_range()) {
      out << "Compiled from code ";
      orig_source_range->highlight(out);
    }
  }
}

} // namespace jit
} // namespace torch
