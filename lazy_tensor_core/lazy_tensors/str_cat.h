#pragma once

#include <c10/util/string_view.h>

#include <string>

namespace lazy_tensors {

// `AlphaNum` acts as the main parameter type for `StrCat()` and `StrAppend()`,
class AlphaNum {
 public:
  // No bool ctor -- bools convert to an integral type.
  // A bool ctor would also convert incoming pointers (bletch).

  AlphaNum(int x) : digits_(std::to_string(x)), piece_(digits_) {}
  AlphaNum(unsigned int x) : digits_(std::to_string(x)), piece_(digits_) {}
  AlphaNum(long x) : digits_(std::to_string(x)), piece_(digits_) {}
  AlphaNum(unsigned long x) : digits_(std::to_string(x)), piece_(digits_) {}
  AlphaNum(long long x) : digits_(std::to_string(x)), piece_(digits_) {}
  AlphaNum(unsigned long long x)
      : digits_(std::to_string(x)), piece_(digits_) {}

  AlphaNum(float f) : digits_(std::to_string(f)), piece_(digits_) {}
  AlphaNum(double f) : digits_(std::to_string(f)), piece_(digits_) {}

  AlphaNum(const char* c_str) : piece_(c_str) {}
  AlphaNum(c10::string_view pc) : piece_(pc) {}

  template <typename Allocator>
  AlphaNum(
      const std::basic_string<char, std::char_traits<char>, Allocator>& str)
      : piece_(str) {}

  // Use string literals ":" instead of character literals ':'.
  AlphaNum(char c) = delete;

  AlphaNum(const AlphaNum&) = delete;
  AlphaNum& operator=(const AlphaNum&) = delete;

  // Normal enums are already handled by the integer formatters.
  // This overload matches only scoped enums.
  template <typename T,
            typename = typename std::enable_if<
                std::is_enum<T>{} && !std::is_convertible<T, int>{}>::type>
  AlphaNum(T e)
      : AlphaNum(static_cast<typename std::underlying_type<T>::type>(e)) {}

  c10::string_view::size_type size() const { return piece_.size(); }
  const char* data() const { return piece_.data(); }
  c10::string_view Piece() const { return piece_; }

  // vector<bool>::reference and const_reference require special help to
  // convert to `AlphaNum` because it requires two user defined conversions.
  template <
      typename T,
      typename std::enable_if<
          std::is_class<T>::value &&
          (std::is_same<T, std::vector<bool>::reference>::value ||
           std::is_same<T, std::vector<bool>::const_reference>::value)>::type* =
          nullptr>
  AlphaNum(T e) : AlphaNum(static_cast<bool>(e)) {}

 private:
  std::string digits_;
  c10::string_view piece_;
};

// Merges given strings or numbers, using no delimiter(s), returning the merged
// result as a string.
inline std::string StrCat() { return std::string(); }

template <typename... AV>
inline std::string StrCat(const AV&... args) {
  std::string result;
  for (const auto& arg : {static_cast<const AlphaNum&>(args).Piece()...}) {
    result.append(arg.begin(), arg.end());
  }
  return result;
}

// Appends a string or set of strings to an existing string, in a similar
// fashion to `StrCat()`.
inline void StrAppend(std::string*) {}

template <typename... AV>
inline void StrAppend(std::string* dest, const AV&... args) {
  for (const auto& arg : {static_cast<const AlphaNum&>(args).Piece()...}) {
    dest->append(arg.data(), arg.size());
  }
}

}  // namespace lazy_tensors
