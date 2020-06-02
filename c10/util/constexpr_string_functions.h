#pragma once

#include <cstdlib>

namespace c10 {
namespace util {

// In general, constexpr algorithms in C++ must be written recursively,
// as looping constructs (like for) are not constexpr.  This code is
// written using ternaries as a holdover from C++11; but as of C++14
// constexpr functions are allowed to have multiple return statements
// (writing them with ternaries is more portable though!)

namespace detail {

  // returns the number of characters in the string, not including
  // overload name
  //
  // strlen("hello") == 5
  // strlen("hello.overload") == 5
  constexpr size_t strlen_wo_overload(const char* string) {
      return (string[0] == '\0' || string[0] == '.')
          ? 0
          : 1 + strlen_wo_overload(string + 1);
  }

  // returns true iff string starts with prefix, not including overload
  // in prefix
  //
  // starts_with_wo_overload("house", "ho") == true
  // starts_with_wo_overload("house", "ho.overload") == true
  constexpr bool starts_with_wo_overload(const char* string, const char* prefix) {
      return (prefix[0] == '\0' || prefix[0] == '.')
          ? true
          : (prefix[0] != string[0])
          ? false
          : starts_with_wo_overload(string+1, prefix+1);
  }
} // namespace detail

// returns a pointer to the position directly after the first occurrence of character in string
// or returns a pointer to the '\0' at the end of the string if the string doesn't contain character.
// skip_until_first_of("house", 'o') == "use"
// skip_until_first_of("ab", 'c') == ""
constexpr const char* skip_until_first_of(const char* string, char character) {
    return (string[0] == '\0')
        ? string
        : (string[0] == character)
        ? string + 1
        : skip_until_first_of(string + 1, character);
}

// compares both strings and returns true iff they are equal
// strequal("house", "house") == true
constexpr bool strequal(const char* lhs, const char* rhs) {
    return (lhs[0] != rhs[0])
        ? false
        : (lhs[0] == '\0')
        ? true
        : strequal(lhs+1, rhs+1);
}

// returns true iff whitelist contains item (not including overload)
// csv_contains("a;bc;d", "bc") == true
// csv_contains("a;bc;d", "bc.foo") == true
constexpr bool op_whitelist_contains(const char* whitelist, const char* item) {
    return (whitelist[0] == '\0' && (item[0] == '\0' || item[0] == '.'))
        ? true
        : (item[0] == '\0' || item[0] == '.')
        ? false
        : (detail::starts_with_wo_overload(whitelist, item) &&
           (whitelist[detail::strlen_wo_overload(item)] == ';' ||
            whitelist[detail::strlen_wo_overload(item)] == '\0'))
        ? true
        : (whitelist[0] == '\0')
        ? false
        : op_whitelist_contains(skip_until_first_of(whitelist, ';'), item);
}

} // namespace util
} // namespace c10
