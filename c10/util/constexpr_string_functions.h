#pragma once

#include <cstdlib>

namespace c10 {
namespace util {

// returns the number of characters in the string
// strlen("hello") == 5
constexpr size_t strlen(const char* string) {
    return (string[0] == '\0')
        ? 0
        : 1 + strlen(string + 1);
}

// returns true iff string starts with prefix
// starts_with("house", "ho") == true
constexpr bool starts_with(const char* string, const char* prefix) {
    return (prefix[0] == '\0')
        ? true
        : (prefix[0] != string[0])
        ? false
        : starts_with(string+1, prefix+1);
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

// returns true iff csv_list contains item
// csv_contains("a,bc,d", "bc") == true
constexpr bool csv_contains(const char* csv_list, const char* item) {
    return (csv_list[0] == '\0' && item[0] == '\0')
        ? true
        : (item[0] == '\0')
        ? false
        : (starts_with(csv_list, item) && (csv_list[strlen(item)] == ',' || csv_list[strlen(item)] == '\0'))
        ? true
        : (csv_list[0] == '\0')
        ? false
        : csv_contains(skip_until_first_of(csv_list, ','), item);
}

}
}
