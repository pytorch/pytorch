#pragma once

#include <cstdlib>

#include <c10/util/string_view.h>

namespace c10 {
namespace util {

// returns true iff whitelist contains item
// op_whitelist_contains("a;bc;d", "bc") == true
constexpr bool op_whitelist_contains(string_view whitelist, string_view item) {
    size_t next = -1;
    for (size_t cur = 0; cur <= whitelist.size(); cur = next) {
      next = whitelist.find(';', cur);
      if (next != string_view::npos) {
        if (whitelist.substr(cur, next - cur) == item) {
          return true;
        }
        next++;
      } else {
        if (whitelist.substr(cur) == item) {
          return true;
        }
        break;
      }
    }
    return false;
}

} // namespace util
} // namespace c10
