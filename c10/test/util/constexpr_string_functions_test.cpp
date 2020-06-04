#include <gtest/gtest.h>
#include <c10/util/constexpr_string_functions.h>

namespace op_whitelist_contains_test {
  static_assert(c10::util::op_whitelist_contains("", ""), "");
  static_assert(!c10::util::op_whitelist_contains("", "a"), "");
  static_assert(!c10::util::op_whitelist_contains("a", ""), "");
  static_assert(!c10::util::op_whitelist_contains("a;bc", ""), "");

  static_assert(c10::util::op_whitelist_contains("a;bc;d", "a"), "");
  static_assert(c10::util::op_whitelist_contains("a;bc;d", "bc"), "");
  static_assert(c10::util::op_whitelist_contains("a;bc;d", "d"), "");
  static_assert(!c10::util::op_whitelist_contains("a;bc;d", "e"), "");
  static_assert(!c10::util::op_whitelist_contains("a;bc;d", ""), "");

  static_assert(c10::util::op_whitelist_contains(";", ""), "");
  static_assert(c10::util::op_whitelist_contains("a;", ""), "");
  static_assert(c10::util::op_whitelist_contains("a;", "a"), "");
}
