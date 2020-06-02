#include <gtest/gtest.h>
#include <c10/util/constexpr_string_functions.h>

namespace strlen_test {
  static_assert(0 == c10::util::detail::strlen_wo_overload(""), "");
  static_assert(1 == c10::util::detail::strlen_wo_overload("a"), "");
  static_assert(1 == c10::util::detail::strlen_wo_overload("a.overload"), "");
  static_assert(10 == c10::util::detail::strlen_wo_overload("0123456789"), "");
}

namespace starts_with_test {
  static_assert(c10::util::detail::starts_with_wo_overload("house", ""), "");
  static_assert(c10::util::detail::starts_with_wo_overload("house", "h"), "");
  static_assert(c10::util::detail::starts_with_wo_overload("house", "ho"), "");
  static_assert(c10::util::detail::starts_with_wo_overload("house", "hou"), "");
  static_assert(c10::util::detail::starts_with_wo_overload("house", "hous"), "");
  static_assert(c10::util::detail::starts_with_wo_overload("house", "house"), "");
  static_assert(c10::util::detail::starts_with_wo_overload("house", "h.overload"), "");
  static_assert(c10::util::detail::starts_with_wo_overload("house", "ho.overload"), "");
  static_assert(c10::util::detail::starts_with_wo_overload("house", "hou.overload"), "");
  static_assert(c10::util::detail::starts_with_wo_overload("house", "hous.overload"), "");
  static_assert(c10::util::detail::starts_with_wo_overload("house", "house.overload"), "");

  static_assert(!c10::util::detail::starts_with_wo_overload("house", "b"), "");
  static_assert(!c10::util::detail::starts_with_wo_overload("house", "b.overload"), "");
  static_assert(!c10::util::detail::starts_with_wo_overload("house", "bouse"), "");
  static_assert(!c10::util::detail::starts_with_wo_overload("house", "houseb"), "");
}

namespace strequal_test {
  static_assert(c10::util::strequal("", ""), "");
  static_assert(c10::util::strequal("a", "a"), "");
  static_assert(c10::util::strequal("ab", "ab"), "");
  static_assert(c10::util::strequal("0123456789", "0123456789"), "");

  static_assert(!c10::util::strequal("", "0"), "");
  static_assert(!c10::util::strequal("0", ""), "");
  static_assert(!c10::util::strequal("0123", "012"), "");
  static_assert(!c10::util::strequal("012", "0123"), "");
  static_assert(!c10::util::strequal("0123456789", "0123556789"), "");
  static_assert(!c10::util::strequal("0123456789", "0123456788"), "");
}

namespace skip_until_first_of_test {
  static_assert(c10::util::strequal("tring", c10::util::skip_until_first_of("string", 's')), "");
  static_assert(c10::util::strequal("ring", c10::util::skip_until_first_of("string", 't')), "");
  static_assert(c10::util::strequal("ing", c10::util::skip_until_first_of("string", 'r')), "");
  static_assert(c10::util::strequal("ng", c10::util::skip_until_first_of("string", 'i')), "");
  static_assert(c10::util::strequal("g", c10::util::skip_until_first_of("string", 'n')), "");
  static_assert(c10::util::strequal("", c10::util::skip_until_first_of("string", 'g')), "");

  static_assert(c10::util::strequal("", c10::util::skip_until_first_of("string", 'a')), "");
  static_assert(c10::util::strequal("", c10::util::skip_until_first_of("", 'a')), "");
}

namespace op_whitelist_contains_test {
  static_assert(c10::util::op_whitelist_contains("", ""), "");
  static_assert(!c10::util::op_whitelist_contains("", "a"), "");
  static_assert(!c10::util::op_whitelist_contains("a", ""), "");
  static_assert(!c10::util::op_whitelist_contains("a;bc", ""), "");

  static_assert(c10::util::op_whitelist_contains("a;bc,d", "a"), "");
  static_assert(c10::util::op_whitelist_contains("a;bc,d", "a.overload"), "");
  static_assert(c10::util::op_whitelist_contains("a;bc;d", "bc"), "");
  static_assert(c10::util::op_whitelist_contains("a;bc;d", "bc.overload"), "");
  static_assert(c10::util::op_whitelist_contains("a;bc;d", "d"), "");
  static_assert(c10::util::op_whitelist_contains("a;bc;d", "d.overload"), "");
  static_assert(!c10::util::op_whitelist_contains("a;bc;d", "e"), "");
  static_assert(!c10::util::op_whitelist_contains("a;bc;d", "e.overload"), "");
  static_assert(!c10::util::op_whitelist_contains("a;bc;d", ""), "");
}
