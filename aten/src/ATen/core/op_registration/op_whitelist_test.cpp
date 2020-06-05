#include <gtest/gtest.h>
#include <ATen/core/op_registration/op_whitelist.h>

namespace op_whitelist_contains_test {
  static_assert(c10::impl::op_whitelist_contains("", ""), "");
  static_assert(!c10::impl::op_whitelist_contains("", "a"), "");
  static_assert(!c10::impl::op_whitelist_contains("a", ""), "");
  static_assert(!c10::impl::op_whitelist_contains("a;bc", ""), "");

  static_assert(c10::impl::op_whitelist_contains("a;bc;d", "a"), "");
  static_assert(c10::impl::op_whitelist_contains("a;bc;d", "bc"), "");
  static_assert(c10::impl::op_whitelist_contains("a;bc;d", "d"), "");
  static_assert(!c10::impl::op_whitelist_contains("a;bc;d", "e"), "");
  static_assert(!c10::impl::op_whitelist_contains("a;bc;d", ""), "");

  static_assert(c10::impl::op_whitelist_contains(";", ""), "");
  static_assert(c10::impl::op_whitelist_contains("a;", ""), "");
  static_assert(c10::impl::op_whitelist_contains("a;", "a"), "");
}
