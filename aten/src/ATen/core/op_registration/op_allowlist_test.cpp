#include <gtest/gtest.h>
#include <ATen/core/op_registration/op_allowlist.h>

namespace allowlist_contains_test {
  static_assert(c10::impl::allowlist_contains("", ""), "");
  static_assert(!c10::impl::allowlist_contains("", "a"), "");
  static_assert(!c10::impl::allowlist_contains("a", ""), "");
  static_assert(!c10::impl::allowlist_contains("a;bc", ""), "");

  static_assert(c10::impl::allowlist_contains("a;bc;d", "a"), "");
  static_assert(c10::impl::allowlist_contains("a;bc;d", "bc"), "");
  static_assert(c10::impl::allowlist_contains("a;bc;d", "d"), "");
  static_assert(!c10::impl::allowlist_contains("a;bc;d", "e"), "");
  static_assert(!c10::impl::allowlist_contains("a;bc;d", ""), "");

  static_assert(c10::impl::allowlist_contains(";", ""), "");
  static_assert(c10::impl::allowlist_contains("a;", ""), "");
  static_assert(c10::impl::allowlist_contains("a;", "a"), "");
}
