#include <c10/util/string_view.h>

#include <gmock/gmock.h>

// NOLINTBEGIN(modernize*, readability*, bugprone-string-constructor)
using string_view = c10::c10_string_view;

namespace {
namespace testutils {
constexpr bool string_equal(const char* lhs, const char* rhs, size_t size) {
  return (size == 0)   ? true
      : (*lhs != *rhs) ? false
                       : string_equal(lhs + 1, rhs + 1, size - 1);
}
static_assert(string_equal("hi", "hi", 2), "");
static_assert(string_equal("", "", 0), "");
static_assert(string_equal("hi", "hi2", 2), "");
static_assert(string_equal("hi2", "hi", 2), "");
static_assert(!string_equal("hi", "hi2", 3), "");
static_assert(!string_equal("hi2", "hi", 3), "");
static_assert(!string_equal("hi", "ha", 2), "");

template <class Exception, class Functor>
inline void expectThrows(Functor&& functor, const char* expectMessageContains) {
  try {
    std::forward<Functor>(functor)();
  } catch (const Exception& e) {
    EXPECT_THAT(e.what(), testing::HasSubstr(expectMessageContains));
    return;
  }
  ADD_FAILURE() << "Expected to throw exception containing \""
                << expectMessageContains << "\" but didn't throw";
}
} // namespace testutils

using testutils::expectThrows;
using testutils::string_equal;

namespace test_starts_with {
static_assert(string_view("hi").starts_with(string_view("hi")), "");
static_assert(string_view("").starts_with(string_view("")), "");
static_assert(string_view("hi2").starts_with(string_view("")), "");
static_assert(!string_view("").starts_with(string_view("hi")), "");
static_assert(string_view("hi2").starts_with(string_view("hi")), "");
static_assert(!string_view("hi").starts_with(string_view("hi2")), "");
static_assert(!string_view("hi").starts_with(string_view("ha")), "");

static_assert(string_view("hi").starts_with("hi"), "");
static_assert(string_view("").starts_with(""), "");
static_assert(string_view("hi2").starts_with(""), "");
static_assert(!string_view("").starts_with("hi"), "");
static_assert(string_view("hi2").starts_with("hi"), "");
static_assert(!string_view("hi").starts_with("hi2"), "");
static_assert(!string_view("hi").starts_with("ha"), "");

static_assert(!string_view("").starts_with('a'), "");
static_assert(!string_view("").starts_with('\0'), "");
static_assert(!string_view("hello").starts_with('a'), "");
static_assert(string_view("hello").starts_with('h'), "");
} // namespace test_starts_with

namespace test_ends_with {
static_assert(string_view("hi").ends_with(string_view("hi")), "");
static_assert(string_view("").ends_with(string_view("")), "");
static_assert(string_view("hi2").ends_with(string_view("")), "");
static_assert(!string_view("").ends_with(string_view("hi")), "");
static_assert(string_view("hi2").ends_with(string_view("i2")), "");
static_assert(!string_view("i2").ends_with(string_view("hi2")), "");
static_assert(!string_view("hi").ends_with(string_view("ha")), "");

static_assert(string_view("hi").ends_with("hi"), "");
static_assert(string_view("").ends_with(""), "");
static_assert(string_view("hi2").ends_with(""), "");
static_assert(!string_view("").ends_with("hi"), "");
static_assert(string_view("hi2").ends_with("i2"), "");
static_assert(!string_view("i2").ends_with("hi2"), "");
static_assert(!string_view("hi").ends_with("ha"), "");

static_assert(!string_view("").ends_with('a'), "");
static_assert(!string_view("").ends_with('\0'), "");
static_assert(!string_view("hello").ends_with('a'), "");
static_assert(string_view("hello").ends_with('o'), "");
} // namespace test_ends_with

} // namespace
// NOLINTEND(modernize*, readability*, bugprone-string-constructor)
