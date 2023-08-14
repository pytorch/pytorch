#include <c10/util/string_view.h>

#include <gmock/gmock.h>

using c10::string_view;

namespace test_starts_with {
static_assert(c10::starts_with(string_view("hi"), string_view("hi")), "");
static_assert(c10::starts_with(string_view(""), string_view("")), "");
static_assert(c10::starts_with(string_view("hi2"), string_view("")), "");
static_assert(!c10::starts_with(string_view(""), string_view("hi")), "");
static_assert(c10::starts_with(string_view("hi2"), string_view("hi")), "");
static_assert(!c10::starts_with(string_view("hi"), string_view("hi2")), "");
static_assert(!c10::starts_with(string_view("hi"), string_view("ha")), "");

static_assert(c10::starts_with(string_view("hi"), "hi"), "");
static_assert(c10::starts_with(string_view(""), ""), "");
static_assert(c10::starts_with(string_view("hi2"), ""), "");
static_assert(!c10::starts_with(string_view(""), "hi"), "");
static_assert(c10::starts_with(string_view("hi2"), "hi"), "");
static_assert(!c10::starts_with(string_view("hi"), "hi2"), "");
static_assert(!c10::starts_with(string_view("hi"), "ha"), "");

static_assert(!c10::starts_with(string_view(""), 'a'), "");
static_assert(!c10::starts_with(string_view(""), '\0'), "");
static_assert(!c10::starts_with(string_view("hello"), 'a'), "");
static_assert(c10::starts_with(string_view("hello"), 'h'), "");
} // namespace test_starts_with

namespace test_ends_with {
static_assert(c10::ends_with(string_view("hi"), string_view("hi")), "");
static_assert(c10::ends_with(string_view(""), string_view("")), "");
static_assert(c10::ends_with(string_view("hi2"), string_view("")), "");
static_assert(!c10::ends_with(string_view(""), string_view("hi")), "");
static_assert(c10::ends_with(string_view("hi2"), string_view("i2")), "");
static_assert(!c10::ends_with(string_view("i2"), string_view("hi2")), "");
static_assert(!c10::ends_with(string_view("hi"), string_view("ha")), "");

static_assert(c10::ends_with(string_view("hi"), "hi"), "");
static_assert(c10::ends_with(string_view(""), ""), "");
static_assert(c10::ends_with(string_view("hi2"), ""), "");
static_assert(!c10::ends_with(string_view(""), "hi"), "");
static_assert(c10::ends_with(string_view("hi2"), "i2"), "");
static_assert(!c10::ends_with(string_view("i2"), "hi2"), "");
static_assert(!c10::ends_with(string_view("hi"), "ha"), "");

static_assert(!c10::ends_with(string_view(""), 'a'), "");
static_assert(!c10::ends_with(string_view(""), '\0'), "");
static_assert(!c10::ends_with(string_view("hello"), 'a'), "");
static_assert(c10::ends_with(string_view("hello"), 'o'), "");
} // namespace test_ends_with
