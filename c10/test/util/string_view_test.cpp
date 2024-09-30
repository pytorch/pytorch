#include <c10/util/string_view.h>

#include <gmock/gmock.h>

// NOLINTBEGIN(modernize*, readability*, bugprone-string-constructor)
using c10::string_view;

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

namespace test_typedefs {
static_assert(std::is_same<char, string_view::value_type>::value, "");
static_assert(std::is_same<char*, string_view::pointer>::value, "");
static_assert(std::is_same<const char*, string_view::const_pointer>::value, "");
static_assert(std::is_same<char&, string_view::reference>::value, "");
static_assert(
    std::is_same<const char&, string_view::const_reference>::value,
    "");
static_assert(std::is_same<std::size_t, string_view::size_type>::value, "");
static_assert(
    std::is_same<std::ptrdiff_t, string_view::difference_type>::value,
    "");
} // namespace test_typedefs

namespace test_default_constructor {
static_assert(string_view().empty());
static_assert(string_view().data() == nullptr, "");
static_assert(string_view() == string_view(""));
} // namespace test_default_constructor

namespace test_constchar_constructor {
static_assert(string_view("").size() == 0, "");
constexpr string_view hello = "hello";
static_assert(5 == hello.size(), "");
static_assert(string_equal("hello", hello.data(), hello.size()), "");
} // namespace test_constchar_constructor

namespace test_sized_constructor {
static_assert(string_view("", 0).size() == 0, "");
constexpr string_view hell("hello", 4);
static_assert(4 == hell.size(), "");
static_assert(string_equal("hell", hell.data(), hell.size()), "");
} // namespace test_sized_constructor

namespace test_string_constructor {
void test_conversion_is_implicit(string_view a) {}
TEST(StringViewTest, testStringConstructor) {
  std::string empty;
  EXPECT_EQ(0, string_view(empty).size());
  std::string hello_str = "hello";
  string_view hello_sv = hello_str;
  EXPECT_EQ(5, hello_sv.size());
  EXPECT_TRUE(string_equal("hello", hello_sv.data(), hello_sv.size()));

  test_conversion_is_implicit(hello_str);
}
} // namespace test_string_constructor

namespace test_conversion_to_string {
TEST(StringViewTest, testConversionToString) {
  string_view empty;
  EXPECT_EQ(0, std::string(empty).size());
  string_view hello_sv = "hello";
  std::string hello_str(hello_sv);
  EXPECT_EQ(5, hello_str.size());
  EXPECT_EQ(std::string("hello"), hello_str);
}
} // namespace test_conversion_to_string

namespace test_copy_constructor {
constexpr string_view hello = "hello";
constexpr string_view copy = hello;
static_assert(5 == copy.size(), "");
static_assert(string_equal("hello", copy.data(), copy.size()), "");
} // namespace test_copy_constructor

namespace test_copy_assignment {
constexpr string_view assign(string_view value) {
  string_view result = "temporary_content";
  result = value; // this is the assignment we're testing
  return result;
}
TEST(StringViewTest, testCopyAssignment) {
#if defined(__cpp_constexpr) && __cpp_constexpr >= 201304
  {
    constexpr string_view hello = assign("hello");
    static_assert(5 == hello.size(), "");
    static_assert(string_equal("hello", hello.data(), hello.size()), "");

    static_assert(5 == (string_view() = "hello").size(), "");
    static_assert(
        string_equal("hello", (string_view() = "hello").data(), 5), "");
  }
#endif
  const string_view hello = assign("hello");
  EXPECT_EQ(5, hello.size());
  EXPECT_EQ("hello", hello);
  EXPECT_EQ(5, (string_view() = "hello").size());
  EXPECT_EQ("hello", (string_view() = "hello"));
}

} // namespace test_copy_assignment

namespace test_iterators {
static_assert('h' == *string_view("hello").begin(), "");
static_assert('h' == *string_view("hello").cbegin(), "");
static_assert('h' == *begin(string_view("hello")), "");
static_assert('o' == *(string_view("hello").end() - 1), "");
static_assert('o' == *(string_view("hello").cend() - 1), "");
static_assert('o' == *(end(string_view("hello")) - 1), "");
static_assert('o' == *string_view("hello").rbegin(), "");
static_assert('o' == *string_view("hello").crbegin(), "");
static_assert('h' == *(string_view("hello").rend() - 1), "");
static_assert('h' == *(string_view("hello").crend() - 1), "");
} // namespace test_iterators

namespace test_forward_iteration {
constexpr string_view hello = "hello";
static_assert('h' == *(hello.begin() + 0), "");
static_assert('e' == *(hello.begin() + 1), "");
static_assert('l' == *(hello.begin() + 2), "");
static_assert('l' == *(hello.begin() + 3), "");
static_assert('o' == *(hello.begin() + 4), "");
static_assert(hello.end() == hello.begin() + 5, "");
} // namespace test_forward_iteration

namespace test_reverse_iteration {
constexpr string_view hello = "hello";
static_assert('o' == *(hello.rbegin() + 0), "");
static_assert('l' == *(hello.rbegin() + 1), "");
static_assert('l' == *(hello.rbegin() + 2), "");
static_assert('e' == *(hello.rbegin() + 3), "");
static_assert('h' == *(hello.rbegin() + 4), "");
static_assert(hello.rend() == hello.rbegin() + 5, "");
} // namespace test_reverse_iteration

namespace test_random_access {
constexpr string_view hello = "hello";
static_assert('h' == hello[0], "");
static_assert('e' == hello[1], "");
static_assert('l' == hello[2], "");
static_assert('l' == hello[3], "");
static_assert('o' == hello[4], "");

static_assert('h' == hello.at(0), "");
static_assert('e' == hello.at(1), "");
static_assert('l' == hello.at(2), "");
static_assert('l' == hello.at(3), "");
static_assert('o' == hello.at(4), "");

TEST(StringViewTest, whenCallingAccessOperatorOutOfRange_thenThrows) {
  expectThrows<std::out_of_range>(
      [] { string_view("").at(1); },
      "string_view::operator[] or string_view::at() out of range. Index: 1, size: 0");

  expectThrows<std::out_of_range>(
      [] { string_view("hello").at(5); },
      "string_view::operator[] or string_view::at() out of range. Index: 5, size: 5");

  expectThrows<std::out_of_range>(
      [] { string_view("hello").at(100); },
      "string_view::operator[] or string_view::at() out of range. Index: 100, size: 5");

  expectThrows<std::out_of_range>(
      [] { string_view("hello").at(string_view::npos); },
      "string_view::operator[] or string_view::at() out of range. Index: 18446744073709551615, size: 5");
}
} // namespace test_random_access

namespace test_front_back {
static_assert('h' == string_view("hello").front(), "");
static_assert('o' == string_view("hello").back(), "");
} // namespace test_front_back

namespace test_data {
static_assert(string_equal("hello", string_view("hello").data(), 5), "");
} // namespace test_data

namespace test_size_length {
static_assert(0 == string_view("").size(), "");
static_assert(5 == string_view("hello").size(), "");

static_assert(0 == string_view("").length(), "");
static_assert(5 == string_view("hello").length(), "");
} // namespace test_size_length

namespace test_empty {
static_assert(string_view().empty(), "");
static_assert(string_view("").empty(), "");
static_assert(!string_view("hello").empty(), "");
} // namespace test_empty

namespace test_remove_prefix {
constexpr string_view remove_prefix(string_view input, size_t len) {
  input.remove_prefix(len);
  return input;
}

TEST(StringViewTest, whenRemovingValidPrefix_thenWorks) {
  static_assert(
      remove_prefix(string_view("hello"), 0) == string_view("hello"), "");
  static_assert(
      remove_prefix(string_view("hello"), 1) == string_view("ello"), "");
  static_assert(remove_prefix(string_view("hello"), 5) == string_view(""), "");

  EXPECT_EQ(remove_prefix(string_view("hello"), 0), string_view("hello"));
  EXPECT_EQ(remove_prefix(string_view("hello"), 1), string_view("ello"));
  EXPECT_EQ(remove_prefix(string_view("hello"), 5), string_view(""));
}

TEST(StringViewTest, whenRemovingTooLargePrefix_thenThrows) {
  expectThrows<std::out_of_range>(
      [] { remove_prefix(string_view("hello"), 6); },
      "basic_string_view::remove_prefix: out of range. PrefixLength: 6, size: 5");
}
} // namespace test_remove_prefix

namespace test_remove_suffix {
constexpr string_view remove_suffix(string_view input, size_t len) {
  input.remove_suffix(len);
  return input;
}

TEST(StringViewTest, whenRemovingValidSuffix_thenWorks) {
  static_assert(
      remove_suffix(string_view("hello"), 0) == string_view("hello"), "");
  static_assert(
      remove_suffix(string_view("hello"), 1) == string_view("hell"), "");
  static_assert(remove_suffix(string_view("hello"), 5) == string_view(""), "");

  EXPECT_EQ(remove_suffix(string_view("hello"), 0), string_view("hello"));
  EXPECT_EQ(remove_suffix(string_view("hello"), 1), string_view("hell"));
  EXPECT_EQ(remove_suffix(string_view("hello"), 5), string_view(""));
}

TEST(StringViewTest, whenRemovingTooLargeSuffix_thenThrows) {
  expectThrows<std::out_of_range>(
      [] { remove_suffix(string_view("hello"), 6); },
      "basic_string_view::remove_suffix: out of range. SuffixLength: 6, size: 5");
}
} // namespace test_remove_suffix

namespace test_swap_function {
constexpr std::pair<string_view, string_view> get() {
  string_view first = "first";
  string_view second = "second";
  swap(first, second);
  return std::make_pair(first, second);
}
TEST(StringViewTest, testSwapFunction) {
  static_assert(string_view("second") == get().first, "");
  static_assert(string_view("first") == get().second, "");

  EXPECT_EQ(string_view("second"), get().first);
  EXPECT_EQ(string_view("first"), get().second);
}
} // namespace test_swap_function

namespace test_swap_method {
constexpr std::pair<string_view, string_view> get() {
  string_view first = "first";
  string_view second = "second";
  first.swap(second);
  return std::make_pair(first, second);
}
TEST(StringViewTest, testSwapMethod) {
  static_assert(string_view("second") == get().first, "");
  static_assert(string_view("first") == get().second, "");

  EXPECT_EQ(string_view("second"), get().first);
  EXPECT_EQ(string_view("first"), get().second);
}
} // namespace test_swap_method

namespace test_copy {
TEST(StringViewTest, whenCopyingFullStringView_thenDestinationHasCorrectData) {
  string_view data = "hello";
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  char result[5];
  size_t num_copied = data.copy(result, 5);
  EXPECT_EQ(5, num_copied);
  EXPECT_TRUE(string_equal("hello", result, 5));
}

TEST(StringViewTest, whenCopyingSubstr_thenDestinationHasCorrectData) {
  string_view data = "hello";
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  char result[2];
  size_t num_copied = data.copy(result, 2, 2);
  EXPECT_EQ(2, num_copied);
  EXPECT_TRUE(string_equal("ll", result, 2));
}

TEST(StringViewTest, whenCopyingTooMuch_thenJustCopiesLess) {
  string_view data = "hello";
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  char result[100];
  size_t num_copied = data.copy(result, 100, 2);
  EXPECT_EQ(3, num_copied);
  EXPECT_TRUE(string_equal("llo", result, 3));
}

TEST(StringViewTest, whenCopyingJustAtRange_thenDoesntCrash) {
  string_view data = "hello";
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  char result[1];
  size_t num_copied = data.copy(result, 2, 5);
  EXPECT_EQ(0, num_copied);
}

TEST(StringViewTest, whenCopyingOutOfRange_thenThrows) {
  string_view data = "hello";
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  char result[2];
  expectThrows<std::out_of_range>(
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
      [&] { data.copy(result, 2, 6); },
      "basic_string_view::copy: out of range. Index: 6, size: 5");
}
} // namespace test_copy

namespace test_substr {
static_assert(string_view("").substr() == string_view(""), "");
static_assert(string_view("").substr(0) == string_view(""), "");
static_assert(string_view("").substr(0, 0) == string_view(""), "");

static_assert(string_view("hello").substr() == string_view("hello"), "");
static_assert(string_view("hello").substr(0) == string_view("hello"), "");
static_assert(string_view("hello").substr(1) == string_view("ello"), "");
static_assert(string_view("hello").substr(5) == string_view(""), "");

static_assert(string_view("hello").substr(0, 0) == string_view(""), "");
static_assert(string_view("hello").substr(0, 2) == string_view("he"), "");
static_assert(string_view("hello").substr(1, 2) == string_view("el"), "");
static_assert(string_view("hello").substr(4, 1) == string_view("o"), "");

static_assert(string_view("hello").substr(0, 100) == string_view("hello"), "");
static_assert(string_view("hello").substr(1, 100) == string_view("ello"), "");
static_assert(string_view("hello").substr(5, 100) == string_view(""), "");

TEST(StringViewTest, whenCallingSubstrWithPosOutOfRange_thenThrows) {
  expectThrows<std::out_of_range>(
      [] { string_view("hello").substr(6); },
      "basic_string_view::substr parameter out of bounds. Index: 6, size: 5");

  expectThrows<std::out_of_range>(
      [] { string_view("hello").substr(6, 0); },
      "basic_string_view::substr parameter out of bounds. Index: 6, size: 5");
}
} // namespace test_substr

namespace test_compare_overload1 {
static_assert(0 == string_view("").compare(string_view("")), "");
static_assert(0 == string_view("a").compare(string_view("a")), "");
static_assert(0 == string_view("hello").compare(string_view("hello")), "");
static_assert(0 < string_view("hello").compare(string_view("")), "");
static_assert(0 < string_view("hello").compare(string_view("aello")), "");
static_assert(0 < string_view("hello").compare(string_view("a")), "");
static_assert(
    0 < string_view("hello").compare(string_view("abcdefghijklmno")),
    "");
static_assert(0 < string_view("hello").compare(string_view("hela")), "");
static_assert(0 < string_view("hello").compare(string_view("helao")), "");
static_assert(
    0 < string_view("hello").compare(string_view("helaobcdefgh")),
    "");
static_assert(0 < string_view("hello").compare(string_view("hell")), "");
static_assert(0 > string_view("").compare(string_view("hello")), "");
static_assert(0 > string_view("hello").compare(string_view("zello")), "");
static_assert(0 > string_view("hello").compare(string_view("z")), "");
static_assert(
    0 > string_view("hello").compare(string_view("zabcdefghijklmno")),
    "");
static_assert(0 > string_view("hello").compare(string_view("helz")), "");
static_assert(0 > string_view("hello").compare(string_view("helzo")), "");
static_assert(
    0 > string_view("hello").compare(string_view("helzobcdefgh")),
    "");
static_assert(0 > string_view("hello").compare(string_view("helloa")), "");
} // namespace test_compare_overload1

namespace test_compare_overload2 {
static_assert(0 == string_view("").compare(0, 0, string_view("")), "");
static_assert(0 == string_view("hello").compare(2, 2, string_view("ll")), "");
static_assert(0 < string_view("hello").compare(2, 2, string_view("l")), "");
static_assert(0 > string_view("hello").compare(2, 2, string_view("lll")), "");
static_assert(0 < string_view("hello").compare(2, 2, string_view("la")), "");
static_assert(0 > string_view("hello").compare(2, 2, string_view("lz")), "");
} // namespace test_compare_overload2

namespace test_compare_overload3 {
static_assert(0 == string_view("").compare(0, 0, string_view(""), 0, 0), "");
static_assert(
    0 == string_view("hello").compare(2, 2, string_view("hello"), 2, 2),
    "");
static_assert(
    0 < string_view("hello").compare(2, 2, string_view("hello"), 2, 1),
    "");
static_assert(
    0 > string_view("hello").compare(2, 2, string_view("hello"), 2, 3),
    "");
static_assert(
    0 < string_view("hello").compare(2, 2, string_view("hellola"), 5, 2),
    "");
static_assert(
    0 > string_view("hello").compare(2, 2, string_view("hellolz"), 5, 2),
    "");
} // namespace test_compare_overload3

namespace test_compare_overload4 {
static_assert(0 == string_view("").compare(""), "");
static_assert(0 == string_view("a").compare("a"), "");
static_assert(0 == string_view("hello").compare("hello"), "");
static_assert(0 < string_view("hello").compare(""), "");
static_assert(0 < string_view("hello").compare("aello"), "");
static_assert(0 < string_view("hello").compare("a"), "");
static_assert(0 < string_view("hello").compare("abcdefghijklmno"), "");
static_assert(0 < string_view("hello").compare("hela"), "");
static_assert(0 < string_view("hello").compare("helao"), "");
static_assert(0 < string_view("hello").compare("helaobcdefgh"), "");
static_assert(0 < string_view("hello").compare("hell"), "");
static_assert(0 > string_view("").compare("hello"), "");
static_assert(0 > string_view("hello").compare("zello"), "");
static_assert(0 > string_view("hello").compare("z"), "");
static_assert(0 > string_view("hello").compare("zabcdefghijklmno"), "");
static_assert(0 > string_view("hello").compare("helz"), "");
static_assert(0 > string_view("hello").compare("helzo"), "");
static_assert(0 > string_view("hello").compare("helzobcdefgh"), "");
static_assert(0 > string_view("hello").compare("helloa"), "");
} // namespace test_compare_overload4

namespace test_compare_overload5 {
static_assert(0 == string_view("").compare(0, 0, ""), "");
static_assert(0 == string_view("hello").compare(2, 2, "ll"), "");
static_assert(0 < string_view("hello").compare(2, 2, "l"), "");
static_assert(0 > string_view("hello").compare(2, 2, "lll"), "");
static_assert(0 < string_view("hello").compare(2, 2, "la"), "");
static_assert(0 > string_view("hello").compare(2, 2, "lz"), "");
} // namespace test_compare_overload5

namespace test_compare_overload6 {
static_assert(0 == string_view("").compare(0, 0, "", 0, 0), "");
static_assert(0 == string_view("hello").compare(2, 2, "hello", 2, 2), "");
static_assert(0 < string_view("hello").compare(2, 2, "hello", 2, 1), "");
static_assert(0 > string_view("hello").compare(2, 2, "hello", 2, 3), "");
static_assert(0 < string_view("hello").compare(2, 2, "hellola", 5, 2), "");
static_assert(0 > string_view("hello").compare(2, 2, "hellolz", 5, 2), "");
} // namespace test_compare_overload6

namespace test_equality_comparison {
static_assert(string_view("hi") == string_view("hi"), "");
static_assert(!(string_view("hi") != string_view("hi")), "");

static_assert(string_view("") == string_view(""), "");
static_assert(!(string_view("") != string_view("")), "");

static_assert(string_view("hi") != string_view("hi2"), "");
static_assert(!(string_view("hi") == string_view("hi2")), "");

static_assert(string_view("hi2") != string_view("hi"), "");
static_assert(!(string_view("hi2") == string_view("hi")), "");

static_assert(string_view("hi") != string_view("ha"), "");
static_assert(!(string_view("hi") == string_view("ha")), "");

static_assert(string_view("ha") != string_view("hi"), "");
static_assert(!(string_view("ha") == string_view("hi")), "");
} // namespace test_equality_comparison

namespace test_less_than {
static_assert(!(string_view("") < string_view("")), "");
static_assert(!(string_view("a") < string_view("a")), "");
static_assert(!(string_view("hello") < string_view("hello")), "");
static_assert(!(string_view("hello") < string_view("")), "");
static_assert(!(string_view("hello") < string_view("aello")), "");
static_assert(!(string_view("hello") < string_view("a")), "");
static_assert(!(string_view("hello") < string_view("abcdefghijklmno")), "");
static_assert(!(string_view("hello") < string_view("hela")), "");
static_assert(!(string_view("hello") < string_view("helao")), "");
static_assert(!(string_view("hello") < string_view("helaobcdefgh")), "");
static_assert(!(string_view("hello") < string_view("hell")), "");
static_assert(string_view("") < string_view("hello"), "");
static_assert(string_view("hello") < string_view("zello"), "");
static_assert(string_view("hello") < string_view("z"), "");
static_assert(string_view("hello") < string_view("zabcdefghijklmno"), "");
static_assert(string_view("hello") < string_view("helz"), "");
static_assert(string_view("hello") < string_view("helzo"), "");
static_assert(string_view("hello") < string_view("helzobcdefgh"), "");
static_assert(string_view("hello") < string_view("helloa"), "");
} // namespace test_less_than

namespace test_less_or_equal_than {
static_assert(string_view("") <= string_view(""), "");
static_assert(string_view("a") <= string_view("a"), "");
static_assert(string_view("hello") <= string_view("hello"), "");
static_assert(!(string_view("hello") <= string_view("")), "");
static_assert(!(string_view("hello") <= string_view("aello")), "");
static_assert(!(string_view("hello") <= string_view("a")), "");
static_assert(!(string_view("hello") <= string_view("abcdefghijklmno")), "");
static_assert(!(string_view("hello") <= string_view("hela")), "");
static_assert(!(string_view("hello") <= string_view("helao")), "");
static_assert(!(string_view("hello") <= string_view("helaobcdefgh")), "");
static_assert(!(string_view("hello") <= string_view("hell")), "");
static_assert(string_view("") <= string_view("hello"), "");
static_assert(string_view("hello") <= string_view("zello"), "");
static_assert(string_view("hello") <= string_view("z"), "");
static_assert(string_view("hello") <= string_view("zabcdefghijklmno"), "");
static_assert(string_view("hello") <= string_view("helz"), "");
static_assert(string_view("hello") <= string_view("helzo"), "");
static_assert(string_view("hello") <= string_view("helzobcdefgh"), "");
static_assert(string_view("hello") <= string_view("helloa"), "");
} // namespace test_less_or_equal_than

namespace test_greater_than {
static_assert(!(string_view("") > string_view("")), "");
static_assert(!(string_view("a") > string_view("a")), "");
static_assert(!(string_view("hello") > string_view("hello")), "");
static_assert(string_view("hello") > string_view(""), "");
static_assert(string_view("hello") > string_view("aello"), "");
static_assert(string_view("hello") > string_view("a"), "");
static_assert(string_view("hello") > string_view("abcdefghijklmno"), "");
static_assert(string_view("hello") > string_view("hela"), "");
static_assert(string_view("hello") > string_view("helao"), "");
static_assert(string_view("hello") > string_view("helaobcdefgh"), "");
static_assert(string_view("hello") > string_view("hell"), "");
static_assert(!(string_view("") > string_view("hello")), "");
static_assert(!(string_view("hello") > string_view("zello")), "");
static_assert(!(string_view("hello") > string_view("z")), "");
static_assert(!(string_view("hello") > string_view("zabcdefghijklmno")), "");
static_assert(!(string_view("hello") > string_view("helz")), "");
static_assert(!(string_view("hello") > string_view("helzo")), "");
static_assert(!(string_view("hello") > string_view("helzobcdefgh")), "");
static_assert(!(string_view("hello") > string_view("helloa")), "");
} // namespace test_greater_than

namespace test_greater_or_equals_than {
static_assert(string_view("") >= string_view(""), "");
static_assert(string_view("a") >= string_view("a"), "");
static_assert(string_view("hello") >= string_view("hello"), "");
static_assert(string_view("hello") >= string_view(""), "");
static_assert(string_view("hello") >= string_view("aello"), "");
static_assert(string_view("hello") >= string_view("a"), "");
static_assert(string_view("hello") >= string_view("abcdefghijklmno"), "");
static_assert(string_view("hello") >= string_view("hela"), "");
static_assert(string_view("hello") >= string_view("helao"), "");
static_assert(string_view("hello") >= string_view("helaobcdefgh"), "");
static_assert(string_view("hello") >= string_view("hell"), "");
static_assert(!(string_view("") >= string_view("hello")), "");
static_assert(!(string_view("hello") >= string_view("zello")), "");
static_assert(!(string_view("hello") >= string_view("z")), "");
static_assert(!(string_view("hello") >= string_view("zabcdefghijklmno")), "");
static_assert(!(string_view("hello") >= string_view("helz")), "");
static_assert(!(string_view("hello") >= string_view("helzo")), "");
static_assert(!(string_view("hello") >= string_view("helzobcdefgh")), "");
static_assert(!(string_view("hello") >= string_view("helloa")), "");
} // namespace test_greater_or_equals_than

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

namespace test_find_overload1 {
static_assert(0 == string_view("").find(string_view("")), "");
static_assert(string_view::npos == string_view("").find(string_view("a")), "");
static_assert(
    string_view::npos == string_view("").find(string_view(""), 1),
    "");
static_assert(0 == string_view("abc").find(string_view("")), "");
static_assert(2 == string_view("abc").find(string_view(""), 2), "");
static_assert(0 == string_view("abc").find(string_view("a")), "");
static_assert(0 == string_view("abc").find(string_view("ab")), "");
static_assert(0 == string_view("abc").find(string_view("abc")), "");
static_assert(1 == string_view("abc").find(string_view("bc")), "");
static_assert(1 == string_view("abc").find(string_view("b")), "");
static_assert(2 == string_view("abc").find(string_view("c")), "");
static_assert(0 == string_view("abc").find(string_view("a")), "");
static_assert(0 == string_view("abc").find(string_view("ab")), "");
static_assert(0 == string_view("abc").find(string_view("abc")), "");
static_assert(1 == string_view("ababa").find(string_view("ba")), "");
static_assert(3 == string_view("ababa").find(string_view("ba"), 2), "");
static_assert(3 == string_view("ababa").find(string_view("ba"), 3), "");
static_assert(
    string_view::npos == string_view("ababa").find(string_view("ba"), 4),
    "");
static_assert(
    string_view::npos == string_view("abc").find(string_view("abcd")),
    "");
} // namespace test_find_overload1

namespace test_find_overload2 {
static_assert(string_view::npos == string_view("").find('a'), "");
static_assert(0 == string_view("a").find('a'), "");
static_assert(0 == string_view("abc").find('a'), "");
static_assert(string_view::npos == string_view("a").find('a', 1), "");
static_assert(1 == string_view("abc").find('b'), "");
static_assert(1 == string_view("abc").find('b', 1), "");
static_assert(string_view::npos == string_view("abc").find('b', 2), "");
static_assert(2 == string_view("abc").find('c'), "");
static_assert(2 == string_view("abc").find('c', 1), "");
static_assert(2 == string_view("abc").find('c', 2), "");
static_assert(string_view::npos == string_view("abc").find('c', 3), "");
static_assert(string_view::npos == string_view("abc").find('a', 100), "");
static_assert(string_view::npos == string_view("abc").find('z'), "");
static_assert(0 == string_view("ababa").find('a'), "");
static_assert(0 == string_view("ababa").find('a', 0), "");
static_assert(2 == string_view("ababa").find('a', 1), "");
static_assert(2 == string_view("ababa").find('a', 2), "");
static_assert(4 == string_view("ababa").find('a', 3), "");
static_assert(4 == string_view("ababa").find('a', 4), "");
static_assert(string_view::npos == string_view("ababa").find('a', 5), "");
} // namespace test_find_overload2

namespace test_find_overload3 {
static_assert(0 == string_view("").find("", 0, 0), "");
static_assert(string_view::npos == string_view("").find("a", 0, 1), "");
static_assert(string_view::npos == string_view("").find("", 1, 0), "");
static_assert(0 == string_view("abc").find("", 0, 0), "");
static_assert(2 == string_view("abc").find("", 2, 0), "");
static_assert(0 == string_view("abc").find("a", 0, 1), "");
static_assert(0 == string_view("abc").find("ab", 0, 2), "");
static_assert(0 == string_view("abc").find("abc", 0, 3), "");
static_assert(1 == string_view("abc").find("bc", 0, 2), "");
static_assert(1 == string_view("abc").find("b", 0, 1), "");
static_assert(2 == string_view("abc").find("c", 0, 1), "");
static_assert(0 == string_view("abc").find("a", 0, 1), "");
static_assert(0 == string_view("abc").find("ab", 0, 2), "");
static_assert(0 == string_view("abc").find("abc", 0, 3), "");
static_assert(1 == string_view("ababa").find("ba", 0, 2), "");
static_assert(3 == string_view("ababa").find("ba", 2, 2), "");
static_assert(3 == string_view("ababa").find("ba", 3, 2), "");
static_assert(string_view::npos == string_view("ababa").find("ba", 4, 2), "");
static_assert(string_view::npos == string_view("abc").find("abcd", 0, 4), "");
} // namespace test_find_overload3

namespace test_find_overload4 {
static_assert(0 == string_view("").find(""), "");
static_assert(string_view::npos == string_view("").find("a"), "");
static_assert(string_view::npos == string_view("").find("", 1), "");
static_assert(0 == string_view("abc").find(""), "");
static_assert(2 == string_view("abc").find("", 2), "");
static_assert(0 == string_view("abc").find("a"), "");
static_assert(0 == string_view("abc").find("ab"), "");
static_assert(0 == string_view("abc").find("abc"), "");
static_assert(1 == string_view("abc").find("bc"), "");
static_assert(1 == string_view("abc").find("b"), "");
static_assert(2 == string_view("abc").find("c"), "");
static_assert(0 == string_view("abc").find("a"), "");
static_assert(0 == string_view("abc").find("ab"), "");
static_assert(0 == string_view("abc").find("abc"), "");
static_assert(1 == string_view("ababa").find("ba"), "");
static_assert(3 == string_view("ababa").find("ba", 2), "");
static_assert(3 == string_view("ababa").find("ba", 3), "");
static_assert(string_view::npos == string_view("ababa").find("ba", 4), "");
static_assert(string_view::npos == string_view("abc").find("abcd"), "");
} // namespace test_find_overload4

namespace test_rfind_overload1 {
static_assert(0 == string_view("").rfind(string_view("")), "");
static_assert(string_view::npos == string_view("").rfind(string_view("a")), "");
static_assert(0 == string_view("").rfind(string_view(""), 1), "");
static_assert(3 == string_view("abc").rfind(string_view("")), "");
static_assert(0 == string_view("abc").rfind(string_view(""), 0), "");
static_assert(0 == string_view("abc").rfind(string_view("a")), "");
static_assert(0 == string_view("abc").rfind(string_view("ab")), "");
static_assert(0 == string_view("abc").rfind(string_view("abc")), "");
static_assert(1 == string_view("abc").rfind(string_view("bc")), "");
static_assert(1 == string_view("abc").rfind(string_view("b")), "");
static_assert(2 == string_view("abc").rfind(string_view("c")), "");
static_assert(0 == string_view("abc").rfind(string_view("a")), "");
static_assert(0 == string_view("abc").rfind(string_view("ab")), "");
static_assert(0 == string_view("abc").rfind(string_view("abc")), "");
static_assert(3 == string_view("ababa").rfind(string_view("ba")), "");
static_assert(1 == string_view("ababa").rfind(string_view("ba"), 2), "");
static_assert(1 == string_view("ababa").rfind(string_view("ba"), 1), "");
static_assert(
    string_view::npos == string_view("ababa").rfind(string_view("ba"), 0),
    "");
static_assert(
    string_view::npos == string_view("abc").rfind(string_view("abcd")),
    "");
} // namespace test_rfind_overload1

namespace test_rfind_overload2 {
static_assert(string_view::npos == string_view("").rfind('a'), "");
static_assert(0 == string_view("a").rfind('a'), "");
static_assert(0 == string_view("abc").rfind('a'), "");
static_assert(0 == string_view("a").rfind('a', 0), "");
static_assert(1 == string_view("abc").rfind('b'), "");
static_assert(string_view::npos == string_view("abc").rfind('b', 0), "");
static_assert(1 == string_view("abc").rfind('b', 1), "");
static_assert(2 == string_view("abc").rfind('c'), "");
static_assert(string_view::npos == string_view("abc").rfind('c', 0), "");
static_assert(string_view::npos == string_view("abc").rfind('c', 1), "");
static_assert(2 == string_view("abc").rfind('c', 2), "");
static_assert(2 == string_view("abc").rfind('c', 3), "");
static_assert(0 == string_view("abc").rfind('a', 100), "");
static_assert(string_view::npos == string_view("abc").rfind('z'), "");
static_assert(4 == string_view("ababa").rfind('a'), "");
static_assert(0 == string_view("ababa").rfind('a', 0), "");
static_assert(0 == string_view("ababa").rfind('a', 1), "");
static_assert(2 == string_view("ababa").rfind('a', 2), "");
static_assert(2 == string_view("ababa").rfind('a', 3), "");
static_assert(4 == string_view("ababa").rfind('a', 4), "");
static_assert(4 == string_view("ababa").rfind('a', 5), "");
} // namespace test_rfind_overload2

namespace test_rfind_overload3 {
static_assert(0 == string_view("").rfind("", string_view::npos, 0), "");
static_assert(
    string_view::npos == string_view("").rfind("a", string_view::npos, 1),
    "");
static_assert(0 == string_view("").rfind("", 1, 0), "");
static_assert(3 == string_view("abc").rfind("", string_view::npos, 0), "");
static_assert(0 == string_view("abc").rfind("", 0, 0), "");
static_assert(0 == string_view("abc").rfind("a", string_view::npos, 1), "");
static_assert(0 == string_view("abc").rfind("ab", string_view::npos, 2), "");
static_assert(0 == string_view("abc").rfind("abc", string_view::npos, 3), "");
static_assert(1 == string_view("abc").rfind("bc", string_view::npos, 2), "");
static_assert(1 == string_view("abc").rfind("b", string_view::npos, 1), "");
static_assert(2 == string_view("abc").rfind("c", string_view::npos, 1), "");
static_assert(0 == string_view("abc").rfind("a", string_view::npos, 1), "");
static_assert(0 == string_view("abc").rfind("ab", string_view::npos, 2), "");
static_assert(0 == string_view("abc").rfind("abc", string_view::npos, 3), "");
static_assert(3 == string_view("ababa").rfind("ba", string_view::npos, 2), "");
static_assert(1 == string_view("ababa").rfind("ba", 2, 2), "");
static_assert(1 == string_view("ababa").rfind("ba", 1, 2), "");
static_assert(string_view::npos == string_view("ababa").rfind("ba", 0, 2), "");
static_assert(
    string_view::npos == string_view("abc").rfind("abcd", string_view::npos, 4),
    "");
} // namespace test_rfind_overload3

namespace test_rfind_overload4 {
static_assert(0 == string_view("").rfind(""), "");
static_assert(string_view::npos == string_view("").rfind("a"), "");
static_assert(0 == string_view("").rfind("", 1), "");
static_assert(3 == string_view("abc").rfind(""), "");
static_assert(0 == string_view("abc").rfind("", 0), "");
static_assert(0 == string_view("abc").rfind("a"), "");
static_assert(0 == string_view("abc").rfind("ab"), "");
static_assert(0 == string_view("abc").rfind("abc"), "");
static_assert(1 == string_view("abc").rfind("bc"), "");
static_assert(1 == string_view("abc").rfind("b"), "");
static_assert(2 == string_view("abc").rfind("c"), "");
static_assert(0 == string_view("abc").rfind("a"), "");
static_assert(0 == string_view("abc").rfind("ab"), "");
static_assert(0 == string_view("abc").rfind("abc"), "");
static_assert(3 == string_view("ababa").rfind("ba"), "");
static_assert(1 == string_view("ababa").rfind("ba", 2), "");
static_assert(1 == string_view("ababa").rfind("ba", 1), "");
static_assert(string_view::npos == string_view("ababa").rfind("ba", 0), "");
static_assert(string_view::npos == string_view("abc").rfind("abcd"), "");
} // namespace test_rfind_overload4

namespace test_find_first_of_overload1 {
static_assert(
    string_view::npos == string_view("").find_first_of(string_view("")),
    "");
static_assert(
    string_view::npos == string_view("").find_first_of(string_view("a")),
    "");
static_assert(
    string_view::npos == string_view("").find_first_of(string_view("abc")),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_of(string_view("")),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_of(string_view("d")),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_of(string_view("def")),
    "");

static_assert(0 == string_view("abcabc").find_first_of(string_view("a")), "");
static_assert(1 == string_view("abcabc").find_first_of(string_view("b")), "");
static_assert(2 == string_view("abcabc").find_first_of(string_view("c")), "");
static_assert(1 == string_view("abcabc").find_first_of(string_view("bc")), "");
static_assert(1 == string_view("abcabc").find_first_of(string_view("cbd")), "");

static_assert(
    string_view::npos == string_view("").find_first_of(string_view(""), 1),
    "");
static_assert(
    string_view::npos == string_view("").find_first_of(string_view("a"), 1),
    "");
static_assert(
    string_view::npos == string_view("").find_first_of(string_view("abc"), 100),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_of(string_view(""), 1),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_of(string_view("d"), 3),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_first_of(string_view("def"), 2),
    "");

static_assert(
    3 == string_view("abcabc").find_first_of(string_view("a"), 1),
    "");
static_assert(
    4 == string_view("abcabc").find_first_of(string_view("b"), 3),
    "");
static_assert(
    5 == string_view("abcabc").find_first_of(string_view("c"), 5),
    "");
static_assert(
    4 == string_view("abcabc").find_first_of(string_view("bc"), 3),
    "");
static_assert(
    4 == string_view("abcabc").find_first_of(string_view("cbd"), 4),
    "");
} // namespace test_find_first_of_overload1

namespace test_find_first_of_overload2 {
static_assert(string_view::npos == string_view("").find_first_of('a'), "");
static_assert(0 == string_view("a").find_first_of('a'), "");
static_assert(0 == string_view("abc").find_first_of('a'), "");
static_assert(string_view::npos == string_view("a").find_first_of('a', 1), "");
static_assert(1 == string_view("abc").find_first_of('b'), "");
static_assert(1 == string_view("abc").find_first_of('b', 1), "");
static_assert(
    string_view::npos == string_view("abc").find_first_of('b', 2),
    "");
static_assert(2 == string_view("abc").find_first_of('c'), "");
static_assert(2 == string_view("abc").find_first_of('c', 1), "");
static_assert(2 == string_view("abc").find_first_of('c', 2), "");
static_assert(
    string_view::npos == string_view("abc").find_first_of('c', 3),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_of('a', 100),
    "");
static_assert(string_view::npos == string_view("abc").find_first_of('z'), "");
static_assert(0 == string_view("ababa").find_first_of('a'), "");
static_assert(0 == string_view("ababa").find_first_of('a', 0), "");
static_assert(2 == string_view("ababa").find_first_of('a', 1), "");
static_assert(2 == string_view("ababa").find_first_of('a', 2), "");
static_assert(4 == string_view("ababa").find_first_of('a', 3), "");
static_assert(4 == string_view("ababa").find_first_of('a', 4), "");
static_assert(
    string_view::npos == string_view("ababa").find_first_of('a', 5),
    "");
} // namespace test_find_first_of_overload2

namespace test_find_first_of_overload3 {
static_assert(
    string_view::npos == string_view("").find_first_of("ab", 0, 0),
    "");
static_assert(
    string_view::npos == string_view("").find_first_of("abc", 0, 1),
    "");
static_assert(
    string_view::npos == string_view("").find_first_of("abcdef", 0, 3),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_of("abcdef", 0, 0),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_of("defa", 0, 1),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_of("defabc", 0, 3),
    "");

static_assert(0 == string_view("abcabc").find_first_of("abc", 0, 1), "");
static_assert(1 == string_view("abcabc").find_first_of("bac", 0, 1), "");
static_assert(2 == string_view("abcabc").find_first_of("cab", 0, 1), "");
static_assert(1 == string_view("abcabc").find_first_of("bccda", 0, 2), "");
static_assert(1 == string_view("abcabc").find_first_of("cbdab", 0, 3), "");

static_assert(
    string_view::npos == string_view("").find_first_of("ab", 1, 0),
    "");
static_assert(
    string_view::npos == string_view("").find_first_of("abc", 1, 1),
    "");
static_assert(
    string_view::npos == string_view("").find_first_of("abcdef", 100, 3),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_of("abcdef", 1, 0),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_of("defa", 3, 1),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_of("defabc", 2, 3),
    "");

static_assert(3 == string_view("abcabc").find_first_of("abc", 1, 1), "");
static_assert(4 == string_view("abcabc").find_first_of("bac", 3, 1), "");
static_assert(5 == string_view("abcabc").find_first_of("cab", 5, 1), "");
static_assert(4 == string_view("abcabc").find_first_of("bccda", 3, 2), "");
static_assert(4 == string_view("abcabc").find_first_of("cbdab", 4, 3), "");
} // namespace test_find_first_of_overload3

namespace test_find_first_of_overload4 {
static_assert(string_view::npos == string_view("").find_first_of(""), "");
static_assert(string_view::npos == string_view("").find_first_of("a"), "");
static_assert(string_view::npos == string_view("").find_first_of("abc"), "");
static_assert(string_view::npos == string_view("abc").find_first_of(""), "");
static_assert(string_view::npos == string_view("abc").find_first_of("d"), "");
static_assert(string_view::npos == string_view("abc").find_first_of("def"), "");

static_assert(0 == string_view("abcabc").find_first_of("a"), "");
static_assert(1 == string_view("abcabc").find_first_of("b"), "");
static_assert(2 == string_view("abcabc").find_first_of("c"), "");
static_assert(1 == string_view("abcabc").find_first_of("bc"), "");
static_assert(1 == string_view("abcabc").find_first_of("cbd"), "");

static_assert(string_view::npos == string_view("").find_first_of("", 1), "");
static_assert(string_view::npos == string_view("").find_first_of("a", 1), "");
static_assert(
    string_view::npos == string_view("").find_first_of("abc", 100),
    "");
static_assert(string_view::npos == string_view("abc").find_first_of("", 1), "");
static_assert(
    string_view::npos == string_view("abc").find_first_of("d", 3),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_of("def", 2),
    "");

static_assert(3 == string_view("abcabc").find_first_of("a", 1), "");
static_assert(4 == string_view("abcabc").find_first_of("b", 3), "");
static_assert(5 == string_view("abcabc").find_first_of("c", 5), "");
static_assert(4 == string_view("abcabc").find_first_of("bc", 3), "");
static_assert(4 == string_view("abcabc").find_first_of("cbd", 4), "");
} // namespace test_find_first_of_overload4

namespace test_find_last_of_overload1 {
static_assert(
    string_view::npos == string_view("").find_last_of(string_view("")),
    "");
static_assert(
    string_view::npos == string_view("").find_last_of(string_view("a")),
    "");
static_assert(
    string_view::npos == string_view("").find_last_of(string_view("abc")),
    "");
static_assert(
    string_view::npos == string_view("abc").find_last_of(string_view("")),
    "");
static_assert(
    string_view::npos == string_view("abc").find_last_of(string_view("d")),
    "");
static_assert(
    string_view::npos == string_view("abc").find_last_of(string_view("def")),
    "");

static_assert(3 == string_view("abcabc").find_last_of(string_view("a")), "");
static_assert(4 == string_view("abcabc").find_last_of(string_view("b")), "");
static_assert(5 == string_view("abcabc").find_last_of(string_view("c")), "");
static_assert(5 == string_view("abcabc").find_last_of(string_view("bc")), "");
static_assert(5 == string_view("abcabc").find_last_of(string_view("cbd")), "");

static_assert(
    string_view::npos == string_view("").find_last_of(string_view(""), 1),
    "");
static_assert(
    string_view::npos == string_view("").find_last_of(string_view("a"), 0),
    "");
static_assert(
    string_view::npos == string_view("").find_last_of(string_view("abc"), 100),
    "");
static_assert(
    string_view::npos == string_view("abc").find_last_of(string_view(""), 1),
    "");
static_assert(
    string_view::npos == string_view("abc").find_last_of(string_view("d"), 3),
    "");
static_assert(
    string_view::npos == string_view("abc").find_last_of(string_view("def"), 2),
    "");

static_assert(0 == string_view("abcabc").find_last_of(string_view("a"), 2), "");
static_assert(1 == string_view("abcabc").find_last_of(string_view("b"), 3), "");
static_assert(2 == string_view("abcabc").find_last_of(string_view("c"), 2), "");
static_assert(
    2 == string_view("abcabc").find_last_of(string_view("bc"), 3),
    "");
static_assert(
    2 == string_view("abcabc").find_last_of(string_view("cbd"), 2),
    "");
} // namespace test_find_last_of_overload1

namespace test_find_last_of_overload2 {
static_assert(string_view::npos == string_view("").find_last_of('a'), "");
static_assert(0 == string_view("a").find_last_of('a'), "");
static_assert(0 == string_view("abc").find_last_of('a'), "");
static_assert(0 == string_view("a").find_last_of('a', 0), "");
static_assert(1 == string_view("abc").find_last_of('b'), "");
static_assert(string_view::npos == string_view("abc").find_last_of('b', 0), "");
static_assert(1 == string_view("abc").find_last_of('b', 1), "");
static_assert(2 == string_view("abc").find_last_of('c'), "");
static_assert(string_view::npos == string_view("abc").find_last_of('c', 0), "");
static_assert(string_view::npos == string_view("abc").find_last_of('c', 1), "");
static_assert(2 == string_view("abc").find_last_of('c', 2), "");
static_assert(2 == string_view("abc").find_last_of('c', 3), "");
static_assert(0 == string_view("abc").find_last_of('a', 100), "");
static_assert(string_view::npos == string_view("abc").find_last_of('z'), "");
static_assert(4 == string_view("ababa").find_last_of('a'), "");
static_assert(0 == string_view("ababa").find_last_of('a', 0), "");
static_assert(0 == string_view("ababa").find_last_of('a', 1), "");
static_assert(2 == string_view("ababa").find_last_of('a', 2), "");
static_assert(2 == string_view("ababa").find_last_of('a', 3), "");
static_assert(4 == string_view("ababa").find_last_of('a', 4), "");
static_assert(4 == string_view("ababa").find_last_of('a', 5), "");
} // namespace test_find_last_of_overload2

namespace test_find_last_of_overload3 {
static_assert(
    string_view::npos ==
        string_view("").find_last_of("ab", string_view::npos, 0),
    "");
static_assert(
    string_view::npos ==
        string_view("").find_last_of("abc", string_view::npos, 1),
    "");
static_assert(
    string_view::npos ==
        string_view("").find_last_of("abcdef", string_view::npos, 3),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_last_of("abcdef", string_view::npos, 0),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_last_of("defa", string_view::npos, 1),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_last_of("defcba", string_view::npos, 3),
    "");

static_assert(
    3 == string_view("abcabc").find_last_of("abc", string_view::npos, 1),
    "");
static_assert(
    4 == string_view("abcabc").find_last_of("bca", string_view::npos, 1),
    "");
static_assert(
    5 == string_view("abcabc").find_last_of("cab", string_view::npos, 1),
    "");
static_assert(
    5 == string_view("abcabc").find_last_of("bcab", string_view::npos, 2),
    "");
static_assert(
    5 == string_view("abcabc").find_last_of("cbdac", string_view::npos, 3),
    "");

static_assert(
    string_view::npos == string_view("").find_last_of("ab", 1, 0),
    "");
static_assert(
    string_view::npos == string_view("").find_last_of("abc", 0, 1),
    "");
static_assert(
    string_view::npos == string_view("").find_last_of("abcdef", 100, 3),
    "");
static_assert(
    string_view::npos == string_view("abc").find_last_of("abcdef", 1, 0),
    "");
static_assert(
    string_view::npos == string_view("abc").find_last_of("defa", 3, 1),
    "");
static_assert(
    string_view::npos == string_view("abc").find_last_of("defcba", 2, 3),
    "");

static_assert(0 == string_view("abcabc").find_last_of("abc", 2, 1), "");
static_assert(1 == string_view("abcabc").find_last_of("bca", 3, 1), "");
static_assert(2 == string_view("abcabc").find_last_of("cab", 2, 1), "");
static_assert(2 == string_view("abcabc").find_last_of("bcab", 3, 2), "");
static_assert(2 == string_view("abcabc").find_last_of("cbdac", 2, 2), "");
} // namespace test_find_last_of_overload3

namespace test_find_last_of_overload4 {
static_assert(string_view::npos == string_view("").find_last_of(""), "");
static_assert(string_view::npos == string_view("").find_last_of("a"), "");
static_assert(string_view::npos == string_view("").find_last_of("abc"), "");
static_assert(string_view::npos == string_view("abc").find_last_of(""), "");
static_assert(string_view::npos == string_view("abc").find_last_of("d"), "");
static_assert(string_view::npos == string_view("abc").find_last_of("def"), "");

static_assert(3 == string_view("abcabc").find_last_of("a"), "");
static_assert(4 == string_view("abcabc").find_last_of("b"), "");
static_assert(5 == string_view("abcabc").find_last_of("c"), "");
static_assert(5 == string_view("abcabc").find_last_of("bc"), "");
static_assert(5 == string_view("abcabc").find_last_of("cbd"), "");

static_assert(string_view::npos == string_view("").find_last_of("", 1), "");
static_assert(string_view::npos == string_view("").find_last_of("a", 0), "");
static_assert(
    string_view::npos == string_view("").find_last_of("abc", 100),
    "");
static_assert(string_view::npos == string_view("abc").find_last_of("", 1), "");
static_assert(string_view::npos == string_view("abc").find_last_of("d", 3), "");
static_assert(
    string_view::npos == string_view("abc").find_last_of("def", 2),
    "");

static_assert(0 == string_view("abcabc").find_last_of("a", 2), "");
static_assert(1 == string_view("abcabc").find_last_of("b", 3), "");
static_assert(2 == string_view("abcabc").find_last_of("c", 2), "");
static_assert(2 == string_view("abcabc").find_last_of("bc", 3), "");
static_assert(2 == string_view("abcabc").find_last_of("cbd", 2), "");
} // namespace test_find_last_of_overload4

namespace test_find_first_not_of_overload1 {
static_assert(
    string_view::npos == string_view("").find_first_not_of(string_view("")),
    "");
static_assert(
    string_view::npos == string_view("").find_first_not_of(string_view("a")),
    "");
static_assert(
    string_view::npos == string_view("").find_first_not_of(string_view("abc")),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_first_not_of(string_view("abc")),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_first_not_of(string_view("acdb")),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_first_not_of(string_view("defabc")),
    "");

static_assert(
    0 == string_view("abcabc").find_first_not_of(string_view("")),
    "");
static_assert(
    0 == string_view("abcabc").find_first_not_of(string_view("bc")),
    "");
static_assert(
    1 == string_view("abcabc").find_first_not_of(string_view("ac")),
    "");
static_assert(
    2 == string_view("abcabc").find_first_not_of(string_view("ab")),
    "");
static_assert(
    1 == string_view("abcabc").find_first_not_of(string_view("a")),
    "");
static_assert(
    1 == string_view("abcabc").find_first_not_of(string_view("da")),
    "");

static_assert(
    string_view::npos == string_view("").find_first_not_of(string_view(""), 1),
    "");
static_assert(
    string_view::npos == string_view("").find_first_not_of(string_view("a"), 1),
    "");
static_assert(
    string_view::npos ==
        string_view("").find_first_not_of(string_view("abc"), 100),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_first_not_of(string_view("abc"), 1),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_first_not_of(string_view("acdb"), 3),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_first_not_of(string_view("defabc"), 2),
    "");

static_assert(
    1 == string_view("abcabc").find_first_not_of(string_view(""), 1),
    "");
static_assert(
    3 == string_view("abcabc").find_first_not_of(string_view("bc"), 1),
    "");
static_assert(
    4 == string_view("abcabc").find_first_not_of(string_view("ac"), 4),
    "");
static_assert(
    5 == string_view("abcabc").find_first_not_of(string_view("ab"), 5),
    "");
static_assert(
    4 == string_view("abcabc").find_first_not_of(string_view("a"), 3),
    "");
static_assert(
    4 == string_view("abcabc").find_first_not_of(string_view("da"), 4),
    "");
} // namespace test_find_first_not_of_overload1

namespace test_find_first_not_of_overload2 {
static_assert(string_view::npos == string_view("").find_first_not_of('a'), "");
static_assert(string_view::npos == string_view("a").find_first_not_of('a'), "");
static_assert(1 == string_view("abc").find_first_not_of('a'), "");
static_assert(
    string_view::npos == string_view("a").find_first_not_of('a', 1),
    "");
static_assert(0 == string_view("abc").find_first_not_of('b'), "");
static_assert(2 == string_view("abc").find_first_not_of('b', 1), "");
static_assert(2 == string_view("abc").find_first_not_of('b', 2), "");
static_assert(
    string_view::npos == string_view("abc").find_first_not_of('b', 3),
    "");
static_assert(0 == string_view("abc").find_first_not_of('c'), "");
static_assert(1 == string_view("abc").find_first_not_of('c', 1), "");
static_assert(
    string_view::npos == string_view("abc").find_first_not_of('c', 2),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_not_of('c', 3),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_not_of('a', 100),
    "");
static_assert(1 == string_view("ababa").find_first_not_of('a'), "");
static_assert(1 == string_view("ababa").find_first_not_of('a', 0), "");
static_assert(1 == string_view("ababa").find_first_not_of('a', 1), "");
static_assert(3 == string_view("ababa").find_first_not_of('a', 2), "");
static_assert(3 == string_view("ababa").find_first_not_of('a', 3), "");
static_assert(
    string_view::npos == string_view("ababa").find_first_not_of('a', 4),
    "");
static_assert(
    string_view::npos == string_view("ababa").find_first_not_of('a', 5),
    "");
} // namespace test_find_first_not_of_overload2

namespace test_find_first_not_of_overload3 {
static_assert(
    string_view::npos == string_view("").find_first_not_of("ab", 0, 0),
    "");
static_assert(
    string_view::npos == string_view("").find_first_not_of("abc", 0, 1),
    "");
static_assert(
    string_view::npos == string_view("").find_first_not_of("abcdef", 0, 3),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_not_of("abcdef", 0, 3),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_not_of("acdbef", 0, 4),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_not_of("defabcas", 0, 6),
    "");

static_assert(0 == string_view("abcabc").find_first_not_of("abc", 0, 0), "");
static_assert(0 == string_view("abcabc").find_first_not_of("bca", 0, 2), "");
static_assert(1 == string_view("abcabc").find_first_not_of("acb", 0, 2), "");
static_assert(2 == string_view("abcabc").find_first_not_of("abc", 0, 2), "");
static_assert(1 == string_view("abcabc").find_first_not_of("abac", 0, 1), "");
static_assert(1 == string_view("abcabc").find_first_not_of("dadab", 0, 2), "");

static_assert(
    string_view::npos == string_view("").find_first_not_of("ab", 1, 0),
    "");
static_assert(
    string_view::npos == string_view("").find_first_not_of("abc", 1, 1),
    "");
static_assert(
    string_view::npos == string_view("").find_first_not_of("abcdef", 100, 3),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_not_of("abcdef", 1, 3),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_not_of("acdbef", 3, 4),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_not_of("defabcas", 2, 6),
    "");

static_assert(1 == string_view("abcabc").find_first_not_of("bca", 1, 0), "");
static_assert(3 == string_view("abcabc").find_first_not_of("bca", 1, 2), "");
static_assert(4 == string_view("abcabc").find_first_not_of("acb", 4, 2), "");
static_assert(5 == string_view("abcabc").find_first_not_of("abc", 5, 2), "");
static_assert(4 == string_view("abcabc").find_first_not_of("abac", 3, 1), "");
static_assert(4 == string_view("abcabc").find_first_not_of("dadab", 4, 2), "");
} // namespace test_find_first_not_of_overload3

namespace test_find_first_not_of_overload4 {
static_assert(string_view::npos == string_view("").find_first_not_of(""), "");
static_assert(string_view::npos == string_view("").find_first_not_of("a"), "");
static_assert(
    string_view::npos == string_view("").find_first_not_of("abc"),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_not_of("abc"),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_not_of("acdb"),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_not_of("defabc"),
    "");

static_assert(0 == string_view("abcabc").find_first_not_of(""), "");
static_assert(0 == string_view("abcabc").find_first_not_of("bc"), "");
static_assert(1 == string_view("abcabc").find_first_not_of("ac"), "");
static_assert(2 == string_view("abcabc").find_first_not_of("ab"), "");
static_assert(1 == string_view("abcabc").find_first_not_of("a"), "");
static_assert(1 == string_view("abcabc").find_first_not_of("da"), "");

static_assert(
    string_view::npos == string_view("").find_first_not_of("", 1),
    "");
static_assert(
    string_view::npos == string_view("").find_first_not_of("a", 1),
    "");
static_assert(
    string_view::npos == string_view("").find_first_not_of("abc", 100),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_not_of("abc", 1),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_not_of("acdb", 3),
    "");
static_assert(
    string_view::npos == string_view("abc").find_first_not_of("defabc", 2),
    "");

static_assert(1 == string_view("abcabc").find_first_not_of("", 1), "");
static_assert(3 == string_view("abcabc").find_first_not_of("bc", 1), "");
static_assert(4 == string_view("abcabc").find_first_not_of("ac", 4), "");
static_assert(5 == string_view("abcabc").find_first_not_of("ab", 5), "");
static_assert(4 == string_view("abcabc").find_first_not_of("a", 3), "");
static_assert(4 == string_view("abcabc").find_first_not_of("da", 4), "");
} // namespace test_find_first_not_of_overload4

namespace test_find_last_not_of_overload1 {
static_assert(
    string_view::npos == string_view("").find_last_not_of(string_view("")),
    "");
static_assert(
    string_view::npos == string_view("").find_last_not_of(string_view("a")),
    "");
static_assert(
    string_view::npos == string_view("").find_last_not_of(string_view("abc")),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_last_not_of(string_view("abc")),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_last_not_of(string_view("acdb")),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_last_not_of(string_view("defabc")),
    "");

static_assert(5 == string_view("abcabc").find_last_not_of(string_view("")), "");
static_assert(
    3 == string_view("abcabc").find_last_not_of(string_view("bc")),
    "");
static_assert(
    4 == string_view("abcabc").find_last_not_of(string_view("ac")),
    "");
static_assert(
    5 == string_view("abcabc").find_last_not_of(string_view("ab")),
    "");
static_assert(
    4 == string_view("abcabc").find_last_not_of(string_view("c")),
    "");
static_assert(
    4 == string_view("abcabc").find_last_not_of(string_view("ca")),
    "");

static_assert(
    string_view::npos == string_view("").find_last_not_of(string_view(""), 1),
    "");
static_assert(
    string_view::npos == string_view("").find_last_not_of(string_view("a"), 0),
    "");
static_assert(
    string_view::npos ==
        string_view("").find_last_not_of(string_view("abc"), 100),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_last_not_of(string_view("abc"), 1),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_last_not_of(string_view("acdb"), 3),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_last_not_of(string_view("defabc"), 2),
    "");

static_assert(
    4 == string_view("abcabc").find_last_not_of(string_view(""), 4),
    "");
static_assert(
    0 == string_view("abcabc").find_last_not_of(string_view("bc"), 2),
    "");
static_assert(
    1 == string_view("abcabc").find_last_not_of(string_view("ac"), 2),
    "");
static_assert(
    2 == string_view("abcabc").find_last_not_of(string_view("ab"), 2),
    "");
static_assert(
    4 == string_view("abcabc").find_last_not_of(string_view("c"), 4),
    "");
static_assert(
    1 == string_view("abcabc").find_last_not_of(string_view("ca"), 2),
    "");
} // namespace test_find_last_not_of_overload1

namespace test_find_last_not_of_overload2 {
static_assert(string_view::npos == string_view("").find_last_not_of('a'), "");
static_assert(string_view::npos == string_view("a").find_last_not_of('a'), "");
static_assert(2 == string_view("abc").find_last_not_of('a'), "");
static_assert(1 == string_view("abc").find_last_not_of('c'), "");
static_assert(
    string_view::npos == string_view("a").find_last_not_of('a', 0),
    "");
static_assert(2 == string_view("abc").find_last_not_of('b'), "");
static_assert(
    string_view::npos == string_view("abc").find_last_not_of('a', 0),
    "");
static_assert(0 == string_view("abc").find_last_not_of('b', 1), "");
static_assert(0 == string_view("abc").find_last_not_of('c', 0), "");
static_assert(1 == string_view("abc").find_last_not_of('c', 1), "");
static_assert(1 == string_view("abc").find_last_not_of('c', 2), "");
static_assert(1 == string_view("abc").find_last_not_of('c', 3), "");
static_assert(2 == string_view("abc").find_last_not_of('a', 100), "");
static_assert(3 == string_view("ababa").find_last_not_of('a'), "");
static_assert(
    string_view::npos == string_view("ababa").find_last_not_of('a', 0),
    "");
static_assert(1 == string_view("ababa").find_last_not_of('a', 1), "");
static_assert(1 == string_view("ababa").find_last_not_of('a', 2), "");
static_assert(3 == string_view("ababa").find_last_not_of('a', 3), "");
static_assert(3 == string_view("ababa").find_last_not_of('a', 4), "");
static_assert(3 == string_view("ababa").find_last_not_of('a', 5), "");
} // namespace test_find_last_not_of_overload2

namespace test_find_last_not_of_overload3 {
static_assert(
    string_view::npos ==
        string_view("").find_last_not_of("ab", string_view::npos, 0),
    "");
static_assert(
    string_view::npos ==
        string_view("").find_last_not_of("abc", string_view::npos, 1),
    "");
static_assert(
    string_view::npos ==
        string_view("").find_last_not_of("abcdef", string_view::npos, 3),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_last_not_of("abcdef", string_view::npos, 3),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_last_not_of("acdbef", string_view::npos, 4),
    "");
static_assert(
    string_view::npos ==
        string_view("abc").find_last_not_of("defabcas", string_view::npos, 6),
    "");

static_assert(
    5 == string_view("abcabc").find_last_not_of("cab", string_view::npos, 0),
    "");
static_assert(
    3 == string_view("abcabc").find_last_not_of("bca", string_view::npos, 2),
    "");
static_assert(
    4 == string_view("abcabc").find_last_not_of("acb", string_view::npos, 2),
    "");
static_assert(
    5 == string_view("abcabc").find_last_not_of("abc", string_view::npos, 2),
    "");
static_assert(
    4 == string_view("abcabc").find_last_not_of("caba", string_view::npos, 1),
    "");
static_assert(
    4 == string_view("abcabc").find_last_not_of("cacab", string_view::npos, 2),
    "");

static_assert(
    string_view::npos == string_view("").find_last_not_of("ab", 1, 0),
    "");
static_assert(
    string_view::npos == string_view("").find_last_not_of("abc", 0, 1),
    "");
static_assert(
    string_view::npos == string_view("").find_last_not_of("abcdef", 100, 3),
    "");
static_assert(
    string_view::npos == string_view("abc").find_last_not_of("abcdef", 1, 3),
    "");
static_assert(
    string_view::npos == string_view("abc").find_last_not_of("acdbef", 3, 4),
    "");
static_assert(
    string_view::npos == string_view("abc").find_last_not_of("defabcas", 2, 6),
    "");

static_assert(4 == string_view("abcabc").find_last_not_of("bca", 4, 0), "");
static_assert(0 == string_view("abcabc").find_last_not_of("bca", 2, 2), "");
static_assert(1 == string_view("abcabc").find_last_not_of("acb", 2, 2), "");
static_assert(2 == string_view("abcabc").find_last_not_of("abc", 2, 2), "");
static_assert(4 == string_view("abcabc").find_last_not_of("caba", 4, 1), "");
static_assert(1 == string_view("abcabc").find_last_not_of("cacab", 2, 2), "");
} // namespace test_find_last_not_of_overload3

namespace test_find_last_not_of_overload4 {
static_assert(string_view::npos == string_view("").find_last_not_of(""), "");
static_assert(string_view::npos == string_view("").find_last_not_of("a"), "");
static_assert(string_view::npos == string_view("").find_last_not_of("abc"), "");
static_assert(
    string_view::npos == string_view("abc").find_last_not_of("abc"),
    "");
static_assert(
    string_view::npos == string_view("abc").find_last_not_of("acdb"),
    "");
static_assert(
    string_view::npos == string_view("abc").find_last_not_of("defabc"),
    "");

static_assert(5 == string_view("abcabc").find_last_not_of(""), "");
static_assert(3 == string_view("abcabc").find_last_not_of("bc"), "");
static_assert(4 == string_view("abcabc").find_last_not_of("ac"), "");
static_assert(5 == string_view("abcabc").find_last_not_of("ab"), "");
static_assert(4 == string_view("abcabc").find_last_not_of("c"), "");
static_assert(4 == string_view("abcabc").find_last_not_of("ca"), "");

static_assert(string_view::npos == string_view("").find_last_not_of("", 1), "");
static_assert(
    string_view::npos == string_view("").find_last_not_of("a", 0),
    "");
static_assert(
    string_view::npos == string_view("").find_last_not_of("abc", 100),
    "");
static_assert(
    string_view::npos == string_view("abc").find_last_not_of("abc", 1),
    "");
static_assert(
    string_view::npos == string_view("abc").find_last_not_of("acdb", 3),
    "");
static_assert(
    string_view::npos == string_view("abc").find_last_not_of("defabc", 2),
    "");

static_assert(4 == string_view("abcabc").find_last_not_of("", 4), "");
static_assert(0 == string_view("abcabc").find_last_not_of("bc", 2), "");
static_assert(1 == string_view("abcabc").find_last_not_of("ac", 2), "");
static_assert(2 == string_view("abcabc").find_last_not_of("ab", 2), "");
static_assert(4 == string_view("abcabc").find_last_not_of("c", 4), "");
static_assert(1 == string_view("abcabc").find_last_not_of("ca", 2), "");
} // namespace test_find_last_not_of_overload4

namespace test_output_operator {
void testOutputIterator(const std::string& str) {
  std::ostringstream stream;
  stream << string_view(str);
  std::string actual = stream.str();
  EXPECT_EQ(str, actual);
}

TEST(StringViewTest, testOutputOperator) {
  testOutputIterator("");
  testOutputIterator("abc");
}
} // namespace test_output_operator

namespace test_hash {
TEST(StringViewTest, testHash) {
  EXPECT_EQ(
      std::hash<string_view>()(string_view()), std::hash<string_view>()(""));
  EXPECT_EQ(
      std::hash<string_view>()(string_view("hello")),
      std::hash<string_view>()("hello"));
  EXPECT_NE(
      std::hash<string_view>()(string_view("hello")),
      std::hash<string_view>()(""));
}
} // namespace test_hash

} // namespace
// NOLINTEND(modernize*, readability*, bugprone-string-constructor)
