#include <c10/util/StringUtil.h>

#include <gtest/gtest.h>

namespace {

namespace test_str_narrow_single {
TEST(StringUtilTest, testStrNarrowSingle) {
  std::string s = "narrow test string";
  EXPECT_EQ(s, c10::str(s));

  const char* c_str = s.c_str();
  EXPECT_EQ(s, c10::str(c_str));

  char c = 'a';
  EXPECT_EQ(std::string(1, c), c10::str(c));
}
} // namespace test_str_narrow_single

namespace test_str_wide_single {
TEST(StringUtilTest, testStrWideSingle) {
  std::wstring s = L"wide test string";
  std::string narrow = "wide test string";
  EXPECT_EQ(narrow, c10::str(s));

  const wchar_t* c_str = s.c_str();
  EXPECT_EQ(narrow, c10::str(c_str));

  wchar_t c = L'a';
  std::string narrowC = "a";
  EXPECT_EQ(narrowC, c10::str(c));
}
} // namespace test_str_wide_single

namespace test_str_wide_single_multibyte {
TEST(StringUtilTest, testStrWideSingleMultibyte) {
  std::wstring s = L"\u00EC blah";
  std::string narrow = "\xC3\xAC blah";
  EXPECT_EQ(narrow, c10::str(s));

  const wchar_t* c_str = s.c_str();
  EXPECT_EQ(narrow, c10::str(c_str));

  wchar_t c = L'\u00EC';
  std::string narrowC = "\xC3\xAC";
  EXPECT_EQ(narrowC, c10::str(c));
}
} // namespace test_str_wide_single_multibyte

namespace test_str_wide_empty {
TEST(StringUtilTest, testStrWideEmpty) {
  std::wstring s = L"";
  std::string narrow = "";
  EXPECT_EQ(narrow, c10::str(s));

  const wchar_t* c_str = s.c_str();
  EXPECT_EQ(narrow, c10::str(c_str));

  wchar_t c = L'\0';
  std::string narrowC(1, '\0');
  EXPECT_EQ(narrowC, c10::str(c));
}
} // namespace test_str_wide_empty

namespace test_str_multi {
TEST(StringUtilTest, testStrMulti) {
  std::string result = c10::str(
      "c_str ",
      'c',
      std::string(" std::string "),
      42,
      L" wide c_str ",
      L'w',
      std::wstring(L" std::wstring "));
  std::string expected = "c_str c std::string 42 wide c_str w std::wstring ";
  EXPECT_EQ(expected, result);
}
} // namespace test_str_multi

} // namespace
