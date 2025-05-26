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
  std::wstring s;
  std::string narrow;
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

namespace test_try_to {
TEST(tryToTest, Int64T) {
  const std::vector<std::pair<const char*, int64_t>> valid_examples = {
      {"123", 123},
      {"+456", 456},
      {"-123", -123},
      {"0x123", 291},
      {"00123", 83},
      {"000", 0},
  };
  for (const auto& [str, num] : valid_examples) {
    EXPECT_EQ(c10::tryToNumber<int64_t>(str), num);
    EXPECT_EQ(c10::tryToNumber<int64_t>(std::string{str}), num);
  }

  const std::vector<const char*> invalid_examples = {
      "123abc",
      "123.45",
      "",
      "12345678901234567890", // overflow
  };
  for (const auto str : invalid_examples) {
    EXPECT_FALSE(c10::tryToNumber<int64_t>(str).has_value());
    EXPECT_FALSE(c10::tryToNumber<int64_t>(std::string{str}).has_value());
  }
  EXPECT_FALSE(c10::tryToNumber<int64_t>(nullptr).has_value());
}

TEST(tryToTest, Double) {
  const std::vector<std::pair<const char*, double>> valid_examples = {
      {"123.45", 123.45},
      {"-123.45", -123.45},
      {"123", 123.},
      {".5", 0.5},
      {"-.02", -0.02},
      {"5e-2", 5e-2},
      {"1e+3", 1e3},
      {"0x123.45", 291.26953125},
  };
  for (const auto& [str, num] : valid_examples) {
    EXPECT_EQ(c10::tryToNumber<double>(str), num);
    EXPECT_EQ(c10::tryToNumber<double>(std::string{str}), num);
  }

  const std::vector<const char*> invalid_examples = {
      "123abc",
      "",
      "1e309", // overflow
  };
  for (const auto str : invalid_examples) {
    EXPECT_FALSE(c10::tryToNumber<double>(str).has_value());
    EXPECT_FALSE(c10::tryToNumber<double>(std::string{str}).has_value());
  }
  EXPECT_FALSE(c10::tryToNumber<double>(nullptr).has_value());
}
} // namespace test_try_to
} // namespace
