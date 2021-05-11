#include <gtest/gtest.h>

#include <ATen/core/jit_type.h>

namespace c10 {
// std::string serializeType(const Type &t);
TypePtr parseType(const std::string& pythonStr);
} // namespace c10

namespace torch {
namespace jit {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MobileTypeParserTest, Empty) {
  std::string empty_ps("");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(empty_ps));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MobileTypeParserTest, RoundTripAnnotationStr) {
  std::string int_ps("int");
  auto int_tp = c10::parseType(int_ps);
  std::string int_tps = int_tp->annotation_str();
  ASSERT_EQ(int_ps, int_tps);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MobileTypeParserTest, NestedContainersAnnotationStr) {
  std::string tuple_ps(
      "Tuple[str, Optional[float], Dict[str, List[Tensor]], int]");
  auto tuple_tp = c10::parseType(tuple_ps);
  std::string tuple_tps = tuple_tp->annotation_str();
  ASSERT_EQ(tuple_ps, tuple_tps);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MobileTypeParserTest, NestedContainersAnnotationStrWithSpaces) {
  std::string tuple_ps(
      "Tuple[str, Optional[float], Dict[str, List[Tensor]], int]");
  std::string tuple_space_ps(
      "Tuple[  str, Optional[float], Dict[str, List[Tensor ]]  , int]");
  auto tuple_space_tp = c10::parseType(tuple_space_ps);
  // tuple_space_tps should not have weird white spaces
  std::string tuple_space_tps = tuple_space_tp->annotation_str();
  ASSERT_EQ(tuple_ps, tuple_space_tps);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MobileTypeParserTest, TypoRaises) {
  std::string typo_token("List[tensor]");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(typo_token));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MobileTypeParserTest, MismatchBracketRaises) {
  std::string mismatch1("List[Tensor");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(mismatch1));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MobileTypeParserTest, MismatchBracketRaises2) {
  std::string mismatch2("List[[Tensor]");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(mismatch2));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MobileTypeParserTest, DictWithoutValueRaises) {
  std::string mismatch3("Dict[Tensor]");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(mismatch3));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MobileTypeParserTest, ListArgCountMismatchRaises) {
  // arg count mismatch
  std::string mismatch4("List[int, str]");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(mismatch4));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MobileTypeParserTest, DictArgCountMismatchRaises) {
  std::string trailing_commm("Dict[str,]");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(trailing_commm));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MobileTypeParserTest, ValidTypeWithExtraStuffRaises) {
  std::string extra_stuff("int int");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(extra_stuff));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MobileTypeParserTest, NonIdentifierRaises) {
  std::string non_id("(int)");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(non_id));
}
} // namespace jit
} // namespace torch
