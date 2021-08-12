#include <gtest/gtest.h>

#include <ATen/core/jit_type.h>

namespace c10 {
// std::string serializeType(const Type &t);
TypePtr parseType(const std::string& pythonStr);
TypePtr parseCustomType(IValue custom_type);
} // namespace c10

namespace torch {
namespace jit {
TEST(MobileTypeParserTest, Empty) {
  std::string empty_ps("");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(empty_ps));
}

TEST(MobileTypeParserTest, RoundTripAnnotationStr) {
  std::string int_ps("int");
  auto int_tp = c10::parseType(int_ps);
  std::string int_tps = int_tp->annotation_str();
  ASSERT_EQ(int_ps, int_tps);
}

TEST(MobileTypeParserTest, NestedContainersAnnotationStr) {
  std::string tuple_ps(
      "Tuple[str, Optional[float], Dict[str, List[Tensor]], int]");
  auto tuple_tp = c10::parseType(tuple_ps);
  std::string tuple_tps = tuple_tp->annotation_str();
  ASSERT_EQ(tuple_ps, tuple_tps);
}

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

TEST(MobileTypeParserTest, CustomTypeNamedTuple) {
  std::vector<at::IValue> namedtuple_definition(
      {c10::ivalue::Tuple::create(std::vector<at::IValue>(
           {at::IValue("id"), at::IValue("List[int]")})),
       c10::ivalue::Tuple::create(std::vector<at::IValue>(
           {at::IValue("name"), at::IValue("List[int]")}))});
  std::vector<at::IValue> namedtuple_vector = std::vector<at::IValue>(
      {"NamedTuple",
       c10::ivalue::Tuple::create(std::move(namedtuple_definition))});
  std::vector<at::IValue> namedtuple_type_vector(
      {at::IValue("mynamedtuple"),
       c10::ivalue::Tuple::create(std::move(namedtuple_vector))});
  at::IValue named_tuple_dummy =
      c10::ivalue::Tuple::create(std::move(namedtuple_type_vector));

  auto named_tuple_type_ptr = c10::parseCustomType(named_tuple_dummy);
  ASSERT_EQ(named_tuple_type_ptr->annotation_str(), "mynamedtuple");
}

TEST(MobileTypeParserTest, TypoRaises) {
  std::string typo_token("List[tensor]");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(typo_token));
}

TEST(MobileTypeParserTest, MismatchBracketRaises) {
  std::string mismatch1("List[Tensor");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(mismatch1));
}

TEST(MobileTypeParserTest, MismatchBracketRaises2) {
  std::string mismatch2("List[[Tensor]");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(mismatch2));
}

TEST(MobileTypeParserTest, DictWithoutValueRaises) {
  std::string mismatch3("Dict[Tensor]");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(mismatch3));
}

TEST(MobileTypeParserTest, ListArgCountMismatchRaises) {
  // arg count mismatch
  std::string mismatch4("List[int, str]");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(mismatch4));
}

TEST(MobileTypeParserTest, DictArgCountMismatchRaises) {
  std::string trailing_commm("Dict[str,]");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(trailing_commm));
}

TEST(MobileTypeParserTest, ValidTypeWithExtraStuffRaises) {
  std::string extra_stuff("int int");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(extra_stuff));
}

TEST(MobileTypeParserTest, NonIdentifierRaises) {
  std::string non_id("(int)");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(non_id));
}

} // namespace jit
} // namespace torch
