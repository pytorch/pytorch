#include <gtest/gtest.h>

#include <ATen/core/jit_type.h>

namespace c10 {
TypePtr parseType(const std::string& pythonStr);
} // namespace c10

namespace torch {
namespace jit {

// Parse Success cases
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

TEST(MobileTypeParserTest, NamedTuple) {
  std::string named_tuple_ps(
      "__torch__.dper3_models.ads_ranking.base_models.sparse_nn.pytorch_preproc_types.PreprocOutputType["
      "    NamedTuple, ["
      "        [float_features, Tensor],"
      "        [id_list_features, List[Tensor]],"
      "        [label,  Tensor],"
      "        [weight, Tensor],"
      "        [prod_prediction, Tuple[Tensor, Tensor]],"
      "        [id_score_list_features, List[Tensor]],"
      "        [embedding_features, List[Tensor]],"
      "        [teacher_label, Tensor]"
      "        ]"
      "    ]");

  c10::TypePtr named_tuple_tp = c10::parseType(named_tuple_ps);
  std::string named_tuple_annotation_str = named_tuple_tp->annotation_str();
  ASSERT_EQ(
      named_tuple_annotation_str,
      "__torch__.dper3_models.ads_ranking.base_models.sparse_nn.pytorch_preproc_types.PreprocOutputType");
}

TEST(MobileTypeParserTest, DictNestedNamedTuple) {
  std::string named_tuple_ps(
      "Dict[str, "
      "  __torch__.dper3_models.ads_ranking.base_models.sparse_nn.pytorch_preproc_types.PreprocOutputType["
      "    NamedTuple, ["
      "        [float_features, Tensor],"
      "        [id_list_features, List[Tensor]],"
      "        [label,  Tensor],"
      "        [weight, Tensor],"
      "        [prod_prediction, Tuple[Tensor, Tensor]],"
      "        [id_score_list_features, List[Tensor]],"
      "        [embedding_features, List[Tensor]],"
      "        [teacher_label, Tensor]"
      "        ]"
      "    ]");

  c10::TypePtr named_tuple_tp = c10::parseType(named_tuple_ps);
  std::string named_tuple_annotation_str = named_tuple_tp->annotation_str();
  ASSERT_EQ(
      named_tuple_annotation_str,
      "Dict[str, __torch__.dper3_models.ads_ranking.base_models.sparse_nn.pytorch_preproc_types.PreprocOutputType]");
}

TEST(MobileTypeParserTest, NamedTupleNestedNamedTuple) {
  std::string named_tuple_ps(
      "__torch__.aaa.xxx["
      "    NamedTuple, ["
      "        [field_name_a, __torch__.bbb.xxx ["
      "            NamedTuple, ["
      "                [field_name_b, __torch__.ccc.xxx ["
      "                    NamedTuple, ["
      "                      [field_name_c_1, Tensor],"
      "                      [field_name_c_2, Tuple[Tensor, Tensor]]"
      "                    ]"
      "                ]"
      "                ]"
      "            ]"
      "        ]"
      "        ]"
      "    ]   "
      "]");

  c10::TypePtr named_tuple_tp = c10::parseType(named_tuple_ps);
  std::string named_tuple_annotation_str = named_tuple_tp->str();
  ASSERT_EQ(named_tuple_annotation_str, "__torch__.aaa.xxx");
}

// Parse throw cases
TEST(MobileTypeParserTest, Empty) {
  std::string empty_ps("");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(c10::parseType(empty_ps));
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
