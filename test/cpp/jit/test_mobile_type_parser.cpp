#include <gtest/gtest.h>
#include <test/cpp/jit/test_utils.h>

#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/mobile/type_parser.h>

namespace torch {
namespace jit {

// Parse Success cases
TEST(MobileTypeParserTest, Int) {
  std::string int_ps("int");
  auto int_tp = c10::parseType(int_ps);
  EXPECT_EQ(*int_tp, *IntType::get());
}

TEST(MobileTypeParserTest, NestedContainersAnnotationStr) {
  std::string tuple_ps(
      "Tuple[str, Optional[float], Dict[str, List[Tensor]], int]");
  auto tuple_tp = c10::parseType(tuple_ps);
  std::vector<TypePtr> args = {
      c10::StringType::get(),
      c10::OptionalType::create(c10::FloatType::get()),
      c10::DictType::create(
          StringType::get(), ListType::create(TensorType::get())),
      IntType::get()};
  auto tp = TupleType::create(std::move(args));
  ASSERT_EQ(*tuple_tp, *tp);
}

TEST(MobileTypeParserTest, TorchBindClass) {
  std::string tuple_ps("__torch__.torch.classes.rnn.CellParamsBase");
  auto tuple_tp = c10::parseType(tuple_ps);
  std::string tuple_tps = tuple_tp->annotation_str();
  ASSERT_EQ(tuple_ps, tuple_tps);
}

TEST(MobileTypeParserTest, ListOfTorchBindClass) {
  std::string tuple_ps("List[__torch__.torch.classes.rnn.CellParamsBase]");
  auto tuple_tp = c10::parseType(tuple_ps);
  EXPECT_TRUE(tuple_tp->isSubtypeOf(AnyListType::get()));
  EXPECT_EQ(
      "__torch__.torch.classes.rnn.CellParamsBase",
      tuple_tp->containedType(0)->annotation_str());
}

TEST(MobileTypeParserTest, NestedContainersAnnotationStrWithSpaces) {
  std::string tuple_space_ps(
      "Tuple[  str, Optional[float], Dict[str, List[Tensor ]]  , int]");
  auto tuple_space_tp = c10::parseType(tuple_space_ps);
  // tuple_space_tps should not have weird white spaces
  std::string tuple_space_tps = tuple_space_tp->annotation_str();
  ASSERT_TRUE(tuple_space_tps.find("[ ") == std::string::npos);
  ASSERT_TRUE(tuple_space_tps.find(" ]") == std::string::npos);
  ASSERT_TRUE(tuple_space_tps.find(" ,") == std::string::npos);
}

TEST(MobileTypeParserTest, NamedTuple) {
  std::string named_tuple_ps(
      "__torch__.base_models.preproc_types.PreprocOutputType["
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
      "__torch__.base_models.preproc_types.PreprocOutputType");
}

TEST(MobileTypeParserTest, DictNestedNamedTupleTypeList) {
  std::string type_str_1(
      "__torch__.base_models.preproc_types.PreprocOutputType["
      "  NamedTuple, ["
      "      [float_features, Tensor],"
      "      [id_list_features, List[Tensor]],"
      "      [label,  Tensor],"
      "      [weight, Tensor],"
      "      [prod_prediction, Tuple[Tensor, Tensor]],"
      "      [id_score_list_features, List[Tensor]],"
      "      [embedding_features, List[Tensor]],"
      "      [teacher_label, Tensor]"
      "      ]");
  std::string type_str_2(
      "Dict[str, __torch__.base_models.preproc_types.PreprocOutputType]");
  std::vector<std::string> type_strs = {type_str_1, type_str_2};
  std::vector<c10::TypePtr> named_tuple_tps = c10::parseType(type_strs);
  EXPECT_EQ(*named_tuple_tps[1]->containedType(0), *c10::StringType::get());
  EXPECT_EQ(*named_tuple_tps[0], *named_tuple_tps[1]->containedType(1));
}

TEST(MobileTypeParserTest, NamedTupleNestedNamedTupleTypeList) {
  std::string type_str_1(
      " __torch__.ccc.xxx ["
      "    NamedTuple, ["
      "      [field_name_c_1, Tensor],"
      "      [field_name_c_2, Tuple[Tensor, Tensor]]"
      "    ]"
      "]");
  std::string type_str_2(
      "__torch__.bbb.xxx ["
      "    NamedTuple,["
      "        [field_name_b, __torch__.ccc.xxx]]"
      "    ]"
      "]");

  std::string type_str_3(
      "__torch__.aaa.xxx["
      "    NamedTuple, ["
      "        [field_name_a, __torch__.bbb.xxx]"
      "    ]"
      "]");

  std::vector<std::string> type_strs = {type_str_1, type_str_2, type_str_3};
  std::vector<c10::TypePtr> named_tuple_tps = c10::parseType(type_strs);
  std::string named_tuple_annotation_str = named_tuple_tps[2]->annotation_str();
  ASSERT_EQ(named_tuple_annotation_str, "__torch__.aaa.xxx");
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

TEST(MobileTypeParserTest, DictNestedNamedTupleTypeListRaises) {
  std::string type_str_1(
      "Dict[str, __torch__.base_models.preproc_types.PreprocOutputType]");
  std::string type_str_2(
      "__torch__.base_models.preproc_types.PreprocOutputType["
      "  NamedTuple, ["
      "      [float_features, Tensor],"
      "      [id_list_features, List[Tensor]],"
      "      [label,  Tensor],"
      "      [weight, Tensor],"
      "      [prod_prediction, Tuple[Tensor, Tensor]],"
      "      [id_score_list_features, List[Tensor]],"
      "      [embedding_features, List[Tensor]],"
      "      [teacher_label, Tensor]"
      "      ]");
  std::vector<std::string> type_strs = {type_str_1, type_str_2};
  std::string error_message =
      R"(Can't find definition for the type: __torch__.base_models.preproc_types.PreprocOutputType)";
  ASSERT_THROWS_WITH_MESSAGE(c10::parseType(type_strs), error_message);
}

} // namespace jit
} // namespace torch
