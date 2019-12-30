#include "test/cpp/jit/test_base.h"
//#include <gtest.h>

namespace c10 {
//std::string serializeType(const Type &t);
TypePtr parseType(const std::string& pythonStr);
}

namespace torch {
namespace jit {
void testMobileTypeParser() {
  std::string empty_ps("");
  ASSERT_ANY_THROW(c10::parseType(empty_ps));

  std::string int_ps("int");
  auto int_tp = c10::parseType(int_ps);
  std::string int_tps = int_tp->python_str();
  ASSERT_EQ(int_ps, int_tps);

  std::string tuple_ps("Tuple[str, Optional[float], Dict[str, List[Tensor]], int]");
  auto tuple_tp = c10::parseType(tuple_ps);
  std::string tuple_tps = tuple_tp->python_str();
  ASSERT_EQ(tuple_ps, tuple_tps);

  std::string tuple_space_ps("Tuple[  str, Optional[float], Dict[str, List[Tensor ]]  , int]");
  auto tuple_space_tp = c10::parseType(tuple_space_ps);
  // tuple_space_tps should not have weird white spaces
  std::string tuple_space_tps = tuple_space_tp->python_str();
  ASSERT_EQ(tuple_ps, tuple_space_tps);

  std::string typo_token("List[tensor]");
  ASSERT_ANY_THROW(c10::parseType(typo_token));

  std::string mismatch1("List[Tensor");
  ASSERT_ANY_THROW(c10::parseType(mismatch1));

  std::string mismatch2("List[[Tensor]");
  ASSERT_ANY_THROW(c10::parseType(mismatch2));

  std::string mismatch3("Dict[Tensor]");
  ASSERT_ANY_THROW(c10::parseType(mismatch3));

  // arg count mismatch
  std::string mismatch4("List[int, str]");
  ASSERT_ANY_THROW(c10::parseType(mismatch4));

  std::string trailing_commm("Dict[str,]");
  ASSERT_ANY_THROW(c10::parseType(trailing_commm));

  std::string extra_stuff("int int");
  ASSERT_ANY_THROW(c10::parseType(extra_stuff));

  std::string non_id("(int)");
  ASSERT_ANY_THROW(c10::parseType(non_id));
}
} // namespace torch
} // namespace jit
