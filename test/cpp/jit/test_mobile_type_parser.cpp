#include "test/cpp/jit/test_base.h"
#include <gtest/gtest.h>

namespace c10 {
//std::string serializeType(const Type &t);
TypePtr parseType(const std::string& pythonStr);
}

namespace torch {
namespace jit {
void testMobileTypeParser() {
  std::string int_ps("int");
  auto int_tp = c10::parseType(int_ps);
  std::string int_tps = int_tp->python_str();
  ASSERT_EQ(int_ps, int_tps);

  std::string tuple_ps("Tuple[str, Optional[float], Dict[str, List[Tensor]], int]");
  auto tuple_tp = c10::parseType(tuple_ps);
  std::string tuple_tps = tuple_tp->python_str();
  ASSERT_EQ(tuple_ps, tuple_tps);

  std::string typo_token("List[tensor]");
  EXPECT_ANY_THROW(c10::parseType(typo_token));

  std::string mismatch1("List[Tensor");
  EXPECT_ANY_THROW(c10::parseType(mismatch1));

  std::string mismatch2("List[[Tensor]");
  EXPECT_ANY_THROW(c10::parseType(mismatch2));

  std::string mismatch3("Dict[Tensor]");
  EXPECT_ANY_THROW(c10::parseType(mismatch2));
}
} // namespace torch
} // namespace jit
