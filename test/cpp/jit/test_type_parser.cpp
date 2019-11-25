#include "test/cpp/jit/test_base.h"

namespace c10 {
//std::string serializeType(const Type &t);
TypePtr parseType(const std::string& pythonStr);
}

namespace torch {
namespace jit {
void testTypeParser() {
  std::string int_ps("int");
  auto int_tp = c10::parseType(int_ps);
  std::string int_tps = int_tp->python_str();
  ASSERT_EQ(int_ps, int_tps);

  std::string tuple_ps("Tuple[str, Optional[float], Dict[str, List[Tensor]], int]");
  auto tuple_tp = c10::parseType(tuple_ps);
  std::string tuple_tps = tuple_tp->python_str();
  ASSERT_EQ(tuple_ps, tuple_tps);
}
} // namespace torch
} // namespace jit
