#include <ATen/core/function_schema.h>

namespace c10 {

void FunctionSchema::dump() const {
  std::cout << *this << "\n";
}

}
