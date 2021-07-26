#include <ATen/core/function_schema.h>

namespace c10 {

TORCH_API void FunctionSchema::dump() const {
  std::cout << *this << "\n";
}

}
