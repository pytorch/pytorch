#include <ATen/core/operator_name.h>

namespace c10 {

std::string toString(const OperatorName& opName) {
  std::string result = opName.name;
  if (opName.overload_name.size() != 0) {
    result += "." + opName.overload_name;
  }
  return result;
}

std::ostream& operator<<(std::ostream& os, const OperatorName& opName) {
  return os << toString(opName);
}

}
