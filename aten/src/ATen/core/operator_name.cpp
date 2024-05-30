#include <ATen/core/operator_name.h>

namespace c10 {

std::string toString(const OperatorName& opName) {
  std::ostringstream oss;
  oss << opName;
  return oss.str();
}

std::ostream& operator<<(std::ostream& os, const OperatorName& opName) {
  os << opName.name;
  if (!opName.overload_name.empty()) {
    os << "." << opName.overload_name;
  }
  return os;
}

} // namespace c10
