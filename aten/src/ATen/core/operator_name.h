#pragma once

#include <string>

namespace c10 {

struct OperatorName final {
  std::string name;
  std::string overload_name;
};

inline bool operator==(const OperatorName& lhs, const OperatorName& rhs) {
  return lhs.name == rhs.name && lhs.overload_name == rhs.overload_name;
}

inline bool operator!=(const OperatorName& lhs, const OperatorName& rhs) {
  return !operator==(lhs, rhs);
}

}
