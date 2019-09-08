#pragma once

#include <string>
#include <utility>

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

namespace std {
  template <>
  struct hash<::c10::OperatorName> {
    size_t operator()(const ::c10::OperatorName& x) const {
      return std::hash<std::string>()(x.name) ^ (~ std::hash<std::string>()(x.overload_name));
    }
  };
}
