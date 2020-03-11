#pragma once

#include <string>
#include <utility>

namespace c10 {

struct OperatorName final {
  std::string name;
  std::string overload_name;
  OperatorName(std::string name, std::string overload_name)
      : name(std::move(name)), overload_name(std::move(overload_name)) {}
};

inline bool operator==(const OperatorName& lhs, const OperatorName& rhs) {
  return lhs.name == rhs.name && lhs.overload_name == rhs.overload_name;
}

inline bool operator!=(const OperatorName& lhs, const OperatorName& rhs) {
  return !operator==(lhs, rhs);
}

inline std::string toString(const OperatorName& opName) {
  std::string result = opName.name;
  if (opName.overload_name.size() != 0) {
    result += "." + opName.overload_name;
  }
  return result;
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
