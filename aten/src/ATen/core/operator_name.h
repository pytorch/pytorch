#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <string>
#include <utility>
#include <ostream>

namespace c10 {

// TODO: consider storing namespace separately too
struct OperatorName final {
  std::string name;
  std::string overload_name;
  OperatorName(std::string name, std::string overload_name)
      : name(std::move(name)), overload_name(std::move(overload_name)) {}

  // Returns true if we successfully set the namespace
  bool setNamespaceIfNotSet(const char* ns) {
    // TODO: slow!  Fix internal data structures so I don't have to paste the
    // names together
    std::ostringstream oss;
    if (name.find("::") == std::string::npos) {
      oss << ns << "::" << name;
      name = oss.str();
      return true;
    } else {
      return false;
    }
  }
};

inline bool operator==(const OperatorName& lhs, const OperatorName& rhs) {
  return lhs.name == rhs.name && lhs.overload_name == rhs.overload_name;
}

inline bool operator!=(const OperatorName& lhs, const OperatorName& rhs) {
  return !operator==(lhs, rhs);
}

CAFFE2_API std::string toString(const OperatorName& opName);
CAFFE2_API std::ostream& operator<<(std::ostream&, const OperatorName&);

} // namespace c10

namespace std {
  template <>
  struct hash<::c10::OperatorName> {
    size_t operator()(const ::c10::OperatorName& x) const {
      return std::hash<std::string>()(x.name) ^ (~ std::hash<std::string>()(x.overload_name));
    }
  };
}
