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

  void setNamespaceIfNotSet(const char* ns) {
    // TODO: slow!  Fix internal data structures so I don't have to paste the
    // names together
    std::ostringstream oss;
    if (name.find("::") == std::string::npos) {
      oss << ns << "::" << name;
      name = oss.str();
    } else {
      // TODO: This error message assumes that this is called only from
      // the op registration API (which is currently true
      TORCH_CHECK(false,
        "Attempted to def/impl operator ", name, " which is explicitly qualified with a namespace, but you were defining a TORCH_LIBRARY for ", ns, ".  This is not allowed; all TORCH_LIBRARY definitions must be unqualified.  Did you mean to use TORCH_LIBRARY_IMPL?");
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
