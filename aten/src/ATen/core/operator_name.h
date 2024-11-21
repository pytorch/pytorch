#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/string_view.h>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

namespace c10 {

// TODO: consider storing namespace separately too
struct OperatorName final {
  std::string name;
  std::string overload_name;
  OperatorName(std::string name, std::string overload_name)
      : name(std::move(name)), overload_name(std::move(overload_name)) {}

  // TODO: These two functions below are slow!  Fix internal data structures so
  // I don't have to manually reconstruct the namespaces!

  // Return the namespace of this OperatorName, if it exists.  The
  // returned string_view is only live as long as the OperatorName
  // exists and name is not mutated
  std::optional<std::string_view> getNamespace() const {
    auto pos = name.find("::");
    if (pos == std::string::npos) {
      return std::nullopt;
    } else {
      return std::make_optional(std::string_view(name.data(), pos));
    }
  }

  // Returns true if we successfully set the namespace
  bool setNamespaceIfNotSet(const char* ns) {
    if (!getNamespace().has_value()) {
      const auto ns_len = strlen(ns);
      const auto old_name_size = name.size();
      name.resize(ns_len + 2 + old_name_size);
      // Shift current value of name to the end of the new space.
      name.replace(
          name.size() - old_name_size, old_name_size, name, 0, old_name_size);
      name.replace(0, ns_len, ns, ns_len);
      name[ns_len] = ':';
      name[ns_len + 1] = ':';
      return true;
    } else {
      return false;
    }
  }
};

// Non-owning view of an OperatorName.  Unlike OperatorName, most of
// its functions are constexpr, so it can be used for compile time
// computations
struct OperatorNameView final {
  std::string_view name;
  std::string_view overload_name;
  constexpr OperatorNameView(
      std::string_view name,
      std::string_view overload_name)
      : name(name), overload_name(overload_name) {}
  // Parses strings like "foo.overload" and also "foo"
  constexpr static OperatorNameView parse(std::string_view full_name) {
    auto i = full_name.find('.');
    if (i == std::string_view::npos) {
      return OperatorNameView(full_name, std::string_view());
    } else {
      return OperatorNameView(full_name.substr(0, i), full_name.substr(i + 1));
    }
  }
};

inline bool operator==(const OperatorName& lhs, const OperatorName& rhs) {
  return lhs.name == rhs.name && lhs.overload_name == rhs.overload_name;
}

inline bool operator!=(const OperatorName& lhs, const OperatorName& rhs) {
  return !operator==(lhs, rhs);
}

TORCH_API std::string toString(const OperatorName& opName);
TORCH_API std::ostream& operator<<(std::ostream&, const OperatorName&);

} // namespace c10

namespace std {
template <>
struct hash<::c10::OperatorName> {
  size_t operator()(const ::c10::OperatorName& x) const {
    return std::hash<std::string>()(x.name) ^
        (~std::hash<std::string>()(x.overload_name));
  }
};
} // namespace std
