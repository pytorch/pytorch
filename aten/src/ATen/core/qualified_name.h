#pragma once

#include <c10/util/intrusive_ptr.h>
#include <string>

namespace c10 {

// Represents names of the form, e.g., self.a.b
struct QualifiedName;
using QualifiedNamePtr = c10::intrusive_ptr<QualifiedName>;
struct QualifiedName : c10::intrusive_ptr_target {
  QualifiedName(QualifiedNamePtr prefix, std::string name)
      : prefix_(std::move(prefix)), name_(std::move(name)) {
    const auto pos = name.find('.');
    AT_ASSERTM(
        pos == std::string::npos,
        "Invalid name for qualified name: '",
        name,
        "'");
  }

  QualifiedNamePtr prefix_;
  std::string name_;

  static QualifiedNamePtr create(QualifiedNamePtr prefix, std::string name) {
    return c10::make_intrusive<QualifiedName>(
        std::move(prefix), std::move(name));
  }
  static QualifiedNamePtr create(std::string name) {
    return c10::make_intrusive<QualifiedName>(
        QualifiedNamePtr(), std::move(name));
  }
};

} // namespace c10
