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
    const auto pos = name_.find('.');
    AT_ASSERTM(
        pos == std::string::npos,
        "Invalid name for qualified name: '",
        name_,
        "'");
  }

  const QualifiedNamePtr prefix_ = QualifiedNamePtr();
  const std::string name_;

  static QualifiedNamePtr create(QualifiedNamePtr prefix, std::string name) {
    return c10::make_intrusive<QualifiedName>(
        std::move(prefix), std::move(name));
  }
  static QualifiedNamePtr create(std::string name) {
    return c10::make_intrusive<QualifiedName>(
        QualifiedNamePtr(), std::move(name));
  }

  // Create a qualified name from a dotted string, splitting it as necessary.
  // Example input: "foo.bar.baz"
  static QualifiedNamePtr createFromDotted(const std::string& name) {
    size_t startSearchFrom = 0;
    size_t pos = name.find('.', startSearchFrom);

    auto qualifiedName = QualifiedNamePtr();
    while (pos != std::string::npos) {
      auto atom = name.substr(startSearchFrom, pos - startSearchFrom);
      AT_ASSERTM(
          atom.size() > 0, "Invalid name for qualified name: '", name, "'");
      qualifiedName = QualifiedName::create(qualifiedName, std::move(atom));
      startSearchFrom = pos + 1;
      pos = name.find('.', startSearchFrom);
    }

    auto finalAtom = name.substr(startSearchFrom, pos - startSearchFrom);
    AT_ASSERTM(
        finalAtom.size() > 0, "Invalid name for qualified name: '", name, "'");
    return QualifiedName::create(qualifiedName, std::move(finalAtom));
  }

  // Flatten this qualified name and convert the whole thing to a string, like
  // "foo.bar.baz".
  std::string toString() const {
    std::ostringstream ss;
    toString(ss);
    return ss.str();
  }

  // Compares `this` with `other` for equality, starting from the furthest
  // prefix.
  bool equals(const QualifiedNamePtr& other) const {
    if (!other) {
      return false;
    }

    if (name_ != other->name_) {
      return false;
    }

    if (!prefix_) {
      return !other->prefix_;
    }
    return prefix_->equals(other->prefix_);
  }

 private:
  std::ostream& toString(std::ostream& ss) const {
    if (!prefix_) {
      ss << name_;
      return ss;
    }
    prefix_->toString(ss) << "." << name_;
    return ss;
  }
};
} // namespace c10
