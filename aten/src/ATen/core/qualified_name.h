#pragma once

#include <c10/util/StringUtil.h>
#include <c10/util/intrusive_ptr.h>
#include <string>

namespace c10 {

// Represents a name of the form "foo.bar.baz"
struct QualifiedName {
  QualifiedName() = default;

  explicit QualifiedName(std::string name) : qualifiedName_(std::move(name)) {
    // Compute the base name based on the qualified name
    const auto pos = qualifiedName_.rfind('.');
    if (pos == std::string::npos) {
      // If there are no delimiters, the qualname and name are the same
      name_ = qualifiedName_;
    } else {
      // Otherwise take the name that trails the last '.'
      name_ = qualifiedName_.substr(pos + 1);
      prefix_ = qualifiedName_.substr(0, pos);
      AT_ASSERTM(
          !name_.empty(),
          "'.' can't be the last character in qualified name: ",
          qualifiedName_);
      AT_ASSERTM(
          !prefix_.empty() && prefix_[0] != '.',
          "'.' can't be the first character in qualified name: ",
          qualifiedName_);
    }
    checkInvariants();
  }

  explicit QualifiedName(const QualifiedName& prefix, std::string name)
      : qualifiedName_(str(prefix.qualifiedName(), '.', name)),
        prefix_(prefix.qualifiedName()),
        name_(std::move(name)) {
    checkInvariants();
  }

  const std::string& qualifiedName() const {
    return qualifiedName_;
  }

  const std::string& prefix() const {
    return prefix_;
  }

  const std::string& name() const {
    return name_;
  }

  bool operator==(const QualifiedName& other) const {
    return this->qualifiedName_ == other.qualifiedName_;
  }

  bool operator!=(const QualifiedName& other) const {
    return !(*this == other);
  }

 private:
  void checkInvariants() {
    if (prefix_.empty()) {
      AT_ASSERT(qualifiedName_ == name_);
    } else {
      AT_ASSERT(qualifiedName_ == str(prefix_, ".", name_));
    }

    // No separators allowed in the base name
    AT_ASSERT(name_.find(".") == std::string::npos);

    // No empty atomic names allowed (e.g. "foo..bar")
    AT_ASSERT(qualifiedName_.find("..") == std::string::npos);
  }
  std::string qualifiedName_;
  std::string prefix_;
  std::string name_;
};
} // namespace c10
