#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <string>

namespace c10 {

// Represents a name of the form "foo.bar.baz"
struct QualifiedName {
  QualifiedName() {}

  // `name` can be a dotted string, like "foo.bar.baz", or just a bare name.
  explicit QualifiedName(std::string name) {
    AT_ASSERT(!name.empty());
    // split the string into its atoms.
    size_t startSearchFrom = 0;
    size_t pos = name.find('.', startSearchFrom);

    while (pos != std::string::npos) {
      auto atom = name.substr(startSearchFrom, pos - startSearchFrom);
      AT_ASSERTM(
          atom.size() > 0, "Invalid name for qualified name: '", name, "'");
      atoms_.push_back(std::move(atom));
      startSearchFrom = pos + 1;
      pos = name.find('.', startSearchFrom);
    }

    auto finalAtom = name.substr(startSearchFrom, pos - startSearchFrom);
    AT_ASSERTM(
        finalAtom.size() > 0, "Invalid name for qualified name: '", name, "'");
    atoms_.push_back(std::move(finalAtom));

    cacheAccessors();
  }

  // `name` must be a bare name (no dots!)
  explicit QualifiedName(const QualifiedName& prefix, std::string name) {
    AT_ASSERT(!name.empty());
    atoms_.insert(atoms_.begin(), prefix.atoms_.begin(), prefix.atoms_.end());
    atoms_.push_back(std::move(name));

    cacheAccessors();
  }

  // The fully qualified name, like "foo.bar.baz"
  const std::string& qualifiedName() const {
    return qualifiedName_;
  }

  // The leading qualifier, like "foo.bar"
  const std::string& prefix() const {
    return prefix_;
  }

  // The base name, like "baz"
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
  void cacheAccessors() {
    qualifiedName_ = Join(".", atoms_);
    if (atoms_.size() > 1) {
      ArrayRef<std::string> view(atoms_);
      const auto prefixView = view.slice(0, view.size() - 1);
      prefix_ = Join(".", prefixView);
    }

    if (atoms_.size() >= 1) {
      name_ = atoms_.back();
    }
    checkInvariants();
  }
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

    // Check individual atoms
    for (const auto& atom : atoms_) {
      AT_ASSERT(!atom.empty());
      AT_ASSERT(atom.find(".") == std::string::npos);
    }
  }

  // The actual list of names, like "{foo, bar, baz}"
  std::vector<std::string> atoms_;

  /*
   * Cached accessors, derived from `atoms_`.
   */
  std::string qualifiedName_;
  std::string prefix_;
  std::string name_;
};
} // namespace c10

namespace std {
template <>
struct hash<c10::QualifiedName> {
  size_t operator()(const c10::QualifiedName& n) const noexcept {
    return std::hash<std::string>()(n.qualifiedName());
  }
};
} // namespace std
