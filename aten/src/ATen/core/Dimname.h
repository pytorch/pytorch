#pragma once

#include <ATen/core/interned_strings.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <ostream>

namespace at {

enum class NameType: uint8_t { BASIC, WILDCARD };

struct CAFFE2_API Dimname {
  static Dimname fromSymbol(Symbol name);
  static Dimname wildcard();
  static bool isValidName(const std::string& name);

  NameType type() const { return type_; }
  Symbol symbol() const { return name_; }

  bool isBasic() const { return type_ == NameType::BASIC; }
  bool isWildcard() const { return type_ == NameType::WILDCARD; }

  bool matches(Dimname other) const;
  optional<Dimname> unify(Dimname other) const;

 private:
  Dimname(Symbol name)
    : name_(name), type_(NameType::BASIC) {}
  Dimname(Symbol name, NameType type)
    : name_(name), type_(type) {}

  Symbol name_;
  NameType type_;
};

using DimnameList = c10::ArrayRef<Dimname>;

CAFFE2_API std::ostream& operator<<(std::ostream& out, const Dimname& dimname);

inline bool operator==(const Dimname& lhs, const Dimname& rhs) {
  return lhs.symbol() == rhs.symbol();
}

inline bool operator!=(const Dimname& lhs, const Dimname& rhs) {
  return !(lhs == rhs);
}

} // namespace at
