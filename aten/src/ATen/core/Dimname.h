#pragma once
#include <ATen/core/EnableNamedTensor.h>

#ifdef BUILD_NAMEDTENSOR
#include <ATen/core/interned_strings.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <ostream>

namespace at {

enum class NameType: uint8_t { NORMAL, WILDCARD };

struct CAFFE2_API Dimname {
  static Dimname fromSymbol(Symbol name);
  static Dimname wildcard();

  NameType type() const { return type_; }
  Symbol symbol() const { return name_; }

  bool is_normal() const { return type_ == NameType::NORMAL; }
  bool is_wildcard() const { return type_ == NameType::WILDCARD; }

 private:
  Dimname(Symbol name)
    : name_(name), type_(NameType::NORMAL) {}
  Dimname(Symbol name, NameType type)
    : name_(name), type_(type) {}

  Symbol name_;
  NameType type_;
};

using DimnameList = c10::ArrayRef<Dimname>;

static Symbol kWildcard = Symbol::dimname("*");
bool CAFFE2_API is_valid_identifier(const std::string& name);

CAFFE2_API c10::optional<Dimname> unify(Dimname dimname, Dimname other);
CAFFE2_API bool match(Dimname dimname, Dimname other);

CAFFE2_API std::ostream& operator<<(std::ostream& out, const Dimname& dimname);

inline bool operator==(const Dimname& lhs, const Dimname& rhs) {
  return lhs.symbol() == rhs.symbol();
}

inline bool operator!=(const Dimname& lhs, const Dimname& rhs) {
  return !(lhs == rhs);
}

} // namespace at
#endif
