#pragma once

#if !defined(C10_MOBILE) || defined(FEATURE_TORCH_MOBILE)
#include <ATen/core/interned_strings.h>
#endif

#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <ostream>

#ifdef BUILD_NAMEDTENSOR
namespace at {

enum class NameType: uint8_t { NORMAL, WILDCARD, TAGGED };

#if !defined(C10_MOBILE) || defined(FEATURE_TORCH_MOBILE)
typedef Symbol InternedString;
#else
typedef uint32_t InternedString;
#endif

struct CAFFE2_API Dimname {
  static Dimname fromSymbol(InternedString name);
  static Dimname wildcard();

  NameType type() const { return type_; }
  InternedString full_name() const { return full_name_; }
  InternedString untagged_name() const { return untagged_name_; }

  bool can_refer_to(const Dimname& other) const;

  bool is_normal() const { return type_ == NameType::NORMAL; }
  bool is_wildcard() const { return type_ == NameType::WILDCARD; }
  bool is_tagged() const { return type_ == NameType::TAGGED; }

 private:
  Dimname(InternedString name)
    : untagged_name_(name), full_name_(name), type_(NameType::NORMAL) {}
  Dimname(NameType type, InternedString full_name, InternedString untagged_name)
    : untagged_name_(untagged_name), full_name_(full_name), type_(type) {}

  // [Dimname Terminology]
  //
  // For "C.in":
  // - "C.in" is the "full name"
  // - "C" is the "untagged name"
  // - "in" is the "tag"
  InternedString untagged_name_;
  InternedString full_name_;
  NameType type_;
  // Will need more fields for other special name types.
};

using DimnameList = c10::ArrayRef<Dimname>;

bool CAFFE2_API is_valid_identifier(const std::string& name);

CAFFE2_API c10::optional<Dimname> unify(Dimname dimname, Dimname other);
CAFFE2_API bool match(Dimname dimname, Dimname other);

CAFFE2_API std::ostream& operator<<(std::ostream& out, const Dimname& dimname);

inline bool operator==(const Dimname& lhs, const Dimname& rhs) {
  return lhs.full_name() == rhs.full_name();
}

inline bool operator!=(const Dimname& lhs, const Dimname& rhs) {
  return !(lhs == rhs);
}

} // namespace at
#endif
