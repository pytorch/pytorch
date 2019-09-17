#include <ATen/core/Dimname.h>
#include <c10/util/Exception.h>
#include <cctype>
#include <ATen/core/EnableNamedTensor.h>

#ifdef BUILD_NAMEDTENSOR
namespace at {

std::ostream& operator<<(std::ostream& out, const Dimname& dimname) {
  if (dimname.type() == NameType::WILDCARD) {
    out << "None";
  } else {
    out << "'" << dimname.full_name().toUnqualString() << "'";
  }
  return out;
}

bool is_valid_identifier(const std::string& name) {
  if (name.length() == 0) {
    return false;
  }
  for (auto it = name.begin(); it != name.end(); ++it) {
    if (std::isalpha(*it) || *it == '_') {
      continue;
    }
    return false;
  }
  return true;
}

bool Dimname::can_refer_to(const Dimname& other) const {
  switch (type()) {
    case NameType::WILDCARD:
      return false;

    // "C" can be used to refer to "C" or "C.in".
    case NameType::NORMAL:
      return untagged_name() == other.untagged_name();

    default:
      return full_name() == other.full_name();
  }
}

static void check_valid_identifier(const std::string& name) {
  TORCH_CHECK(
      is_valid_identifier(name),
      "Invalid name: a valid identifier must contain alphabetical characters and/or underscore, got: '",
      name, "'.");
}

Dimname Dimname::fromSymbol(Symbol full_name) {
  TORCH_INTERNAL_ASSERT(full_name.is_dimname());
  if (full_name == kWildcard) {
    return Dimname::wildcard();
  }
  const std::string delimiter = ".";
  const std::string str(full_name.toUnqualString());
  auto it = str.find(delimiter);

  // Check for normal name
  if (it == std::string::npos) {
    check_valid_identifier(str);
    return Dimname(full_name);
  }

  // Check for tagged name
  auto second_dot = str.find(delimiter, it + 1);
  TORCH_CHECK(
      second_dot == std::string::npos,
      "Invalid name '", str, "': A tagged name can only contain one '.'");
  auto untagged_name = str.substr(0, it);
  auto tag = str.substr(it + 1);
  check_valid_identifier(untagged_name); 
  check_valid_identifier(tag);
  return Dimname(NameType::TAGGED, full_name, Symbol::dimname(untagged_name));
}

Dimname Dimname::wildcard() {
  static Dimname result(NameType::WILDCARD, kWildcard, kWildcard);
  return result;
}

optional<Dimname> unify(Dimname dimname, Dimname other) {
  if (other.type() == NameType::WILDCARD) {
    return dimname;
  }
  if (dimname.type() == NameType::WILDCARD) {
    return other;
  }
  if (dimname.full_name() == other.full_name()) {
    return dimname;
  }
  if (dimname.untagged_name() == other.untagged_name()) {
    return Dimname::fromSymbol(dimname.untagged_name());
  }
  return c10::nullopt;
}

bool match(Dimname dimname, Dimname other) {
  return unify(dimname, other).has_value();
}

} // namespace at
#endif
