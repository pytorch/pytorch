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
    out << "'" << dimname.symbol().toUnqualString() << "'";
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

static void check_valid_identifier(const std::string& name) {
  TORCH_CHECK(
      is_valid_identifier(name),
      "Invalid name: a valid identifier must contain alphabetical characters and/or underscore, got: '",
      name, "'.");
}

Dimname Dimname::fromSymbol(Symbol name) {
  TORCH_INTERNAL_ASSERT(name.is_dimname());
  if (name == kWildcard) {
    return Dimname::wildcard();
  }
  check_valid_identifier(name.toUnqualString());
  return Dimname(name);
}

Dimname Dimname::wildcard() {
  static Dimname result(kWildcard, NameType::WILDCARD);
  return result;
}

optional<Dimname> unify(Dimname dimname, Dimname other) {
  if (other.type() == NameType::WILDCARD) {
    return dimname;
  }
  if (dimname.type() == NameType::WILDCARD) {
    return other;
  }
  if (dimname.symbol() == other.symbol()) {
    return dimname;
  }
  return c10::nullopt;
}

bool match(Dimname dimname, Dimname other) {
  return unify(dimname, other).has_value();
}

} // namespace at
#endif
