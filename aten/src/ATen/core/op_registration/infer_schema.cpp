#include "infer_schema.h"
#include <sstream>

namespace c10 {

C10_EXPORT c10::optional<std::string> findSchemaDifferences(const FunctionSchema& lhs, const FunctionSchema& rhs) {
  if (lhs.arguments().size() != rhs.arguments().size()) {
    return "The number of arguments is different. " + guts::to_string(lhs.arguments().size()) +
             " vs " + guts::to_string(rhs.arguments().size()) + ".";
  }
  if (lhs.returns().size() != rhs.returns().size()) {
    return "The number of returns is different. " + guts::to_string(lhs.returns().size()) +
             " vs " + guts::to_string(rhs.returns().size());
  }

  for (size_t i = 0; i < lhs.arguments().size(); ++i) {
    if (*lhs.arguments()[i].type() != *rhs.arguments()[i].type()) {
      return "Type mismatch in argument " + guts::to_string(i+1) + ": " + lhs.arguments()[i].type()->str() +
               " vs " + rhs.arguments()[i].type()->str();
    }
  }

  for (size_t i = 0; i < lhs.returns().size(); ++i) {
    if (*lhs.returns()[i].type() != *rhs.returns()[i].type()) {
      return "Type mismatch in return " + guts::to_string(i+1) + ": " + lhs.returns()[i].type()->str() +
               " vs " + rhs.returns()[i].type()->str();
    }
  }

  // no differences found
  return c10::nullopt;
}

}
