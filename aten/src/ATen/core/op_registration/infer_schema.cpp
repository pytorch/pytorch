#include <ATen/core/op_registration/infer_schema.h>
#include <c10/util/irange.h>
#include <sstream>

namespace c10 {

namespace detail {
namespace infer_schema {
namespace {

std::string fastToString(size_t x) {
  if (C10_LIKELY(x < 10)) {
    std::string result;
    result.push_back('_');
    result.push_back('0' + x);
    return result;
  }
  return "_" + c10::guts::to_string(x);
}

std::vector<Argument> createArgumentVector(c10::ArrayRef<ArgumentDef> args) {
  std::vector<Argument> result;
  result.reserve(args.size());
  for (const auto i : c10::irange(args.size())) {
    // Arguments are named "_<index>"
    result.emplace_back(fastToString(i), (*args[i].getFakeTypeFn)(), (*args[i].getTypeFn)());
  }
  return result;
}
}
// This is intentionally a separate function and in a .cpp file
// because then the template is smaller and that benefits binary size
FunctionSchema make_function_schema(std::string&& name, std::string&& overload_name, c10::ArrayRef<ArgumentDef> arguments, c10::ArrayRef<ArgumentDef> returns) {
  return FunctionSchema(std::move(name), std::move(overload_name), createArgumentVector(arguments), createArgumentVector(returns));
}

FunctionSchema make_function_schema(c10::ArrayRef<ArgumentDef> arguments, c10::ArrayRef<ArgumentDef> returns) {
  return make_function_schema("", "", arguments, returns);
}
}
}

c10::optional<std::string> findSchemaDifferences(const FunctionSchema& lhs, const FunctionSchema& rhs) {
  if (lhs.arguments().size() != rhs.arguments().size()) {
    return "The number of arguments is different. " + guts::to_string(lhs.arguments().size()) +
             " vs " + guts::to_string(rhs.arguments().size()) + ".";
  }
  if (lhs.returns().size() != rhs.returns().size()) {
    return "The number of returns is different. " + guts::to_string(lhs.returns().size()) +
             " vs " + guts::to_string(rhs.returns().size());
  }

  for (const auto i : c10::irange(lhs.arguments().size())) {
    const TypePtr& leftType = lhs.arguments()[i].type();
    const TypePtr& rightType = rhs.arguments()[i].type();
    // Type::operator== is virtual. Comparing pointers first is
    // cheaper, particularly when one of the types is a singleton like
    // NumberType or AnyType.
    if (leftType.get() != rightType.get() && *leftType != *rightType) {
      return "Type mismatch in argument " + guts::to_string(i+1) + ": " + lhs.arguments()[i].type()->str() +
               " vs " + rhs.arguments()[i].type()->str();
    }
  }

  for (const auto i : c10::irange(lhs.returns().size())) {
    const TypePtr& leftType = lhs.returns()[i].type();
    const TypePtr& rightType = rhs.returns()[i].type();
    // See above about comparing pointers first.
    if (leftType.get() != rightType.get() && *leftType != *rightType) {
      return "Type mismatch in return " + guts::to_string(i+1) + ": " + lhs.returns()[i].type()->str() +
               " vs " + rhs.returns()[i].type()->str();
    }
  }

  // no differences found
  return c10::nullopt;
}

}
