#include <ATen/core/op_registration/infer_schema.h>
#include <c10/util/irange.h>
#include <fmt/format.h>

namespace c10 {

namespace detail::infer_schema {
namespace {

std::vector<Argument> createArgumentVector(c10::ArrayRef<ArgumentDef> args) {
  std::vector<Argument> result;
  result.reserve(args.size());
  for (const auto i : c10::irange(args.size())) {
    // Arguments are named "_<index>"
    result.emplace_back(
        fmt::format("_{}", i),
        (*args[i].getFakeTypeFn)(),
        (*args[i].getTypeFn)());
  }
  return result;
}
} // namespace
// This is intentionally a separate function and in a .cpp file
// because then the template is smaller and that benefits binary size
FunctionSchema make_function_schema(
    std::string&& name,
    std::string&& overload_name,
    c10::ArrayRef<ArgumentDef> arguments,
    c10::ArrayRef<ArgumentDef> returns) {
  return FunctionSchema(
      std::move(name),
      std::move(overload_name),
      createArgumentVector(arguments),
      createArgumentVector(returns));
}

FunctionSchema make_function_schema(
    c10::ArrayRef<ArgumentDef> arguments,
    c10::ArrayRef<ArgumentDef> returns) {
  return make_function_schema("", "", arguments, returns);
}
} // namespace detail

std::optional<std::string> findSchemaDifferences(
    const FunctionSchema& lhs,
    const FunctionSchema& rhs) {
  if (lhs.arguments().size() != rhs.arguments().size()) {
    return fmt::format(
        "The number of arguments is different. {} vs {}.",
        lhs.arguments().size(),
        rhs.arguments().size());
  }
  if (lhs.returns().size() != rhs.returns().size()) {
    return fmt::format(
        "The number of returns is different. {} vs {}.",
        lhs.returns().size(),
        rhs.returns().size());
  }

  for (const auto i : c10::irange(lhs.arguments().size())) {
    const TypePtr& leftType = lhs.arguments()[i].type();
    const TypePtr& rightType = rhs.arguments()[i].type();
    // Type::operator== is virtual. Comparing pointers first is
    // cheaper, particularly when one of the types is a singleton like
    // NumberType or AnyType.
    if (leftType.get() != rightType.get() && *leftType != *rightType) {
      return fmt::format(
          "Type mismatch in argument {}: {} vs {}.",
          i + 1,
          lhs.arguments()[i].type()->str(),
          rhs.arguments()[i].type()->str());
    }
  }

  for (const auto i : c10::irange(lhs.returns().size())) {
    const TypePtr& leftType = lhs.returns()[i].type();
    const TypePtr& rightType = rhs.returns()[i].type();
    // See above about comparing pointers first.
    if (leftType.get() != rightType.get() && *leftType != *rightType) {
      return fmt::format(
          "Type mismatch in return {}: {} vs {}.",
          i + 1,
          lhs.returns()[i].type()->str(),
          rhs.returns()[i].type()->str());
    }
  }

  // no differences found
  return std::nullopt;
}

} // namespace c10
