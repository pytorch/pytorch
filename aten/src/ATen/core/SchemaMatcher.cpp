#include <ATen/core/SchemaMatcher.h>

#include <ATen/core/jit_type.h>
#include <fmt/format.h>

namespace c10 {
namespace {
c10::optional<size_t> findInputWithName(
    const std::string& name,
    const FunctionSchema& schema) {
  for (size_t i = 0; i < schema.arguments().size(); i++) {
    if (schema.arguments()[i].name() == name) {
      return i;
    }
  }
  return c10::nullopt;
}
} // namespace

SchemaMatcher::SchemaMatcher(
    const FunctionSchema& schema,
    const std::vector<TypePtr>& args,
    const std::unordered_map<std::string, TypePtr>& kwargs)
    : schema_(schema), args_(args), kwargs_(kwargs) {
  argToInputs_.resize(args_.size());
  doMatch();
}

bool SchemaMatcher::isMatch() const {
  return isMatch_;
}

const std::vector<TypePtr>& SchemaMatcher::outputs() const {
  TORCH_INTERNAL_ASSERT(isMatch_);
  return outputs_;
}

std::vector<size_t> SchemaMatcher::argToInputs() const {
  TORCH_INTERNAL_ASSERT(isMatch_);
  return argToInputs_;
}

std::unordered_map<std::string, size_t> SchemaMatcher::kwargToInputs() const {
  TORCH_INTERNAL_ASSERT(isMatch_);
  return kwargToInputs_;
}

// This logic is adapted from PEP 3102, see:
//   https://www.python.org/dev/peps/pep-3102/#function-calling-behavior
void SchemaMatcher::doMatch() {
  // Input arguments are matched against formal (schema) parameters as follows:
  // * For each formal parameter, there is a slot used to contain the argument
  //   type.
  // * Slots that are nullptr are "empty". Slots with any other TypePtr are
  //   "filled".
  // * All slots start as "empty".
  std::vector<TypePtr> parameterSlots{schema_.arguments().size()};

  // First, place all positional arguments in slots.
  for (size_t i = 0; i < args_.size(); i++) {
    if (i >= schema_.arguments().size()) {
      err_ << fmt::format(
          "{}() takes {} positional argument(s) but {} were given\n",
          schema_.name(),
          i,
          args_.size());
      isMatch_ = false;
      return;
    }

    const auto& formal = schema_.arguments()[i];
    if (formal.kwarg_only()) {
      err_ << fmt::format(
          "{}() takes {} positional argument(s) but {} were given\n",
          schema_.name(),
          i,
          args_.size());
      isMatch_ = false;
      return;
    }

    // NOTE: this is a hacky divergence from the PEP handling of varargs,
    // since our schema language does not have a way to mark a parameter
    // vararg.
    //
    // The strategy is to look ahead and check whether all remaining
    // positional arguments can converted into a list and used to satisfy this
    // formal.
    if (canConvertToVararg(schema_, i)) {
      // The above function also does type checking, so we can fill in the
      // paramter slot with a trivially matching type.
      parameterSlots[i] = formal.type();
      // Map the rest of the arguments to this slot
      for (size_t argIdx = i; argIdx < args_.size(); argIdx++) {
        argToInputs_[argIdx] = i;
      }

      // Then exit this loop, since we've "consumed" the rest of the
      // positional arguments.
      break;
    }

    // Common case: the positional argument fills the corresponding slot.
    parameterSlots[i] = args_[i];
    argToInputs_[i] = i;
  }

  // Then, place all kwargs in slots.
  for (const auto& pr : kwargs_) {
    const auto& argName = pr.first;
    const auto& argType = pr.second;
    // Find corresponding parameter slot
    auto slot = findInputWithName(argName, schema_);
    if (!slot) {
      err_ << fmt::format(
          "{}() got an unexpected keyword argument '{}'\n",
          schema_.name(),
          argName);
      isMatch_ = false;
      return;
    }

    if (parameterSlots[*slot] != nullptr) {
      err_ << fmt::format(
          "{}() got multiple values for argument '{}'\n",
          schema_.name(),
          argName);
      isMatch_ = false;
      return;
    }

    // Fill the appropriate
    parameterSlots[*slot] = argType;
    kwargToInputs_[argName] = *slot;
  }

  // For any unfilled slots, if the schema specifies a default value, then
  // we can consider them filled.
  for (size_t i = 0; i < parameterSlots.size(); i++) {
    if (parameterSlots[i] != nullptr) {
      continue;
    }

    const auto& formal = schema_.arguments()[i];
    if (formal.default_value()) {
      parameterSlots[i] = formal.type();
    }
  }

  // At this point, if there are any unfilled slots then the matching is a
  // failure.
  for (size_t i = 0; i < parameterSlots.size(); i++) {
    const auto& slot = parameterSlots[i];
    std::vector<std::string> missingNames;
    if (slot == nullptr) {
      missingNames.push_back(
          fmt::format("'{}'", schema_.arguments().at(i).name()));
    }
    if (!missingNames.empty()) {
      err_ << fmt::format(
          "{}() missing {} required argument(s): {}",
          schema_.name(),
          missingNames.size(),
          fmt::join(missingNames, ","));
      isMatch_ = false;
      return;
    }
  }

  // Now, perform type checking. For each slot, we should check that the type
  // can
  for (size_t i = 0; i < schema_.arguments().size(); i++) {
    const auto& formalArg = schema_.arguments()[i];
    const auto& actualType = parameterSlots[i];
    if (!isMatchingArgument(formalArg, actualType)) {
      // err_ populated by `isMatchingArgument`
      // TODO this flow is not that clear
      isMatch_ = false;
      return;
    }
  }

  isMatch_ = true;
  inputs_ = fmap(schema_.arguments(), [&](const Argument& r) {
    TypePtr result = tryEvalTypeVariables(r.type(), typeEnv_);
    TORCH_INTERNAL_ASSERT(
        result, r.type()->repr_str(), " has unbound type variables.");
    return result;
  });
  outputs_ = fmap(schema_.returns(), [&](const Argument& r) {
    TypePtr result = tryEvalTypeVariables(r.type(), typeEnv_);
    TORCH_INTERNAL_ASSERT(
        result, r.type()->repr_str(), " has unbound type variables.");
    return result;
  });
}

// Check if it is possible to convert all the remaining non-kwarg
// arguments to a list. This allows zeros(IntArrayRef sizes) to work with
// zeros(1, 2) or zeros(1)
bool SchemaMatcher::canConvertToVararg(
    const FunctionSchema& schema,
    size_t arg_index) {
  const auto& formal = schema.arguments().at(arg_index);
  // The arg must be the last one in the arg list that is not a kwarg
  bool is_last_positional_formal = arg_index + 1 == schema.arguments().size() ||
      schema.arguments()[arg_index + 1].kwarg_only();
  if (!is_last_positional_formal) {
    return false;
  }

  // The formal must be a list
  bool argument_is_list = formal.type()->kind() == TypeKind::ListType;
  if (!argument_is_list) {
    return false;
  }

  // matching varargs of typevar list nyi
  bool typevar_list = argument_is_list &&
      formal.type()->cast<ListType>()->getElementType()->cast<VarType>();
  if (typevar_list) {
    return false;
  }

  // it must not be a broadcasting list like int[3],
  // otherwise a single int is a valid input
  bool arg_is_broadcasting_list = bool(formal.N());
  if (arg_is_broadcasting_list) {
    return false;
  }

  // The rest of the arguments in provided must match this type.
  Argument elem_arg(
      "<varargs>", formal.type()->expect<ListType>()->getElementType());
  bool rest_of_args_match = std::all_of(
      args_.begin() + arg_index, args_.end(), [&](const TypePtr& t) {
        return isMatchingArgument(elem_arg, t);
      });

  if (!rest_of_args_match) {
    return false;
  }

  return true;
}

std::string SchemaMatcher::err() const {
  TORCH_INTERNAL_ASSERT(!isMatch_);
  return err_.str();
}

const std::vector<TypePtr>& SchemaMatcher::inputs() const {
  TORCH_INTERNAL_ASSERT(isMatch_);
  return inputs_;
}

// Check that a value of `actualType` is allowed to be used as the argument
// `arg`. Performs implicit conversions.
bool SchemaMatcher::isMatchingArgument(
    const Argument& arg,
    const TypePtr& actualType) {
  TypePtr schemaType = arg.type();

  // Handle the type variable resolution, i.e. the case where we have a
  // generic List[T].
  if (schemaType->hasFreeVariables()) {
    const MatchTypeReturn matched =
        matchTypeVariables(arg.type(), actualType, typeEnv_);
    if (!matched.success()) {
      err_ << fmt::format(
          "Could not match type '{}' to '{}' in argument '{}': {}.\n",
          actualType->repr_str(),
          arg.type()->repr_str(),
          arg.name(),
          matched.reason());
      return false;
    }
    schemaType = tryEvalTypeVariables(arg.type(), typeEnv_);
    if (!schemaType) {
      err_ << fmt::format(
          "Type variables in type '{}' could not be inferred from actual type '{}'\n",
          arg.type()->repr_str(),
          actualType->repr_str());
      return false;
    }
  }

  // Easy case: do the types trivially match?
  if (actualType->isSubtypeOf(schemaType)) {
    return true;
  }

  // If not, we have to apply the implicit conversion rules:
  // * floats/ints can be broadcast to fixed-size lists (e.g. `int` can be used
  // for `int[3]`)
  if (*actualType == *IntType::get() || *actualType == *FloatType::get()) {
    if (auto listType = schemaType->cast<ListType>()) {
      if (listType->getElementType() == actualType && arg.N()) {
        return true;
      }
    }
  }

  // * Homogenous tuples can be converted to lists containing the same type.
  if (actualType->cast<TupleType>() && schemaType->cast<ListType>()) {
    auto schemaContainedType = schemaType->expect<ListType>()->getElementType();
    auto tuple = actualType->expect<TupleType>();
    if (std::all_of(
            tuple->elements().cbegin(),
            tuple->elements().cend(),
            [&](const auto& el) {
              return el->isSubtypeOf(schemaContainedType);
            })) {
      return true;
    }
  }

  bool argIsFloat = *schemaType == *FloatType::get();
  bool argIsInt = *schemaType == *IntType::get();
  bool argIsNumber = *schemaType == *NumberType::get();

  // * Tensor is implicitly convertible to:
  if (actualType->isSubtypeOf(TensorType::get())) {
    // - Float
    // - Int
    // - Scalar
    if (argIsFloat || argIsInt || argIsNumber) {
      return true;
    }
  }

  // * Scalar is implicitly convertible to:
  if (*actualType == *NumberType::get()) {
    // - Float
    // - Int
    if (argIsFloat || argIsInt) {
      return true;
    }
  }

  // * String is implicitly convertible to device
  if (*actualType == *StringType::get() &&
      schemaType->isSubtypeOf(DeviceObjType::get())) {
    return true;
  }

  // We've exhausted any implicit conversion rules, so the provided type does
  // not match the formal argument.
  auto isSubtype = actualType->isSubtypeOfExt(schemaType, &err_);
  // This should always fail, we're just callign the subtyping check to get an
  // error msg.
  // TODO: we are checking the subtype twice for not really any good reason.
  TORCH_INTERNAL_ASSERT(!isSubtype);

  err_ << arg.formatTypeMismatchMsg(actualType->repr_str());
  auto tensorType = actualType->cast<TensorType>();
  if (tensorType && tensorType->isInferredType()) {
    err_ << fmt::format(
        "Inferred the value for argument '{}' to be of type 'Tensor' "
        "because it was not annotated with an explicit type.\n",
        arg.name());
  }
  auto listType = actualType->cast<ListType>();
  if (listType && listType->getElementType()->isSubtypeOf(TensorType::get())) {
    err_ << "Empty lists default to List[Tensor]. Add a variable "
            "annotation to the assignment to create an empty list "
            "of another type.";
  }

  return false;
}

TORCH_API bool isMatchingSchema(
    const FunctionSchema& schema,
    const std::vector<TypePtr>& args,
    const std::unordered_map<std::string, TypePtr>& kwargs) {
  auto resolver = SchemaMatcher(schema, args, kwargs);
  return resolver.isMatch();
}

} // namespace c10
