#pragma once

#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <string>
#include <vector>

namespace c10 {

/**
 * Determines whether the provided argument types match the provided schema, and
 * contains metadata about the results of the match. Implicit conversions are
 * performed during this matching.
 */
class TORCH_API SchemaMatcher {
 public:
  SchemaMatcher(
      const FunctionSchema& schema,
      const std::vector<TypePtr>& args,
      const std::unordered_map<std::string, TypePtr>& kwargs);

  /**
   * Did the arguments match the schema?
   */
  bool isMatch() const;

  /**
   * Contains the error message for a potential mismatch.
   *
   * \note Calling this method on a successful match is an error. Use isMatch()
   * first to determine whether the match was successful.
   */
  std::string err() const;

  /**
   * Returns a list of fully-resolved types for the inputs/outputs.
   *
   * This list is similar to the one returned by schema.arguments() or
   * schema.returns(), except with all VarTypes resolved to concrete types
   * based on the args/kwargs provided.
   */
  const std::vector<TypePtr>& inputs() const;
  const std::vector<TypePtr>& outputs() const;

  /**
   * Returns a mapping of args/kwargs to the index of the corresponding element
   * of inputs().
   *
   * This is useful when, e.g. the arguments `int, int, int` are matched to the
   * formal parameter `List[int]`, so that we know that all the input arguments
   * match to the same parameter.
   */
  std::vector<size_t> argToInputs() const;
  std::unordered_map<std::string, size_t> kwargToInputs() const;

 private:
  void doMatch();
  bool canConvertToVararg(const FunctionSchema& schema, size_t arg_index);
  bool isMatchingArgument(const Argument& arg, const TypePtr& actualType);

  void setErr(std::string err);

  const FunctionSchema& schema_;
  const std::vector<TypePtr>& args_;
  const std::unordered_map<std::string, TypePtr>& kwargs_;

  std::vector<TypePtr> inputs_;
  std::vector<TypePtr> outputs_;

  std::vector<size_t> argToInputs_;
  std::unordered_map<std::string, size_t> kwargToInputs_ ;

  bool isMatch_;
  std::ostringstream err_;

  TypeEnv typeEnv_;
};

TORCH_API bool isMatchingSchema(
    const FunctionSchema& schema,
    const std::vector<TypePtr>& args,
    const std::unordered_map<std::string, TypePtr>& kwargs);

} // namespace c10
