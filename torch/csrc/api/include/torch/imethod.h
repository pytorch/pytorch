#pragma once
#include <ATen/core/ivalue.h>

namespace torch {

class IMethod {
  /*
  IMethod provides a portable interface for torch methods, whether
  they are backed by torchscript or python/deploy.

  This is helpful since torchscript methods provide additional information
  (e.g. FunctionSchema, Graph) which aren't available in pure python methods.

  Higher level APIs should prefer depending on this interface rather
  than a specific implementation of it, to promote portability and reuse, and
  avoid unintentional dependencies on e.g. script methods.

  Note: This API is experimental, and may evolve.
  */
 public:
  using IValueList = std::vector<c10::IValue>;
  using IValueMap = std::unordered_map<std::string, at::IValue>;

  virtual ~IMethod() = default;

  virtual c10::IValue operator()(
      std::vector<c10::IValue> args,
      const IValueMap& kwargs = IValueMap()) = 0;

  // Returns an ordered list of argument names, possible in both
  // script and python methods.  This is a more portable dependency
  // than a ScriptMethod FunctionSchema, which has more information
  // than can be generally expected from a python method.
  virtual std::vector<std::string> getArgumentNames() = 0;
};

} // namespace torch
