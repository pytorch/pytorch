#include <torch/imethod.h>

namespace torch {

const std::vector<std::string>& IMethod::getArgumentNames() const
{
  // TODO(jwtan): Deal with empty parameter list.
  if (!argumentNames_.empty()) {
    return argumentNames_;
  }

  setArgumentNames(argumentNames_);
  return argumentNames_;
}

} // namespace torch
