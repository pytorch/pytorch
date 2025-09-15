#include <torch/imethod.h>

namespace torch {

const std::vector<std::string>& IMethod::getArgumentNames() const {
  if (isArgumentNamesInitialized_) {
    return argumentNames_;
  }

  isArgumentNamesInitialized_ = true;
  setArgumentNames(argumentNames_);
  return argumentNames_;
}

} // namespace torch
