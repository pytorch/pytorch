#pragma once

#include <torch/csrc/deploy/environment.h>

namespace torch {
namespace deploy {

class NoopEnvironment : public Environment {
 public:
  void configureInterpreter(Interpreter* /* interp */) override {}
};

} // namespace deploy
} // namespace torch
