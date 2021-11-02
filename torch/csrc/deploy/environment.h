#pragma once

namespace torch {
namespace deploy {

class Interpreter;

/*
 * An environment is the concept to decribe the circumstances in which a
 * torch::deploy interpreter runs. In can be an xar file embedded in the binary,
 * a filesystem path for the installed libraries etc.
 */
class Environment {
 public:
  virtual ~Environment() = default;
  virtual void configureInterpreter(Interpreter* interp) = 0;
};

} // namespace deploy
} // namespace torch
