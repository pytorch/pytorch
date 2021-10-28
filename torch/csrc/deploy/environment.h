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
  virtual ~Environment() {}
  /*
   * Setup the environment. E.g., for the embedded xar, we need extract it out
   * to the file system.
   */
  virtual void setup() = 0;
  /*
   * Cleanup the environment.
   */
  virtual void teardown() = 0;
  /*
   * Do the configuration on the interpreter. E.g., append to sys.path.
   */
  virtual void configureInterpreter(Interpreter* interp) = 0;
};

} // namespace deploy
} // namespace torch
