#pragma once
#include <memory>
#include <vector>

namespace at {
  struct Tensor;
}
namespace torch { namespace jit {

struct NotImplementedException : public std::logic_error {
  NotImplementedException()
  : std::logic_error("Function not yet implemented.") {}
};

// The interpreter run Graphs with Tensor inputs and Tensor outputs
// a separate component in the autograd handles unwrapping and wrapping
// variable objects for use in the interpreter.

struct FunctionImpl;
struct InterpreterImpl;
struct Graph;

struct Function {
  Function()
  : pImpl(nullptr) {}
  Function(std::shared_ptr<Graph> & graph);
  ~Function();
  operator bool() const {
    return pImpl != nullptr;
  }
private:
  std::shared_ptr<FunctionImpl> pImpl;
  friend class InterpreterImpl;
};

struct Interpreter {
  Interpreter(const Function & function);
  void runOneStage(
    const std::vector<at::Tensor> & inputs,
    std::vector<at::Tensor> & outputs);
  ~Interpreter();
  // create a copy of Interpreter with its current state
  // used when retain_graph=True so that stages can be re-run
  Interpreter clone() const;
private:
  Interpreter(InterpreterImpl * pImpl);
  std::shared_ptr<InterpreterImpl> pImpl;
};

}}
