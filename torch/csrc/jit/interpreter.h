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

struct CodeImpl;
struct InterpreterStateImpl;
struct Graph;

struct Code {
  Code()
  : pImpl(nullptr) {}
  Code(std::shared_ptr<Graph> & graph);
  ~Code();
  operator bool() const {
    return pImpl != nullptr;
  }
private:
  std::shared_ptr<CodeImpl> pImpl;
  friend class InterpreterStateImpl;
};

struct InterpreterState {
  InterpreterState(const Code & code);
  // advance the interpreter state by running one stage. Returning the
  // outputs for that stage, suspending the computation.
  // Call this function again continues computation where it left off.
  void runOneStage(
    const std::vector<at::Tensor> & inputs,
    std::vector<at::Tensor> & outputs);
  ~InterpreterState();
  // create a copy of InterpreterState with its current state
  // used when retain_graph=True so that stages can be re-run
  InterpreterState clone() const;
private:
  InterpreterState(InterpreterStateImpl * pImpl);
  std::shared_ptr<InterpreterStateImpl> pImpl;
};

}}
