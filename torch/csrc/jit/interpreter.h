#pragma once
#include <memory>
#include <vector>
#include "ATen/optional.h"

namespace at {
  struct Tensor;
}
namespace torch { namespace jit {

// The interpreter run Graphs with Tensor inputs and Tensor outputs
// a separate component in the autograd handles unwrapping and wrapping
// variable objects for use in the interpreter.

struct Node;
struct GraphExecutor;
struct CodeImpl;
struct InterpreterStateImpl;
struct Graph;
struct Node;
struct TensorType;
struct IValue;
using Stack = std::vector<IValue>;

struct Code {
  Code()
    : pImpl(nullptr) {}
  Code(std::shared_ptr<Graph>& graph);
  ~Code();

  // Returns pointers to GraphExecutors created to run GraphExecutor nodes in the given graph.
  const std::vector<GraphExecutor*>& executors();

  explicit operator bool() const {
    return pImpl != nullptr;
  }

private:
  std::shared_ptr<CodeImpl> pImpl;
  friend struct InterpreterStateImpl;
  friend std::ostream & operator<<(std::ostream & out, const Code & code);
};

struct InterpreterState {
  InterpreterState(const Code & code);
  // advance the interpreter state by running one stage. Returning the
  // outputs for that stage, suspending the computation.
  // Call this function again continues computation where it left off.
  void runOneStage(Stack & stack);
  const TensorType & tensorTypeForInput(size_t i) const;
  ~InterpreterState();
  // create a copy of InterpreterState with its current state
  // used when retain_graph=True so that stages can be re-run
  InterpreterState clone() const;
private:
  InterpreterState(InterpreterStateImpl * pImpl);
  std::shared_ptr<InterpreterStateImpl> pImpl;
};

}}
