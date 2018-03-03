#pragma once
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/autograd/function.h"

#include <functional>

// ${generated_comment}

namespace torch { namespace jit {



using Stack = std::vector<at::Tensor>;
using Operation = std::function<int(Stack&)>;

// An operation with N inputs and M outputs pops the last N inputs off
// the stack and pushes its M inputs onto the stack
// before: <other stack items> I0, I1, ... IN <- stack.back()
// after: <other stack items> O0, O1, ... OM
// operations are defined this way so that ownership of inputs can be transferred
// to the operation and it can incrementally drop ownership of tensors
// when they become unneeded. For large operations, like 'run an entire subgraph',
// this functionality is very important for minimizing gpu memory usage
// return value is the relative 'offset' to jump to for the next operation:
// pc += 1 + offset
// so a return value of 0 goes to the next instruction

// treat the last N elements of the stack as a list, looking up
// element i
static inline at::Tensor & peek(Stack & stack, size_t i, size_t N) {
  return *(stack.end() - N + i);
}
// treat the last N elements of the stack as a list, looking up the
// slice starting at index i and having length len
static inline ArrayRef<at::Tensor> peekSlice(Stack & stack, size_t i, size_t len, size_t N) {
  return ArrayRef<at::Tensor>(stack).slice(stack.size() - N + i, len);
}
static inline ArrayRef<at::Tensor> last(Stack & stack, size_t N) {
  return peekSlice(stack, 0, N, N);
}
static inline void drop(Stack & stack, size_t n) {
  stack.erase(stack.end() - n, stack.end());
}
static inline at::Tensor pop(Stack & stack) {
  auto r = std::move(stack.back());
  stack.pop_back();
  return r;
}

struct TensorOp {
  TensorOp(Operation op, std::string name, size_t num_inputs)
    : op(op)
    , name(name)
    , num_inputs(num_inputs) {}

  const Operation op;
  const std::string name;
  const size_t num_inputs;
};

using operator_constructor = std::function<TensorOp(jit::Node*)>;
using ConstructorsMap = std::unordered_map<std::string, operator_constructor>;

ConstructorsMap::iterator findTensorOp(jit::Node* n);
bool hasTensorOp(jit::Node* n);
TensorOp getTensorOp(jit::Node* n);

}} // namespace torch::jit;
