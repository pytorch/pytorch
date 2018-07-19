#pragma once
#include "ATen/ATen.h"
#include "torch/csrc/jit/tensor_conversions.h"
#include "torch/csrc/jit/ivalue.h"

namespace torch { namespace jit {

using Stack = std::vector<IValue>;
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
static inline IValue & peek(Stack & stack, size_t i, size_t N) {
  return *(stack.end() - N + i);
}
// treat the last N elements of the stack as a list, looking up the
// slice starting at index i and having length len
static inline at::ArrayRef<IValue> peekSlice(Stack & stack, size_t i, size_t len, size_t N) {
  return at::ArrayRef<IValue>(stack).slice(stack.size() - N + i, len);
}
static inline at::ArrayRef<IValue> last(Stack & stack, size_t N) {
  return peekSlice(stack, 0, N, N);
}
static inline void drop(Stack & stack, size_t n) {
  stack.erase(stack.end() - n, stack.end());
}
static inline IValue pop(Stack & stack) {
  auto r = std::move(stack.back());
  stack.pop_back();
  return r;
}

// The packer here is carefully written not to make any unnecessary
// copies.

// pack takes the return values of aten functions pushes them onto the stack
template<typename T>
inline void pack(Stack & stack, T&& v) {
  stack.push_back(IValue(as_variable(std::move(v))));
}
template<>
inline void pack(Stack & stack, at::Tensor&& v) {
  stack.push_back(IValue(std::move(v)));
}

template<>
inline void pack(Stack & stack, autograd::Variable&& v) {
  stack.push_back(IValue(std::move(v)));
}

template<>
inline void pack(Stack & stack, std::vector<at::Tensor>&& ts) {
  for(auto& t : ts) {
    stack.push_back(IValue(std::move(t)));
  }
}

template<std::size_t remaining, typename... Args>
struct TuplePacker
{
  // NB: *Not* a universal reference.
  static void execute(Stack & stack, std::tuple<Args...> && t)
  {
    // NB: The move here does not "destroy" the entire tuple, that is
    // not what std::move does; only the particular tuple index
    // processed here gets stolen.
    pack(stack, std::get<sizeof...(Args) - remaining>(std::move(t)));
    TuplePacker<remaining - 1, Args...>::execute(stack, std::move(t));
  }
};

template<typename... Args>
struct TuplePacker<0, Args...>
{
  static void execute(Stack & stack, std::tuple<Args...> && t) {};
};

template<typename... Args>
inline void pack(Stack & stack, std::tuple<Args...> && t) {
  TuplePacker<sizeof...(Args), Args...>::execute(stack, std::move(t));
}

}}
