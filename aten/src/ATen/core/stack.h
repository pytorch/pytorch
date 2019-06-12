#pragma once

#include <ATen/core/ivalue.h>

// TODO move this to c10 namespace

namespace torch {
namespace jit {

using c10::IValue;
using Stack = std::vector<IValue>;
using Operation = std::function<int(Stack&)>;

// An operation with N inputs and M outputs pops the last N inputs off
// the stack and pushes its M inputs onto the stack
// before: <other stack items> I0, I1, ... IN <- stack.back()
// after: <other stack items> O0, O1, ... OM
// operations are defined this way so that ownership of inputs can be
// transferred to the operation and it can incrementally drop ownership of
// tensors when they become unneeded. For large operations, like 'run an entire
// subgraph', this functionality is very important for minimizing gpu memory
// usage return value is the relative 'offset' to jump to for the next
// operation:
//   pc += 1 + offset
// so a return value of 0 goes to the next instruction

// treat the last N elements of the stack as a list, looking up
// element i
static inline IValue& peek(Stack& stack, size_t i, size_t N) {
  return *(stack.end() - N + i);
}
static inline const IValue& peek(const Stack& stack, size_t i, size_t N) {
  return *(stack.end() - N + i);
}
// treat the last N elements of the stack as a list, looking up the
// slice starting at index i and having length len
static inline at::ArrayRef<IValue> peekSlice(
    const Stack& stack,
    size_t i,
    size_t len,
    size_t N) {
  return at::ArrayRef<IValue>(stack).slice(stack.size() - N + i, len);
}
static inline at::ArrayRef<IValue> last(const Stack& stack, size_t N) {
  return peekSlice(stack, 0, N, N);
}
static inline void drop(Stack& stack, size_t n) {
  stack.erase(stack.end() - n, stack.end());
}
static inline IValue pop(Stack& stack) {
  auto r = std::move(stack.back());
  stack.pop_back();
  return r;
}
static inline std::vector<IValue> pop(Stack& stack, size_t n) {
  std::vector<IValue> result;
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    result.push_back(std::move(peek(stack, i, n)));
  }
  drop(stack, n);
  return result;
}

// variadic pop:
// int64_t a; at::Tensor b;
// pop(stack, a, b);
// equivalent to:
// b = pop(stack).toTensor();
// a = pop(stack).toInt();
template <typename... Types>
static inline void pop(Stack& stack, Types&... args) {
  size_t i = 0;
  constexpr size_t N = sizeof...(args);
  int result[N] = {
      (args = std::move(peek(stack, i++, N)).template to<Types>(), 0)...};
  (void)result;
  drop(stack, N);
}
template <typename... Types>
static inline void push(Stack& stack, Types&&... args) {
  (void)std::initializer_list<int>{(stack.emplace_back(std::forward<Types>(args)), 0)...};
}

// The packer here is carefully written not to make any unnecessary
// copies.

// pack takes the return values of aten functions pushes them onto the stack
template <typename T>
inline void pack(Stack& stack, T&& v) {
  stack.emplace_back(std::forward<T>(v));
}

template <std::size_t remaining, typename... Args>
struct TuplePacker {
  // NB: *Not* a universal reference.
  static void execute(Stack& stack, std::tuple<Args...>&& t) {
    // NB: The move here does not "destroy" the entire tuple, that is
    // not what std::move does; only the particular tuple index
    // processed here gets stolen.
    pack(stack, std::get<sizeof...(Args) - remaining>(std::move(t)));
    TuplePacker<remaining - 1, Args...>::execute(stack, std::move(t));
  }
};

template <typename... Args>
struct TuplePacker<0, Args...> {
  static void execute(Stack& stack, std::tuple<Args...>&& t){};
};

template <typename... Args>
inline void pack(Stack& stack, std::tuple<Args...>&& t) {
  TuplePacker<sizeof...(Args), Args...>::execute(stack, std::move(t));
}

} // namespace jit
} // namespace torch
