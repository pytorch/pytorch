#pragma once

#include <type_traits>

#include <ATen/core/ivalue.h>
#include <c10/util/Deprecated.h>
#include <c10/util/irange.h>

// TODO move this to c10 namespace


namespace torch::jit {

using c10::IValue;
using Stack = std::vector<IValue>;

class Operation {
  template <typename F, typename Arg>
  using accepts = std::is_constructible<std::function<void(Arg)>, F&&>;

 public:
  template <typename F,
            std::enable_if_t<accepts<F, Stack*>::value, int> = 0>
  C10_DEPRECATED_MESSAGE("Please use void(Stack&) to register operator instead.")
  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
  Operation(F&& raw): op_([raw = std::forward<F>(raw)](Stack& stack) {
    raw(&stack);
  }) {}

  template <typename F,
            std::enable_if_t<accepts<F, Stack&>::value &&
                !std::is_same_v<std::decay_t<F>, Operation>, int> = 0>
  Operation(F&& op): op_(std::forward<F>(op)) {}

  Operation(std::nullptr_t) noexcept {}

  explicit operator bool() const noexcept {
    return op_ ? true : false;
  }

  void operator()(Stack& stack) {
    op_(stack);
  }

  template <typename T>
  T* target() noexcept {
    return op_.target<T>();
  }

 private:
  std::function<void(Stack&)> op_;
};

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
inline IValue& peek(Stack& stack, size_t i, size_t N) {
  // NOLINTNEXTLINE(*-narrowing-conversions)
  return *(stack.end() - N + i);
}
inline IValue& peek(Stack* stack, size_t i, size_t N) {
  return peek(*stack, i, N);
}
inline const IValue& peek(const Stack& stack, size_t i, size_t N) {
  // NOLINTNEXTLINE(*-narrowing-conversions)
  return *(stack.end() - N + i);
}
inline const IValue& peek(const Stack* stack, size_t i, size_t N) {
  return peek(*stack, i, N);
}
// treat the last N elements of the stack as a list, looking up the
// slice starting at index i and having length len
inline at::ArrayRef<IValue> peekSlice(
    const Stack& stack,
    size_t i,
    size_t len,
    size_t N) {
  return at::ArrayRef<IValue>(stack).slice(stack.size() - N + i, len);
}
inline at::ArrayRef<IValue> last(const Stack& stack, size_t N) {
  return peekSlice(stack, 0, N, N);
}
inline at::ArrayRef<IValue> last(const Stack* stack, size_t N) {
  return last(*stack, N);
}
inline void drop(Stack& stack, size_t n) {
  // NOLINTNEXTLINE(*-narrowing-conversions)
  stack.erase(stack.end() - n, stack.end());
}
inline void drop(Stack* stack, size_t n) {
  drop(*stack, n);
}
inline IValue pop(Stack& stack) {
  if (stack.empty()) {
    throw std::runtime_error("pop() called on empty stack");
  }
  auto r = std::move(stack.back());
  stack.pop_back();
  return r;
}
inline IValue pop(Stack* stack) {
  return pop(*stack);
}
inline std::vector<IValue> pop(Stack& stack, size_t n) {
  std::vector<IValue> result;
  result.reserve(n);
  for (const auto i : c10::irange(n)) {
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
inline void pop(Stack& stack, Types&... args) {
  size_t i = 0;
  constexpr size_t N = sizeof...(args);
  (void)std::initializer_list<int>{
      (args = std::move(peek(stack, i++, N)).template to<Types>(), 0)...};
  drop(stack, N);
}
template <typename... Types>
inline void pop(Stack* stack, Types&... args) {
  pop(*stack, args...);
}
template <typename Type>
inline void push_one(Stack& stack, Type&& arg) {
  stack.emplace_back(std::forward<Type>(arg));
}

inline void push_one(Stack& stack, c10::TensorOptions options) {
  stack.emplace_back(c10::typeMetaToScalarType(options.dtype()));
  stack.emplace_back(options.layout());
  stack.emplace_back(options.device());
  stack.emplace_back(options.pinned_memory());
}

template <typename... Types>
inline void push(Stack& stack, Types&&... args) {
  (void)std::initializer_list<int>{(push_one(stack, std::forward<Types>(args)), 0)...};
}
template <typename... Types>
inline void push(Stack* stack, Types&&... args) {
  return push(*stack, std::forward<Types>(args)...);
}
template <class T>
inline void push_list_elements(Stack& stack, const c10::List<T>& elements) {
  for (T elem : elements) {
    stack.push_back(std::move(elem));
  }
}

// The packer here is carefully written not to make any unnecessary
// copies.

// pack takes the return values of aten functions pushes them onto the stack
template <typename T>
inline void pack(Stack& stack, T&& v) {
  stack.emplace_back(std::forward<T>(v));
}
template <typename T>
inline void pack(Stack* stack, T&& v) {
  pack(*stack, std::forward<T>(v));
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
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  static void execute(Stack& /*stack*/, std::tuple<Args...>&& /*t*/){}
};

template <typename... Args>
inline void pack(Stack& stack, std::tuple<Args...>&& t) {
  TuplePacker<sizeof...(Args), Args...>::execute(stack, std::move(t));
}

} // namespace torch::jit
