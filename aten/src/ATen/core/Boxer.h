#pragma once

#include <c10/util/C++17.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>

namespace at {

template <class R, class... Ts>
using are_all_constructible = c10::guts::conjunction<std::is_constructible<R, guts::decay_t<Ts>>...>;

struct Boxer {
  Boxer(torch::jit::Stack* stack) : stack_(stack) {}

  template <
      typename... Args,
      typename c10::guts::enable_if_t<!are_all_constructible<IValue, Args...>::value, std::nullptr_t> = nullptr>
  bool operator()(Args&&... as) {
    return false;
  }

  template <
      typename... Args,
      typename c10::guts::enable_if_t<are_all_constructible<IValue, Args...>::value, std::nullptr_t> = nullptr>
  bool operator()(Args&&... as) {
    // torch::jit::push(*stack_, std::forward<Args>(as)...);
    return true;
  }

  torch::jit::Stack* stack_;
};

}
