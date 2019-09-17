#pragma once

#include <c10/util/C++17.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>

namespace at {

// Assume T is decayed
template <typename T>
using not_ok_to_box =
  c10::guts::disjunction<
    c10::guts::negation<std::is_constructible<IValue, T>>,
    // some constructors are templated (and therefore pass
    // is_constructible), but do not actually work with all
    // template arguments, so we must blacklist them explicitly
    // TODO: The correct fix is to sfinae based on is_constructible of T
    std::is_same<optional<ArrayRef<Dimname>>, T>,
    std::is_same<optional<ScalarType>, T>,
    std::is_same<optional<MemoryFormat>, T>
  >;

template <class... Args>
using all_ok_to_box = c10::guts::negation<c10::guts::disjunction<not_ok_to_box<guts::decay_t<Args>>...>>;

struct Boxer {
  Boxer(torch::jit::Stack* stack) : stack_(stack) {}

  template <
      typename... Args,
      typename c10::guts::enable_if_t<!all_ok_to_box<Args...>::value, std::nullptr_t> = nullptr>
  bool operator()(Args&&... as) {
    return false;
  }

  template <
      typename... Args,
      typename c10::guts::enable_if_t<all_ok_to_box<Args...>::value, std::nullptr_t> = nullptr>
  bool operator()(Args&&... as) {
    torch::jit::push(*stack_, std::forward<Args>(as)...);
    return true;
  }

  torch::jit::Stack* stack_;
};

}
