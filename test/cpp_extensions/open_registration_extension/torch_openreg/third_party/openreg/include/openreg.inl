#ifndef OPENREG_H
#error "Don`t include openreg.inl directly, include openreg.h instead."
#endif

#include <functional>
#include <tuple>
#include <utility>

namespace openreg {
OPENREG_EXPORT orError_t
addTaskToStream(orStream* stream, std::function<void()> task);
}

template <typename Func, typename... Args>
OPENREG_EXPORT inline orError_t orLaunchKernel(
    orStream* stream,
    Func&& kernel_func,
    Args&&... args) {
  if (!stream) {
    return orErrorUnknown;
  }

/*
 * Some tests in PyTorch still use C++11, so we use conditional macro to
 * select different approaches for different C++ version.
 *
 * Std::apply is only supported in C++17, so for C++11/14, std::bind is
 * a more appropriate approach, but the former has better performance.
 */
#if __cplusplus >= 201703L
  auto task = [func = std::forward<Func>(kernel_func),
               args_tuple =
                   std::make_tuple(std::forward<Args>(args)...)]() mutable {
    std::apply(func, std::move(args_tuple));
  };
#else
  auto task =
      std::bind(std::forward<Func>(kernel_func), std::forward<Args>(args)...);
#endif

  return openreg::addTaskToStream(stream, std::move(task));
}
