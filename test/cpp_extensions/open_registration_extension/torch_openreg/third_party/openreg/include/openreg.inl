#ifndef OPENREG_H
#error "Don`t include openreg.inl directly, include openreg.h instead."
#endif

#include <functional>
#include <tuple>
#include <utility>

namespace internal {
orError_t addTaskToStream(orStream* stream, std::function<void()> task);
}

template <typename Func, typename... Args>
inline orError_t orLaunchKernel(
    orStream* stream,
    Func&& kernel_func,
    Args&&... args) {
  if (!stream)
    return orErrorUnknown;

  auto task = [func = std::forward<Func>(kernel_func),
               args_tuple =
                   std::make_tuple(std::forward<Args>(args)...)]() mutable {
    std::apply(func, std::move(args_tuple));
  };

  return internal::addTaskToStream(stream, std::move(task));
}
