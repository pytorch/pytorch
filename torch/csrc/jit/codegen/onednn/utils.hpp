#pragma once

#include <memory>
#include <type_traits>
#include <utility>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {
namespace impl {
namespace utils {

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

} // namespace utils
} // namespace impl
} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch