#ifndef CAFFE2_C10_UTIL_SSIZE_H
#define CAFFE2_C10_UTIL_SSIZE_H

#include <cassert>
#include <cstddef>
#include <limits>

namespace c10 {

template <typename Container>
std::ptrdiff_t ssize(Container const& container) {
  std::size_t size = container.size();
  assert(size < std::size_t{std::numeric_limits<std::ptrdiff_t>::max()});
  return static_cast<std::ptrdiff_t>(size);
}

} // namespace c10

#endif // CAFFE2_C10_UTIL_SSIZE_H
