#ifndef COMPUTATION_CLIENT_UTIL_H_
#define COMPUTATION_CLIENT_UTIL_H_

#include <c10/util/Optional.h>
#include <torch/csrc/lazy/core/hash.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

namespace lazy_tensors {
namespace util {

// The following is only used within computation_client.
template <typename T, typename G>
const typename T::mapped_type& MapInsert(T* cont,
                                         const typename T::key_type& key,
                                         const G& gen) {
  auto it = cont->find(key);
  if (it == cont->end()) {
    it = cont->emplace(key, gen()).first;
  }
  return it->second;
}

}  // namespace util
}  // namespace lazy_tensors

#endif  // COMPUTATION_CLIENT_UTIL_H_
