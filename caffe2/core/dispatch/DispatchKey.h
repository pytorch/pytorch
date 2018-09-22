#pragma once

#include "caffe2/core/dispatch/DeviceId.h"
#include "caffe2/core/dispatch/LayoutId.h"
#include "caffe2/core/typeid.h"

#include <vector>
#include <functional>
#include <sstream>
#include "caffe2/utils/Array.h"

namespace c10 {

namespace details {

struct TensorParameterDispatchKey final {
  // note: This dispatch key structure is not final yet and will change. Don't rely on it.
  DeviceTypeId deviceTypeId;
  LayoutId layoutId;
  // TODO Move this TypeIdentifier to c10 namespace
  caffe2::TypeIdentifier dataType;
};
inline constexpr bool operator==(const TensorParameterDispatchKey& lhs, const TensorParameterDispatchKey& rhs) {
  return lhs.deviceTypeId == rhs.deviceTypeId && lhs.layoutId == rhs.layoutId && lhs.dataType == rhs.dataType;
}

inline std::ostream& operator<<(std::ostream& stream, const TensorParameterDispatchKey& key) {
  return stream << "TensorKey(" << key.deviceTypeId << ", " << key.layoutId.value() << ", " << key.dataType << ")";
}

}  // namespace details
}  // namespace c10

namespace std {
  template<>
  struct hash<c10::details::TensorParameterDispatchKey> {
    // TODO constexpr hashing
    size_t operator()(const c10::details::TensorParameterDispatchKey& obj) const {
      return std::hash<c10::DeviceTypeId>()(obj.deviceTypeId) ^ std::hash<c10::LayoutId>()(obj.layoutId) ^ std::hash<caffe2::TypeIdentifier>()(obj.dataType);
    }
  };
}  // namespace std

namespace c10 {
/**
 * The dispatch key encodes the runtime type identity of a function call arguments,
 * specifying what aspects of this identity can be dynamically dispatched on.
 *
 * Intuitively, given a function signature like f(Tensor, int), a valid dispatch
 * key for the arguments might be [CPUFloatTensor] (notice that 'f' is NOT included
 * in the dispatch key, and the runtime type of 'int' is NOT considered for dispatch
 * (since it is trivial).
 *
 * Dispatch keys permit equality tests and are hashable.
 *
 * @tparam num_dispatch_args The number of dispatchable arguments
 */
template<size_t num_dispatch_args>
struct DispatchKey final {
  guts::array<details::TensorParameterDispatchKey, num_dispatch_args> argTypes;
};

template<size_t num_dispatch_args>
inline constexpr bool operator==(const DispatchKey<num_dispatch_args> &lhs, const DispatchKey<num_dispatch_args>& rhs) {
  // TODO: Use AVX instructions to perform this equality test more quickly
  return lhs.argTypes == rhs.argTypes;
}

template<size_t num_dispatch_args>
inline std::ostream& operator<<(std::ostream& stream, const DispatchKey<num_dispatch_args>& key) {
  stream << "DispatchKey(";
  if (num_dispatch_args > 0) {
      stream << "DispatchKey(" << key.argTypes[0];
      for (size_t i = 1; i < num_dispatch_args; ++i) {
          stream << ", " << key.argTypes[i];
      }
      stream << ")";
  }
  return stream << ")";
}

}  // namespace c10

namespace std {
  template<size_t num_dispatch_args>
  struct hash<c10::DispatchKey<num_dispatch_args>> {
    // TODO constexpr hashing
    size_t operator()(const c10::DispatchKey<num_dispatch_args>& obj) const {
      size_t hash_value = 0;
      for (const auto& argType : obj.argTypes) {
        hash_value *= 10883; // prime
        hash_value += std::hash<c10::details::TensorParameterDispatchKey>()(argType);
      }
      return hash_value;
    }
  };
}  // namespace std
