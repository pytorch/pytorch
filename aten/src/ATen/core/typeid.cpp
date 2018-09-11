#include <ATen/core/typeid.h>
#include <ATen/core/Error.h>

#include <atomic>

#if !defined(_MSC_VER)
#include <cxxabi.h>
#endif

using std::string;

namespace caffe2 {

std::unordered_map<TypeIdentifier, string>& gTypeNames() {
  static std::unordered_map<TypeIdentifier, string> g_type_names;
  return g_type_names;
}

std::unordered_set<string>& gRegisteredTypeNames() {
  static std::unordered_set<string> g_registered_type_names;
  return g_registered_type_names;
}

std::mutex& gTypeRegistrationMutex() {
  static std::mutex g_type_registration_mutex;
  return g_type_registration_mutex;
}

void TypeMeta::_ThrowRuntimeTypeLogicError(const std::string& msg) {
  // In earlier versions it used to be std::abort() but it's a bit hard-core
  // for a library
  AT_ERROR(msg);
}

TypeIdentifier TypeIdentifier::createTypeId() {
  static std::atomic<TypeIdentifier::underlying_type> counter(
      TypeMeta::Id<_CaffeHighestPreallocatedTypeId>().underlyingId());
  const TypeIdentifier::underlying_type new_value = ++counter;
  if (new_value ==
      std::numeric_limits<TypeIdentifier::underlying_type>::max()) {
    throw std::logic_error(
        "Ran out of available type ids. If you need more than 2^16 CAFFE_KNOWN_TYPEs, we need to increase TypeIdentifier to use more than 16 bit.");
  }
  return TypeIdentifier(new_value);
}

} // namespace caffe2
