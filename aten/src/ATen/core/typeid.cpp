#include <ATen/core/typeid.h>
#include <ATen/core/Error.h>

#include <atomic>

#if !defined(_MSC_VER)
#include <cxxabi.h>
#endif

using std::string;

namespace caffe2 {

void _ThrowRuntimeTypeLogicError(const string& msg) {
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
