#include <ATen/core/typeid.h>
#include <ATen/core/Error.h>

#include <atomic>

#if !defined(_MSC_VER)
#include <cxxabi.h>
#endif

using std::string;

namespace caffe2 {
namespace detail {
C10_EXPORT void _ThrowRuntimeTypeLogicError(const string& msg) {
  // In earlier versions it used to be std::abort() but it's a bit hard-core
  // for a library
  AT_ERROR(msg);
}
} // namespace detail

constexpr detail::TypeMetaData TypeMeta::uninitialized_;

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

CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(uint8_t)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(int8_t)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(int16_t)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(int)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(int64_t)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(at::Half)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(float)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(double)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(at::ComplexHalf)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(std::complex<float>)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(std::complex<double>)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(std::string)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(bool)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(uint16_t)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(char)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(std::unique_ptr<std::mutex>)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(std::unique_ptr<std::atomic<bool>>)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(std::vector<int32_t>)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(std::vector<int64_t>)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(std::vector<unsigned long>)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(bool*)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(char*)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(int*)

// see typeid.h for details.
#if defined(_MSC_VER) || defined(__APPLE__) || \
    (defined(__ANDROID__) && !defined(__LP64__))
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(long);
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(std::vector<long>);
#endif

CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(_CaffeHighestPreallocatedTypeId)

} // namespace caffe2
