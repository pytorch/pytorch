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

CAFFE_DEFINE_KNOWN_TYPE(float);
CAFFE_DEFINE_KNOWN_TYPE(int);
CAFFE_DEFINE_KNOWN_TYPE(std::string);
CAFFE_DEFINE_KNOWN_TYPE(bool);
CAFFE_DEFINE_KNOWN_TYPE(uint8_t);
CAFFE_DEFINE_KNOWN_TYPE(int8_t);
CAFFE_DEFINE_KNOWN_TYPE(uint16_t);
CAFFE_DEFINE_KNOWN_TYPE(int16_t);
CAFFE_DEFINE_KNOWN_TYPE(int64_t);
CAFFE_DEFINE_KNOWN_TYPE(double);
CAFFE_DEFINE_KNOWN_TYPE(char);
CAFFE_DEFINE_KNOWN_TYPE(at::Half);
CAFFE_DEFINE_KNOWN_TYPE(std::unique_ptr<std::mutex>);
CAFFE_DEFINE_KNOWN_TYPE(std::unique_ptr<std::atomic<bool>>);
CAFFE_DEFINE_KNOWN_TYPE(std::vector<int32_t>);
CAFFE_DEFINE_KNOWN_TYPE(std::vector<int64_t>);
CAFFE_DEFINE_KNOWN_TYPE(std::vector<unsigned long>);
CAFFE_DEFINE_KNOWN_TYPE(bool*);
CAFFE_DEFINE_KNOWN_TYPE(char*);
CAFFE_DEFINE_KNOWN_TYPE(int*);

// see typeid.h for details.
#if defined(_MSC_VER) || defined(__APPLE__) || \
    (defined(__ANDROID__) && !defined(__LP64__))
CAFFE_DEFINE_KNOWN_TYPE(long);
CAFFE_DEFINE_KNOWN_TYPE(std::vector<long>);
#endif

CAFFE_DEFINE_KNOWN_TYPE(_CaffeHighestPreallocatedTypeId);

namespace {
// This single registerer exists solely for us to be able to name a TypeMeta
// for unintializied blob. You should not use this struct yourself - it is
// intended to be only instantiated once here.
struct UninitializedTypeNameRegisterer {
  UninitializedTypeNameRegisterer() {
    gTypeNames()[TypeIdentifier::uninitialized()] = "nullptr (uninitialized)";
  }
};
static UninitializedTypeNameRegisterer g_uninitialized_type_name_registerer;

} // namespace
} // namespace caffe2
