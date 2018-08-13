#include <ATen/core/typeid.h>
#include <ATen/core/Error.h>

#include <atomic>

#if !defined(_MSC_VER)
#include <cxxabi.h>
#endif

using std::string;

namespace at {

at::DataType DataType::createTypeId() {
  static std::atomic<at::DataType::underlying_type> counter(
      caffe2::TypeMeta::Id<caffe2::_CaffeHighestPreallocatedTypeId>().underlyingId());
  const at::DataType::underlying_type new_value = ++counter;
  if (new_value ==
      std::numeric_limits<at::DataType::underlying_type>::max()) {
    throw std::logic_error(
        "Ran out of available type ids. If you need more than 2^16 CAFFE_KNOWN_TYPEs, we need to increase at::DataType to use more than 16 bit.");
  }
  return at::DataType(new_value);
}

}

namespace caffe2 {

std::unordered_map<at::DataType, string>& gTypeNames() {
  static std::unordered_map<at::DataType, string> g_type_names;
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

string GetExceptionString(const std::exception& e) {
#ifdef __GXX_RTTI
  return at::demangle(typeid(e).name()) + ": " + e.what();
#else
  return string("Exception (no RTTI available): ") + e.what();
#endif // __GXX_RTTI
}

void TypeMeta::_ThrowRuntimeTypeLogicError(const std::string& msg) {
  // In earlier versions it used to be std::abort() but it's a bit hard-core
  // for a library
  AT_ERROR(msg);
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
CAFFE_DEFINE_KNOWN_TYPE(std::unique_ptr<std::mutex>);
CAFFE_DEFINE_KNOWN_TYPE(std::unique_ptr<std::atomic<bool>>);
CAFFE_DEFINE_KNOWN_TYPE(std::vector<int32_t>);
CAFFE_DEFINE_KNOWN_TYPE(std::vector<int64_t>);
CAFFE_DEFINE_KNOWN_TYPE(std::vector<unsigned long>);
CAFFE_DEFINE_KNOWN_TYPE(bool*);
CAFFE_DEFINE_KNOWN_TYPE(char*);
CAFFE_DEFINE_KNOWN_TYPE(int*);

#ifdef CAFFE2_UNIQUE_LONG_TYPEMETA
CAFFE_DEFINE_KNOWN_TYPE(long);
CAFFE_DEFINE_KNOWN_TYPE(std::vector<long>);
#endif // CAFFE2_UNIQUE_LONG_TYPEMETA

CAFFE_DEFINE_KNOWN_TYPE(_CaffeHighestPreallocatedTypeId);

namespace {
// This single registerer exists solely for us to be able to name a TypeMeta
// for unintializied blob. You should not use this struct yourself - it is
// intended to be only instantiated once here.
struct UninitializedTypeNameRegisterer {
  UninitializedTypeNameRegisterer() {
    gTypeNames()[at::DataType::uninitialized()] = "nullptr (uninitialized)";
  }
};
static UninitializedTypeNameRegisterer g_uninitialized_type_name_registerer;

} // namespace
} // namespace caffe2
