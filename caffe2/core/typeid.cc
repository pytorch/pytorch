#include "caffe2/core/typeid.h"
#include "caffe2/core/scope_guard.h"

#if !defined(_MSC_VER)
#include <cxxabi.h>
#endif

namespace caffe2 {
std::map<CaffeTypeId, string>& gTypeNames() {
  static std::map<CaffeTypeId, string> g_type_names;
  return g_type_names;
}

std::set<string>& gRegisteredTypeNames() {
  static std::set<string> g_registered_type_names;
  return g_registered_type_names;
}

std::mutex& gCaffe2TypeRegistrationMutex() {
  static std::mutex g_caffe2_type_registration_mutex;
  return g_caffe2_type_registration_mutex;
}

#if defined(_MSC_VER)
// Windows does not have cxxabi.h, so we will simply return the original.
string Demangle(const char* name) {
  return string(name);
}
#else
string Demangle(const char* name) {
  int status = 0;
  auto demangled = ::abi::__cxa_demangle(name, nullptr, nullptr, &status);
  if (demangled) {
    auto guard = MakeGuard([demangled]() { free(demangled); });
    return string(demangled);
  }
  return name;
}
#endif

string GetExceptionString(const std::exception& e) {
#ifdef __GXX_RTTI
  return Demangle(typeid(e).name()) + ": " + e.what();
#else
  return string("Exception (no RTTI available): ") + e.what();
#endif // __GXX_RTTI
}

namespace {
// This single registerer exists solely for us to be able to name a TypeMeta
// for unintializied blob. You should not use this struct yourself - it is
// intended to be only instantiated once here.
struct UninitializedTypeNameRegisterer {
  UninitializedTypeNameRegisterer() {
    gTypeNames()[0] = "nullptr (uninitialized)";
  }
};
static UninitializedTypeNameRegisterer g_uninitialized_type_name_registerer;

} // namespace
} // namespace caffe2
