#include "caffe2/core/typeid.h"
#include "caffe2/core/scope_guard.h"

#include <cxxabi.h>

namespace caffe2 {
std::map<CaffeTypeId, string>& gTypeNames() {
  static std::map<CaffeTypeId, string> g_type_names;
  return g_type_names;
}

string Demangle(const char* name) {
  int status = 0;
  auto demangled = ::abi::__cxa_demangle(name, nullptr, nullptr, &status);
  if (demangled) {
    auto guard = MakeGuard([demangled]() { free(demangled); });
    return string(demangled);
  }
  return name;
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

}  // namespace
}  // namespace caffe2
