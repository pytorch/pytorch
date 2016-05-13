#include "caffe2/core/typeid.h"

#include <cxxabi.h>

namespace caffe2 {
std::map<CaffeTypeId, const char*>& gTypeNames() {
  static std::map<CaffeTypeId, const char*> g_type_names;
  return g_type_names;
}

const char* Demangle(const char* name) {
  int status = 0;
  size_t len = 0;
  const char* demangled = ::abi::__cxa_demangle(name, nullptr, &len, &status);
  return (status == 0 ? demangled : name);
}

namespace {
// This single registerer exists solely for us to be able to name a TypeMet
// object of unknown type. You should not use this struct yourself - it is
// intended to be only instantiated once here.
struct UnknownTypeNameRegisterer {
  UnknownTypeNameRegisterer() {
    gTypeNames()[0] = "Unknown Type";
  }
};
static UnknownTypeNameRegisterer g_unknown_type_name_registerer;

}  // namespace
}  // namespace caffe2
