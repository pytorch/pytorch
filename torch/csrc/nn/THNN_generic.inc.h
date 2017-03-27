#include "THNN_generic.h"

#include <sstream>
#include <stdarg.h>

#include <TH/TH.h>
#include <THNN/THNN.h>
#ifdef THNN_
#undef THNN_
#endif

#ifdef WITH_CUDA
#include <THC/THC.h>
#include <THCUNN/THCUNN.h>
#ifdef THNN_
#undef THNN_
#endif
#endif

#ifdef WITH_CUDA
extern THCState* state;
#endif

namespace {

static std::runtime_error invalid_tensor(const char* expected, const char* got) {
  std::stringstream ss;
  ss << "expected " << expected << " tensor (got " << got << " tensor)";
  return std::runtime_error(ss.str());
}

void checkTypes(bool isCuda, thpp::Type type, ...) {
  va_list args;
  va_start(args, type);

  const char* name;
  while ((name = va_arg(args, const char*))) {
    bool optional = false;
    if (name[0] == '?') {
      name++;
      optional = true;
    }
    thpp::Tensor* tensor = va_arg(args, thpp::Tensor*);
    if (!tensor) {
      if (optional) {
        continue;
      }
      throw std::runtime_error(std::string("missing required argument '") + name + "'");
    }
    if (tensor->isCuda() != isCuda) {
      throw invalid_tensor(isCuda ? "CUDA" : "CPU", tensor->isCuda() ? "CUDA" : "CPU");
    }
  }
}

}
