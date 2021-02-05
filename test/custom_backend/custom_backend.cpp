#include "custom_backend.h"

namespace torch {
namespace custom_backend {
namespace {
constexpr auto kBackendName = "custom_backend";
static auto cls = torch::jit::backend<CustomBackend>(kBackendName, backendPreprocessFunction);
}

std::string getBackendName() {
  return std::string(kBackendName);
}
}
}
