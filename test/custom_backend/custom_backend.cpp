#include "custom_backend.h"
#include <torch/csrc/jit/backends/backend_preprocess.h>

namespace torch {
namespace custom_backend {
namespace {
constexpr auto kBackendName = "custom_backend";
static auto cls = torch::jit::backend<CustomBackend>(kBackendName);
static auto pre_reg = torch::jit::backend_preprocess_register(kBackendName, preprocess);
}

std::string getBackendName() {
  return std::string(kBackendName);
}
}
}
