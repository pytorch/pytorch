#include <torch/csrc/jit/backends/coreml/cpp/context.h>
#include <atomic>

namespace torch {
namespace jit {
namespace mobile {
namespace coreml {

std::atomic<ContextInterface*> g_coreml_ctx_registry;

BackendRegistrar::BackendRegistrar(ContextInterface* ctx) {
  g_coreml_ctx_registry.store(ctx);
}

void setModelCacheDirectory(std::string path) {
  auto p = g_coreml_ctx_registry.load();
  if (p) {
    p->setModelCacheDirectory(path);
  }
}

} // namespace coreml
} // namespace mobile
} // namespace jit
} // namespace torch
