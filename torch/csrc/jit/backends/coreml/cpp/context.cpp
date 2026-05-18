#include <torch/csrc/jit/backends/coreml/cpp/context.h>
#include <atomic>
#include <utility>

namespace torch::jit::mobile::coreml {

std::atomic<ContextInterface*> g_coreml_ctx_registry;

BackendRegistrar::BackendRegistrar(ContextInterface* ctx) {
  g_coreml_ctx_registry.store(ctx);
}

void setModelCacheDirectory(std::string path) {
  auto p = g_coreml_ctx_registry.load();
  if (p) {
    p->setModelCacheDirectory(std::move(path));
  }
}

} // namespace torch::jit::mobile::coreml
