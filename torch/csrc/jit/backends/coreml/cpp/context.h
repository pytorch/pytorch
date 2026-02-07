#ifndef PTM_COREML_Context_h
#define PTM_COREML_Context_h

#include <string>

namespace torch::jit::mobile::coreml {

struct ContextInterface {
  virtual ~ContextInterface() = default;
  virtual void setModelCacheDirectory(std::string path) = 0;
};

class BackendRegistrar {
 public:
  explicit BackendRegistrar(ContextInterface* ctx);
};

void setModelCacheDirectory(std::string path);

#if __has_include(<xplat/lazy_static/lazy_static.h>) && defined(LAZY_GATE_coreml_backend)
// Register the CoreML backend. Call this before using the backend.
// With LAZY_STATIC gating, this triggers deferred initialization.
void registerBackend();
#endif

} // namespace torch::jit::mobile::coreml

#endif
