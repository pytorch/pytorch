#ifndef PTM_COREML_Context_h
#define PTM_COREML_Context_h

#include <atomic>
#include <string>

namespace torch {
namespace jit {
namespace mobile {
namespace coreml {

struct ContextInterface {
  virtual ~ContextInterface() = default;
  virtual void setModelCacheDirectory(std::string path) = 0;
};

class BackendRegistrar {
 public:
  explicit BackendRegistrar(ContextInterface* ctx);
};

void setModelCacheDirectory(std::string path);

} // namespace coreml
} // namespace mobile
} // namespace jit
} // namespace torch

#endif
