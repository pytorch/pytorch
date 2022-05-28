#import <Foundation/Foundation.h>

#import <torch/csrc/jit/backends/backend.h>

namespace torch {
namespace jit {
namespace mobile {
namespace coreml {

struct PTMCoreMLContext : public ContextInterface {
  void setModelCacheDirectory(std::string dir) override {
    NSString* directory = [NSString stringWithCString:dir.c_str()];
    [PTMCoreMLCompiler setModelCacheDirectory:directory];
  }
};

API_AVAILABLE(ios(11.0), macos(10.13))
static BackendRegistrar g_coreml_backend(new PTMCoreMLContext());

} // namespace
}
}
}
