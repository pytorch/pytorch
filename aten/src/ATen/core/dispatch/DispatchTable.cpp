#include <ATen/core/dispatch/DispatchTable.h>

#include <sstream>

namespace c10 {

namespace impl {
  std::string KernelFunctionTable::dumpState() const {
    std::ostringstream oss;
    for (uint8_t i = 0; i < static_cast<uint8_t>(DispatchKey::NumDispatchKeys); i++) {
      if (kernels_[i].isValid()) oss << "  " << kernels_[i].dumpState() << "\n";
    }
    return oss.str();
  }
}

std::string DispatchTable::dumpState() const {
  std::ostringstream oss;
  oss << "table:\n";
  oss << kernels_.dumpState();
  oss << "  catchall: " << catchallKernel_.dumpState() << "\n";
  oss << "  extractor: " << dispatchKeyExtractor_.dumpState() << "\n";
  oss << "  name: " << operatorName_ << "\n";
  return oss.str();
}

} // namespace c10
