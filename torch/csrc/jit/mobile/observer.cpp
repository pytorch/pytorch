#include <torch/csrc/jit/mobile/observer.h>

namespace torch {

MobileObserverConfig& observerConfig() {
  static MobileObserverConfig instance;
  return instance;
}

} // namespace torch
