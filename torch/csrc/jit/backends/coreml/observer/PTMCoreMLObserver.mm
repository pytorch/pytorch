#import <torch/csrc/jit/backends/coreml/observer/PTMCoreMLObserver.h>

PTMCoreMLObserverConfig::PTMCoreMLObserverConfig() : observer_{nullptr} {}

PTMCoreMLObserverConfig& coreMLObserverConfig() {
  static PTMCoreMLObserverConfig global_instance;
  return global_instance;
}
