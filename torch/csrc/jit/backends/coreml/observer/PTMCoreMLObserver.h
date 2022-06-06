#include <memory>

class PTMCoreMLObserver {
 public:
  virtual ~PTMCoreMLObserver() = default;

  virtual size_t getRemainingMemory() {
    return 0;
  }

  virtual void onEnterCompileModel(const int32_t, const int32_t) {}
  virtual void onExitCompileModel(const int32_t, bool, bool) {}

  virtual void onEnterExecuteModel(
      const int32_t,
      const int32_t,
      const size_t,
      const int32_t) {}
  virtual void onExitExecuteModel(const int32_t, const int32_t, bool, bool) {}
};

class PTMCoreMLObserverConfig {
 public:
  PTMCoreMLObserverConfig();

  // Do not allow copying/moving.
  // There should be only one global instance of this class.
  PTMCoreMLObserverConfig(const PTMCoreMLObserverConfig&) = delete;
  PTMCoreMLObserverConfig& operator=(const PTMCoreMLObserverConfig&) = delete;

  PTMCoreMLObserverConfig(PTMCoreMLObserverConfig&&) = delete;
  PTMCoreMLObserverConfig& operator=(PTMCoreMLObserverConfig&&) = delete;

 private:
  std::unique_ptr<PTMCoreMLObserver> observer_;

 public:
  void setCoreMLObserver(std::unique_ptr<PTMCoreMLObserver> observer) {
    observer_ = std::move(observer);
  }

  PTMCoreMLObserver* getCoreMLObserver() {
    return observer_.get();
  }
};

PTMCoreMLObserverConfig& coreMLObserverConfig();
