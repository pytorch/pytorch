#pragma once

#include <string>
#include <ATen/ThreadLocalDebugInfo.h>

namespace torch {

class MobileDebugInfo
    : public at::ThreadLocalDebugInfoBase {
public:
  const std::string& getModelName() {
    return model_name_;
  }

  void setModelName(const std::string& model_name) {
    model_name_ = model_name;
  }

  const std::string& getMethodName() {
    return method_name_;
  }

  void setMethodName(const std::string& method_name) {
    method_name_ = method_name;
  }

  size_t getOpIdx() {
    return op_idx_;
  }

  void setOpIdx(size_t op_idx) {
    op_idx_ = op_idx;
  }

  virtual ~MobileDebugInfo() {}

private:
  std::string model_name_;
  std::string method_name_;
  size_t op_idx_ = 0;
};

class MobileModuleObserver {
    public:
    virtual ~MobileModuleObserver() = default;

    virtual void onEnter(const std::string& model_name, const std::string& method_name) {}
    virtual void onExit() {}
};

class MobileObserverConfig {
  public:
  void setModuleObserver(std::unique_ptr<MobileModuleObserver> reporter) {
    module_observer_ = std::move(reporter);
  }
  MobileModuleObserver* getModuleObserver() {
    return module_observer_.get();
  }

private:
  std::unique_ptr<MobileModuleObserver> module_observer_;
};

MobileObserverConfig& observerConfig();

} // namespace torch
