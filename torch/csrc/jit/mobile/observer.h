#pragma once

#include <c10/util/ThreadLocalDebugInfo.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {

class MobileDebugInfo : public c10::DebugInfoBase {
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

 private:
  std::string model_name_;
  std::string method_name_;
  // TODO: Kimish
  // If we launch a thread such as for at::launch, interepter continuation
  // and if the caching allocator is enabled in the base thread
  // then, in order to propagate this information, that is caching allocator
  // is enabled, across thread boundaries we can use the mechanism provided
  // by ThreadLocalDebugInfo
  // Once the thread local MobileDebugInfo is accessible in the launched
  // thread, it can be accessed in that thread and that thread can set
  // its own thread local CachingAllocatorInfo.
  // However, we cannot expect every launched thread to extract and set
  // its own thread local copy of CachingAllocatorInfo.
  // But this can be done in lite interpreter, where in the run method
  // it can do info =
  // c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::MOBILE_RUNTIME_INFO))
  // .get_caching_allocator_info();
  // GetThreadLocalCachingAllocatorInfo() = info;
  // Other option is to have MobileDebugInfo itself be the place where thread
  // local copy of CachingAllocatorInfo is stored. Then
  // DefaultMobileCPUAllocator inspects this to decide if to use
  // CachingAllocator. However, current lite interpreter does not support FORK,
  // thus from the run method of lite interpreter we are not really gonna launch
  // another instance of lite interpreter in a different thread. So for now not
  // getting bothered about passing CachingAllocatorInfo across thread
  // boundaries. c10::CachingAllocatorInfo caching_allocator_info;
  size_t op_idx_ = 0;
};

class MobileModuleObserver {
 public:
  virtual ~MobileModuleObserver() = default;

  virtual void onEnterRunMethod(const int32_t) {}
  virtual void onExitRunMethod(
      const std::unordered_map<std::string, std::string>&,
      const std::string&,
      const int32_t) {}
  virtual void onFailRunMethod(
      const std::unordered_map<std::string, std::string>&,
      const std::string&,
      const int32_t,
      const char*) {}
  virtual void onEnterLoadModel(const int32_t) {}
  virtual void onExitLoadModel(
      const int32_t,
      const std::unordered_map<std::string, std::string>&) {
  } // key: filename, value: file content
  virtual void onFailLoadModel(const int32_t, const char*) {}
  virtual void onFailLoadModel(
      const int32_t,
      const char*,
      const std::unordered_map<std::string, std::string>&) {}
  virtual std::vector<std::string> getDefaultExtraFiles() = 0;
  virtual std::unordered_map<std::string, std::string> processMetadataFromExtra(
      const std::unordered_map<std::string, std::string>&) = 0;
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
