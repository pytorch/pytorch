#include "ThreadLocalPtr.h"
#include <algorithm>

namespace caffe2 {

// meyer's singleton
AllThreadLocalHelperVector* getAllThreadLocalHelperVector() {
  // leak the pointer to avoid dealing with destruction order issues
  static auto* instance = new AllThreadLocalHelperVector();
  return instance;
}

ThreadLocalHelper* getThreadLocalHelper() {
  static thread_local ThreadLocalHelper instance;
  return &instance;
}

// AllThreadLocalHelperVector

void AllThreadLocalHelperVector::push_back(ThreadLocalHelper* helper) {
  std::lock_guard<std::mutex> lg(mutex_);
  vector_.push_back(helper);
}

void AllThreadLocalHelperVector::erase(ThreadLocalHelper* helper) {
  std::lock_guard<std::mutex> lg(mutex_);
  vector_.erase(
      std::remove(vector_.begin(), vector_.end(), helper), vector_.end());
}

void AllThreadLocalHelperVector::erase_tlp(ThreadLocalPtrImpl* ptr) {
  std::lock_guard<std::mutex> lg(mutex_);
  for (auto* ins : vector_) {
    ins->erase(ptr);
  }
}

// ThreadLocalHelper
ThreadLocalHelper::ThreadLocalHelper() {
  getAllThreadLocalHelperVector()->push_back(this);
}

ThreadLocalHelper::~ThreadLocalHelper() {
  getAllThreadLocalHelperVector()->erase(this);
}

void ThreadLocalHelper::insert(
    ThreadLocalPtrImpl* tl_ptr,
    std::shared_ptr<void> ptr) {
  std::lock_guard<std::mutex> lg(mutex_);
  mapping_.insert(std::make_pair(tl_ptr, std::move(ptr)));
}

void* ThreadLocalHelper::get(ThreadLocalPtrImpl* key) {
  /* Grabbing the mutex for the thread local map protecting the case
   * when other object exits(~ThreadLocalPtrImpl()), and removes the
   * element in the map, which will change the iterator returned
   * by find.
   */
  std::lock_guard<std::mutex> lg(mutex_);
  auto it = mapping_.find(key);

  if (it == mapping_.end()) {
    return nullptr;
  } else {
    return it->second.get();
  }
}

void ThreadLocalHelper::erase(ThreadLocalPtrImpl* key) {
  std::lock_guard<std::mutex> lg(mutex_);
  mapping_.erase(key);
}

ThreadLocalPtrImpl::~ThreadLocalPtrImpl() {
  getAllThreadLocalHelperVector()->erase_tlp(this);
}

} // namespace caffe2
