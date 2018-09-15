#pragma once

#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include "caffe2/core/logging.h"

namespace caffe2 {

/**
 * thread_local pointer in C++ is a per thread pointer. However, sometimes
 * we want to have a thread local state that is per thread and also per
 * instance. e.g. we have the following class:
 * class A {
 *   ThreadLocalPtr<int> x;
 * }
 * We would like to have a copy of x per thread and also per instance of class A
 * This can be applied to storing per instance thread local state of some class,
 * when we could have multiple instances of the class in the same thread.
 * We implemented a subset of functions in folly::ThreadLocalPtr that's enough
 * to support BlackBoxPredictor.
 */

class ThreadLocalPtrImpl;
class ThreadLocalHelper;

/**
 * Map of object pointer to instance in each thread
 * to achieve per thread(using thread_local) per object(using the map)
 * thread local pointer
 */
typedef std::unordered_map<ThreadLocalPtrImpl*, std::shared_ptr<void>>
    UnsafeThreadLocalMap;

ThreadLocalHelper* getThreadLocalHelper();

typedef std::vector<ThreadLocalHelper*> UnsafeAllThreadLocalHelperVector;

/**
 * A thread safe vector of all ThreadLocalHelper, this will be used
 * to encapuslate the locking in the APIs for the changes to the global
 * AllThreadLocalHelperVector instance.
 */
class AllThreadLocalHelperVector {
 public:
  AllThreadLocalHelperVector() {}

  // Add a new ThreadLocalHelper to the vector
  void push_back(ThreadLocalHelper* helper);

  // Erase a ThreadLocalHelper to the vector
  void erase(ThreadLocalHelper* helper);

  // Erase object in all the helpers stored in vector
  // Called during destructor of a ThreadLocalPtrImpl
  void erase_tlp(ThreadLocalPtrImpl* ptr);

 private:
  UnsafeAllThreadLocalHelperVector vector_;
  std::mutex mutex_;
};

/**
 * ThreadLocalHelper is per thread
 */
class ThreadLocalHelper {
 public:
  ThreadLocalHelper();

  // When the thread dies, we want to clean up *this*
  // in AllThreadLocalHelperVector
  ~ThreadLocalHelper();

  // Insert a (object, ptr) pair into the thread local map
  void insert(ThreadLocalPtrImpl* tl_ptr, std::shared_ptr<void> ptr);
  // Get the ptr by object
  void* get(ThreadLocalPtrImpl* key);
  // Erase the ptr associated with the object in the map
  void erase(ThreadLocalPtrImpl* key);

 private:
  // mapping of object -> ptr in each thread
  UnsafeThreadLocalMap mapping_;
  std::mutex mutex_;
}; // ThreadLocalHelper

/** ThreadLocalPtrImpl is per object
 */
class ThreadLocalPtrImpl {
 public:
  ThreadLocalPtrImpl() {}
  // Delete copy and move constructors
  ThreadLocalPtrImpl(const ThreadLocalPtrImpl&) = delete;
  ThreadLocalPtrImpl(ThreadLocalPtrImpl&&) = delete;
  ThreadLocalPtrImpl& operator=(const ThreadLocalPtrImpl&) = delete;
  ThreadLocalPtrImpl& operator=(const ThreadLocalPtrImpl&&) = delete;

  // In the case when object dies first, we want to
  // clean up the states in all child threads
  ~ThreadLocalPtrImpl();

  template <typename T>
  T* get() {
    return static_cast<T*>(getThreadLocalHelper()->get(this));
  }

  template <typename T>
  void reset(T* newPtr = nullptr) {
    VLOG(2) << "In Reset(" << newPtr << ")";
    auto* wrapper = getThreadLocalHelper();
    // Cleaning up the objects(T) stored in the ThreadLocalPtrImpl in the thread
    wrapper->erase(this);
    if (newPtr != nullptr) {
      std::shared_ptr<void> sharedPtr(newPtr);
      // Deletion of newPtr is handled by shared_ptr
      // as it implements type erasure
      wrapper->insert(this, std::move(sharedPtr));
    }
  }

}; // ThreadLocalPtrImpl

template <typename T>
class ThreadLocalPtr {
 public:
  auto* operator-> () {
    return get();
  }

  auto& operator*() {
    return *get();
  }

  auto* get() {
    return impl_.get<T>();
  }

  auto* operator-> () const {
    return get();
  }

  auto& operator*() const {
    return *get();
  }

  auto* get() const {
    return impl_.get<T>();
  }

  void reset(unique_ptr<T> ptr = nullptr) {
    impl_.reset<T>(ptr.release());
  }

 private:
  ThreadLocalPtrImpl impl_;
};

} // namespace caffe2
