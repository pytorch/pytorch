#pragma once

/**
 * Android versions with libgnustl incorrectly handle thread_local C++
 * qualifier with composite types. NDK up to r17 version is affected.
 *
 * (A fix landed on Jun 4 2018:
 * https://android-review.googlesource.com/c/toolchain/gcc/+/683601)
 *
 * In such cases, use c10::ThreadLocal<T> wrapper
 * which is `pthread_*` based with smart pointer semantics.
 *
 * In addition, convenient macro C10_DEFINE_TLS_static is available.
 * To define static TLS variable of type std::string, do the following
 * ```
 *  C10_DEFINE_TLS_static(std::string, str_tls_);
 *  ///////
 *  {
 *    *str_tls_ = "abc";
 *    assert(str_tls_->length(), 3);
 *  }
 * ```
 *
 * (see c10/test/util/ThreadLocal_test.cpp for more examples)
 */
#if !defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)

#if defined(C10_ANDROID) && defined(__GLIBCXX__) && __GLIBCXX__ < 20180604
#define C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE
#endif // defined(C10_ANDROID) && defined(__GLIBCXX__) && __GLIBCXX__ < 20180604

#endif // !defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)

#if defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
#include <c10/util/Exception.h>
#include <pthread.h>
#include <memory>
namespace c10 {

/**
 * @brief Temporary thread_local C++ qualifier replacement for Android
 * based on `pthread_*`.
 * To be used with composite types that provide default ctor.
 */
template <typename Type>
class ThreadLocal {
 public:
  ThreadLocal() {
    pthread_key_create(
        &key_, [](void* buf) { delete static_cast<Type*>(buf); });
  }

  ~ThreadLocal() {
    if (void* current = pthread_getspecific(key_)) {
      delete static_cast<Type*>(current);
    }

    pthread_key_delete(key_);
  }

  ThreadLocal(const ThreadLocal&) = delete;
  ThreadLocal& operator=(const ThreadLocal&) = delete;

  Type& get() {
    if (void* current = pthread_getspecific(key_)) {
      return *static_cast<Type*>(current);
    }

    std::unique_ptr<Type> ptr = std::make_unique<Type>();
    if (0 == pthread_setspecific(key_, ptr.get())) {
      return *ptr.release();
    }

    int err = errno;
    TORCH_INTERNAL_ASSERT(false, "pthread_setspecific() failed, errno = ", err);
  }

  Type& operator*() {
    return get();
  }

  Type* operator->() {
    return &get();
  }

 private:
  pthread_key_t key_;
};

} // namespace c10

#define C10_DEFINE_TLS_static(Type, Name) static ::c10::ThreadLocal<Type> Name

#define C10_DECLARE_TLS_class_static(Class, Type, Name) \
  static ::c10::ThreadLocal<Type> Name

#define C10_DEFINE_TLS_class_static(Class, Type, Name) \
  ::c10::ThreadLocal<Type> Class::Name

#else // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)

namespace c10 {

/**
 * @brief Default thread_local implementation for non-Android cases.
 * To be used with composite types that provide default ctor.
 */
template <typename Type>
class ThreadLocal {
 public:
  using Accessor = Type* (*)();
  explicit ThreadLocal(Accessor accessor) : accessor_(accessor) {}

  ThreadLocal(const ThreadLocal&) = delete;
  ThreadLocal& operator=(const ThreadLocal&) = delete;

  Type& get() {
    return *accessor_();
  }

  Type& operator*() {
    return get();
  }

  Type* operator->() {
    return &get();
  }

 private:
  Accessor accessor_;
};

} // namespace c10

#define C10_DEFINE_TLS_static(Type, Name)     \
  static ::c10::ThreadLocal<Type> Name([]() { \
    static thread_local Type var;             \
    return &var;                              \
  })

#define C10_DECLARE_TLS_class_static(Class, Type, Name) \
  static ::c10::ThreadLocal<Type> Name

#define C10_DEFINE_TLS_class_static(Class, Type, Name) \
  ::c10::ThreadLocal<Type> Class::Name([]() {          \
    static thread_local Type var;                      \
    return &var;                                       \
  })

#endif // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
