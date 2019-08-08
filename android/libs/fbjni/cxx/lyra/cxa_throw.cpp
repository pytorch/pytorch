/*
 *  Copyright (c) 2004-present, Facebook, Inc.
 *
 *  This source code is licensed under the MIT license found in the LICENSE
 *  file in the root directory of this source tree.
 *
 */
#include <atomic>
#include <stdexcept>
#include <cxxabi.h>
#include <unwind.h>
#include <cassert>

#include <lyra/lyra_exceptions.h>

namespace facebook {
namespace lyra {

namespace {
std::atomic<bool> enableBacktraces{true};
}

void enableCxaThrowHookBacktraces(bool enable) {
  enableBacktraces.store(enable, std::memory_order_relaxed);
}

[[gnu::noreturn]] void (*original_cxa_throw)(void*, const std::type_info*, void (*) (void *));

#if defined(_LIBCPP_VERSION)
[[noreturn]] void cxa_throw(void* obj, const std::type_info* type, void (*destructor) (void *)) {
  // lyra doesn't have support yet for libc++.
  original_cxa_throw(obj, type, destructor);
}
#else

using namespace detail;

namespace {

const auto traceHolderType =
  static_cast<const abi::__class_type_info*>(&typeid(ExceptionTraceHolder));

// lyra's __cxa_throw attaches stack trace information to thrown exceptions. It basically does:
//   1. capture stack trace
//   2. construct a new type_info struct that:
//     a. holds the ExceptionTraceHolder
//     b. supports upcasting to lyra::ExceptionTraceHolder* (by just returning the holder member)
//     c. acts like the original exception type_info otherwise
//   3. call original __cxa_throw() with original exception pointer, the
//      HijackedExceptionTypeInfo, and HijackedExceptionTypeInfo::destructor
//      (which will both delete the constructed type info and call the original
//      destructor).
struct HijackedExceptionTypeInfo : public abi::__class_type_info {
  HijackedExceptionTypeInfo(void* obj, const std::type_info* base, void(*destructor)(void*))
      : abi::__class_type_info{base->name()}, base_{base}, orig_dest_{destructor} {
  }

  bool __is_pointer_p() const override {
    return base_->__is_pointer_p();
  }

  bool __is_function_p() const override {
    return base_->__is_function_p();
  }

  bool __do_catch(const type_info *__thr_type, void **__thr_obj, unsigned __outer) const override {
    return base_->__do_catch(__thr_type, __thr_obj, __outer);
  }

  bool __do_upcast(const abi::__class_type_info *__target, void **__obj_ptr) const override {
    if (__target == traceHolderType) {
      *__obj_ptr = (void*)&stack_;
      return true;
    }
    return base_->__do_upcast(__target, __obj_ptr);
  }

  static void destructor(void* obj) {
    auto exc_ptr = reinterpret_cast<std::exception_ptr*>(&obj);
    auto info = reinterpret_cast<const::std::type_info*>(exc_ptr->__cxa_exception_type());
    auto mutable_info = static_cast<HijackedExceptionTypeInfo*>(const_cast<std::type_info*>(info));
    mutable_info->orig_dest_(obj);
    delete mutable_info;
  }

 private:
  const std::type_info* base_;
  void (*orig_dest_)(void*);
  ExceptionTraceHolder stack_;
};

} // namespace

[[noreturn]] void cxa_throw(void* obj, const std::type_info* type, void (*destructor) (void *)) {
  if (enableBacktraces.load(std::memory_order_relaxed)) {
    if (!type->__do_upcast(traceHolderType, &obj)) {
      type = new HijackedExceptionTypeInfo(obj, type, destructor);
      destructor = HijackedExceptionTypeInfo::destructor;
    }
  }
  original_cxa_throw(obj, type, destructor);
}

#endif // libc++

} // namespace lyra
} // namespace facebook
