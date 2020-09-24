
#pragma once

#include <c10/util/Exception.h>

namespace torch {
namespace jit {
namespace fuser {

// Simple mixin for suppressing copy & move operations, ex:
//
//  class Foo : public NonCopyable {
//   ...
//  };
//
class NonCopyable {
 public:
  NonCopyable() = default;

  // No copy/move semantics
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
};

// A generic root for a hierarchy of polymorphic classes:
// - It ensures virtual destructors
// - Provides the base->as<Derived>() and node->isA<T>() notation
class PolymorphicBase {
 public:
  virtual ~PolymorphicBase() = default;

  // Replacement for static_cast<T*>(ptr): ptr->as<T>()
  // (checked in DEBUG builds)
  template <class T>
  T* as() {
#ifdef NDEBUG
    auto downcast_ptr = static_cast<T*>(this);
#else
    auto downcast_ptr = dynamic_cast<T*>(this);
    TORCH_INTERNAL_ASSERT(downcast_ptr != nullptr);
#endif
    return downcast_ptr;
  }

  template <class T>
  const T* as() const {
#ifdef NDEBUG
    auto downcast_ptr = static_cast<const T*>(this);
#else
    auto downcast_ptr = dynamic_cast<const T*>(this);
    TORCH_INTERNAL_ASSERT(downcast_ptr != nullptr);
#endif
    return downcast_ptr;
  }

  template <class T>
  bool isA() const {
    return dynamic_cast<const T*>(this) != nullptr;
  }
};

} // namespace fuser
} // namespace jit
} // namespace torch
