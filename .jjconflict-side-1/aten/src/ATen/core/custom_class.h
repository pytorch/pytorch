#pragma once

#include <typeindex>
#include <memory>

#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

namespace c10 {

struct ClassType;
using ClassTypePtr = std::shared_ptr<ClassType>;

TORCH_API c10::ClassTypePtr getCustomClassTypeImpl(const std::type_index &tindex);

template <typename T>
const c10::ClassTypePtr& getCustomClassType() {
  // Classes are never unregistered from getCustomClassTypeMap and the
  // hash lookup can be a hot path, so just cache.
  // For the same reason, it's fine If this ends up getting duplicated across
  // DSO boundaries for whatever reason.
  static c10::ClassTypePtr cache = getCustomClassTypeImpl(
      std::type_index(typeid(T)));
  return cache;
}

}
