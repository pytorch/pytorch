#pragma once

#include <typeindex>
#include <memory>
#include <unordered_map>

#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/python_stub.h>

namespace c10 {

struct ClassType;
using ClassTypePtr = std::shared_ptr<ClassType>;

TORCH_API ska::flat_hash_map<std::type_index, c10::ClassTypePtr>&
getCustomClassTypeMap();

template <typename T>
c10::ClassTypePtr getCustomClassTypeImpl() {
  auto& tmap = c10::getCustomClassTypeMap();
  auto tindex = std::type_index(typeid(T));
  auto res = tmap.find(tindex);
  if (C10_UNLIKELY(res == tmap.end())) {
    // type_index is not guaranteed to be unique across shared libraries on some platforms
    // For example see https://github.com/llvm-mirror/libcxx/blob/78d6a7767ed57b50122a161b91f59f19c9bd0d19/include/typeinfo#L133
    // Also, this is not the case if RTLD_LOCAL option is used, see
    // https://github.com/pybind/pybind11/blob/f791dc8648e1f6ec33f402d679b6b116a76d4e1b/include/pybind11/detail/internals.h#L101-L106
    // Take a slow path of iterating over all registered types and compare their names
    auto class_name = std::string(tindex.name());
    for(const auto &it: tmap) {
      if (class_name == it.first.name()) {
          // Do not modify existing type map here as this template is supposed to be called only once per type
          // from getCustomClassTypeImpl()
          return it.second;
      }
    }
    TORCH_CHECK(false, "Can't find class id in custom class type map for ", tindex.name());
  }
  return res->second;
}

template <typename T>
const c10::ClassTypePtr& getCustomClassType() {
  // Classes are never unregistered from getCustomClassTypeMap and the
  // hash lookup can be a hot path, so just cache.
  // For the same reason, it's fine If this ends up getting duplicated across
  // DSO boundaries for whatever reason.
  static c10::ClassTypePtr cache = getCustomClassTypeImpl<T>();
  return cache;
}

TORCH_API std::unordered_map<std::string, std::function<PyObject*(void*)>>&
getClassConverter();
}
