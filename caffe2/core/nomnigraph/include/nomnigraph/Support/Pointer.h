//===- nomnigraph/Support/Pointer.h - Smart pointer helpers -----*- C++ -*-===//
//
// TODO Licensing.
//
//===----------------------------------------------------------------------===//
//
// This file defines a C++11 compatible make_unique
//
//===----------------------------------------------------------------------===//

#ifndef NOM_SUPPORT_POINTER_H
#define NOM_SUPPORT_POINTER_H

#include <memory>

namespace nom {
namespace util {

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

} // namespace util
} // namespace nom

#endif // NOM_SUPPORT_POINTER_H
