#ifndef CAFFE2_CORE_COMMON_H_
#define CAFFE2_CORE_COMMON_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <sstream>
#include <type_traits>
#include <vector>

namespace caffe2 {

// Note(Yangqing): NVCC does not play well with unordered_map on some platforms,
// forcing us to use std::map instead of unordered_map. This may affect speed
// in some cases, but in most of the computation code we do not access map very
// often, so it should be fine for us. I am putting a CaffeMap alias so we can
// change it more easily if things work out for unordered_map down the road.
template <typename Key, typename Value>
using CaffeMap = std::map<Key, Value>;
// using CaffeMap = std::unordered_map;

// Using statements for common classes that we refer to in caffe2 very often.
// Note that we only place it inside caffe2 so the global namespace is not
// polluted.
/* using override */
using std::set;
using std::string;
using std::unique_ptr;
using std::vector;

// Just in order to mark things as not implemented. Do not use in final code.
#define CAFFE_NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented."

// suppress an unused variable.
#define UNUSED_VARIABLE __attribute__((unused))

// Disable the copy and assignment operator for a class. Note that this will
// disable the usage of the class in std containers.
#ifndef DISABLE_COPY_AND_ASSIGN
#define DISABLE_COPY_AND_ASSIGN(classname)                              \
private:                                                                       \
  classname(const classname&) = delete;                                        \
  classname& operator=(const classname&) = delete
#endif

// make_unique is a C++14 feature. If we don't have 14, we will emulate
// its behavior. This is copied from folly/Memory.h
#if __cplusplus >= 201402L ||                                              \
    (defined __cpp_lib_make_unique && __cpp_lib_make_unique >= 201304L) || \
    (defined(_MSC_VER) && _MSC_VER >= 1900)
/* using override */ using std::make_unique;
#else

template<typename T, typename... Args>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Allows 'make_unique<T[]>(10)'. (N3690 s20.9.1.4 p3-4)
template<typename T>
typename std::enable_if<std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(const size_t n) {
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}

// Disallows 'make_unique<T[10]>()'. (N3690 s20.9.1.4 p5)
template<typename T, typename... Args>
typename std::enable_if<
  std::extent<T>::value != 0, std::unique_ptr<T>>::type
make_unique(Args&&...) = delete;

#endif

// to_string implementation for Android related stuff.
#ifndef __ANDROID__
using std::to_string;
#else 
template <typename T>
std::string to_string(T value)
{
  std::ostringstream os;
  os << value;
  return os.str();
}
#endif

// dynamic cast reroute: if RTTI is disabled, go to reinterpret_cast
template <typename Dst, typename Src>
inline Dst dynamic_cast_if_rtti(Src ptr) {
#ifdef __GXX_RTTI
  return dynamic_cast<Dst>(ptr);
#else
  return reinterpret_cast<Dst>(ptr);
#endif
}

}  // namespace caffe2

#endif  // CAFFE2_CORE_COMMON_H_
