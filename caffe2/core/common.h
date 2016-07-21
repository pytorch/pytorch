#ifndef CAFFE2_CORE_COMMON_H_
#define CAFFE2_CORE_COMMON_H_

#include <map>
#include <memory>
#include <set>
#include <string>
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

}  // namespace caffe2

#endif  // CAFFE2_CORE_COMMON_H_
