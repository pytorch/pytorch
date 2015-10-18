#ifndef CAFFE2_CORE_COMMON_H_
#define CAFFE2_CORE_COMMON_H_

#include <memory>
#include <string>
#include <map>
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
using std::string;
using std::unique_ptr;
using std::vector;

// Just in order to mark things as not implemented. Do not use in final code.
#define NOT_IMPLEMENTED CAFFE_LOG_FATAL << "Not Implemented."

// suppress an unused variable.
#define UNUSED_VARIABLE __attribute__((unused))

// Disable the copy and assignment operator for a class. Note that this will
// disable the usage of the class in std containers.
#define DISABLE_COPY_AND_ASSIGN(classname)                                     \
private:                                                                       \
  classname(const classname&);                                                 \
  classname& operator=(const classname&)


/**
 * Gets the gradient name for a blob.
 *
 * This is a legacy function that I placed here in the initial refactoring. In
 * retrospect, the gradient name probably should not be a first-class citizen
 * especially in a file as common as common.h. As a result it is very likely
 * that we will remove it or move it to a more proper location.
 */
inline string GetGradientName(const string& name) {
  return name + ".grad";
}

}  // namespace caffe2
#endif  // CAFFE2_CORE_COMMON_H_
