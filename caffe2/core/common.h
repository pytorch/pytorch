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

// Half float definition. Currently half float operators are mainly on CUDA
// gpus.
// The reason we do not directly use the cuda __half data type is because that
// requires compilation with nvcc. The float16 data type should be compatible
// with the cuda __half data type, but will allow us to refer to the data type
// without the need of cuda.
static_assert(sizeof(unsigned short) == 2,
              "Short on this platform is not 16 bit.");
typedef struct __f16 {
  unsigned short x;
} __attribute__((aligned(2))) float16;
static_assert(sizeof(float16) == 2,
              "This should always be satisfied - float16 safeguard.");

// Just in order to mark things as not implemented. Do not use in final code.
#define CAFFE_NOT_IMPLEMENTED CAFFE_LOG_FATAL << "Not Implemented."

// suppress an unused variable.
#define UNUSED_VARIABLE __attribute__((unused))

// Disable the copy and assignment operator for a class. Note that this will
// disable the usage of the class in std containers.
#define DISABLE_COPY_AND_ASSIGN(classname)                                     \
private:                                                                       \
  classname(const classname&) = delete;                                        \
  classname& operator=(const classname&) = delete

}  // namespace caffe2


namespace std {
template<>
struct is_fundamental<caffe2::__f16> : std::integral_constant<bool, true> {
};
}  // namespace std

#endif  // CAFFE2_CORE_COMMON_H_
