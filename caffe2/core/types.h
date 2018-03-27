#ifndef CAFFE2_CORE_TYPES_H_
#define CAFFE2_CORE_TYPES_H_

#include <cstdint>
#include <string>
#include <type_traits>

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/typeid.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

// Storage orders that are often used in the image applications.
enum StorageOrder {
  UNKNOWN = 0,
  NHWC = 1,
  NCHW = 2,
};

inline StorageOrder StringToStorageOrder(const string& str) {
  if (str == "NHWC" || str == "nhwc") {
    return StorageOrder::NHWC;
  } else if (str == "NCHW" || str == "nchw") {
    return StorageOrder::NCHW;
  } else {
    LOG(ERROR) << "Unknown storage order string: " << str;
    return StorageOrder::UNKNOWN;
  }
}

inline constexpr char NameScopeSeparator() { return '/'; }

// From TypeMeta to caffe2::DataType protobuffer enum.
TensorProto::DataType TypeMetaToDataType(const TypeMeta& meta);

// From caffe2::DataType protobuffer enum to TypeMeta
const TypeMeta& DataTypeToTypeMeta(const TensorProto::DataType& dt);

}  // namespace caffe2

///////////////////////////////////////////////////////////////////////////////
// Half float definition. Currently half float operators are mainly on CUDA
// gpus.
// The reason we do not directly use the cuda __half data type is because that
// requires compilation with nvcc. The float16 data type should be compatible
// with the cuda __half data type, but will allow us to refer to the data type
// without the need of cuda.
static_assert(sizeof(unsigned short) == 2,
              "Short on this platform is not 16 bit.");
namespace caffe2 {
typedef struct CAFFE2_ALIGNED(2) __f16 { uint16_t x; } float16;

// Helpers to avoid using typeinfo with -rtti
template <typename T>
inline bool fp16_type();
// explicit instantation for float16 defined in types.cc.
template <>
inline bool fp16_type<float16>() {
  return true;
}
// The rest.
template <typename T>
inline bool fp16_type() {
  return false;
}

}  // namespace caffe2

// Make __f16 a fundamental type.
namespace std {
template<>
struct is_fundamental<caffe2::__f16> : std::integral_constant<bool, true> {
};
}  // namespace std

#endif  // CAFFE2_CORE_TYPES_H_
