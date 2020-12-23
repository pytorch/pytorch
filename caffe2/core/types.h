#ifndef CAFFE2_CORE_TYPES_H_
#define CAFFE2_CORE_TYPES_H_

#include <cstdint>
#include <string>
#include <type_traits>

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include <c10/util/typeid.h>
#include "caffe2/proto/caffe2_pb.h"
#include <c10/util/Half.h>

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

inline int32_t GetDimFromOrderString(const std::string& str) {
  auto order = StringToStorageOrder(str);
  switch (order) {
    case StorageOrder::NHWC:
      return 3;
    case StorageOrder::NCHW:
      return 1;
    default:
      CAFFE_THROW("Unsupported storage order: ", str);
      return -1;
  }
}

inline constexpr char NameScopeSeparator() { return '/'; }

// From TypeMeta to caffe2::DataType protobuffer enum.
TORCH_API TensorProto::DataType TypeMetaToDataType(const TypeMeta meta);

// From caffe2::DataType protobuffer enum to TypeMeta
TORCH_API const TypeMeta DataTypeToTypeMeta(const TensorProto::DataType& dt);

}  // namespace caffe2

///////////////////////////////////////////////////////////////////////////////
// at::Half is defined in c10/util/Half.h. Currently half float operators are
// mainly on CUDA gpus.
// The reason we do not directly use the cuda __half data type is because that
// requires compilation with nvcc. The float16 data type should be compatible
// with the cuda __half data type, but will allow us to refer to the data type
// without the need of cuda.
static_assert(sizeof(unsigned short) == 2,
              "Short on this platform is not 16 bit.");
namespace caffe2 {
// Helpers to avoid using typeinfo with -rtti
template <typename T>
inline bool fp16_type();

template <>
inline bool fp16_type<at::Half>() {
  return true;
}

template <typename T>
inline bool fp16_type() {
  return false;
}

}  // namespace caffe2

#endif  // CAFFE2_CORE_TYPES_H_
