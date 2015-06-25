#ifndef CAFFE2_CORE_TYPES_H_
#define CAFFE2_CORE_TYPES_H_

#include <string>

namespace caffe2 {

// Storage orders that are often used in the image applications.
enum StorageOrder {
  UNKNOWN = 0,
  NHWC = 1,
  NCHW = 2,
};

inline StorageOrder StringToStorageOrder(const string& str) {
  if (str == "NHWC") {
    return StorageOrder::NHWC;
  } else if (str == "NCHW") {
    return StorageOrder::NCHW;
  } else {
    return StorageOrder::UNKNOWN;
  }
}

}  // namespace caffe2

#endif  // CAFFE2_CORE_TYPES_H_
