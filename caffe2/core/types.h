#ifndef CAFFE2_CORE_TYPES_H_
#define CAFFE2_CORE_TYPES_H_

#include <string>

#include "caffe2/core/logging.h"

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
    CAFFE_LOG_ERROR << "Unknown storage order string: " << str;
    return StorageOrder::UNKNOWN;
  }
}

}  // namespace caffe2

#endif  // CAFFE2_CORE_TYPES_H_
