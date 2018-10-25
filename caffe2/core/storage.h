#ifndef CAFFE2_CORE_STORAGE_H_
#define CAFFE2_CORE_STORAGE_H_

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "caffe2/core/allocator.h"
#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/typeid.h"

#include <ATen/core/Allocator.h>
#include <c10/Device.h>
#include <c10/DeviceType.h>
#include <ATen/core/intrusive_ptr.h>
#include <ATen/core/Storage.h>
#include <ATen/core/StorageImpl.h>

namespace caffe2 {

using StorageImpl = at::StorageImpl;
using Storage = at::Storage;

} // namespace caffe2

#endif // CAFFE2_CORE_STORAGE_H_
