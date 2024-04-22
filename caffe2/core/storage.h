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
#include <c10/util/typeid.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>

namespace caffe2 {

using StorageImpl = at::StorageImpl;
using Storage = at::Storage;

} // namespace caffe2

#endif // CAFFE2_CORE_STORAGE_H_
