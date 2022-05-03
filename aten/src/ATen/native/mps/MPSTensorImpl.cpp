//  Copyright © 2022 Apple Inc.

#include <ATen/native/mps/MPSTensorImpl.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/ArrayRef.h>

namespace at {
namespace native {
namespace mps {

using at::Device;
using at::DeviceType;

const char* MPSTensorImpl::tensorimpl_type_name() const {
  return "MPSTensorImpl";
}

const at::Storage& MPSTensorImpl::storage() const {
  return this->storage_;
}

void MPSTensorImpl::release_resources() {
  TensorImpl::release_resources();
}

} // namespace mps
} // namespace native
} // namespace at
