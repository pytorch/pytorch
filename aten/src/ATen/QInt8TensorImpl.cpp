#include <ATen/ATen.h>
#include <ATen/QInt8TensorImpl.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/core/LegacyTypeDispatch.h>

#include <iostream>
namespace at {
/*
 * TODO: add qInt8DataTypeCheck Here
 */

QInt8TensorImpl::QInt8TensorImpl(Storage&& storage, at::TensorTypeId type_id)
    : TensorImpl(std::move(storage), type_id, false)
    , scale_(1.0)
    , zero_point_(0) {
    }

QInt8TensorImpl::QInt8TensorImpl(Storage&& storage, at::TensorTypeId type_id, float scale, int32_t zero_point)
    : TensorImpl(std::move(storage), type_id, false)
    , scale_(scale)
    , zero_point_(zero_point) {
    }

} // namespace c10
