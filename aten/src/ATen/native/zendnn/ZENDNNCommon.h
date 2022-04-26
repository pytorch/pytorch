#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

#if AT_ZENDNN_ENABLED()
#include <ATen/native/zendnn/AbstractTypes.hpp>
#include <ATen/native/zendnn/Tensor.hpp>

namespace at {
namespace native {

// Mapping ScalarType to zendnn tensor data_type
zendnn::tensor::data_type get_zendnn_dtype(ScalarType type);

// Construct aten ZENDNN tensor given an zendnn tensor
Tensor new_with_itensor_zendnn(
    zendnn::tensor&& it,
    c10::optional<ScalarType> dtype,
    c10::optional<Device> device);

// Retrieve `zendnn::tensor` from ZENDNN tensor
zendnn::tensor& itensor_from_zendnn(const Tensor& zendnn_tensor);

// Construct an `zendnn::tensor` "view" from dense tensor, note the
// zendnn::tensor will share the underlying buffer
zendnn::tensor itensor_view_from_dense(const Tensor& tensor);

// Helper function for getting an zendnn tensor out of an aten Tensor or ZENDNN
// tensor.
zendnn::tensor itensor_from_tensor(const Tensor& tensor);

// helper functions for pytorch to zendnn tensor conversion without wrapping
// into abstract tensor
zendnn::tensor zendnn_tensor_view_from_dense(const Tensor& ttensor);
Tensor new_dense_from_zendnn(
    const zendnn::tensor& zendnn_tensor,
    const TensorOptions& options);

} // namespace native
} // namespace at

#endif // AT_ZENDNN_ENABLED
