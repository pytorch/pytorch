#pragma once

#if !defined(_MSC_VER) && __cplusplus < 201703L
#error C++17 or later compatible compiler is required to use ATen.
#endif

#include <ATen/Context.h>
#include <ATen/Device.h>
#include <ATen/DeviceGuard.h>
#include <ATen/DimVector.h>
#include <ATen/Dispatch.h>
#include <ATen/Formatting.h>
#include <ATen/Functions.h>
#include <ATen/NamedTensor.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/TensorGeometry.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>
#include <ATen/Version.h>
#include <ATen/core/ATenGeneral.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Scalar.h>
#include <ATen/core/UnsafeFromTH.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/core/Allocator.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/Layout.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>

// TODO: try to remove this
// There is some back story, see https://github.com/pytorch/pytorch/issues/48684
#include <ATen/NativeFunctions.h>
