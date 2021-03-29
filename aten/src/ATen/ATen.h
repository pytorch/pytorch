#pragma once

#if !defined(_MSC_VER) && __cplusplus < 201402L
#error C++14 or later compatible compiler is required to use ATen.
#endif

#include <c10/core/Allocator.h>
#include <ATen/core/ATenGeneral.h>
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
#include <c10/core/Layout.h>
#include <ATen/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/Reduction.h>
#include <c10/util/Exception.h>
#include <ATen/core/UnsafeFromTH.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
