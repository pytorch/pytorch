// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#pragma once

#include <ATen/ATen.h>
#include <NvInfer.h>
#include <stdexcept>
#include <c10/core/ScalarType.h>

namespace fx2trt::common {

inline c10::ScalarType fieldTypeToScalarType(
    const nvinfer1::PluginFieldType ftype) {
  switch (ftype) {
    case nvinfer1::PluginFieldType::kFLOAT32: {
      return c10::kFloat;
    }
    case nvinfer1::PluginFieldType::kFLOAT16: {
      return c10::kHalf;
    }
    case nvinfer1::PluginFieldType::kINT32: {
      return c10::kInt;
    }
    case nvinfer1::PluginFieldType::kINT8: {
      return c10::kChar;
    }
    default:
      throw std::invalid_argument(
          "No corresponding datatype for plugin field type");
  }
}

} // namespace fx2trt::common
