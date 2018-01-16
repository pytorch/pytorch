#pragma once

#include <ATen/ATen.h>
#include <Python.h>

#include "python_numbers.h"
#include "torch/csrc/Exceptions.h"

namespace torch { namespace utils {

inline void store_scalar(void* data, at::ScalarType scalarType, PyObject* obj) {
  switch (scalarType) {
    case at::kByte: *(uint8_t*)data = (uint8_t)THPUtils_unpackLong(obj); break;
    case at::kChar: *(char*)data = (char)THPUtils_unpackLong(obj); break;
    case at::kShort: *(int16_t*)data = (int16_t)THPUtils_unpackLong(obj); break;
    case at::kInt: *(int32_t*)data = (int32_t)THPUtils_unpackLong(obj); break;
    case at::kLong: *(int64_t*)data = THPUtils_unpackLong(obj); break;
    case at::kHalf:
      *(at::Half*)data = at::convert<at::Half, double>(THPUtils_unpackDouble(obj));
      break;
    case at::kFloat: *(float*)data = (float)THPUtils_unpackDouble(obj); break;
    case at::kDouble: *(double*)data = THPUtils_unpackDouble(obj); break;
    default: throw std::runtime_error("invalid type");
  }
}

inline PyObject* load_scalar(void* data, at::ScalarType scalarType) {
  switch (scalarType) {
    case at::kByte: return THPUtils_packInt64(*(uint8_t*)data);
    case at::kChar: return THPUtils_packInt64(*(char*)data);
    case at::kShort: return THPUtils_packInt64(*(int16_t*)data);
    case at::kInt: return THPUtils_packInt64(*(int32_t*)data);
    case at::kLong: return THPUtils_packInt64(*(int64_t*)data);
    case at::kHalf: return PyFloat_FromDouble(at::convert<double, at::Half>(*(at::Half*)data));
    case at::kFloat: return PyFloat_FromDouble(*(float*)data);
    case at::kDouble: return PyFloat_FromDouble(*(double*)data);
    default: throw std::runtime_error("invalid type");
  }
}

}}  // namespace torch::utils
