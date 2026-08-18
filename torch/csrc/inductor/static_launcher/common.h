#pragma once

#include <c10/util/bit_cast.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
#include <cstdint>

namespace torch::inductor::static_launcher {

inline uint16_t unpackTritonFp16(PyObject* obj) {
  uint16_t bits = 0;
  if (PyFloat_Pack2(
          THPUtils_unpackDouble(obj), reinterpret_cast<char*>(&bits), 1) < 0) {
    throw python_error();
  }
  return bits;
}

inline uint16_t unpackTritonBf16(PyObject* obj) {
  float value = static_cast<float>(THPUtils_unpackDouble(obj));
  return static_cast<uint16_t>(c10::bit_cast<uint32_t>(value) >> 16);
}

} // namespace torch::inductor::static_launcher
