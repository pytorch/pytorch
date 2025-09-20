#pragma once

#include <ATen/ATen.h>
#include <c10/util/TypeCast.h>
#include <torch/csrc/python_headers.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_numbers.h>

namespace torch::utils {

template <typename T>
inline T unpackIntegral(PyObject* obj, const char* type) {
#if PY_VERSION_HEX >= 0x030a00f0
  // In Python-3.10 floats can no longer be silently converted to integers
  // Keep backward compatible behavior for now
  if (PyFloat_Check(obj)) {
    return c10::checked_convert<T>(THPUtils_unpackDouble(obj), type);
  }
  return c10::checked_convert<T>(THPUtils_unpackLong(obj), type);
#else
  return static_cast<T>(THPUtils_unpackLong(obj));
#endif
}

inline void store_scalar(void* data, at::ScalarType scalarType, PyObject* obj) {
  switch (scalarType) {
    case at::kByte:
      *(uint8_t*)data = unpackIntegral<uint8_t>(obj, "uint8");
      break;
    case at::kUInt16:
      *(uint16_t*)data = unpackIntegral<uint16_t>(obj, "uint16");
      break;
    case at::kUInt32:
      *(uint32_t*)data = unpackIntegral<uint32_t>(obj, "uint32");
      break;
    case at::kUInt64:
      // NB: This doesn't allow implicit conversion of float to int
      *(uint64_t*)data = THPUtils_unpackUInt64(obj);
      break;
    case at::kChar:
      *(int8_t*)data = unpackIntegral<int8_t>(obj, "int8");
      break;
    case at::kShort:
      *(int16_t*)data = unpackIntegral<int16_t>(obj, "int16");
      break;
    case at::kInt:
      *(int32_t*)data = unpackIntegral<int32_t>(obj, "int32");
      break;
    case at::kLong:
      *(int64_t*)data = unpackIntegral<int64_t>(obj, "int64");
      break;
    case at::kHalf:
      *(at::Half*)data =
          at::convert<at::Half, double>(THPUtils_unpackDouble(obj));
      break;
    case at::kFloat:
      *(float*)data = (float)THPUtils_unpackDouble(obj);
      break;
    case at::kDouble:
      *(double*)data = THPUtils_unpackDouble(obj);
      break;
    case at::kComplexHalf:
      *(c10::complex<at::Half>*)data =
          (c10::complex<at::Half>)static_cast<c10::complex<float>>(
              THPUtils_unpackComplexDouble(obj));
      break;
    case at::kComplexFloat:
      *(c10::complex<float>*)data =
          (c10::complex<float>)THPUtils_unpackComplexDouble(obj);
      break;
    case at::kComplexDouble:
      *(c10::complex<double>*)data = THPUtils_unpackComplexDouble(obj);
      break;
    case at::kBool:
      *(bool*)data = THPUtils_unpackNumberAsBool(obj);
      break;
    case at::kBFloat16:
      *(at::BFloat16*)data =
          at::convert<at::BFloat16, double>(THPUtils_unpackDouble(obj));
      break;
    // TODO(#146647): simplify below with macros
    case at::kFloat8_e5m2:
      *(at::Float8_e5m2*)data =
          at::convert<at::Float8_e5m2, double>(THPUtils_unpackDouble(obj));
      break;
    case at::kFloat8_e5m2fnuz:
      *(at::Float8_e5m2fnuz*)data =
          at::convert<at::Float8_e5m2fnuz, double>(THPUtils_unpackDouble(obj));
      break;
    case at::kFloat8_e4m3fn:
      *(at::Float8_e4m3fn*)data =
          at::convert<at::Float8_e4m3fn, double>(THPUtils_unpackDouble(obj));
      break;
    case at::kFloat8_e4m3fnuz:
      *(at::Float8_e4m3fnuz*)data =
          at::convert<at::Float8_e4m3fnuz, double>(THPUtils_unpackDouble(obj));
      break;
    case at::kFloat8_e8m0fnu:
      *(at::Float8_e8m0fnu*)data =
          at::convert<at::Float8_e8m0fnu, double>(THPUtils_unpackDouble(obj));
      break;
    default:
      TORCH_CHECK(false, "store_scalar: invalid type");
  }
}

inline PyObject* load_scalar(const void* data, at::ScalarType scalarType) {
  switch (scalarType) {
    case at::kByte:
      return THPUtils_packInt64(*(uint8_t*)data);
    case at::kUInt16:
      return THPUtils_packInt64(*(uint16_t*)data);
    case at::kUInt32:
      return THPUtils_packUInt32(*(uint32_t*)data);
    case at::kUInt64:
      return THPUtils_packUInt64(*(uint64_t*)data);
    case at::kChar:
      return THPUtils_packInt64(*(int8_t*)data);
    case at::kShort:
      return THPUtils_packInt64(*(int16_t*)data);
    case at::kInt:
      return THPUtils_packInt64(*(int32_t*)data);
    case at::kLong:
      return THPUtils_packInt64(*(int64_t*)data);
    case at::kHalf:
      return PyFloat_FromDouble(
          at::convert<double, at::Half>(*(at::Half*)data));
    case at::kFloat:
      return PyFloat_FromDouble(*(float*)data);
    case at::kDouble:
      return PyFloat_FromDouble(*(double*)data);
    case at::kComplexHalf: {
      auto data_ = reinterpret_cast<const c10::complex<at::Half>*>(data);
      return PyComplex_FromDoubles(data_->real(), data_->imag());
    }
    case at::kComplexFloat: {
      auto data_ = reinterpret_cast<const c10::complex<float>*>(data);
      return PyComplex_FromDoubles(data_->real(), data_->imag());
    }
    case at::kComplexDouble:
      return PyComplex_FromCComplex(
          *reinterpret_cast<Py_complex*>((c10::complex<double>*)data));
    case at::kBool:
      // Don't use bool*, since it may take out-of-range byte as bool.
      // Instead, we cast explicitly to avoid ASAN error.
      return PyBool_FromLong(static_cast<bool>(*(uint8_t*)data));
    case at::kBFloat16:
      return PyFloat_FromDouble(
          at::convert<double, at::BFloat16>(*(at::BFloat16*)data));
    // TODO(#146647): simplify below with macros
    case at::kFloat8_e5m2:
      return PyFloat_FromDouble(
          at::convert<double, at::Float8_e5m2>(*(at::Float8_e5m2*)data));
    case at::kFloat8_e4m3fn:
      return PyFloat_FromDouble(
          at::convert<double, at::Float8_e4m3fn>(*(at::Float8_e4m3fn*)data));
    case at::kFloat8_e5m2fnuz:
      return PyFloat_FromDouble(at::convert<double, at::Float8_e5m2fnuz>(
          *(at::Float8_e5m2fnuz*)data));
    case at::kFloat8_e4m3fnuz:
      return PyFloat_FromDouble(at::convert<double, at::Float8_e4m3fnuz>(
          *(at::Float8_e4m3fnuz*)data));
    case at::kFloat8_e8m0fnu:
      return PyFloat_FromDouble(
          at::convert<double, at::Float8_e8m0fnu>(*(at::Float8_e8m0fnu*)data));
    default:
      TORCH_CHECK(false, "load_scalar: invalid type");
  }
}

} // namespace torch::utils
