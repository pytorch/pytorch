#pragma once

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/csrc/Storage.h>
#include <torch/csrc/THConcat.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/utils/python_numbers.h>
#include <string>
#include <type_traits>
#include <vector>

#define THPUtils_(NAME) TH_CONCAT_4(THP, Real, Utils_, NAME)

#define THPUtils_typename(obj) (Py_TYPE(obj)->tp_name)

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define THP_EXPECT(x, y) (__builtin_expect((x), (y)))
#else
#define THP_EXPECT(x, y) (x)
#endif

#define THPUtils_unpackReal_FLOAT(object)           \
  (PyFloat_Check(object) ? PyFloat_AsDouble(object) \
       : PyLong_Check(object)                       \
       ? PyLong_AsLongLong(object)                  \
       : (throw std::runtime_error("Could not parse real"), 0))

#define THPUtils_checkReal_INT(object) PyLong_Check(object)

#define THPUtils_unpackReal_INT(object) \
  (PyLong_Check(object)                 \
       ? PyLong_AsLongLong(object)      \
       : (throw std::runtime_error("Could not parse real"), 0))

#define THPUtils_unpackReal_BOOL(object) \
  (PyBool_Check(object)                  \
       ? object                          \
       : (throw std::runtime_error("Could not parse real"), Py_False))

#define THPUtils_unpackReal_COMPLEX(object)                                   \
  (PyComplex_Check(object)                                                    \
       ? (c10::complex<double>(                                               \
             PyComplex_RealAsDouble(object), PyComplex_ImagAsDouble(object))) \
       : PyFloat_Check(object)                                                \
       ? (c10::complex<double>(PyFloat_AsDouble(object), 0))                  \
       : PyLong_Check(object)                                                 \
       ? (c10::complex<double>(PyLong_AsLongLong(object), 0))                 \
       : (throw std::runtime_error("Could not parse real"),                   \
          c10::complex<double>(0, 0)))

#define THPBoolUtils_unpackReal(object) THPUtils_unpackReal_BOOL(object)
#define THPBoolUtils_checkAccreal(object) THPUtils_checkReal_BOOL(object)
#define THPByteUtils_checkReal(object) THPUtils_checkReal_INT(object)
#define THPByteUtils_unpackReal(object) \
  (unsigned char)THPUtils_unpackReal_INT(object)

/*
   From https://github.com/python/cpython/blob/v3.7.0/Modules/xxsubtype.c
   If compiled as a shared library, some compilers don't allow addresses of
   Python objects defined in other libraries to be used in static PyTypeObject
   initializers. The DEFERRED_ADDRESS macro is used to tag the slots where such
   addresses appear; the module init function that adds the PyTypeObject to the
   module must fill in the tagged slots at runtime. The argument is for
   documentation -- the macro ignores it.
*/
#define DEFERRED_ADDRESS(ADDR) nullptr

TORCH_PYTHON_API void THPUtils_setError(const char* format, ...);
TORCH_PYTHON_API void THPUtils_invalidArguments(
    PyObject* given_args,
    PyObject* given_kwargs,
    const char* function_name,
    size_t num_options,
    ...);

bool THPUtils_checkIntTuple(PyObject* arg);
std::vector<int> THPUtils_unpackIntTuple(PyObject* arg);

TORCH_PYTHON_API void THPUtils_addPyMethodDefs(
    std::vector<PyMethodDef>& vector,
    const PyMethodDef* methods);

int THPUtils_getCallable(PyObject* arg, PyObject** result);

typedef THPPointer<THPGenerator> THPGeneratorPtr;
typedef class THPPointer<THPStorage> THPStoragePtr;

TORCH_PYTHON_API std::vector<int64_t> THPUtils_unpackLongs(PyObject* arg);
PyObject* THPUtils_dispatchStateless(
    PyObject* tensor,
    const char* name,
    PyObject* args,
    PyObject* kwargs);

template <typename _real, typename = void>
struct mod_traits {};

template <typename _real>
struct mod_traits<_real, std::enable_if_t<std::is_floating_point_v<_real>>> {
  static _real mod(_real a, _real b) {
    return fmod(a, b);
  }
};

template <typename _real>
struct mod_traits<_real, std::enable_if_t<std::is_integral_v<_real>>> {
  static _real mod(_real a, _real b) {
    return a % b;
  }
};

void setBackCompatBroadcastWarn(bool warn);
bool getBackCompatBroadcastWarn();

void setBackCompatKeepdimWarn(bool warn);
bool getBackCompatKeepdimWarn();
bool maybeThrowBackCompatKeepdimWarn(char* func);

void storage_fill(const at::Storage& self, uint8_t value);
void storage_set(const at::Storage& self, ptrdiff_t idx, uint8_t value);
uint8_t storage_get(const at::Storage& self, ptrdiff_t idx);

std::string uuid_to_string(const char* uuid_bytes);
