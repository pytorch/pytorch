#ifndef THP_UTILS_H
#define THP_UTILS_H

#include <vector>
#include <string>
#include <type_traits>
#include <ATen/ATen.h>
#include <torch/csrc/THConcat.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_compat.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

#define THPUtils_(NAME) TH_CONCAT_4(THP,Real,Utils_,NAME)

#define THPUtils_typename(obj) (Py_TYPE(obj)->tp_name)

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define THP_EXPECT(x, y) (__builtin_expect((x), (y)))
#else
#define THP_EXPECT(x, y) (x)
#endif

#define THPUtils_checkReal_FLOAT(object)                                       \
    (PyFloat_Check(object) || PyLong_Check(object))

#define THPUtils_unpackReal_FLOAT(object)                                      \
    (PyFloat_Check(object) ? PyFloat_AsDouble(object) :                        \
    PyLong_Check(object) ? PyLong_AsLongLong(object) :                         \
    (throw std::runtime_error("Could not parse real"), 0))

#define THPUtils_checkReal_INT(object)                                         \
    PyLong_Check(object)

#define THPUtils_unpackReal_INT(object)                                        \
    (PyLong_Check(object) ? PyLong_AsLongLong(object) :                        \
    (throw std::runtime_error("Could not parse real"), 0))

#define THPUtils_unpackReal_BOOL(object)                                       \
    (PyBool_Check(object) ? object :                                           \
    (throw std::runtime_error("Could not parse real"), Py_False))

#define THPUtils_unpackReal_COMPLEX(object)                                                                        \
    (PyComplex_Check(object) ?                                                                                     \
    (c10::complex<double>(PyComplex_RealAsDouble(object), PyComplex_ImagAsDouble(object))) :                       \
    PyFloat_Check(object) ? (c10::complex<double>(PyFloat_AsDouble(object), 0)) :                                  \
    PyLong_Check(object) ? (c10::complex<double>(PyLong_AsLongLong(object), 0)) :                                  \
    (throw std::runtime_error("Could not parse real"), c10::complex<double>(0,0)))                                 \

#define THPUtils_checkReal_BOOL(object)                                        \
    PyBool_Check(object)

#define THPUtils_checkReal_COMPLEX(object)                                        \
    PyComplex_Check(object) || PyFloat_Check(object) || PyLong_Check(object) || PyInt_Check(object)

#define THPUtils_newReal_FLOAT(value) PyFloat_FromDouble(value)
#define THPUtils_newReal_INT(value) PyInt_FromLong(value)

#define THPUtils_newReal_BOOL(value) PyBool_FromLong(value)

#define THPUtils_newReal_COMPLEX(value) PyComplex_FromDoubles(value.real(), value.imag())

#define THPDoubleUtils_checkReal(object)             THPUtils_checkReal_FLOAT(object)
#define THPDoubleUtils_unpackReal(object)            (double)THPUtils_unpackReal_FLOAT(object)
#define THPDoubleUtils_newReal(value)                THPUtils_newReal_FLOAT(value)
#define THPFloatUtils_checkReal(object)              THPUtils_checkReal_FLOAT(object)
#define THPFloatUtils_unpackReal(object)             (float)THPUtils_unpackReal_FLOAT(object)
#define THPFloatUtils_newReal(value)                 THPUtils_newReal_FLOAT(value)
#define THPHalfUtils_checkReal(object)               THPUtils_checkReal_FLOAT(object)
#define THPHalfUtils_unpackReal(object)              (at::Half)THPUtils_unpackReal_FLOAT(object)
#define THPHalfUtils_newReal(value)                  PyFloat_FromDouble(value)
#define THPHalfUtils_newAccreal(value)               THPUtils_newReal_FLOAT(value)
#define THPComplexDoubleUtils_checkReal(object)      THPUtils_checkReal_COMPLEX(object)
#define THPComplexDoubleUtils_unpackReal(object)     THPUtils_unpackReal_COMPLEX(object)
#define THPComplexDoubleUtils_newReal(value)         THPUtils_newReal_COMPLEX(value)
#define THPComplexFloatUtils_checkReal(object)       THPUtils_checkReal_COMPLEX(object)
#define THPComplexFloatUtils_unpackReal(object)      (c10::complex<float>)THPUtils_unpackReal_COMPLEX(object)
#define THPComplexFloatUtils_newReal(value)          THPUtils_newReal_COMPLEX(value)
#define THPBFloat16Utils_checkReal(object)           THPUtils_checkReal_FLOAT(object)
#define THPBFloat16Utils_unpackReal(object)          (at::BFloat16)THPUtils_unpackReal_FLOAT(object)
#define THPBFloat16Utils_newReal(value)              PyFloat_FromDouble(value)
#define THPBFloat16Utils_newAccreal(value)           THPUtils_newReal_FLOAT(value)

#define THPBoolUtils_checkReal(object)        THPUtils_checkReal_BOOL(object)
#define THPBoolUtils_unpackReal(object)       THPUtils_unpackReal_BOOL(object)
#define THPBoolUtils_newReal(value)           THPUtils_newReal_BOOL(value)
#define THPBoolUtils_checkAccreal(object)     THPUtils_checkReal_BOOL(object)
#define THPBoolUtils_unpackAccreal(object)    (int64_t)THPUtils_unpackReal_BOOL(object)
#define THPBoolUtils_newAccreal(value)        THPUtils_newReal_BOOL(value)
#define THPLongUtils_checkReal(object)        THPUtils_checkReal_INT(object)
#define THPLongUtils_unpackReal(object)       (int64_t)THPUtils_unpackReal_INT(object)
#define THPLongUtils_newReal(value)           THPUtils_newReal_INT(value)
#define THPIntUtils_checkReal(object)         THPUtils_checkReal_INT(object)
#define THPIntUtils_unpackReal(object)        (int)THPUtils_unpackReal_INT(object)
#define THPIntUtils_newReal(value)            THPUtils_newReal_INT(value)
#define THPShortUtils_checkReal(object)       THPUtils_checkReal_INT(object)
#define THPShortUtils_unpackReal(object)      (short)THPUtils_unpackReal_INT(object)
#define THPShortUtils_newReal(value)          THPUtils_newReal_INT(value)
#define THPCharUtils_checkReal(object)        THPUtils_checkReal_INT(object)
#define THPCharUtils_unpackReal(object)       (char)THPUtils_unpackReal_INT(object)
#define THPCharUtils_newReal(value)           THPUtils_newReal_INT(value)
#define THPByteUtils_checkReal(object)        THPUtils_checkReal_INT(object)
#define THPByteUtils_unpackReal(object)       (unsigned char)THPUtils_unpackReal_INT(object)
#define THPByteUtils_newReal(value)           THPUtils_newReal_INT(value)
// quantized types
#define THPQUInt8Utils_checkReal(object)         THPUtils_checkReal_INT(object)
#define THPQUInt8Utils_unpackReal(object)        (int)THPUtils_unpackReal_INT(object)
#define THPQUInt8Utils_newReal(value)           THPUtils_newReal_INT(value)
#define THPQInt8Utils_checkReal(object)         THPUtils_checkReal_INT(object)
#define THPQInt8Utils_unpackReal(object)        (int)THPUtils_unpackReal_INT(object)
#define THPQInt8Utils_newReal(value)           THPUtils_newReal_INT(value)
#define THPQInt32Utils_checkReal(object)         THPUtils_checkReal_INT(object)
#define THPQInt32Utils_unpackReal(object)        (int)THPUtils_unpackReal_INT(object)
#define THPQInt32Utils_newReal(value)           THPUtils_newReal_INT(value)
#define THPQUInt4x2Utils_checkReal(object)      THPUtils_checkReal_INT(object)
#define THPQUInt4x2Utils_unpackReal(object)     (int)THPUtils_unpackReal_INT(object)
#define THPQUInt4x2Utils_newReal(value)         THPUtils_newReal_INT(value)
#define THPQUInt2x4Utils_checkReal(object)      THPUtils_checkReal_INT(object)
#define THPQUInt2x4Utils_unpackReal(object)     (int)THPUtils_unpackReal_INT(object)
#define THPQUInt2x4Utils_newReal(value)         THPUtils_newReal_INT(value)


#define THPUtils_assert(cond, ...) THPUtils_assertRet(nullptr, cond, __VA_ARGS__)
#define THPUtils_assertRet(value, cond, ...)                                   \
if (THP_EXPECT(!(cond), 0)) { THPUtils_setError(__VA_ARGS__); return value; }
TORCH_PYTHON_API void THPUtils_setError(const char *format, ...);
TORCH_PYTHON_API void THPUtils_invalidArguments(
        PyObject *given_args, PyObject *given_kwargs,
        const char *function_name, size_t num_options, ...);

bool THPUtils_checkIntTuple(PyObject *arg);
std::vector<int> THPUtils_unpackIntTuple(PyObject *arg);

void THPUtils_addPyMethodDefs(std::vector<PyMethodDef>& vector, PyMethodDef* methods);

int THPUtils_getCallable(PyObject *arg, PyObject **result);

#define THWTensorPtr  TH_CONCAT_3(TH,Real,TensorPtr)
#define THPStoragePtr TH_CONCAT_3(THP,Real,StoragePtr)
#define THPTensorPtr  TH_CONCAT_3(THP,Real,TensorPtr)
#define THSPTensorPtr  TH_CONCAT_3(THSP,Real,TensorPtr)

typedef THPPointer<THPGenerator> THPGeneratorPtr;

template <typename T>
struct THPUtils_typeTraits {};

#include <torch/csrc/generic/utils.h>
#include <torch/csrc/THGenerateByteType.h>

std::vector<int64_t> THPUtils_unpackLongs(PyObject *arg);
PyObject * THPUtils_dispatchStateless(PyObject *tensor, const char *name, PyObject *args, PyObject *kwargs);

template<typename _real, typename = void>
struct mod_traits {};

template<typename _real>
struct mod_traits<_real, typename std::enable_if<std::is_floating_point<_real>::value>::type> {
  static _real mod(_real a, _real b) { return fmod(a, b); }
};

template<typename _real>
struct mod_traits<_real, typename std::enable_if<std::is_integral<_real>::value>::type> {
  static _real mod(_real a, _real b) { return a % b; }
};

void setBackCompatBroadcastWarn(bool warn);
bool getBackCompatBroadcastWarn();

void setBackCompatKeepdimWarn(bool warn);
bool getBackCompatKeepdimWarn();
bool maybeThrowBackCompatKeepdimWarn(char *func);

// NB: This is in torch/csrc/cuda/utils.cpp, for whatever reason
#ifdef USE_CUDA
std::vector<c10::optional<at::cuda::CUDAStream>> THPUtils_PySequence_to_CUDAStreamList(PyObject *obj);
#endif

void storage_copy(at::Storage dst, at::Storage src, bool non_blocking=false);
void storage_fill(at::Storage self, uint8_t value);
void storage_set(at::Storage self, ptrdiff_t idx, uint8_t value);
uint8_t storage_get(at::Storage self, ptrdiff_t idx);

#endif
