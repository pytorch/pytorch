#ifndef THP_UTILS_H
#define THP_UTILS_H

#include <vector>
#include <string>

#define THPUtils_(NAME) TH_CONCAT_4(THP,Real,Utils_,NAME)

#define THPUtils_typename(obj) (Py_TYPE(obj)->tp_name)


#if PY_MAJOR_VERSION == 2
#define THPUtils_checkLong(obj) (PyLong_Check(obj) || PyInt_Check(obj))
#define THPUtils_unpackLong(obj)                                               \
    (PyLong_Check(obj) ? PyLong_AsLong(obj) :                                  \
    PyInt_Check(obj) ? PyInt_AsLong(obj) :                                     \
    (throw std::runtime_error("Could not unpack long"), 0))
#else
#define THPUtils_checkLong(obj) PyLong_Check(obj)
#define THPUtils_unpackLong(obj)                                               \
    (PyLong_Check(obj) ? PyLong_AsLong(obj) :                                  \
    (throw std::runtime_error("Could not unpack long"), 0))
#endif


#if PY_MAJOR_VERSION == 2
#define THPUtils_bytesFromString(c_string)   PyString_FromString(c_string)
#define THPUtils_checkBytes(obj)             PyString_Check(obj)
#define THPUtils_bytesAsString(obj)          PyString_AS_STRING(obj)
#else
#define THPUtils_bytesFromString(c_string)   PyBytes_FromString(c_string)
#define THPUtils_checkBytes(obj)             PyBytes_Check(obj)
#define THPUtils_bytesAsString(obj)          PyBytes_AS_STRING(obj)
#endif


#if PY_MAJOR_VERSION == 2
#define THPUtils_checkReal_FLOAT(object)                                       \
    (PyFloat_Check(object) || PyLong_Check(object) || PyInt_Check(object))

#define THPUtils_unpackReal_FLOAT(object)                                      \
    (PyFloat_Check(object) ? PyFloat_AsDouble(object) :                        \
    PyLong_Check(object) ? PyLong_AsLongLong(object) :                         \
    PyInt_Check(object) ? PyInt_AsLong(object) :                               \
    (throw std::runtime_error("Could not parse real"), 0))

#define THPUtils_checkReal_INT(object)                                         \
    (PyLong_Check(object) || PyInt_Check(object))

#define THPUtils_unpackReal_INT(object)                                        \
    (PyLong_Check(object) ? PyLong_AsLongLong(object) :                        \
    PyInt_Check(object) ? PyInt_AsLong(object) :                               \
    (throw std::runtime_error("Could not parse real"), 0))
#else /* PY_MAJOR_VERSION == 2 */
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
#endif

#define THPUtils_newReal_FLOAT(value) PyFloat_FromDouble(value)
#define THPUtils_newReal_HALF(value) PyFloat_FromDouble(THC_half2float(value))
// TODO: handle int overflows for py2
#define THPUtils_newReal_INT(value) PyInt_FromLong(value)

#define THPDoubleUtils_checkReal(object)      THPUtils_checkReal_FLOAT(object)
#define THPDoubleUtils_unpackReal(object)     (double)THPUtils_unpackReal_FLOAT(object)
#define THPDoubleUtils_newReal(value)         THPUtils_newReal_FLOAT(value)
#define THPDoubleUtils_checkAccreal(object)   THPUtils_checkReal_FLOAT(object)
#define THPDoubleUtils_unpackAccreal(object)  (double)THPUtils_unpackReal_FLOAT(object)
#define THPDoubleUtils_newAccreal(value)      THPUtils_newReal_FLOAT(value)
#define THPFloatUtils_checkReal(object)       THPUtils_checkReal_FLOAT(object)
#define THPFloatUtils_unpackReal(object)      (float)THPUtils_unpackReal_FLOAT(object)
#define THPFloatUtils_newReal(value)          THPUtils_newReal_FLOAT(value)
#define THPFloatUtils_checkAccreal(object)    THPUtils_checkReal_FLOAT(object)
#define THPFloatUtils_unpackAccreal(object)   (double)THPUtils_unpackReal_FLOAT(object)
#define THPFloatUtils_newAccreal(value)       THPUtils_newReal_FLOAT(value)
#define THPHalfUtils_checkReal(object)        THPUtils_checkReal_FLOAT(object)
#define THPHalfUtils_unpackReal(object)       (half)THC_float2half(THPUtils_unpackReal_FLOAT(object))
#define THPHalfUtils_newReal(value)           THPUtils_newReal_HALF(value)
#define THPHalfUtils_checkAccreal(object)     THPUtils_checkReal_FLOAT(object)
#define THPHalfUtils_unpackAccreal(object)    (double)THPUtils_unpackReal_FLOAT(object)
#define THPHalfUtils_newAccreal(value)        THPUtils_newReal_FLOAT(value)

#define THPLongUtils_checkReal(object)        THPUtils_checkReal_INT(object)
#define THPLongUtils_unpackReal(object)       (long)THPUtils_unpackReal_INT(object)
#define THPLongUtils_newReal(value)           THPUtils_newReal_INT(value)
#define THPLongUtils_checkAccreal(object)     THPUtils_checkReal_INT(object)
#define THPLongUtils_unpackAccreal(object)    (long)THPUtils_unpackReal_INT(object)
#define THPLongUtils_newAccreal(value)        THPUtils_newReal_INT(value)
#define THPIntUtils_checkReal(object)         THPUtils_checkReal_INT(object)
#define THPIntUtils_unpackReal(object)        (int)THPUtils_unpackReal_INT(object)
#define THPIntUtils_newReal(value)            THPUtils_newReal_INT(value)
#define THPIntUtils_checkAccreal(object)      THPUtils_checkReal_INT(object)
#define THPIntUtils_unpackAccreal(object)     (long)THPUtils_unpackReal_INT(object)
#define THPIntUtils_newAccreal(value)         THPUtils_newReal_INT(value)
#define THPShortUtils_checkReal(object)       THPUtils_checkReal_INT(object)
#define THPShortUtils_unpackReal(object)      (short)THPUtils_unpackReal_INT(object)
#define THPShortUtils_newReal(value)          THPUtils_newReal_INT(value)
#define THPShortUtils_checkAccreal(object)    THPUtils_checkReal_INT(object)
#define THPShortUtils_unpackAccreal(object)   (long)THPUtils_unpackReal_INT(object)
#define THPShortUtils_newAccreal(value)       THPUtils_newReal_INT(value)
#define THPCharUtils_checkReal(object)        THPUtils_checkReal_INT(object)
#define THPCharUtils_unpackReal(object)       (char)THPUtils_unpackReal_INT(object)
#define THPCharUtils_newReal(value)           THPUtils_newReal_INT(value)
#define THPCharUtils_checkAccreal(object)     THPUtils_checkReal_INT(object)
#define THPCharUtils_unpackAccreal(object)    (long)THPUtils_unpackReal_INT(object)
#define THPCharUtils_newAccreal(value)        THPUtils_newReal_INT(value)
#define THPByteUtils_checkReal(object)        THPUtils_checkReal_INT(object)
#define THPByteUtils_unpackReal(object)       (unsigned char)THPUtils_unpackReal_INT(object)
#define THPByteUtils_newReal(value)           THPUtils_newReal_INT(value)
#define THPByteUtils_checkAccreal(object)     THPUtils_checkReal_INT(object)
#define THPByteUtils_unpackAccreal(object)    (long)THPUtils_unpackReal_INT(object)
#define THPByteUtils_newAccreal(value)        THPUtils_newReal_INT(value)

#define THPUtils_assert(cond, ...)                                             \
if (!(cond)) { THPUtils_setError(__VA_ARGS__); return NULL; }
THP_API void THPUtils_setError(const char *format, ...);
THP_API void THPUtils_invalidArguments(PyObject *given_args,
            const char *function_name, size_t num_options, ...);

#ifdef _THP_CORE

THLongStorage * THPUtils_getLongStorage(PyObject *args, int ignore_first=0);
int THPUtils_getCallable(PyObject *arg, PyObject **result);
bool THPUtils_parseSlice(PyObject *slice, Py_ssize_t len, Py_ssize_t *ostart,
        Py_ssize_t *ostop, Py_ssize_t *oslicelength);

#define THStoragePtr TH_CONCAT_3(TH,Real,StoragePtr)
#define THTensorPtr  TH_CONCAT_3(TH,Real,TensorPtr)
#define THPStoragePtr TH_CONCAT_3(THP,Real,StoragePtr)
#define THPTensorPtr  TH_CONCAT_3(THP,Real,TensorPtr)

template<class T>
class THPPointer {
public:
  THPPointer(): ptr(nullptr) {};
  THPPointer(T *ptr): ptr(ptr) {};
  THPPointer(THPPointer &&p) { free(); ptr = p.ptr; p.ptr = nullptr; };

  ~THPPointer() { free(); };
  T * get() { return ptr; }
  T * release() { T *tmp = ptr; ptr = NULL; return tmp; }
  operator T*() { return ptr; }
  THPPointer& operator =(T *new_ptr) { free(); ptr = new_ptr; return *this; }
  THPPointer& operator =(THPPointer &&p) { free(); ptr = p.ptr; p.ptr = nullptr; return *this; }
  T * operator ->() { return ptr; }
  operator bool() { return ptr != nullptr; }

private:
  void free();
  T *ptr = nullptr;
};

typedef THPPointer<PyObject> THPObjectPtr;
typedef THPPointer<THPGenerator> THPGeneratorPtr;

template <typename T>
struct THPUtils_typeTraits {};

#include "generic/utils.h"
#include <TH/THGenerateAllTypes.h>

#endif /* _THP_CORE */

#endif

