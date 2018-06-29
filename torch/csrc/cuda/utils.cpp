#include "torch/csrc/python_headers.h"
#include <stdarg.h>
#include <string>
#include "THCP.h"

#include "override_macros.h"

#define THC_GENERIC_FILE "torch/csrc/generic/utils.cpp"
#include <THC/THCGenerateAllTypes.h>

#ifdef USE_CUDA
std::vector <THCStream*> THPUtils_PySequence_to_THCStreamList(PyObject *obj) {
  if (!PySequence_Check(obj)) {
    throw std::runtime_error("Expected a sequence in THPUtils_PySequence_to_THCStreamList");
  }
  THPObjectPtr seq = THPObjectPtr(PySequence_Fast(obj, NULL));
  if (seq.get() == NULL) {
    throw std::runtime_error("expected PySequence, but got " + std::string(THPUtils_typename(obj)));
  }

  std::vector<THCStream*> streams;
  Py_ssize_t length = PySequence_Fast_GET_SIZE(seq.get());
  for (Py_ssize_t i = 0; i < length; i++) {
    PyObject *stream = PySequence_Fast_GET_ITEM(seq.get(), i);

    if (PyObject_IsInstance(stream, THCPStreamClass)) {
      streams.push_back( ((THCPStream *)stream)->cdata);
    } else if (stream == Py_None) {
      streams.push_back(NULL);
    } else {
      std::runtime_error("Unknown data type found in stream list. Need THCStream or None");
    }
  }
  return streams;
}

template<>
void THPPointer<THCTensor>::free() {
  if (ptr)
    THCTensor_free(LIBRARY_STATE ptr);
}

#endif
