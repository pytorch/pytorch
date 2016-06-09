#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/utils.h"
#else


bool THPUtils_(parseSlice)(PyObject *slice, Py_ssize_t len, Py_ssize_t *ostart, Py_ssize_t *ostop, Py_ssize_t *oslicelength);
bool THPUtils_(parseReal)(PyObject *value, real *result);
PyObject * THPUtils_(newReal)(real value);
bool THPUtils_(checkReal)(PyObject *value);

#endif
