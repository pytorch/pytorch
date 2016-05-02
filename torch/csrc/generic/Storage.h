#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Storage.h"
#else

bool THPStorage_(init)(PyObject *module);
PyObject * THPStorage_(newObject)(THStorage *storage);

#endif
