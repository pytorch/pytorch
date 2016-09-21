#ifndef THP_GENERATOR_H
#define THP_GENERATOR_H

struct THPGenerator {
  PyObject_HEAD
  THGenerator *cdata;
};

#define THPGenerator_Check(obj) \
  PyObject_IsInstance(obj, THPGeneratorClass)

#define THPGenerator_CData(obj) \
  ((THPGenerator*)obj)->cdata

THP_API PyObject * THPGenerator_New();
extern PyObject *THPGeneratorClass;

#ifdef _THP_CORE
bool THPGenerator_init(PyObject *module);
#endif

#endif
