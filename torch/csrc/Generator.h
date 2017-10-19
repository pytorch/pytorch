#ifndef THP_GENERATOR_H
#define THP_GENERATOR_H

struct THPGenerator {
  PyObject_HEAD
  THGenerator *cdata;
  bool owner;  // if true, frees cdata in destructor
};

#define THPGenerator_Check(obj) \
  PyObject_IsInstance(obj, THPGeneratorClass)

#define THPGenerator_CData(obj) \
  ((THPGenerator*)obj)->cdata

THP_API PyObject * THPGenerator_New();

// Creates a new Python object wrapping the THGenerator. The reference is
// borrowed. The caller should ensure that the THGenerator* object lifetime
// last at least as long as the Python wrapper.
THP_API PyObject * THPGenerator_NewWithGenerator(THGenerator *cdata);

extern PyObject *THPGeneratorClass;

#ifdef _THP_CORE
bool THPGenerator_init(PyObject *module);
#endif

#endif
