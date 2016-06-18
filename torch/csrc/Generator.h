struct THPGenerator {
  PyObject_HEAD
  THGenerator *cdata;
};

extern PyTypeObject THPGeneratorType;

bool THPGenerator_init(PyObject *module);
PyObject *THPGenerator_newObject();
bool THPGenerator_Check(PyObject *obj);
