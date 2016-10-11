#ifndef THP_VARIABLE_H
#define THP_VARIABLE_H

struct THPVariableVersion {
  THPVariableVersion() {
    version_block = new int[2];
    version_block[0] = 0;
    version_block[1] = 1;
  };

  int operator++(int) { return version_block[0]++; }

  int operator*() { return *version_block; }

  void join_with(THPVariableVersion &other) {
    cleanup();
    version_block = other.version_block;
    version_block[1]++;
  }

  void cleanup() {
    if (--version_block[1])
      return;
    delete[] version_block;
    version_block = nullptr;
  }

  ~THPVariableVersion() { cleanup(); }

  int *version_block;
};

struct THPVariable {
    PyObject_HEAD
    PyObject *creator;
    PyObject *data;
    PyObject *grad;
    PyObject *backward_hooks;
    THPVariableVersion *version_counter;
    int output_nr;
    char is_volatile;
    char requires_grad;
};

bool THPVariable_initModule(PyObject *module);
extern PyObject *THPVariableClass;
PyObject * THPVariable_NewVolatile(PyObject *data);
PyObject * THPVariable_New(PyObject *data, PyObject *creator, char requires_grad);

#define THPVariable_Check(obj) PyObject_IsInstance(obj, THPVariableClass)

#endif
