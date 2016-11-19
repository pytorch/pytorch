#ifndef THP_VARIABLE_H
#define THP_VARIABLE_H

struct THPVariableVersion {
  THPVariableVersion() {
    saved_ref = false;
    version_block = new int[3];
    version_block[0] = 0; // version
    version_block[1] = 1; // refcount
    version_block[2] = 1; // number of variables currently using the counter
  };

  int operator++(int) { return version_block[0]++; }

  int operator*() { return *version_block; }

  int var_refcnt() { return version_block[2]; }

  void join_with(THPVariableVersion &other) {
    cleanup();
    version_block = other.version_block;
    version_block[1]++;
    version_block[2]++;
  }

  THPVariableVersion* new_saved_ref() {
    auto new_ver = new THPVariableVersion();
    new_ver->cleanup();
    new_ver->version_block = version_block;
    version_block[1]++;
    new_ver->saved_ref = true;
    return new_ver;
  }

  void cleanup() {
    if (!saved_ref) --version_block[2];
    if (--version_block[1]) return;
    delete[] version_block;
    version_block = nullptr;
  }

  ~THPVariableVersion() { cleanup(); }

  int *version_block;
  bool saved_ref;
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

#define THPVariable_Check(obj)                                                 \
    (THPVariableClass &&                                                       \
     PyObject_IsInstance(obj, THPVariableClass))

#endif
