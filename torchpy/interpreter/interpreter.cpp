#include <Python.h>
#include <iostream>
#include "interpreter_impl.h"

// static wchar_t* program;

__attribute__((constructor)) void init() {
  // program = Py_DecodeLocale("main", NULL);
  // if (program == NULL) {
  // fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
  // exit(1);
  // }
  // Py_SetProgramName(program);

  Py_Initialize();
  PyRun_SimpleString("print('hello from python')");
  PyEval_ReleaseThread(PyThreadState_Get());
}

__attribute__((destructor)) void deinit() {}

static void teardown() {
  PyGILState_Ensure();
  if (Py_FinalizeEx() < 0) {
    std::cout << "IT BROKE SO WE ARE EXITING\n";
    exit(120);
  }
  // PyMem_RawFree(program);
}

extern "C" void initialize_interface(InterpreterImpl* s) {
#define INITIALIZE_MEMBER(func) s->func = func;
  FOREACH_INTERFACE_FUNCTION(INITIALIZE_MEMBER)
#undef INITIALIZE_MEMBER
}

static void run_some_python(const char* code) {
  PyGILState_STATE gstate = PyGILState_Ensure();

  if (PyRun_SimpleString(code) == -1) {
    throw std::runtime_error("python eval failed\n");
  }

  PyGILState_Release(gstate);
}

static void run_python_file(const char* code) {
  PyGILState_STATE gstate = PyGILState_Ensure();

  FILE* f = fopen(code, "r");
  if (PyRun_SimpleFile(f, code) == -1) {
    throw std::runtime_error("python eval failed\n");
  }
  fclose(f);

  PyGILState_Release(gstate);
}
