#ifndef THP_EXCEPTIONS_H
#define THP_EXCEPTIONS_H

#include <exception>
#include <stdexcept>
#include <string>

#include "THP.h"

#define HANDLE_TH_ERRORS                                                       \
  try {

#define END_HANDLE_TH_ERRORS_RET(retval)                                       \
  } catch (std::exception &e) {                                                \
    PyErr_SetString(PyExc_RuntimeError, e.what());                             \
    return retval;                                                             \
  }

#define END_HANDLE_TH_ERRORS END_HANDLE_TH_ERRORS_RET(NULL)

extern PyObject *THPException_FatalError;

#ifdef _THP_CORE
struct THException: public std::exception {
  THException(const char* msg): msg(msg) {};

  virtual const char* what() const throw() {
    return msg.c_str();
  }

  std::string msg;
};

struct THArgException: public THException {
  THArgException(const char* msg, int argNumber): THException(msg), argNumber(argNumber) {};

  const int argNumber;
};

bool THPException_init(PyObject *module);
#endif

#endif
