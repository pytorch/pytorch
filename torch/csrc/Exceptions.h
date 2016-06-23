#include <exception>
#include <string>

#define HANDLE_TH_ERRORS                                                       \
  try {

#define END_HANDLE_TH_ERRORS_RET(retval)                                       \
  } catch (std::exception &e) {                                                \
    PyErr_SetString(PyExc_RuntimeError, e.what());                             \
    return retval;                                                             \
  }

#define END_HANDLE_TH_ERRORS END_HANDLE_TH_ERRORS_RET(NULL)

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

