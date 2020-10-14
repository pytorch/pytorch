#include <torch/csrc/utils/pycfunction_helpers.h>

PyCFunction convertPyCFunctionWithKeywords(PyCFunctionWithKeywords func) {
  return (PyCFunction)(void(*)(void))func;
}
