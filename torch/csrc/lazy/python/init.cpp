#include <torch/csrc/lazy/python/init.h>

#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/python/python_util.h>

namespace torch {
namespace lazy {

void initLazyBindings(PyObject* /* module */){
#ifndef USE_DEPLOY
  // When libtorch_python is loaded, we register the python frame getter
  // otherwise, debug util simply omits python frames
  // TODO(whc) can we make this work inside torch deploy interpreter?
  // it doesn't work as-is, possibly becuase GetPythonFrames resolves to external
  // cpython rather than embedded cpython
  GetPythonFramesFunction() = GetPythonFrames;
#endif
}

}  // namespace lazy
}  // namespace torch
