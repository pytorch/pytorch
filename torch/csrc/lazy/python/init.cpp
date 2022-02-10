#include <torch/csrc/lazy/python/init.h>

#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/python/python_util.h>

namespace torch {
namespace lazy {

void initLazyBindings(PyObject* /* module */){
  // When libtorch_python is loaded, we register the python frame getter
  // otherwise, debug util simply omits python frames
  GetPythonFramesFunction() = GetPythonFrames;
}

}  // namespace lazy
}  // namespace torch
