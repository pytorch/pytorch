#ifndef THP_CUDNN_TYPES_INC
#define THP_CUDNN_TYPES_INC

#include <Python.h>
#include <cstddef>
#include <cudnn.h>

namespace torch { namespace cudnn {

PyObject * getTensorClass(PyObject *args);
cudnnDataType_t getCudnnDataType(PyObject *tensorClass);

}}  // namespace torch::cudnn

#endif
