#ifndef THCP_GDSFILE_INC
#define THCP_GDSFILE_INC

#include <torch/csrc/python_headers.h>

namespace torch::cuda::shared {
void initGdsBindings(PyObject* module);
}
#endif // THCP_GDSFILE_INC
