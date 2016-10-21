#ifndef THP_CUDNN_EXCEPTIONS_INC
#define THP_CUDNN_EXCEPTIONS_INC

#include <cudnn.h>
#include <stdexcept>

#define END_HANDLE_CUDNN_ERRORS                                              \
} catch (std::cudnn_exception &e) {                                          \
  PyErr_SetString(PyExc_RuntimeError, e.what());                             \
  return retval;                                                             \
}

namespace torch { namespace cudnn {

class cudnn_exception : public std::runtime_error {
public:
  cudnnStatus_t status;
  cudnn_exception(cudnnStatus_t status, const char* msg) : std::runtime_error(msg), status(status) {
  }
};

inline void CHECK(cudnnStatus_t status)
{
  if (status != CUDNN_STATUS_SUCCESS) {
    throw cudnn_exception(status, cudnnGetErrorString(status));
  }
}

inline void CUDA_CHECK(cudaError_t error)
{
  if (error) {
    throw std::runtime_error("CUDA error");
  }
}

}}  // namespace torch::cudnn

#endif
