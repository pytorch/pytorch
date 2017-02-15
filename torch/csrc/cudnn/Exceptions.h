#ifndef THP_CUDNN_EXCEPTIONS_INC
#define THP_CUDNN_EXCEPTIONS_INC

#include <THC/THC.h>
#include <cudnn.h>
#include <string>
#include <stdexcept>
#include <sstream>

#include "Types.h"

#define CHECK_ARG(cond) _CHECK_ARG(cond, #cond, __FILE__, __LINE__)

extern THCState* state;

namespace torch { namespace cudnn {

template<typename ...T>
void assertSameGPU(cudnnDataType_t dataType, T* ... tensors) {
  static_assert(std::is_same<THVoidTensor, typename std::common_type<T...>::type>::value,
      "all arguments to assertSameGPU have to be THVoidTensor*");
  int is_same;
  if (dataType == CUDNN_DATA_FLOAT) {
    is_same = THCudaTensor_checkGPU(state, sizeof...(T),
        reinterpret_cast<THCudaTensor*>(tensors)...);
  } else if (dataType == CUDNN_DATA_HALF) {
    is_same = THCudaHalfTensor_checkGPU(state, sizeof...(T),
        reinterpret_cast<THCudaHalfTensor*>(tensors)...);
  } else if (dataType == CUDNN_DATA_DOUBLE) {
    is_same = THCudaDoubleTensor_checkGPU(state, sizeof...(T),
        reinterpret_cast<THCudaDoubleTensor*>(tensors)...);
  } else {
    throw std::runtime_error("unknown cuDNN data type");
  }
  if (!is_same) {
    throw std::runtime_error("tensors are on different GPUs");
  }
}

class cudnn_exception : public std::runtime_error {
public:
  cudnnStatus_t status;
  cudnn_exception(cudnnStatus_t status, const char* msg)
      : std::runtime_error(msg)
      , status(status) {}
  cudnn_exception(cudnnStatus_t status, const std::string& msg)
      : std::runtime_error(msg)
      , status(status) {}
};

inline void CHECK(cudnnStatus_t status)
{
  if (status != CUDNN_STATUS_SUCCESS) {
    if (status == CUDNN_STATUS_NOT_SUPPORTED) {
        throw cudnn_exception(status, std::string(cudnnGetErrorString(status)) +
                ". This error may appear if you passed in a non-contiguous input.");
    }
    throw cudnn_exception(status, cudnnGetErrorString(status));
  }
}

inline void CUDA_CHECK(cudaError_t error)
{
  if (error) {
    std::string msg("CUDA error: ");
    msg += cudaGetErrorString(error);
    throw std::runtime_error(msg);
  }
}

inline void _CHECK_ARG(bool cond, const char* code, const char* f, int line) {
  if (!cond) {
    std::stringstream ss;
    ss << "CHECK_ARG(" << code << ") failed at " << f << ":" << line;
    throw std::runtime_error(ss.str());
  }
}

}}  // namespace torch::cudnn

#endif
