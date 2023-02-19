#pragma once
#ifndef C10_MOBILE

#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Exception.h>
#include <cuda.h>

namespace c10 {
namespace cuda {
typedef CUresult (*_cuDevicePrimaryCtxGetState)(
    CUdevice dev,
    unsigned int* flags,
    int* active);
class C10_CUDA_API CUDADriverAPI {
 public:
  CUDADriverAPI();
  ~CUDADriverAPI();
  bool hasPrimaryContext(int device);

 private:
  void* handle = nullptr;
  _cuDevicePrimaryCtxGetState _hasPrimaryContext_funcptr = nullptr;
};
} // namespace cuda
} // namespace c10
#endif // C10_MOBILE