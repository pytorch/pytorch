#include <cstdio>

#include <cuda_runtime.h>

static bool cmakeCompilerCUDAArch()
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    std::fprintf(stderr, "No CUDA devices found.\n");
    return -1;
  }

  bool found = false;
  char const* sep = "";
  for (int device = 0; device < count; ++device) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
      std::printf("%s%d%d", sep, prop.major, prop.minor);
      sep = ";";
      found = true;
    }
  }

  if (!found) {
    std::fprintf(stderr, "No CUDA architecture detected from any devices.\n");
  }

  return found;
}
