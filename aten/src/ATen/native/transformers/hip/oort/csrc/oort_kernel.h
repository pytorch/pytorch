#ifndef OORT_KERNEL_H
#define OORT_KERNEL_H

#include <stdint.h>
#include <vector>
#include <hip/hip_runtime.h>
#include <incbin.h>

#define OORT_HIP_CHECK_RETURN(expr)                                     \
    do {                                                                \
        auto r = (expr);                                                \
        if (r != hipSuccess)                                            \
            throw std::runtime_error("FAILURE at Line " INCBIN_STRINGIZE(__LINE__) );   \
    } while(0)

namespace oort {

class OortKernel {
public:
  OortKernel(const char* kernel_name,
             const void* image,
             int shared_memory_size)
    : shared_memory_size_(shared_memory_size)
  {
    hipJitOption opt[] = {hipJitOptionErrorLogBufferSizeBytes,
                          hipJitOptionErrorLogBuffer,
                          hipJitOptionInfoLogBufferSizeBytes,
                          hipJitOptionInfoLogBuffer, hipJitOptionLogVerbose};
    const unsigned int errbufsize = 8192;
    const unsigned int logbufsize = 8192;
    std::vector<char> err(errbufsize, 0);
    std::vector<char> log(errbufsize, 0);
    void *optval[] = {(void *)(uintptr_t)err.size(), err.data(),
                      (void *)(uintptr_t)log.size(), log.data(), (void *)(uintptr_t)1};

    OORT_HIP_CHECK_RETURN(hipModuleLoadDataEx(&mod_, image, 5, opt, optval));
    OORT_HIP_CHECK_RETURN(hipModuleGetFunction(&fun_, mod_, kernel_name));
  }

  hipError_t invoke(dim3 grid, dim3 block,
                    std::vector<void*>& args,
                    hipStream_t stream)
  {
    return hipModuleLaunchKernel(fun_,
                                 grid.x, grid.y, grid.z,
                                 block.x, block.y, block.z,
                                 shared_memory_size_, stream, args.data(), 0);
  }
private:
  hipModule_t mod_;
  hipFunction_t fun_;
  int shared_memory_size_;
};

} // namespace oort

#endif
