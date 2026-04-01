// cuda-shader-program.cpp
#include "cuda-shader-program.h"

namespace gfx
{
#ifdef GFX_ENABLE_CUDA
using namespace Slang;

namespace cuda
{

ShaderProgramImpl::~ShaderProgramImpl()
{
    if (cudaModule)
        cuModuleUnload(cudaModule);
}

} // namespace cuda
#endif
} // namespace gfx
