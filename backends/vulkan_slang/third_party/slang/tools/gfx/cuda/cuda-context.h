// cuda-context.h
#pragma once
#include "cuda-base.h"

namespace gfx
{
#ifdef GFX_ENABLE_CUDA
using namespace Slang;

namespace cuda
{

class CUDAContext : public RefObject
{
public:
    CUcontext m_context = nullptr;
    ~CUDAContext() { cuCtxDestroy(m_context); }
};

} // namespace cuda
#endif
} // namespace gfx
