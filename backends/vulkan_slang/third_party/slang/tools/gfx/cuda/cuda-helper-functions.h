// cuda-helper-functions.h
#pragma once

#include "../../../source/core/slang-list.h"
#include "cuda-base.h"
#include "slang-gfx.h"

namespace gfx
{
using namespace Slang;

#ifdef GFX_ENABLE_CUDA
namespace cuda
{
SLANG_FORCE_INLINE bool _isError(CUresult result)
{
    return result != 0;
}

// A enum used to control if errors are reported on failure of CUDA call.
enum class CUDAReportStyle
{
    Normal,
    Silent,
};

struct CUDAErrorInfo
{
    CUDAErrorInfo(
        const char* filePath,
        int lineNo,
        const char* errorName = nullptr,
        const char* errorString = nullptr)
        : m_filePath(filePath), m_lineNo(lineNo), m_errorName(errorName), m_errorString(errorString)
    {
    }
    SlangResult handle() const;

    const char* m_filePath;
    int m_lineNo;
    const char* m_errorName;
    const char* m_errorString;
};

// If this code path is enabled, CUDA errors will be reported directly to StdWriter::out stream.

SlangResult _handleCUDAError(CUresult cuResult, const char* file, int line);

#define SLANG_CUDA_HANDLE_ERROR(x) _handleCUDAError(x, __FILE__, __LINE__)

#define SLANG_CUDA_RETURN_ON_FAIL(x)              \
    {                                             \
        auto _res = x;                            \
        if (_isError(_res))                       \
            return SLANG_CUDA_HANDLE_ERROR(_res); \
    }

#define SLANG_CUDA_RETURN_WITH_REPORT_ON_FAIL(x, r)                                             \
    {                                                                                           \
        auto _res = x;                                                                          \
        if (_isError(_res))                                                                     \
        {                                                                                       \
            return (r == CUDAReportStyle::Normal) ? SLANG_CUDA_HANDLE_ERROR(_res) : SLANG_FAIL; \
        }                                                                                       \
    }

#define SLANG_CUDA_ASSERT_ON_FAIL(x)           \
    {                                          \
        auto _res = x;                         \
        if (_isError(_res))                    \
        {                                      \
            SLANG_ASSERT(!"Failed CUDA call"); \
        };                                     \
    }

#ifdef RENDER_TEST_OPTIX

bool _isError(OptixResult result);

#if 1
SlangResult _handleOptixError(OptixResult result, char const* file, int line);

#define SLANG_OPTIX_HANDLE_ERROR(RESULT) _handleOptixError(RESULT, __FILE__, __LINE__)
#else
#define SLANG_OPTIX_HANDLE_ERROR(RESULT) SLANG_FAIL
#endif

#define SLANG_OPTIX_RETURN_ON_FAIL(EXPR)           \
    do                                             \
    {                                              \
        auto _res = EXPR;                          \
        if (_isError(_res))                        \
            return SLANG_OPTIX_HANDLE_ERROR(_res); \
    } while (0)

void _optixLogCallback(unsigned int level, const char* tag, const char* message, void* userData);

#endif

AdapterLUID getAdapterLUID(int deviceIndex);

} // namespace cuda
#endif

Result SLANG_MCALL getCUDAAdapters(List<AdapterInfo>& outAdapters);

Result SLANG_MCALL createCUDADevice(const IDevice::Desc* desc, IDevice** outDevice);

} // namespace gfx
