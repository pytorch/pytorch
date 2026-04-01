#ifndef SLANG_CORE_PERFORMANCE_PROFILER_H
#define SLANG_CORE_PERFORMANCE_PROFILER_H

#include "../core/slang-list.h"
#include "slang-com-helper.h"
#include "slang-string.h"

#include <chrono>
#include <vector>

namespace Slang
{

struct FuncProfileInfo
{
    int invocationCount = 0;
    std::chrono::nanoseconds duration = std::chrono::nanoseconds::zero();
};

struct FuncProfileContext
{
    const char* funcName = nullptr;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};

class PerformanceProfiler
{
public:
    virtual FuncProfileContext enterFunction(const char* funcName) = 0;
    virtual void exitFunction(FuncProfileContext context) = 0;
    virtual void getResult(StringBuilder& out) = 0;
    virtual void clear() = 0;
    virtual void dispose() = 0;

public:
    static PerformanceProfiler* getProfiler();
};

struct PerformanceProfilerFuncRAIIContext
{
    FuncProfileContext context;
    PerformanceProfilerFuncRAIIContext(const char* funcName)
    {
        context = PerformanceProfiler::getProfiler()->enterFunction(funcName);
    }
    ~PerformanceProfilerFuncRAIIContext()
    {
        PerformanceProfiler::getProfiler()->exitFunction(context);
    }
};

struct SlangProfiler : public ISlangProfiler, public RefObject
{
public:
    SLANG_REF_OBJECT_IUNKNOWN_ALL
    struct ProfileInfo
    {
        char funcName[256] = {0};
        int invocationCount = 0;
        std::chrono::nanoseconds duration = std::chrono::nanoseconds::zero();
    };
    SlangProfiler(PerformanceProfiler* profiler);
    ISlangUnknown* getInterface(const Guid& guid);

    virtual SLANG_NO_THROW size_t SLANG_MCALL getEntryCount() override;
    virtual SLANG_NO_THROW const char* SLANG_MCALL getEntryName(uint32_t index) override;
    virtual SLANG_NO_THROW long SLANG_MCALL getEntryTimeMS(uint32_t index) override;
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL getEntryInvocationTimes(uint32_t index) override;

private:
    List<ProfileInfo> m_profilEntries;
};

#define SLANG_PROFILE PerformanceProfilerFuncRAIIContext _profileContext(__func__)
#define SLANG_PROFILE_SECTION(s) PerformanceProfilerFuncRAIIContext _profileContext##s(#s)

} // namespace Slang

#endif
