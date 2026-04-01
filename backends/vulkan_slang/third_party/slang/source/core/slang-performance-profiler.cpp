#include "slang-performance-profiler.h"

#include "slang-dictionary.h"

namespace Slang
{
class PerformanceProfilerImpl : public PerformanceProfiler
{
public:
    OrderedDictionary<const char*, FuncProfileInfo> data;

    virtual FuncProfileContext enterFunction(const char* funcName) override
    {
        auto entry = data.tryGetValue(funcName);
        if (!entry)
        {
            data.add(funcName, FuncProfileInfo());
            entry = data.tryGetValue(funcName);
        }
        entry->invocationCount++;
        FuncProfileContext ctx;
        ctx.funcName = funcName;
        ctx.startTime = std::chrono::high_resolution_clock::now();
        return ctx;
    }
    virtual void exitFunction(FuncProfileContext ctx) override
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = endTime - ctx.startTime;
        auto entry = data.tryGetValue(ctx.funcName);
        entry->duration += duration;
    }
    virtual void getResult(StringBuilder& out) override
    {
        char buffer[512];
        for (const auto& func : data)
        {
            memset(buffer, 0, sizeof(buffer));
            snprintf(buffer, sizeof(buffer), "[*] %30s", func.key);
            out << buffer << " \t";
            auto milliseconds =
                std::chrono::duration_cast<std::chrono::milliseconds>(func.value.duration);
            out << func.value.invocationCount << " \t"
                << static_cast<uint64_t>(milliseconds.count()) << "ms\n";
        }
    }
    virtual void clear() override { data.clear(); }
    virtual void dispose() override { data = decltype(data)(); }
};

PerformanceProfiler* Slang::PerformanceProfiler::getProfiler()
{
    thread_local static PerformanceProfilerImpl profiler = PerformanceProfilerImpl();
    return &profiler;
}

SlangProfiler::SlangProfiler(PerformanceProfiler* profiler)
{
    PerformanceProfilerImpl* profilerImpl = static_cast<PerformanceProfilerImpl*>(profiler);
    size_t entryCount = profilerImpl->data.getCount();

    m_profilEntries.reserve(entryCount);

    int index = 0;
    for (auto func : profilerImpl->data)
    {
        ProfileInfo profileEntry{};
        size_t strSize = std::min(sizeof(profileEntry.funcName) - 1, strlen(func.key));

        if (strSize > 0)
        {
            memcpy(profileEntry.funcName, func.key, strSize);
        }
        profileEntry.invocationCount = func.value.invocationCount;
        profileEntry.duration = func.value.duration;

        m_profilEntries.insert(index, profileEntry);
        index++;
    }
}

ISlangUnknown* SlangProfiler::getInterface(const Guid& guid)
{
    if (guid == SlangProfiler::getTypeGuid())
        return static_cast<ISlangUnknown*>(this);
    else
        return nullptr;
}

size_t SlangProfiler::getEntryCount()
{
    return m_profilEntries.getCount();
}

const char* SlangProfiler::getEntryName(uint32_t index)
{
    if (index >= (uint32_t)m_profilEntries.getCount())
        return nullptr;

    return m_profilEntries[index].funcName;
}

long SlangProfiler::getEntryTimeMS(uint32_t index)
{
    if (index >= (uint32_t)m_profilEntries.getCount())
        return 0;

    auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(m_profilEntries[index].duration);
    return (long)milliseconds.count();
}

uint32_t SlangProfiler::getEntryInvocationTimes(uint32_t index)
{
    if (index >= (uint32_t)m_profilEntries.getCount())
        return 0;

    return m_profilEntries[index].invocationCount;
}
} // namespace Slang
