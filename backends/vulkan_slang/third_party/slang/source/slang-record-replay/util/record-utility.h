#ifndef RECORD_UTILITY_H
#define RECORD_UTILITY_H

// in gcc and clang, __PRETTY_FUNCTION__ is the function signature,
// while MSVC uses __FUNCSIG__
#ifdef _MSC_VER
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

namespace SlangRecord
{
enum LogLevel : unsigned int
{
    Silent = 0,
    Error = 1,
    Debug = 2,
    Verbose = 3,
};

bool isRecordLayerEnabled();
void slangRecordLog(LogLevel logLevel, const char* fmt, ...);
void setLogLevel();
} // namespace SlangRecord

#define SLANG_RECORD_ASSERT(VALUE)                \
    do                                            \
    {                                             \
        if (!(VALUE))                             \
        {                                         \
            SlangRecord::slangRecordLog(          \
                SlangRecord::LogLevel::Error,     \
                "Assertion failed: %s, %s, %d\n", \
                #VALUE,                           \
                __FILE__,                         \
                __LINE__);                        \
            std::abort();                         \
        }                                         \
    } while (0)

#define SLANG_RECORD_CHECK(VALUE)                 \
    do                                            \
    {                                             \
        SLANG_RECORD_ASSERT((VALUE) == SLANG_OK); \
    } while (0)
#endif // RECORD_UTILITY_H
