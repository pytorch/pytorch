
#include "record-utility.h"

#include "../../core/slang-string-util.h"
#include "../../core/slang-string.h"

#include <mutex>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

constexpr const char* kRecordLayerEnvVar = "SLANG_RECORD_LAYER";
constexpr const char* kRecordLayerLogLevel = "SLANG_RECORD_LOG_LEVEL";

namespace SlangRecord
{
static thread_local unsigned int g_logLevel = LogLevel::Silent;

static bool getEnvironmentVariable(const char* name, Slang::String& out)
{
#ifdef _WIN32
    char* envVar = nullptr;
    size_t sz = 0;
    if (_dupenv_s(&envVar, &sz, name) == 0 && envVar != nullptr)
    {
        out = envVar;
        free(envVar);
    }
#else
    if (const char* envVar = std::getenv(name))
    {
        out = envVar;
    }
#endif
    return out.getLength() > 0;
}

bool isRecordLayerEnabled()
{
    Slang::String envVarStr;
    if (getEnvironmentVariable(kRecordLayerEnvVar, envVarStr))
    {
        if (envVarStr == "1")
        {
            return true;
        }
    }
    return false;
}

void setLogLevel()
{
    // We only want to set the log level once
    if (g_logLevel != LogLevel::Silent)
    {
        return;
    }

    Slang::String envVarStr;
    if (getEnvironmentVariable(kRecordLayerLogLevel, envVarStr))
    {
        unsigned int logLevel = Slang::stringToUInt(envVarStr);
        g_logLevel = std::min((unsigned int)(LogLevel::Verbose), logLevel);
        return;
    }
}

void slangRecordLog(LogLevel logLevel, const char* fmt, ...)
{
    if (logLevel > g_logLevel)
    {
        return;
    }

    Slang::StringBuilder builder;

    va_list args;
    va_start(args, fmt);
    Slang::StringUtil::append(fmt, args, builder);
    va_end(args);

    fprintf(stdout, "[slang-record-replay]: %s", builder.begin());
}
} // namespace SlangRecord
