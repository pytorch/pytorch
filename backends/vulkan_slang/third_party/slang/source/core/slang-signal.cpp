#include "slang-signal.h"

#include "slang-exception.h"
#include "stdio.h"

namespace Slang
{

thread_local String g_lastSignalMessage;

static const char* _getSignalTypeAsText(SignalType type)
{
    switch (type)
    {
    case SignalType::AssertFailure:
        return "assert failure";
    case SignalType::Unimplemented:
        return "unimplemented";
    case SignalType::Unreachable:
        return "hit unreachable code";
    case SignalType::Unexpected:
        return "unexpected";
    case SignalType::InvalidOperation:
        return "invalid operation";
    case SignalType::AbortCompilation:
        return "abort compilation";
    default:
        return "unhandled";
    }
}

String _getMessage(SignalType type, char const* message)
{
    StringBuilder buf;
    const char* const typeText = _getSignalTypeAsText(type);
    buf << typeText;
    if (message)
    {
        buf << ": " << message;
    }

    return buf.produceString();
}

// One point of having as a single function is a choke point both for handling (allowing different
// handling scenarios) as well as a choke point to set a breakpoint to catch 'signal' types
[[noreturn]] void handleSignal(SignalType type, char const* message)
{
    StringBuilder buf;
    const char* const typeText = _getSignalTypeAsText(type);
    buf << typeText << ": " << message;

    // Can be useful to enable during debug when problem is on CI
    if (false)
    {
        printf("%s\n", _getMessage(type, message).getBuffer());
    }

    g_lastSignalMessage = _getMessage(type, message);

#if SLANG_HAS_EXCEPTIONS
    switch (type)
    {
    case SignalType::InvalidOperation:
        throw InvalidOperationException(_getMessage(type, message));
    case SignalType::AbortCompilation:
        throw AbortCompilationException(_getMessage(type, message));
    default:
        throw InternalError(_getMessage(type, message));
    }
#else
    // Attempt to drop out into the debugger. If a debugger isn't attached this will likely crash -
    // which is probably the best we can do.

    SLANG_BREAKPOINT(0);

    // 'panic'. Exit with an error code as we can't throw or catch.
    exit(-1);
#endif
}

const char* getLastSignalMessage()
{
    return g_lastSignalMessage.getBuffer();
}

} // namespace Slang
