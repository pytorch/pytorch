#ifndef SLANG_CORE_SIGNAL_H
#define SLANG_CORE_SIGNAL_H

#include "slang-common.h"

namespace Slang
{

enum class SignalType
{
    Unexpected,
    Unimplemented,
    AssertFailure,
    Unreachable,
    InvalidOperation,
    AbortCompilation,
};


// Note that message can be passed as nullptr for no message.
[[noreturn]] void handleSignal(SignalType type, char const* message);

#define SLANG_UNEXPECTED(reason) ::Slang::handleSignal(::Slang::SignalType::Unexpected, reason)

#define SLANG_UNIMPLEMENTED_X(what) ::Slang::handleSignal(::Slang::SignalType::Unimplemented, what)

#define SLANG_UNREACHABLE(msg) ::Slang::handleSignal(::Slang::SignalType::Unreachable, msg)

#define SLANG_ASSERT_FAILURE(msg) ::Slang::handleSignal(::Slang::SignalType::AssertFailure, msg)

#define SLANG_INVALID_OPERATION(msg) \
    ::Slang::handleSignal(::Slang::SignalType::InvalidOperation, msg)

#define SLANG_ABORT_COMPILATION(msg) \
    ::Slang::handleSignal(::Slang::SignalType::AbortCompilation, msg)


const char* getLastSignalMessage();

} // namespace Slang

#endif
