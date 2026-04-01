#ifndef SLANG_CORE_EXCEPTION_H
#define SLANG_CORE_EXCEPTION_H

#include "slang-common.h"
#include "slang-string.h"

namespace Slang
{
// NOTE!
// Exceptions should not generally be used in core/compiler-core, use the 'signal' mechanism
// ideally using the macros in the slang-signal.h such as `SLANG_UNEXPECTED`
//
// If core/compiler-core libraries are compiled with SLANG_DISABLE_EXCEPTIONS,
// these classes will *never* be thrown.

class Exception
{
public:
    String Message;
    Exception() {}
    Exception(const String& message)
        : Message(message)
    {
    }

    virtual ~Exception() {}
};

class InvalidOperationException : public Exception
{
public:
    InvalidOperationException() {}
    InvalidOperationException(const String& message)
        : Exception(message)
    {
    }
};

class InternalError : public Exception
{
public:
    InternalError() {}
    InternalError(const String& message)
        : Exception(message)
    {
    }
};

class AbortCompilationException : public Exception
{
public:
    AbortCompilationException() {}
    AbortCompilationException(const String& message)
        : Exception(message)
    {
    }
};
} // namespace Slang

#endif
