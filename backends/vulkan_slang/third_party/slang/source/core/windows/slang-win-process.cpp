// slang-win-process-util.cpp
#include "../slang-process-util.h"
#include "../slang-process.h"
#include "../slang-string-escape-util.h"
#include "../slang-string-util.h"
#include "../slang-string.h"
#include "slang-com-helper.h"

#ifdef _WIN32
// TODO: We could try to avoid including this at all, but it would
// mean trying to hide certain struct layouts, which would add
// more dynamic allocation.
#include <windows.h>
#endif

#include <process.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef SLANG_RETURN_FAIL_ON_FALSE
#define SLANG_RETURN_FAIL_ON_FALSE(x) \
    if (!(x))                         \
        return SLANG_FAIL;
#endif

namespace Slang
{

// Has behavior very similar to unique_ptr - assignment is a move.
class WinHandle
{
public:
    /// Detach the encapsulated handle. Returns the handle (which now must be externally handled)
    HANDLE detach()
    {
        HANDLE handle = m_handle;
        m_handle = nullptr;
        return handle;
    }

    /// Return as a handle
    operator HANDLE() const { return m_handle; }

    /// Assign
    void operator=(HANDLE handle)
    {
        setNull();
        m_handle = handle;
    }
    void operator=(WinHandle&& rhs)
    {
        HANDLE handle = m_handle;
        m_handle = rhs.m_handle;
        rhs.m_handle = handle;
    }

    /// Get ready for writing
    SLANG_FORCE_INLINE HANDLE* writeRef()
    {
        setNull();
        return &m_handle;
    }
    /// Get for read access
    SLANG_FORCE_INLINE const HANDLE* readRef() const { return &m_handle; }

    void setNull()
    {
        if (m_handle)
        {
            CloseHandle(m_handle);
            m_handle = nullptr;
        }
    }
    bool isNull() const { return m_handle == nullptr; }

    /// Ctor
    WinHandle(HANDLE handle = nullptr)
        : m_handle(handle)
    {
    }
    WinHandle(WinHandle&& rhs)
        : m_handle(rhs.m_handle)
    {
        rhs.m_handle = nullptr;
    }

    /// Dtor
    ~WinHandle() { setNull(); }

private:
    WinHandle(const WinHandle&) = delete;
    void operator=(const WinHandle& rhs) = delete;

    HANDLE m_handle;
};

/* A simple Stream implementation of a File HANDLE (or Pipe). Note that currently does not allow
 * getPosition/seek/atEnd */
class WinPipeStream : public Stream
{
public:
    typedef WinPipeStream ThisType;

    // Stream
    virtual Int64 getPosition() SLANG_OVERRIDE { return 0; }
    virtual SlangResult seek(SeekOrigin origin, Int64 offset) SLANG_OVERRIDE
    {
        SLANG_UNUSED(origin);
        SLANG_UNUSED(offset);
        return SLANG_E_NOT_AVAILABLE;
    }
    virtual SlangResult read(void* buffer, size_t length, size_t& outReadBytes) SLANG_OVERRIDE;
    virtual SlangResult write(const void* buffer, size_t length) SLANG_OVERRIDE;
    virtual bool isEnd() SLANG_OVERRIDE { return m_streamHandle.isNull(); }
    virtual bool canRead() SLANG_OVERRIDE
    {
        return _has(FileAccess::Read) && !m_streamHandle.isNull();
    }
    virtual bool canWrite() SLANG_OVERRIDE
    {
        return _has(FileAccess::Write) && !m_streamHandle.isNull();
    }
    virtual void close() SLANG_OVERRIDE;
    virtual SlangResult flush() SLANG_OVERRIDE;

    WinPipeStream(HANDLE handle, FileAccess access, bool isOwned = true);

    ~WinPipeStream() { close(); }

protected:
    bool _has(FileAccess access) const { return (Index(access) & Index(m_access)) != 0; }

    SlangResult _updateState(BOOL res);

    FileAccess m_access = FileAccess::None;
    WinHandle m_streamHandle;
    bool m_isOwned;
    bool m_isPipe;
};

class WinProcess : public Process
{
public:
    // Process
    virtual bool isTerminated() SLANG_OVERRIDE;
    virtual bool waitForTermination(Int timeInMs) SLANG_OVERRIDE;
    virtual void terminate(int32_t returnCode) SLANG_OVERRIDE;
    virtual void kill(int32_t returnCode) SLANG_OVERRIDE;

    WinProcess(HANDLE handle, Stream* const* streams)
        : m_processHandle(handle)
    {
        for (Index i = 0; i < Index(StdStreamType::CountOf); ++i)
        {
            m_streams[i] = streams[i];
        }
    }

protected:
    void _hasTerminated();
    WinHandle m_processHandle; ///< If not set the process has terminated
};

/* !!!!!!!!!!!!!!!!!!!!!!!!!!! WinPipeStream !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

WinPipeStream::WinPipeStream(HANDLE handle, FileAccess access, bool isOwned)
    : m_streamHandle(handle), m_access(access), m_isOwned(isOwned)
{

    // On Win32 a HANDLE has to be handled differently if it's a PIPE or FILE, so first determine
    // if it really is a pipe.
    // http://msdn.microsoft.com/en-us/library/aa364960(VS.85).aspx
    m_isPipe = ::GetFileType(handle) == FILE_TYPE_PIPE;

    if (m_isPipe)
    {
        // It might be handy to get information about the handle
        // https://docs.microsoft.com/en-us/windows/win32/api/namedpipeapi/nf-namedpipeapi-getnamedpipeinfo

        DWORD flags, outBufferSize, inBufferSize, maxInstances;
        // It appears that by default windows pipe buffer size is 4k.
        if (GetNamedPipeInfo(handle, &flags, &outBufferSize, &inBufferSize, &maxInstances))
        {
        }
    }
}

SlangResult WinPipeStream::_updateState(BOOL res)
{
    if (res)
    {
        return SLANG_OK;
    }
    else
    {
        const auto err = GetLastError();

        if (err == ERROR_BROKEN_PIPE)
        {
            m_streamHandle.setNull();
            return SLANG_OK;
        }

        SLANG_UNUSED(err);
        return SLANG_FAIL;
    }
}

SlangResult WinPipeStream::read(void* buffer, size_t length, size_t& outReadBytes)
{
    outReadBytes = 0;
    if (!_has(FileAccess::Read))
    {
        return SLANG_E_NOT_AVAILABLE;
    }

    if (m_streamHandle.isNull())
    {
        return SLANG_OK;
    }

    DWORD bytesRead = 0;

    // Check if there is any data, so won't block
    if (m_isPipe)
    {
        DWORD pipeBytesRead = 0;
        DWORD pipeTotalBytesAvailable = 0;
        DWORD pipeRemainingBytes = 0;

        // Works on anonymous pipes too
        // https://docs.microsoft.com/en-us/windows/win32/api/namedpipeapi/nf-namedpipeapi-peeknamedpipe

        SLANG_RETURN_ON_FAIL(_updateState(::PeekNamedPipe(
            m_streamHandle,
            nullptr,
            DWORD(0),
            &pipeBytesRead,
            &pipeTotalBytesAvailable,
            &pipeRemainingBytes)));
        // If there is nothing to read we are done
        // If we don't do this ReadFile will *block* if there is nothing available
        if (pipeTotalBytesAvailable == 0)
        {
            return SLANG_OK;
        }

        SLANG_RETURN_ON_FAIL(
            _updateState(::ReadFile(m_streamHandle, buffer, DWORD(length), &bytesRead, nullptr)));
    }
    else
    {
        SLANG_RETURN_ON_FAIL(
            _updateState(::ReadFile(m_streamHandle, buffer, DWORD(length), &bytesRead, nullptr)));

        // If it's not a pipe, and there is nothing left, then we are done.
        if (length > 0 && bytesRead == 0)
        {
            close();
        }
    }

    outReadBytes = size_t(bytesRead);
    return SLANG_OK;
}

SlangResult WinPipeStream::write(const void* buffer, size_t length)
{
    if (!_has(FileAccess::Write))
    {
        return SLANG_E_NOT_AVAILABLE;
    }

    if (m_streamHandle.isNull())
    {
        // Writing to closed stream
        return SLANG_FAIL;
    }

    DWORD numWritten = 0;
    BOOL writeResult = ::WriteFile(m_streamHandle, buffer, DWORD(length), &numWritten, nullptr);

    if (!writeResult)
    {
        auto err = ::GetLastError();

        if (err == ERROR_BROKEN_PIPE)
        {
            close();
            return SLANG_FAIL;
        }

        SLANG_UNUSED(err);
        return SLANG_FAIL;
    }

    if (numWritten != length)
    {
        return SLANG_FAIL;
    }

    return SLANG_OK;
}

void WinPipeStream::close()
{
    if (!m_isOwned)
    {
        // If we don't own it just detach it
        m_streamHandle.detach();
    }
    m_streamHandle.setNull();
}

SlangResult WinPipeStream::flush()
{
    if ((Index(m_access) & Index(FileAccess::Write)) == 0 || m_streamHandle.isNull())
    {
        return SLANG_E_NOT_AVAILABLE;
    }

    // https://docs.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-flushfilebuffers
    if (!::FlushFileBuffers(m_streamHandle))
    {
        auto err = GetLastError();
        SLANG_UNUSED(err);
    }
    return SLANG_OK;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!! WinProcess !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void WinProcess::_hasTerminated()
{
    if (!m_processHandle.isNull())
    {
        // get exit code for process
        // https://docs.microsoft.com/en-us/windows/desktop/api/processthreadsapi/nf-processthreadsapi-getexitcodeprocess

        DWORD childExitCode = 0;
        if (::GetExitCodeProcess(m_processHandle, &childExitCode))
        {
            m_returnValue = int32_t(childExitCode);
        }
        m_processHandle.setNull();
    }
}

bool WinProcess::waitForTermination(Int timeInMs)
{
    if (m_processHandle.isNull())
    {
        return true;
    }

    const DWORD timeOutTime = (timeInMs < 0) ? INFINITE : DWORD(timeInMs);

    // wait for the process to exit
    // TODO: set a timeout as a safety measure...
    auto res = ::WaitForSingleObject(m_processHandle, timeOutTime);

    if (res == WAIT_TIMEOUT)
    {
        return false;
    }

    _hasTerminated();
    return true;
}

bool WinProcess::isTerminated()
{
    return waitForTermination(0);
}

void WinProcess::terminate(int32_t returnCode)
{
    if (!isTerminated())
    {
        // If it's not terminated, try terminating.
        // Might take time, so use isTerminated to check
        ::TerminateProcess(m_processHandle, UINT32(returnCode));
    }
}

void WinProcess::kill(int32_t returnCode)
{
    if (!isTerminated())
    {
        // https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-terminateprocess
        ::TerminateProcess(m_processHandle, UINT32(returnCode));

        // Just assume it's done and set the return code
        m_returnValue = returnCode;
        m_processHandle.setNull();
    }
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* static */ StringEscapeHandler* Process::getEscapeHandler()
{
    return StringEscapeUtil::getHandler(StringEscapeUtil::Style::Space);
}

/* static */ UnownedStringSlice Process::getExecutableSuffix()
{
    return UnownedStringSlice::fromLiteral(".exe");
}

/* static */ SlangResult Process::getStdStream(StdStreamType type, RefPtr<Stream>& out)
{
    switch (type)
    {
    case StdStreamType::In:
        {
            out = new WinPipeStream(GetStdHandle(STD_INPUT_HANDLE), FileAccess::Read, false);
            return SLANG_OK;
        }
    case StdStreamType::Out:
        {
            out = new WinPipeStream(GetStdHandle(STD_OUTPUT_HANDLE), FileAccess::Write, false);
            return SLANG_OK;
        }
    case StdStreamType::ErrorOut:
        {
            out = new WinPipeStream(GetStdHandle(STD_ERROR_HANDLE), FileAccess::Write, false);
            return SLANG_OK;
        }
    }

    return SLANG_FAIL;
}

/* static */ SlangResult Process::create(
    const CommandLine& commandLine,
    Process::Flags flags,
    RefPtr<Process>& outProcess)
{
    WinHandle childStdOutRead;
    WinHandle childStdErrRead;
    WinHandle childStdInWrite;

    WinHandle processHandle;
    {
        WinHandle childStdOutWrite;
        WinHandle childStdErrWrite;
        WinHandle childStdInRead;

        SECURITY_ATTRIBUTES securityAttributes;
        securityAttributes.nLength = sizeof(securityAttributes);
        securityAttributes.lpSecurityDescriptor = nullptr;
        securityAttributes.bInheritHandle = true;

        // 0 means use the 'system default'
        // const DWORD bufferSize = 64 * 1024;
        const DWORD bufferSize = 0;

        {
            WinHandle childStdOutReadTmp;
            WinHandle childStdErrReadTmp;
            WinHandle childStdInWriteTmp;
            // create stdout pipe for child process
            SLANG_RETURN_FAIL_ON_FALSE(CreatePipe(
                childStdOutReadTmp.writeRef(),
                childStdOutWrite.writeRef(),
                &securityAttributes,
                bufferSize));
            if ((flags & Process::Flag::DisableStdErrRedirection) == 0)
            {
                // create stderr pipe for child process
                SLANG_RETURN_FAIL_ON_FALSE(CreatePipe(
                    childStdErrReadTmp.writeRef(),
                    childStdErrWrite.writeRef(),
                    &securityAttributes,
                    bufferSize));
            }
            // create stdin pipe for child process
            SLANG_RETURN_FAIL_ON_FALSE(CreatePipe(
                childStdInRead.writeRef(),
                childStdInWriteTmp.writeRef(),
                &securityAttributes,
                bufferSize));

            const HANDLE currentProcess = GetCurrentProcess();

            // https://docs.microsoft.com/en-us/windows/win32/api/handleapi/nf-handleapi-duplicatehandle

            // create a non-inheritable duplicate of the stdout reader
            SLANG_RETURN_FAIL_ON_FALSE(DuplicateHandle(
                currentProcess,
                childStdOutReadTmp,
                currentProcess,
                childStdOutRead.writeRef(),
                0,
                FALSE,
                DUPLICATE_SAME_ACCESS));
            // create a non-inheritable duplicate of the stderr reader
            if (childStdErrReadTmp)
                SLANG_RETURN_FAIL_ON_FALSE(DuplicateHandle(
                    currentProcess,
                    childStdErrReadTmp,
                    currentProcess,
                    childStdErrRead.writeRef(),
                    0,
                    FALSE,
                    DUPLICATE_SAME_ACCESS));
            // create a non-inheritable duplicate of the stdin writer
            SLANG_RETURN_FAIL_ON_FALSE(DuplicateHandle(
                currentProcess,
                childStdInWriteTmp,
                currentProcess,
                childStdInWrite.writeRef(),
                0,
                FALSE,
                DUPLICATE_SAME_ACCESS));
        }

        // TODO: switch to proper wide-character versions of these...
        STARTUPINFOW startupInfo;
        ZeroMemory(&startupInfo, sizeof(startupInfo));
        startupInfo.cb = sizeof(startupInfo);
        startupInfo.hStdError = childStdErrWrite;
        startupInfo.hStdOutput = childStdOutWrite;
        startupInfo.hStdInput = childStdInRead;
        startupInfo.dwFlags = STARTF_USESTDHANDLES;

        OSString pathBuffer;
        LPCWSTR path = nullptr;

        const auto& exe = commandLine.m_executableLocation;
        if (exe.m_type == ExecutableLocation::Type::Path)
        {
            // If it 'Path' specified we pass in as the lpApplicationName to limit
            // searching.
            pathBuffer = exe.m_pathOrName.toWString();
            path = pathBuffer.begin();
        }

        // Produce the command line string
        String cmdString = commandLine.toString();
        OSString cmdStringBuffer = cmdString.toWString();

        // Now we can actually get around to starting a process
        PROCESS_INFORMATION processInfo;
        ZeroMemory(&processInfo, sizeof(processInfo));

        // https://docs.microsoft.com/en-us/windows/win32/procthread/process-creation-flags

        DWORD createFlags = CREATE_NO_WINDOW;

        if (flags & Process::Flag::AttachDebugger)
        {
            createFlags |= CREATE_SUSPENDED;
        }

        // From docs:
        // If both lpApplicationName and lpCommandLine are non-NULL, the null-terminated string
        // pointed to by lpApplicationName specifies the module to execute, and the null-terminated
        // string pointed to by lpCommandLine specifies the command line.

        // JS:
        // Somewhat confusingly this means that even if lpApplicationName is specified, it muse
        // *ALSO* be included as the first whitespace delimited arg must *also* be the (possibly)
        // quoted executable

        // https://docs.microsoft.com/en-us/windows/desktop/api/processthreadsapi/nf-processthreadsapi-createprocessa
        // `CreateProcess` requires write access to this, for some reason...
        BOOL success = CreateProcessW(
            path,
            (LPWSTR)cmdStringBuffer.begin(),
            nullptr,
            nullptr,
            true,
            createFlags,
            nullptr, // TODO: allow specifying environment variables?
            nullptr,
            &startupInfo,
            &processInfo);

        if (!success)
        {
            DWORD err = GetLastError();
            SLANG_UNUSED(err);
            return SLANG_FAIL;
        }

        if (flags & Process::Flag::AttachDebugger)
        {
            // Lets see if we can set up to debug
            // https://docs.microsoft.com/en-us/windows/win32/debug/debugging-a-running-process

            // DebugActiveProcess(processInfo.dwProcessId);

            // Resume the thread
            ResumeThread(processInfo.hThread);
        }

        // close handles we are now done with
        CloseHandle(processInfo.hThread);

        // Save the process handle
        processHandle = processInfo.hProcess;
    }

    RefPtr<Stream> streams[Index(StdStreamType::CountOf)];

    if (childStdErrRead)
        streams[Index(StdStreamType::ErrorOut)] =
            new WinPipeStream(childStdErrRead.detach(), FileAccess::Read);
    streams[Index(StdStreamType::Out)] =
        new WinPipeStream(childStdOutRead.detach(), FileAccess::Read);
    streams[Index(StdStreamType::In)] =
        new WinPipeStream(childStdInWrite.detach(), FileAccess::Write);
    outProcess = new WinProcess(processHandle.detach(), streams[0].readRef());

    return SLANG_OK;
}

/* static */ void Process::sleepCurrentThread(Int timeInMs)
{
    ::Sleep(DWORD(timeInMs));
}

static uint64_t _getClockFrequency()
{
    LARGE_INTEGER timerFrequency;
    QueryPerformanceFrequency(&timerFrequency);
    return timerFrequency.QuadPart;
}

static const uint64_t g_frequency = _getClockFrequency();

/* static */ uint64_t Process::getClockFrequency()
{
    return g_frequency;
}

/* static */ uint64_t Process::getClockTick()
{
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return counter.QuadPart;
}

uint32_t Process::getId()
{
    return _getpid();
}


} // namespace Slang
