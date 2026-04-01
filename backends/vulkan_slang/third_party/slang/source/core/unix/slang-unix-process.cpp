// slang-unix-process.cpp
#include "../slang-common.h"
#include "../slang-memory-arena.h"
#include "../slang-process.h"
#include "../slang-string-escape-util.h"
#include "../slang-string-util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// #include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#if SLANG_OSX
#include <signal.h>
#endif

#include <time.h>

namespace Slang
{

class UnixProcess : public Process
{
public:
    // Process
    virtual bool isTerminated() SLANG_OVERRIDE;
    virtual bool waitForTermination(Int timeInMs) SLANG_OVERRIDE;
    virtual void terminate(int32_t returnValue) SLANG_OVERRIDE;
    virtual void kill(int32_t returnValue) SLANG_OVERRIDE;

    UnixProcess(pid_t pid, Stream* const* streams);

protected:
    /// Returns true if terminated
    bool _updateTerminationState(int options);

    bool m_isTerminated = false; ///< True if ths process is terminated
    pid_t m_pid;                 ///< The process id
};

class UnixPipeStream : public Stream
{
public:
    typedef UnixPipeStream ThisType;

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
    virtual bool isEnd() SLANG_OVERRIDE { return m_isClosed; }
    virtual bool canRead() SLANG_OVERRIDE { return _has(FileAccess::Read) && !m_isClosed; }
    virtual bool canWrite() SLANG_OVERRIDE { return _has(FileAccess::Write) && !m_isClosed; }
    virtual void close() SLANG_OVERRIDE;
    virtual SlangResult flush() SLANG_OVERRIDE;

    UnixPipeStream(int fd, FileAccess access, bool isOwned)
        : m_fd(fd), m_access(access), m_isOwned(isOwned), m_isClosed(false)
    {
    }

protected:
    /// This read file descriptor non blocking. Doing so will change the behavior of
    /// read - it can fail and return an error indicating there is no data, instead of blocking.
    /// Currently this mechanism isn't used, as checking via poll seemed to work.
    void _setReadNonBlocking()
    {
        // Makes non blocking
        if (_has(FileAccess::Read))
        {
            // Make non blocking, for read
            fcntl(m_fd, F_SETFL, fcntl(m_fd, F_GETFL) | O_NONBLOCK);
        }
    }
    bool _has(FileAccess access) const { return (Index(access) & Index(m_access)) != 0; }

    bool m_isClosed;     ///< If true this stream has been closed (ie cannot read/write to anymore)
    bool m_isOwned;      ///< True if m_fd is owned by this object.
    FileAccess m_access; ///< Access allowed to this stream - either Read or Write
    int m_fd;            /// The 'file descriptor' for the pipe
};

/* !!!!!!!!!!!!!!!!!!!!!! UnixProcess !!!!!!!!!!!!!!!!!!!!!!!!!!!! */

UnixProcess::UnixProcess(pid_t pid, Stream* const* streams)
    : m_pid(pid)
{
    // Set to an 'odd value'
    m_returnValue = -1;

    for (Index i = 0; i < SLANG_COUNT_OF(m_streams); ++i)
    {
        m_streams[i] = streams[i];
    }
}

bool UnixProcess::_updateTerminationState(int options)
{
    if (!m_isTerminated)
    {
        int childStatus;
        const pid_t terminatedPid = waitpid(m_pid, &childStatus, options);
        if (terminatedPid == -1)
        {
            // Guess we should just mark as terminated
            m_isTerminated = true;

            fprintf(stderr, "error: `waitpid` failed\n");
        }
        else if (terminatedPid == m_pid)
        {
            if (WIFEXITED(childStatus))
            {
                m_returnValue = (int)(int8_t)WEXITSTATUS(childStatus);
            }
            m_isTerminated = true;
        }
    }
    return m_isTerminated;
}

bool UnixProcess::isTerminated()
{
    if (m_isTerminated)
    {
        return true;
    }
    return _updateTerminationState(WNOHANG);
}

bool UnixProcess::waitForTermination(Int timeInMs)
{
    // If < 0 we will wait blocking until terminated
    if (timeInMs < 0)
    {
        while (!_updateTerminationState(0))
            ;
        return true;
    }

    // Note that the amount of time waiting is very approximate (we are relying on sleeps time and
    // don't take into account time outside of sleeping)

    // How often to test
    const Int checkRateMs = 100; /// Check every 0.1 seconds

    while (timeInMs > 0)
    {
        if (_updateTerminationState(WNOHANG))
        {
            return true;
        }

        // Work out how long to sleep for
        const Int sleepMs = (timeInMs >= checkRateMs) ? checkRateMs : timeInMs;

        // Sleep
        sleepCurrentThread(sleepMs);

        timeInMs -= sleepMs;
    }

    return _updateTerminationState(WNOHANG);
}

void UnixProcess::terminate(int32_t returnValue)
{
    // Using this mechanism, we can't set a returnValue so just ignore
    SLANG_UNUSED(returnValue);

    if (!isTerminated())
    {
        // Request the process terminates
        ::kill(m_pid, SIGTERM);
    }
}

void UnixProcess::kill(int32_t returnValue)
{
    if (!isTerminated())
    {
        // We waited, lets just terminate with kill
        ::kill(m_pid, SIGKILL);

        // Set the return value
        m_returnValue = returnValue;
        // Mark as terminated
        m_isTerminated = true;
    }
}

/* !!!!!!!!!!!!!!!!!!!!!! UnixPipeStream !!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void UnixPipeStream::close()
{
    if (!m_isClosed)
    {
        if (m_isOwned)
        {
            ::close(m_fd);
        }

        m_isClosed = true;
        // Make something hopefully invalid
        m_fd = -1;
    }
}

SlangResult UnixPipeStream::flush()
{
#if 0
    // https://stackoverflow.com/questions/43184035/flushing-pipe-without-closing-in-c
    // Makes the case that flushing is not applicable with pipes.
    if (canWrite())
    {
        // We might want to use
        ::fsync(m_fd);
    }
#endif
    return SLANG_OK;
}

SlangResult UnixPipeStream::read(void* buffer, size_t length, size_t& outReadBytes)
{
    outReadBytes = 0;

    if (!_has(FileAccess::Read))
    {
        return SLANG_E_NOT_AVAILABLE;
    }
    if (m_isClosed)
    {
        return SLANG_OK;
    }

    // Check if it's hung up.
    pollfd pollInfo;

    pollInfo.fd = m_fd;
    pollInfo.events = POLLIN | POLLHUP;
    pollInfo.revents = 0;

    // https://linux.die.net/man/2/poll

    // Return immediately
    const int pollTimeout = 0;

    const int pollResult = ::poll(&pollInfo, 1, pollTimeout);
    if (pollResult < 0)
    {
        return SLANG_FAIL;
    }

    // If there are no poll events, we are done
    if (pollResult == 0)
    {
        return SLANG_OK;
    }

    // If there is data read that first
    if (pollInfo.revents & POLLIN)
    {
        auto count = ::read(m_fd, buffer, length);

        // If it's -1 it seems like an error
        if (count == -1)
        {
            const int err = errno;

            // On non blocking pipe these indicate there could be more to come
            if (err == EAGAIN || err == EWOULDBLOCK)
            {
                return SLANG_OK;
            }
            // Okay - guess we have an error then
            return SLANG_FAIL;
        }

        outReadBytes = size_t(count);

        // If no bytes were wanted, then there could still be bytes in the pipe
        // before a HUP. So don't fall through to check for HUP.
        //
        // If some bytes *were* wanted and none were read, we can allow fall through to
        // handle HUP.
        if (length == 0 || count > 0)
        {
            return SLANG_OK;
        }
    }

    if (pollInfo.revents & POLLHUP)
    {
        close();
    }

    return SLANG_OK;
}

SlangResult UnixPipeStream::write(const void* buffer, size_t length)
{
    if (!_has(FileAccess::Write))
    {
        return SLANG_E_NOT_AVAILABLE;
    }
    if (m_isClosed)
    {
        // The pipe is closed
        return SLANG_FAIL;
    }

    pollfd pollInfo;

    pollInfo.fd = m_fd;
    pollInfo.events = POLLHUP;
    pollInfo.revents = 0;

    // https://linux.die.net/man/2/poll

    // Return immediately
    const int pollTimeout = 0;

    int pollResult = ::poll(&pollInfo, 1, pollTimeout);
    if (pollResult < 0)
    {
        return SLANG_FAIL;
    }

    if (pollInfo.revents & POLLHUP)
    {
        close();
        return SLANG_FAIL;
    }

    const ssize_t writeResult = ::write(m_fd, buffer, length);

    if (writeResult < 0 || size_t(writeResult) != length)
    {
        return SLANG_FAIL;
    }

    return SLANG_OK;
}

/* !!!!!!!!!!!!!!!!!!!!!! Process !!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* static */ UnownedStringSlice Process::getExecutableSuffix()
{
#if __CYGWIN__
    return UnownedStringSlice::fromLiteral(".exe");
#else
    return UnownedStringSlice::fromLiteral("");
#endif
}

/* static */ StringEscapeHandler* Process::getEscapeHandler()
{
    return StringEscapeUtil::getHandler(StringEscapeUtil::Style::Space);
}

static const int kCannotExecute = 126;

static int pipeCLOEXEC(int pipefd[2])
{
#if SLANG_APPLE_FAMILY
    // without pipe2 on macOS, there's an unavoidable race here where
    // another process could fork and execv with execWatchPipe before we
    // can set CLOEXEC on it...
    if (pipe(pipefd) == -1 || fcntl(pipefd[1], F_SETFD, FD_CLOEXEC) == -1 ||
        fcntl(pipefd[0], F_SETFD, FD_CLOEXEC) == -1)
    {
        return -1;
    }
    return 0;
#else
    return pipe2(pipefd, O_CLOEXEC);
#endif
}

/* static */ SlangResult Process::create(
    const CommandLine& commandLine,
    Process::Flags,
    RefPtr<Process>& outProcess)
{
    const char* whatFailed = nullptr;
    pid_t childPid;

    //
    // Set up command line
    //
    List<char const*> argPtrs;

    const auto& exe = commandLine.m_executableLocation;

    // Add the command
    argPtrs.add(exe.m_pathOrName.getBuffer());

    // Add all the args - they don't need any explicit escaping
    for (auto arg : commandLine.m_args)
    {
        // All args for this target must be unescaped (as they are in CommandLine)
        argPtrs.add(arg.getBuffer());
    }

    // Terminate with a null
    argPtrs.add(nullptr);

    //
    // Set up pipes
    //
    int stdinPipe[2] = {-1, -1};
    int stdoutPipe[2] = {-1, -1};
    int stderrPipe[2] = {-1, -1};

    // We will create this pipe with O_CLOEXEC, so that it gets closed
    // automatically if the child's exec succeeds
    int execWatchPipe[2] = {-1, -1};

    if (pipe(stdinPipe) == -1 || pipe(stdoutPipe) == -1 || pipe(stderrPipe) == -1 ||
        pipeCLOEXEC(execWatchPipe) == -1)
    {
        whatFailed = "pipe";
        goto reportErr;
    }

    // Make sure that none of our pipes are going to be clobbered by dup2 to
    // 0,1,2 in the child.
    whatFailed = "fcntl";
    int next;
    if (stdinPipe[0] < 3)
    {
        if (-1 == (next = fcntl(stdinPipe[0], F_DUPFD, 3)))
        {
            goto reportErr;
        }
        close(stdinPipe[0]);
        stdinPipe[0] = next;
    }
    if (stdoutPipe[1] < 3)
    {
        if (-1 == (next = fcntl(stdoutPipe[1], F_DUPFD, 3)))
        {
            goto reportErr;
        }
        close(stdoutPipe[1]);
        stdoutPipe[1] = next;
    }
    if (stderrPipe[1] < 3)
    {
        if (-1 == (next = fcntl(stderrPipe[1], F_DUPFD, 3)))
        {
            goto reportErr;
        }
        close(stderrPipe[1]);
        stderrPipe[1] = next;
    }
    if (execWatchPipe[1] < 3)
    {
        if (-1 == (next = fcntl(execWatchPipe[1], F_DUPFD_CLOEXEC, 3)))
        {
            goto reportErr;
        }
        close(execWatchPipe[1]);
        execWatchPipe[1] = next;
    }
    whatFailed = nullptr;

    childPid = fork();
    if (childPid == -1)
    {
        whatFailed = "fork";
        goto reportErr;
    }

    if (childPid == 0)
    {
        // We are the child process.

        // Close unused fds and duplicate into standard handles

        ::close(execWatchPipe[0]);
        ::close(stdinPipe[1]);
        ::close(stdoutPipe[0]);
        ::close(stderrPipe[0]);

        dup2(stdinPipe[0], STDIN_FILENO);
        ::close(stdinPipe[0]);
        dup2(stdoutPipe[1], STDOUT_FILENO);
        ::close(stdoutPipe[1]);
        dup2(stderrPipe[1], STDERR_FILENO);
        ::close(stderrPipe[1]);

        // Reset locale to ensure the output can be parsed regardless of user's
        // locale.
        setenv("LC_ALL", "C", 1);

        if (exe.m_type == ExecutableLocation::Type::Path)
        {
            // Use the specified path (ie don't search)
            ::execv(argPtrs[0], (char* const*)&argPtrs[0]);
        }
        else
        {
            // Search for the executable
            ::execvp(argPtrs[0], (char* const*)&argPtrs[0]);
        }

        // If we get here, then `exec` failed

        // Signal the failure to our parent
        int execErr = errno;
        if (::write(execWatchPipe[1], &execErr, sizeof(execErr)))
            fprintf(stderr, "error: `exec` watch pipe write failed\n");

        // NOTE! Because we have dup2 into STDERR_FILENO, this error will *not* generally appear on
        // the terminal but in the stderrPipe.
        fprintf(stderr, "error: `exec` failed\n");

        // Terminate with failure.
        // Call _exit() rather than exit() so we don't run anything registered with atexit()
        ::_exit(kCannotExecute);
    }
    else
    {
        // We are the parent process
        ::close(execWatchPipe[1]);
        ::close(stdinPipe[0]);
        ::close(stdoutPipe[1]);
        ::close(stderrPipe[1]);

        RefPtr<Stream> streams[Index(StdStreamType::CountOf)];

        // Previously code didn't need to close, so we'll make stream now own the handles
        streams[Index(StdStreamType::Out)] =
            new UnixPipeStream(stdoutPipe[0], FileAccess::Read, true);
        stdoutPipe[0] = -1;
        streams[Index(StdStreamType::ErrorOut)] =
            new UnixPipeStream(stderrPipe[0], FileAccess::Read, true);
        stderrPipe[0] = -1;
        streams[Index(StdStreamType::In)] =
            new UnixPipeStream(stdinPipe[1], FileAccess::Write, true);
        stdinPipe[1] = -1;

        // Check that the exec actually succeeded
        int execErrCode;
        // Our success is if we read zero bytes, indicating that the pipe was
        // closed by the child's exec and O_CLOEXEC. (and us just above)
        const int readRes = ::read(execWatchPipe[0], &execErrCode, sizeof(execErrCode));
        if (readRes < 0)
        {
            whatFailed = "read from forked process";
            goto reportErr;
        }
        else if (readRes > 0)
        {
            // exec failed, and the child reported back to us
            // don't print messages by default, as we do some speculative
            // execution of processes to see if they exist and it gets noisy
            const bool verbose = false;
            if (verbose)
            {
                fprintf(
                    stderr,
                    "error: exec for \"%s\" failed: %s\n",
                    argPtrs[0],
                    ::strerror(execErrCode));
            }
            whatFailed = "exec";
            // Don't report the exec as we expect some of them to fail
            goto closePipes;
        }

        outProcess = new UnixProcess(childPid, streams[0].readRef());
    }

    goto closePipes;

    // Report any error and then cleanup
reportErr:
    fprintf(stderr, "error: `%s` failed (%s)\n", whatFailed, strerror(errno));
closePipes:
    ::close(execWatchPipe[0]);
    ::close(execWatchPipe[1]);
    ::close(stdinPipe[0]);
    ::close(stdinPipe[1]);
    ::close(stderrPipe[0]);
    ::close(stderrPipe[1]);
    ::close(stdoutPipe[0]);
    ::close(stdoutPipe[1]);

    return whatFailed ? SLANG_FAIL : SLANG_OK;
}

/* static */ uint64_t Process::getClockFrequency()
{
    return 1000000000;
}

/* static */ uint64_t Process::getClockTick()
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return uint64_t(now.tv_sec) * 1000000000 + now.tv_nsec;
}

/* static */ void Process::sleepCurrentThread(Int timeInMs)
{
    struct timespec timeSpec;

    if (timeInMs >= 1000)
    {
        timeSpec.tv_sec = timeInMs / 1000;
        timeSpec.tv_nsec = (timeInMs % 1000) * 1000 * 1000;
    }
    else if (timeInMs > 0)
    {
        timeSpec.tv_sec = 0;
        timeSpec.tv_nsec = timeInMs * 1000 * 1000;
    }
    else
    {
        timeSpec.tv_sec = 0;
        timeSpec.tv_nsec = 0;
    }
    nanosleep(&timeSpec, nullptr);
}

/* static */ SlangResult Process::getStdStream(StdStreamType type, RefPtr<Stream>& out)
{
    switch (type)
    {
    case StdStreamType::In:
        {
            out = new UnixPipeStream(STDIN_FILENO, FileAccess::Read, false);
            break;
        }
    case StdStreamType::Out:
        {
            out = new UnixPipeStream(STDOUT_FILENO, FileAccess::Write, false);
            break;
        }
    case StdStreamType::ErrorOut:
        {
            out = new UnixPipeStream(STDERR_FILENO, FileAccess::Write, false);
            break;
        }
    default:
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

uint32_t Process::getId()
{
    return getpid();
}

} // namespace Slang
