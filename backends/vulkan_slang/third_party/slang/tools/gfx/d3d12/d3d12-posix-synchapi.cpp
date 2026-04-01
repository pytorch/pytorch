#include "d3d12-posix-synchapi.h"

#include "slang.h"

#if SLANG_LINUX_FAMILY

#include "core/slang-common.h"

#include <cerrno>
#include <fcntl.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/poll.h>
#include <sys/timerfd.h>
#include <unistd.h>

// To keep aligned with the d3d12 API, we store file descriptors in the low 32
// bits of HANDLEs.
static int _handleToFD(HANDLE h)
{
    auto i = reinterpret_cast<std::intptr_t>(h);
    int fd = static_cast<int>(i);
    return fd;
}

static int _handleToFlags(HANDLE h)
{
    auto i = reinterpret_cast<std::intptr_t>(h) >> 32;
    int flags = static_cast<int>(i);
    return flags;
}

static HANDLE _fdToHandle(int fd, int flags)
{
    static_assert(sizeof(int) <= 4);
    static_assert(sizeof(std::intptr_t) >= 8);
    return reinterpret_cast<HANDLE>(static_cast<std::intptr_t>(flags) << 32 | fd);
}


HANDLE CreateEventEx(
    LPSECURITY_ATTRIBUTES lpEventAttributes,
    LPCSTR lpName,
    DWORD dwFlags,
    DWORD dwDesiredAccess)
{
    int fd = ::eventfd(dwFlags & CREATE_EVENT_INITIAL_SET ? 1 : 0, EFD_CLOEXEC | EFD_NONBLOCK);
    // Make sure not to return a zero handle, duplicate the fd if necessary
    if (fd == 0)
    {
        int nextFd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
        if (fcntl(nextFd, F_SETFL, O_NONBLOCK) == -1)
        {
            close(nextFd);
            nextFd = -1;
        }
        close(fd);
        fd = nextFd;
    }
    return fd == -1 ? nullptr : _fdToHandle(fd, dwFlags);
}

BOOL CloseHandle(HANDLE h)
{
    if (h == 0)
    {
        return 1;
    }
    // TODO: Windows does reference counting, how to, dupfd?
    return ::close(_handleToFD(h)) == 0;
    return 1;
}

BOOL ResetEvent(HANDLE h)
{
    int fd = _handleToFD(h);
    pollfd pfd{fd, POLLIN, 0};
    uint64_t x;
    int r = 0;
    int nEvents = poll(&pfd, 1, 0);
    if (pfd.revents != POLLIN)
    {
        // Nothing to read, already reset
        return 1;
    }
    if (nEvents != 1)
    {
        return 0;
    }
    r = read(fd, &x, sizeof(x));
    if (r == sizeof(x))
    {
        // We reset it
        return 1;
    }
    if (r == -1 && errno == EAGAIN)
    {
        // Something else reset it
        return 1;
    }
    return 0;
}

BOOL SetEvent(HANDLE h)
{
    int fd = _handleToFD(h);
    pollfd pfd{fd, POLLOUT, 0};
    for (;;)
    {
        int nEvents = poll(&pfd, 1, -1);
        SLANG_ASSERT(nEvents != -1);
        SLANG_ASSERT(nEvents != 0); // shouldn't have timed out
        const uint64_t one = 1;
        int w = ::write(fd, &one, sizeof(one));
        if (w == sizeof(one))
        {
            return 1;
        }
        if (errno != EAGAIN)
        {
            return 0;
        }
    }
}

DWORD WaitForSingleObject(const HANDLE h, const DWORD ms)
{
    int fd = _handleToFD(h);
    bool manualReset = _handleToFlags(h) & CREATE_EVENT_MANUAL_RESET;
    pollfd pfd{fd, POLLIN, 0};
    uint64_t x;
    int r = 0;
    // Implement unlimited waits as timing out with WAIT_FAILED after 5
    // seconds. It's probably something fishy with d3dvk-proton or our synchapi
    // implementation
    const bool isInfinite = ms == INFINITE;
    const DWORD fiveSeconds = 5000;
    int nEvents = poll(&pfd, 1, isInfinite ? fiveSeconds : ms);
    if (pfd.revents != POLLIN)
    {
        return WAIT_FAILED;
    }
    if (nEvents == -1)
    {
        return WAIT_FAILED;
    }
    if (nEvents == 0)
    {
        return isInfinite ? WAIT_FAILED : WAIT_TIMEOUT;
    }
    if (manualReset)
    {
        return WAIT_OBJECT_0;
    }
    r = read(fd, &x, sizeof(x));
    if (r == sizeof(x))
    {
        return WAIT_OBJECT_0;
    }
    if (r == -1 && errno == EAGAIN)
    {
        return isInfinite ? WAIT_FAILED : WAIT_TIMEOUT;
    }
    return WAIT_FAILED;
}

DWORD WaitForMultipleObjects(DWORD n, const HANDLE* hs, BOOL bWaitAll, DWORD requestedMs)
{
    if (n == 0)
    {
        return bWaitAll ? WAIT_OBJECT_0 : WAIT_FAILED;
    }

    // Bail out of infinite waits after 5 seconds as it's probably a
    // driver/vkd3d-proton/synchapi bug
    const bool isInfinite = requestedMs == INFINITE;
    const DWORD fiveSeconds = 5000;
    const auto dwMilliseconds = isInfinite ? fiveSeconds : requestedMs;

    DWORD res;
    int fds[n];
    int flagss[n];
    epoll_event evs[n + 1]; // +1 for our timer
    int ufd = -1;
    int epfd = epoll_create1(EPOLL_CLOEXEC);
    if (epfd == -1)
    {
        goto fail;
    }

    for (int i = 0; i < n; ++i)
    {
        fds[i] = _handleToFD(hs[i]);
        flagss[i] = _handleToFlags(hs[i]);
        epoll_event ev;
        ev.data.fd = fds[i];
        ev.events = EPOLLIN | EPOLLONESHOT;
        if (epoll_ctl(epfd, EPOLL_CTL_ADD, fds[i], &ev) == -1)
        {
            goto fail;
        }
    }

    // The wait all case can't be made correct on linux, as we can't atomically
    // read from several fds and eventfd is the interface available from
    // vkd3d-proton.
    //
    // As a best-effort we wait until they're all free, then grab them on one
    // after the other, and put the values back if we can't claim them all, it
    // sucks.
    //
    if (bWaitAll)
    {
        // Use a timer to easily know for sure when we've timed out
        if (dwMilliseconds != INFINITE)
        {
            ufd = timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC);
            if (ufd == -1)
            {
                goto fail;
            }
            itimerspec spec;
            spec.it_interval.tv_sec = 0;
            spec.it_interval.tv_nsec = 0;
            spec.it_value.tv_sec = 0;
            spec.it_value.tv_nsec = 1000000 * dwMilliseconds;
            if (timerfd_settime(ufd, 0, &spec, nullptr) == -1)
            {
                goto fail;
            }
            evs[n].data.fd = ufd;
            evs[n].events = EPOLLIN | EPOLLONESHOT;
            if (epoll_ctl(epfd, EPOLL_CTL_ADD, ufd, &evs[n]) == -1)
            {
                goto fail;
            }
        }

        bool timesUp = false;
        int nSeenEvents = 0;
        // Repeatedly call epoll_wait to eliminate read fds until the timer
        // expires or we have elimininated all our fds
        do
        {
            do
            {
                // Wait until epoll tells us they're all available, or the timer is
                const int nEvents = epoll_wait(epfd, evs, n + 1, -1);
                // We didn't specify a timeout, so 0 results is abnormal
                if (nEvents < 1)
                {
                    goto fail;
                }

                // Process all the returned fds
                for (int i = 0; i < nEvents; ++i)
                {
                    if (!(evs[i].events & EPOLLIN))
                    {
                        // Something exceptional happened on the fd
                        // Possibly we could just continue and hope it doesn't
                        // happen again?
                        goto fail;
                    }
                    if (evs[i].data.fd == ufd)
                    {
                        // We're out of time, make this the last loop
                        uint64_t x;
                        int r = read(ufd, &x, sizeof(x));
                        if (r == sizeof(x))
                        {
                            timesUp = true;
                        }
                        else
                        {
                            goto fail;
                        }
                    }
                    else
                    {
                        // EPOLLONESHOT has removed this fd
                        ++nSeenEvents;
                    }
                }
            } while (!(timesUp || nSeenEvents == n));

            // If we got here without seeing enough events, we must have timed out
            if (nSeenEvents < n)
            {
                res = isInfinite ? WAIT_FAILED : WAIT_TIMEOUT;
                goto end;
            }

            // See if all the events are readable.
            // This isn't strictly necessary from a correctness point of view,
            // but since we're not correct anything we can do helps, and it
            // makes the code a bit cleaner.
            // Put all the events back in our epoll instance and see if they're
            // all readable.
            for (int i = 0; i < n; ++i)
            {
                epoll_event modEv;
                modEv.data.fd = fds[i];
                modEv.events = EPOLLIN | EPOLLONESHOT;
                if (epoll_ctl(epfd, EPOLL_CTL_MOD, fds[i], &modEv) == -1)
                {
                    goto fail;
                }
            }
            // Remove the timer if we're using it
            if (dwMilliseconds != INFINITE && epoll_ctl(epfd, EPOLL_CTL_DEL, ufd, nullptr) == -1)
            {
                goto fail;
            }
            int nEvents = epoll_wait(epfd, evs, n, 0);
            if (nEvents < 0)
            {
                goto fail;
            }
            else if (nEvents < n)
            {
                // They're not all still available :(
                // Put our timer back in and try again from the top
                if (dwMilliseconds != INFINITE &&
                    epoll_ctl(epfd, EPOLL_CTL_ADD, ufd, &evs[n]) == -1)
                {
                    goto fail;
                }
                // Put back the any fds which did trigger
                for (int i = 0; i < nEvents; ++i)
                {
                    epoll_event modEv = evs[i];
                    modEv.events = EPOLLIN | EPOLLONESHOT;
                    if (epoll_ctl(epfd, EPOLL_CTL_MOD, modEv.data.fd, &modEv) == -1)
                    {
                        goto fail;
                    }
                }
                continue;
            }
            else if (nEvents == n)
            {
                for (int i = 0; i < nEvents; ++i)
                {
                    if (!(evs->events & EPOLLIN))
                    {
                        goto fail;
                    }
                }
            }

            // Try to grab all the events
            uint64_t vs[n];
            int i;
            bool failure = false;
            for (i = 0; i < n; ++i)
            {
                if (flagss[i] & CREATE_EVENT_MANUAL_RESET)
                {
                    // We don't need to read this to unset it
                    continue;
                }
                int r = read(fds[i], &vs[i], sizeof(vs[i]));
                if (r == sizeof(vs[i]))
                {
                    continue;
                }
                else if (r == -1 && errno == EAGAIN)
                {
                    // contention, put things back and try again
                    break;
                }
                else
                {
                    // failure, put things back and fail
                    failure = true;
                    break;
                }
            }
            if (i < n)
            {
                // contention or failure
                for (int j = 0; j < i; ++j)
                {
                    if (flagss[i] & CREATE_EVENT_MANUAL_RESET)
                    {
                        // We didn't read, so we shouldn't write
                        continue;
                    }
                    // TODO: If this doesn't succeed then another thread has jumped
                    // in between our non-atomic reads earlier, oops!
                    //
                    // This is just one case of failure we can detect,
                    // arbitrarily many things may have happened between reads
                    // and we just wouldn't know...
                    int w = write(fds[j], &vs[j], sizeof(vs[j]));
                    SLANG_ASSERT(w == sizeof(vs[j]));
                }
                if (failure)
                {
                    goto fail;
                }
            }
            else
            {
                // success
                res = WAIT_OBJECT_0;
                goto end;
            }

            // If we get here then we've got some contention, go back to the top and try again (or
            // timeout)
        } while (!timesUp);
    }
    else
    {
        // Wait any
        const int nEvents =
            epoll_wait(epfd, evs, n, dwMilliseconds == INFINITE ? -1 : dwMilliseconds);
        if (nEvents == -1)
        {
            goto fail;
        }
        if (nEvents == 0)
        {
            res = isInfinite ? WAIT_FAILED : WAIT_TIMEOUT;
            goto end;
        }
        // Try reads until we get one
        for (int i = 0; i < nEvents; ++i)
        {
            uint64_t x;
            if (!evs[i].events & EPOLLIN)
            {
                continue;
            }
            const int r = ::read(evs[i].data.fd, &x, sizeof(x));
            if (r == sizeof(x))
            {
                res = WAIT_OBJECT_0;
                goto end;
            }
            if (errno != EAGAIN)
            {
                goto fail;
            }
            // Some other waiter got this one first
        }
    }

    goto end;
fail:
    res = WAIT_FAILED;
end:
    close(ufd);
    close(epfd);
    return res;
}

#endif // SLANG_LINUX_FAMILY
