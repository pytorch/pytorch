#pragma once

#include "slang.h"

#if SLANG_LINUX_FAMILY

#pragma push_macro("WIN32_LEAN_AND_MEAN")
#pragma push_macro("NOMINMAX")
#undef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#undef NOMINMAX
#define NOMINMAX
#include <windows.h>
#pragma pop_macro("NOMINMAX")
#pragma pop_macro("WIN32_LEAN_AND_MEAN")

////////////////////////////////////////////////////////////////
//
// It's important to note that due to platform constraints this can't be a
// totally faithful implementation of the Windows API.
//
// Notably, the "wait all" case in WaitForMultipleObjects can't be made correct
// on linux, as we can't atomically read from several fds and eventfd is the
// interface available from vkd3d-proton.
//
////////////////////////////////////////////////////////////////

//
// The synchapi types and macros used in gfx
//
#define INFINITE 0xffffffff

#define WAIT_FAILED 0xffffffff
#define WAIT_OBJECT_0 0

typedef struct _SECURITY_ATTRIBUTES* LPSECURITY_ATTRIBUTES;

#define CREATE_EVENT_MANUAL_RESET 1
#define CREATE_EVENT_INITIAL_SET 2

#define SYNCHRONIZE 0x00100000
#define STANDARD_RIGHTS_REQUIRED 0x000f0000
#define EVENT_ALL_ACCESS (STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 0x3)

HANDLE CreateEventEx(
    LPSECURITY_ATTRIBUTES lpEventAttributes,
    LPCSTR lpName,
    DWORD dwFlags,
    DWORD dwDesiredAccess);

BOOL CloseHandle(HANDLE h);

BOOL ResetEvent(HANDLE h);

BOOL SetEvent(HANDLE h);

DWORD WaitForSingleObject(HANDLE h, DWORD ms);

DWORD WaitForMultipleObjects(
    DWORD nHandles,
    const HANDLE* handles,
    BOOL bWaitAll,
    DWORD dwMilliseconds);

#endif // SLANG_LINUX_FAMILY
