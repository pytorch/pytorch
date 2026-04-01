#ifndef _WIN32
#ifndef SLANG_CORE_SECURE_CRT_H
#define SLANG_CORE_SECURE_CRT_H
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <wchar.h>

inline void memcpy_s(void* dest, [[maybe_unused]] size_t destSize, const void* src, size_t count)
{
    assert(destSize >= count);
    memcpy(dest, src, count);
}

#define _TRUNCATE ((size_t)-1)
#define _stricmp strcasecmp

inline void fopen_s(FILE** f, const char* fileName, const char* mode)
{
    *f = fopen(fileName, mode);
}

inline size_t fread_s(
    void* buffer,
    [[maybe_unused]] size_t bufferSize,
    size_t elementSize,
    size_t count,
    FILE* stream)
{
    assert(bufferSize >= elementSize * count);
    return fread(buffer, elementSize, count, stream);
}

inline size_t wcsnlen_s(const wchar_t* str, size_t /*numberofElements*/)
{
    return wcslen(str);
}

inline size_t strnlen_s(const char* str, size_t numberOfElements)
{
#if defined(__CYGWIN__)
    const char* cur = str;
    if (str)
    {
        const char* const end = str + numberOfElements;
        while (*cur && cur < end)
            cur++;
    }
    return size_t(cur - str);
#else
    return strnlen(str, numberOfElements);
#endif
}

__attribute__((format(printf, 3, 4))) inline int sprintf_s(
    char* buffer,
    size_t sizeOfBuffer,
    const char* format,
    ...)
{
    va_list argptr;
    va_start(argptr, format);
    int rs = vsnprintf(buffer, sizeOfBuffer, format, argptr);
    va_end(argptr);
    return rs;
}

// A patch was submitted to GCC wchar_t support in 2001, so I'm sure we can
// enable this any day now...
// __attribute__((format(wprintf, 3, 4)))
inline int swprintf_s(wchar_t* buffer, size_t sizeOfBuffer, const wchar_t* format, ...)
{
    va_list argptr;
    va_start(argptr, format);
    int rs = vswprintf(buffer, sizeOfBuffer, format, argptr);
    va_end(argptr);
    return rs;
}

inline void wcscpy_s(wchar_t* strDestination, size_t /*numberOfElements*/, const wchar_t* strSource)
{
    wcscpy(strDestination, strSource);
}
inline void strcpy_s(char* strDestination, size_t /*numberOfElements*/, const char* strSource)
{
    strcpy(strDestination, strSource);
}

inline void wcsncpy_s(
    wchar_t* strDestination,
    size_t /*numberOfElements*/,
    const wchar_t* strSource,
    size_t count)
{
    wcsncpy(strDestination, strSource, count);
}
inline void strncpy_s(
    char* strDestination,
    size_t /*numberOfElements*/,
    const char* strSource,
    size_t count)
{
    strncpy(strDestination, strSource, count);
}
#endif
#endif
