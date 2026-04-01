#pragma once

#include <stddef.h>
#include <stdint.h>

struct tint_CompileRequest
{
    const char* wgslCode;
    size_t wgslCodeLength;
};

struct tint_CompileResult
{
    const uint8_t* buffer;
    size_t bufferSize;
    const char* error;
};


typedef int (*tint_CompileFunc)(tint_CompileRequest* request, tint_CompileResult* result);

typedef void (*tint_FreeResultFunc)(tint_CompileResult* result);
