// TEST(smoke):CPP_COMPILER_COMPILE: -pass-through c -entry test -target callable

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __attribute__((__visibility__("default")))
#endif

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

EXTERN_C DLL_EXPORT int test(int intValue, const char* textValue, char* outTextValue)
{
    strcpy(outTextValue, textValue);
    return intValue;
}
