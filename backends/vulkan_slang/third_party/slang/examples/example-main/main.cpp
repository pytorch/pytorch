#include "../stacktrace-windows/common.h"

#include <stdio.h>
#include <stdlib.h>

extern int exampleMain(int argc, char** argv);

#if defined(_WIN32)

#include <windows.h>

int main(int argc, char** argv)
{
    __try
    {
        return exampleMain(argc, argv);
    }
    __except (exceptionFilter(stdout, GetExceptionInformation()))
    {
        ::exit(1);
    }
}

#else // defined(_WIN32)

int main(int argc, char** argv)
{
    // TODO: Catch exception and print stack trace also on non-Windows platforms.
    return exampleMain(argc, argv);
}

#endif
