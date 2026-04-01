#define _CRT_SECURE_NO_WARNINGS
#include "../stacktrace-windows/common.h"

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

extern int exampleMain(int argc, char** argv);
extern const char* const g_logFileName;

int WinMain(
    HINSTANCE /* instance */,
    HINSTANCE /* prevInstance */,
    LPSTR /* commandLine */,
    int /*showCommand*/)

{
    FILE* logFile = fopen(g_logFileName, "w");
    __try
    {
        int argc = 0;
        char** argv = nullptr;
        return exampleMain(argc, argv);
    }
    __except (exceptionFilter(logFile, GetExceptionInformation()))
    {
        ::exit(1);
    }
}
