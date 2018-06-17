/*
    Copyright (c) 2005-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#define HARNESS_DEFAULT_MIN_THREADS 4
#define HARNESS_DEFAULT_MAX_THREADS 4

// Need to include "tbb/tbb_config.h" to obtain the definition of __TBB_DEFINE_MIC.
#include "tbb/tbb_config.h"

#if !__TBB_TODO || __TBB_WIN8UI_SUPPORT
#define HARNESS_NO_PARSE_COMMAND_LINE 1
#include "harness.h"
int TestMain() {
    return Harness::Skipped;
}
#else /* __TBB_TODO */
// TODO: There are a lot of problems with unloading DLL which uses TBB with automatic initialization

#if __TBB_DEFINE_MIC

#ifndef _USRDLL
#define HARNESS_NO_PARSE_COMMAND_LINE 1
#include "harness.h"
int TestMain() {
    return Harness::Skipped;
}
#endif

#else /* !__MIC__ */

#if _WIN32 || _WIN64
#include "tbb/machine/windows_api.h"
#else
#include <dlfcn.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdexcept>

#if TBB_USE_EXCEPTIONS
    #include "harness_report.h"
#endif

#ifdef _USRDLL
#include "tbb/task_scheduler_init.h"

class CModel {
public:
    CModel(void) {};
    static tbb::task_scheduler_init tbb_init;

    void init_and_terminate( int );
};

tbb::task_scheduler_init CModel::tbb_init(1);

//! Test that task::initialize and task::terminate work when doing nothing else.
/** maxthread is treated as the "maximum" number of worker threads. */
void CModel::init_and_terminate( int maxthread ) {
    for( int i=0; i<200; ++i ) {
        switch( i&3 ) {
            default: {
                tbb::task_scheduler_init init( rand() % maxthread + 1 );
                break;
            }
            case 0: {
                tbb::task_scheduler_init init;
                break;
            }
            case 1: {
                tbb::task_scheduler_init init( tbb::task_scheduler_init::automatic );
                break;
            }
            case 2: {
                tbb::task_scheduler_init init( tbb::task_scheduler_init::deferred );
                init.initialize( rand() % maxthread + 1 );
                init.terminate();
                break;
            }
        }
    }
}

extern "C"
#if _WIN32 || _WIN64
__declspec(dllexport)
#endif
void plugin_call(int maxthread)
{
    srand(2);
    __TBB_TRY {
        CModel model;
        model.init_and_terminate(maxthread);
    } __TBB_CATCH( std::runtime_error& error ) {
#if TBB_USE_EXCEPTIONS
        REPORT("ERROR: %s\n", error.what());
#endif /* TBB_USE_EXCEPTIONS */
    }
}

#else /* _USRDLL undefined */

#include "harness.h"
#include "harness_dynamic_libs.h"
#include "harness_tls.h"

extern "C" void plugin_call(int);

void report_error_in(const char* function_name)
{
#if _WIN32 || _WIN64
    char* message;
    int code = GetLastError();

    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
        NULL, code,MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (char*)&message, 0, NULL );
#else
    char* message = (char*)dlerror();
    int code = 0;
#endif
    REPORT( "%s failed with error %d: %s\n", function_name, code, message);

#if _WIN32 || _WIN64
    LocalFree(message);
#endif
}

typedef void (*PLUGIN_CALL)(int);

#if __linux__
    #define RML_LIBRARY_NAME(base) TEST_LIBRARY_NAME(base) ".1"
#else
    #define RML_LIBRARY_NAME(base) TEST_LIBRARY_NAME(base)
#endif

int TestMain () {
#if !RML_USE_WCRM
    PLUGIN_CALL my_plugin_call;

    LimitTLSKeysTo limitTLS(10);

    Harness::LIBRARY_HANDLE hLib;
#if _WIN32 || _WIN64
    hLib = LoadLibrary("irml.dll");
    if ( !hLib )
        hLib = LoadLibrary("irml_debug.dll");
    if ( !hLib )
        return Harness::Skipped; // No shared RML, skip the test
    FreeLibrary(hLib);
#else /* !WIN */
#if __TBB_ARENA_PER_MASTER
    hLib = dlopen(RML_LIBRARY_NAME("libirml"), RTLD_LAZY);
    if ( !hLib )
        hLib = dlopen(RML_LIBRARY_NAME("libirml_debug"), RTLD_LAZY);
    if ( !hLib )
        return Harness::Skipped;
    dlclose(hLib);
#endif /* __TBB_ARENA_PER_MASTER */
#endif /* OS */
    for( int i=1; i<100; ++i ) {  
        REMARK("Iteration %d, loading plugin library...\n", i);
        hLib = Harness::OpenLibrary(TEST_LIBRARY_NAME("test_model_plugin_dll"));
        if ( !hLib ) {
#if !__TBB_NO_IMPLICIT_LINKAGE
#if _WIN32 || _WIN64
            report_error_in("LoadLibrary");
#else
            report_error_in("dlopen");
#endif
            return -1;
#else
            return Harness::Skipped;
#endif
        }
        my_plugin_call = (PLUGIN_CALL)Harness::GetAddress(hLib, "plugin_call");
        if (my_plugin_call==NULL) {
#if _WIN32 || _WIN64
            report_error_in("GetProcAddress");
#else
            report_error_in("dlsym");
#endif
            return -1;
        }
        REMARK("Calling plugin method...\n");
        my_plugin_call(MaxThread);

        REMARK("Unloading plugin library...\n");
        Harness::CloseLibrary(hLib);
    } // end for(1,100)

    return Harness::Done;
#else
    return Harness::Skipped;
#endif /* !RML_USE_WCRM */
}

#endif//_USRDLL
#endif//__MIC__

#endif /*__TBB_WIN8UI_SUPPORT*/
