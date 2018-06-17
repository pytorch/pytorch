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

// IMPORTANT: To use assertion handling in TBB, exactly one of the TBB source files
// should #include tbb_assert_impl.h thus instantiating assertion handling routines.
// The intent of putting it to a separate file is to allow some tests to use it
// as well in order to avoid dependency on the library.

// include headers for required function declarations
#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#if _MSC_VER
#include <crtdbg.h>
#endif

#if _MSC_VER >= 1400
#define __TBB_EXPORTED_FUNC   __cdecl
#else
#define __TBB_EXPORTED_FUNC
#endif

using namespace std;

#if __TBBMALLOC_BUILD
namespace rml { namespace internal {
#else
namespace tbb {
#endif
    //! Type for an assertion handler
    typedef void(*assertion_handler_type)( const char* filename, int line, const char* expression, const char * comment );

    static assertion_handler_type assertion_handler;

    assertion_handler_type __TBB_EXPORTED_FUNC set_assertion_handler( assertion_handler_type new_handler ) {
        assertion_handler_type old_handler = assertion_handler;
        assertion_handler = new_handler;
        return old_handler;
    }

    void __TBB_EXPORTED_FUNC assertion_failure( const char* filename, int line, const char* expression, const char* comment ) {
        if( assertion_handler_type a = assertion_handler ) {
            (*a)(filename,line,expression,comment);
        } else {
            static bool already_failed;
            if( !already_failed ) {
                already_failed = true;
                fprintf( stderr, "Assertion %s failed on line %d of file %s\n",
                         expression, line, filename );
                if( comment )
                    fprintf( stderr, "Detailed description: %s\n", comment );
#if _MSC_VER && _DEBUG
                if(1 == _CrtDbgReport(_CRT_ASSERT, filename, line, "tbb_debug.dll", "%s\r\n%s", expression, comment?comment:""))
                        _CrtDbgBreak();
#else
                fflush(stderr);
                abort();
#endif
            }
        }
    }

#if defined(_MSC_VER)&&_MSC_VER<1400
#   define vsnprintf _vsnprintf
#endif

#if !__TBBMALLOC_BUILD
    namespace internal {
        //! Report a runtime warning.
        void __TBB_EXPORTED_FUNC runtime_warning( const char* format, ... )
        {
            char str[1024]; memset(str, 0, 1024);
            va_list args; va_start(args, format);
            vsnprintf( str, 1024-1, format, args);
            va_end(args);
            fprintf( stderr, "TBB Warning: %s\n", str);
        }
    } // namespace internal
#endif

#if __TBBMALLOC_BUILD
}} // namespaces rml::internal
#else
}  // namespace tbb
#endif
