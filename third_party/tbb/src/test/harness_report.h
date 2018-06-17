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

// Just the tracing portion of the harness.
//
// This header defines TRACE and TRACENL macros, which use REPORT like syntax and
// are useful for duplicating trace output to the standard debug output on Windows.
// It is possible to add the ability of automatic extending messages with additional
// info (file, line, function, time, thread ID, ...).
//
// Macros output nothing when test app runs in non-verbose mode (default).
//

#ifndef tbb_tests_harness_report_H
#define tbb_tests_harness_report_H

#if defined(MAX_TRACE_SIZE) && MAX_TRACE_SIZE < 1024
    #undef MAX_TRACE_SIZE
#endif
#ifndef MAX_TRACE_SIZE
    #define MAX_TRACE_SIZE  1024
#endif

#if __SUNPRO_CC
#include <stdio.h>
#else
#include <cstdio>
#endif

#include <cstdarg>

// Need to include "tbb/tbb_config.h" to obtain the definition of __TBB_DEFINE_MIC.
#include "tbb/tbb_config.h"

#if __TBB_DEFINE_MIC
#include "harness_mic.h"
#endif

#ifdef HARNESS_INCOMPLETE_SOURCES
#error Source files are not complete. Check the build environment
#endif

#if _MSC_VER
    #define snprintf _snprintf
#if _MSC_VER<=1400
    #define vsnprintf _vsnprintf
#endif
#endif

namespace Harness {
    namespace internal {

#ifndef TbbHarnessReporter
    struct TbbHarnessReporter {
        void Report ( const char* msg ) {
            printf( "%s", msg );
            fflush(stdout);
#ifdef _WINDOWS_
            OutputDebugStringA(msg);
#endif
        }
    }; // struct TbbHarnessReporter
#endif /* !TbbHarnessReporter */

    class Tracer {
        int         m_flags;
        const char  *m_file;
        const char  *m_func;
        size_t      m_line;

        TbbHarnessReporter m_reporter;

    public:
        enum  {
            prefix = 1,
            need_lf = 2
        };

        Tracer(): m_flags(0), m_file(NULL), m_func(NULL), m_line(0) {}

        Tracer*  set_trace_info ( int flags, const char *file, size_t line, const char *func ) {
            m_flags = flags;
            m_line = line;
            m_file = file;
            m_func = func;
            return  this;
        }

        void  trace ( const char* fmt, ... ) {
            char    msg[MAX_TRACE_SIZE];
            char    msg_fmt_buf[MAX_TRACE_SIZE];
            const char  *msg_fmt = fmt;
            if ( m_flags & prefix ) {
                snprintf (msg_fmt_buf, MAX_TRACE_SIZE, "[%s] %s", m_func, fmt);
                msg_fmt = msg_fmt_buf;
            }
            std::va_list argptr;
            va_start (argptr, fmt);
            int len = vsnprintf (msg, MAX_TRACE_SIZE, msg_fmt, argptr);
            va_end (argptr);
            if ( m_flags & need_lf &&
                 len < MAX_TRACE_SIZE - 1  &&  msg_fmt[len-1] != '\n' )
            {
                msg[len] = '\n';
                msg[len + 1] = 0;
            }
            m_reporter.Report(msg);
        }
    }; // class Tracer

    static Tracer tracer;

    template<int>
    bool not_the_first_call () {
        static bool first_call = false;
        bool res = first_call;
        first_call = true;
        return res;
    }

    } // namespace internal
} // namespace Harness

#if defined(_MSC_VER)  &&  _MSC_VER >= 1300  ||  defined(__GNUC__)  ||  defined(__GNUG__)
    #define HARNESS_TRACE_ORIG_INFO __FILE__, __LINE__, __FUNCTION__
#else
    #define HARNESS_TRACE_ORIG_INFO __FILE__, __LINE__, ""
    #define __FUNCTION__ ""
#endif


//! printf style tracing macro
/** This variant of TRACE adds trailing line-feed (new line) character, if it is absent. **/
#define TRACE Harness::internal::tracer.set_trace_info(Harness::internal::Tracer::need_lf, HARNESS_TRACE_ORIG_INFO)->trace

//! printf style tracing macro without automatic new line character adding
#define TRACENL Harness::internal::tracer.set_trace_info(0, HARNESS_TRACE_ORIG_INFO)->trace

//! printf style tracing macro with additional information prefix (e.g. current function name)
#define TRACEP Harness::internal::tracer.set_trace_info(Harness::internal::Tracer::prefix | \
                                    Harness::internal::Tracer::need_lf, HARNESS_TRACE_ORIG_INFO)->trace

//! printf style remark macro
/** Produces output only when the test is run with the -v (verbose) option. **/
#define REMARK  !Verbose ? (void)0 : TRACENL

//! printf style remark macro
/** Produces output only when invoked first time.
    Only one instance of this macro is allowed per source code line. **/
#define REMARK_ONCE (!Verbose || Harness::internal::not_the_first_call<__LINE__>()) ? (void)0 : TRACE

//! printf style reporting macro
/** On heterogeneous platforms redirects its output to the host side. **/
#define REPORT TRACENL

//! printf style reporting macro
/** Produces output only when invoked first time.
    Only one instance of this macro is allowed per source code line. **/
#define REPORT_ONCE (Harness::internal::not_the_first_call<__LINE__>()) ? (void)0 : TRACENL

#endif /* tbb_tests_harness_report_H */
