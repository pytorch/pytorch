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

// This file is compiled with C++, but linked with a program written in C.
// The intent is to find dependencies on the C++ run-time.

#include <stdlib.h>
#include "../../../include/tbb/tbb_stddef.h" // __TBB_override
#include "harness_defs.h"
#define RML_PURE_VIRTUAL_HANDLER abort

#if _MSC_VER==1500 && !defined(__INTEL_COMPILER)
// VS2008/VC9 seems to have an issue;
#pragma warning( push )
#pragma warning( disable: 4100 ) 
#elif __TBB_MSVC_UNREACHABLE_CODE_IGNORED
// VS2012-2013 issues "warning C4702: unreachable code" for the code which really
// shouldn't be reached according to the test logic: rml::client has the
// implementation for the "pure" virtual methods to be aborted if they are
// called.
#pragma warning( push )
#pragma warning( disable: 4702 )
#endif
#include "rml_omp.h"
#if ( _MSC_VER==1500 && !defined(__INTEL_COMPILER)) || __TBB_MSVC_UNREACHABLE_CODE_IGNORED
#pragma warning( pop )
#endif

rml::versioned_object::version_type Version;

class MyClient: public __kmp::rml::omp_client {
public:
    rml::versioned_object::version_type version() const __TBB_override {return 0;}
    size_type max_job_count() const __TBB_override {return 1024;}
    size_t min_stack_size() const __TBB_override {return 1<<20;}
    rml::job* create_one_job() __TBB_override {return NULL;}
    void acknowledge_close_connection() __TBB_override {}
    void cleanup(job&) __TBB_override {}
    policy_type policy() const __TBB_override {return throughput;}
    void process( job&, void*, __kmp::rml::omp_client::size_type ) __TBB_override {}

};

//! Never actually set, because point of test is to find linkage issues.
__kmp::rml::omp_server* MyServerPtr;

#define HARNESS_NO_PARSE_COMMAND_LINE 1
#define HARNESS_CUSTOM_MAIN 1
#include "harness.h"

extern "C" void Cplusplus() {
    MyClient client;
    Version = client.version();
    REPORT("done\n");
}
