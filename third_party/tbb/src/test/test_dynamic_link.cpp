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

enum FOO_TYPE {
    FOO_DUMMY,
    FOO_IMPLEMENTATION
};

#if _WIN32 || _WIN64
#define TEST_EXPORT
#else
#define TEST_EXPORT extern "C"
#endif /* _WIN32 || _WIN64 */

// foo "implementations".
TEST_EXPORT FOO_TYPE foo1() { return FOO_IMPLEMENTATION; }
TEST_EXPORT FOO_TYPE foo2() { return FOO_IMPLEMENTATION; }
// foo "dummies".
FOO_TYPE dummy_foo1() { return FOO_DUMMY; }
FOO_TYPE dummy_foo2() { return FOO_DUMMY; }

// Handlers.
static FOO_TYPE (*foo1_handler)() = &dummy_foo1;
static FOO_TYPE (*foo2_handler)() = &dummy_foo2;

#include "tbb/tbb_config.h"
// Suppress the weak symbol mechanism to avoid surplus compiler warnings.
#ifdef __TBB_WEAK_SYMBOLS_PRESENT
#undef __TBB_WEAK_SYMBOLS_PRESENT
#endif
// Use of harness assert to avoid the dependency on TBB
#include "harness_assert.h"
#define LIBRARY_ASSERT(p,message) ASSERT(p,message)
#include "tbb/dynamic_link.h"
// Table describing how to link the handlers.
static const tbb::internal::dynamic_link_descriptor LinkTable[] = {
    { "foo1", (tbb::internal::pointer_to_handler*)(void*)(&foo1_handler) },
    { "foo2", (tbb::internal::pointer_to_handler*)(void*)(&foo2_handler) }
};

// The direct include since we want to test internal functionality.
#include "tbb/dynamic_link.cpp"
#include "harness_dynamic_libs.h"
#include "harness.h"

#if !HARNESS_SKIP_TEST
int TestMain () {
#if !_WIN32
    // Check if the executable exports its symbols.
    ASSERT( Harness::GetAddress( Harness::OpenLibrary(NULL), "foo1" ) && Harness::GetAddress( Harness::OpenLibrary(NULL), "foo2" ),
            "The executable doesn't export its symbols. Is the -rdynamic switch set during linking?" );
#endif /* !_WIN32 */
    // We want to link (or fail to link) to the symbols available from the
    // executable so it doesn't matter what the library name is specified in
    // the dynamic_link call - let it be an empty string.
    // Generally speaking the test has sense only on Linux but on Windows it
    // checks the dynamic_link graceful behavior with incorrect library name.
    if ( tbb::internal::dynamic_link( "", LinkTable, sizeof(LinkTable)/sizeof(LinkTable[0]) ) ) {
        ASSERT( foo1_handler && foo2_handler, "The symbols are corrupted by dynamic_link" );
        ASSERT( foo1_handler() == FOO_IMPLEMENTATION && foo2_handler() == FOO_IMPLEMENTATION,
                "dynamic_link returned the successful code but symbol(s) are wrong" );
    } else {
        ASSERT( foo1_handler==dummy_foo1 && foo2_handler==dummy_foo2, "The symbols are corrupted by dynamic_link" );
    }
    return Harness::Done;
}
#endif // HARNESS_SKIP_TEST
