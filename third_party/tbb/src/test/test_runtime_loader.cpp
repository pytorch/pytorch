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

#if !(_WIN32||_WIN64) || (__MINGW64__||__MINGW32__)

#include "harness.h"

int TestMain () {
    return Harness::Skipped;
}

#else // !(_WIN32||_WIN64)

#define TBB_PREVIEW_RUNTIME_LOADER 1
#include "tbb/runtime_loader.h"
#include "tbb/tbb_stddef.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tbb_exception.h"

#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <vector>
#include <string>
#include <utility>
#include <typeinfo>
#include <stdexcept>

#ifdef HARNESS_USE_RUNTIME_LOADER
    #undef HARNESS_USE_RUNTIME_LOADER    // We do not want harness to preload tbb.
#endif
#include "harness.h"

static int errors = 0;

#define CHECK( cond ) {                                                  \
    if ( ! (cond) ) {                                                    \
        ++ errors;                                                       \
        REPORT( "%s:%d: --- TEST FAILED ---\n", __FILE__, __LINE__ );    \
    };                                                                   \
}

#define SAY( msg ) \
    REMARK( "%s:%d: %s\n", __FILE__, __LINE__, msg )

typedef int (*int_func_t)();

namespace tbb {
namespace interface6 {
namespace internal {
namespace runtime_loader {
    extern tbb::runtime_loader::error_mode stub_mode;
} } } } // namespaces runtime_loader, internal, interface6, tbb

using tbb::interface6::internal::runtime_loader::stub_mode;

#define _CHECK_TBB( code ) {                           \
    stub_mode = tbb::runtime_loader::em_status;        \
    int ver = tbb::TBB_runtime_interface_version();    \
    stub_mode = tbb::runtime_loader::em_abort;         \
    CHECK( ver == code );                              \
}

#define CHECK_TBB_IS_LOADED()                          \
    _CHECK_TBB( TBB_INTERFACE_VERSION )

#define CHECK_TBB_IS_NOT_LOADED()                      \
    _CHECK_TBB( tbb::runtime_loader::ec_no_lib )

int TestMain() {


    __TBB_TRY {

        {
            SAY( "Call a function when library is not yet loaded, stub should return a error." );
            CHECK_TBB_IS_NOT_LOADED();
        }

        {
            SAY( "Create a runtime_loader object, do not load library but make some bad calls." );
            tbb::runtime_loader rtl( tbb::runtime_loader::em_status );
            SAY( "After creating status should be ok." );
            CHECK( rtl.status() == tbb::runtime_loader::ec_ok );
            SAY( "Call a function, stub should return a error." );
            CHECK_TBB_IS_NOT_LOADED();
        }

        {
            SAY( "Create a runtime_loader object and call load() with bad arguments." );
            char const * path[] = { ".", NULL };
            tbb::runtime_loader rtl( tbb::runtime_loader::em_status );
            SAY( "Min version is bad." );
            rtl.load( path, -1 );
            CHECK( rtl.status() == tbb::runtime_loader::ec_bad_arg );
            SAY( "Max version is bad." );
            rtl.load( path, TBB_INTERFACE_VERSION, -1 );
            CHECK( rtl.status() == tbb::runtime_loader::ec_bad_arg );
            SAY( "Both versions are bad." );
            rtl.load( path, -1, -1 );
            CHECK( rtl.status() == tbb::runtime_loader::ec_bad_arg );
            SAY( "Min is bigger than max." );
            rtl.load( path, TBB_INTERFACE_VERSION + 1, TBB_INTERFACE_VERSION - 1 );
            CHECK( rtl.status() == tbb::runtime_loader::ec_bad_arg );
        }

        {
            SAY( "Create a proxy object and call load() with good arguments but not available version." );
            char const * path[] = { ".", NULL };
            tbb::runtime_loader rtl( tbb::runtime_loader::em_status );
            SAY( "Min version too big." );
            rtl.load( path, TBB_INTERFACE_VERSION + 1, TBB_INTERFACE_VERSION + 1 );
            CHECK( rtl.status() == tbb::runtime_loader::ec_no_lib );
            SAY( "Max version is too small." );
            rtl.load( path, TBB_INTERFACE_VERSION - 1, TBB_INTERFACE_VERSION - 1 );
            CHECK( rtl.status() == tbb::runtime_loader::ec_no_lib );
        }

        {
            SAY( "Test em_throw mode." );
            char const * path[] = { ".", NULL };
            tbb::runtime_loader rtl( tbb::runtime_loader::em_throw );
            tbb::runtime_loader::error_code code = tbb::runtime_loader::ec_ok;
            __TBB_TRY {
                rtl.load( path, -1 );
            } __TBB_CATCH ( tbb::runtime_loader::error_code c ) {
                code = c;
            }; // __TBB_TRY
            CHECK( code == tbb::runtime_loader::ec_bad_arg );
            __TBB_TRY {
                rtl.load( path, TBB_INTERFACE_VERSION + 1 );
            } __TBB_CATCH ( tbb::runtime_loader::error_code c ) {
                code = c;
            }; // __TBB_TRY
            CHECK( code == tbb::runtime_loader::ec_no_lib );
        }

        {
            SAY( "Load current version, but specify wrong directories." );
            tbb::runtime_loader rtl( tbb::runtime_loader::em_status );
            SAY( "Specify no directories." );
            char const * path0[] = { NULL };
            rtl.load( path0 );
            CHECK( rtl.status() == tbb::runtime_loader::ec_no_lib );
            SAY( "Specify directories without library." );
            char const * path1[] = { "..", "/", NULL };
            rtl.load( path1 );
            CHECK( rtl.status() == tbb::runtime_loader::ec_no_lib );
        }

        {
            SAY( "Now really load library and do various tests." );
            char const * path[] = { ".", NULL };
            tbb::runtime_loader rtl( tbb::runtime_loader::em_status );
            SAY( "Load current version." );
            rtl.load( path, TBB_INTERFACE_VERSION, TBB_INTERFACE_VERSION );
            CHECK( rtl.status() == tbb::runtime_loader::ec_ok );
            if ( rtl.status() == tbb::runtime_loader::ec_ok ) {
                {
                    SAY( "Make sure the library really loaded." );
                    CHECK_TBB_IS_LOADED();
                }
                SAY( "Call load() again, it should return a error." );
                rtl.load( path, TBB_INTERFACE_VERSION, TBB_INTERFACE_VERSION );
                CHECK( rtl.status() == tbb::runtime_loader::ec_bad_call );
                {
                    SAY( "Initialize task_scheduler." );
                    tbb::task_scheduler_init init( 1 );
                    // Check what?
                }

                // There was a problem on Linux* OS, and still a problem on macOS*.
                SAY( "Throw an exception." );
                // Iterate through all the ids first.
                for ( int id = 1; id < tbb::internal::eid_max; ++ id ) {
                    bool ex_caught = false;
                    __TBB_TRY {
                        tbb::internal::throw_exception( tbb::internal::exception_id( id ) );
                    } __TBB_CATCH ( std::exception const & ) {
                        SAY( "Expected exception caught." );
                        ex_caught = true;
                    } __TBB_CATCH ( ... ) {
                        SAY( "Unexpected exception caught." );
                    }; // try
                    CHECK( ex_caught );
                }; // for
                // Now try to catch exceptions of specific types.
                #define CHECK_EXCEPTION( id, type )                                 \
                    {                                                               \
                        SAY( "Trowing " #id " exception of " #type " type..." );    \
                        bool ex_caught = false;                                     \
                        __TBB_TRY {                                                 \
                            tbb::internal::throw_exception( tbb::internal::id );    \
                        } __TBB_CATCH ( type const & ) {                            \
                            SAY( #type " exception caught." );                      \
                            ex_caught = true;                                       \
                        } __TBB_CATCH ( ... ) {                                     \
                            SAY( "Unexpected exception caught." );                  \
                        }; /* try */                                                \
                        CHECK( ex_caught );                                         \
                    }
                CHECK_EXCEPTION( eid_bad_alloc,                   std::bad_alloc                   );
                CHECK_EXCEPTION( eid_bad_last_alloc,              tbb::bad_last_alloc              );
                CHECK_EXCEPTION( eid_nonpositive_step,            std::invalid_argument            );
                CHECK_EXCEPTION( eid_out_of_range,                std::out_of_range                );
                CHECK_EXCEPTION( eid_segment_range_error,         std::range_error                 );
                CHECK_EXCEPTION( eid_missing_wait,                tbb::missing_wait                );
                CHECK_EXCEPTION( eid_invalid_multiple_scheduling, tbb::invalid_multiple_scheduling );
                CHECK_EXCEPTION( eid_improper_lock,               tbb::improper_lock               );
                CHECK_EXCEPTION( eid_possible_deadlock,           std::runtime_error               );
                CHECK_EXCEPTION( eid_reservation_length_error,    std::length_error                );
                CHECK_EXCEPTION( eid_user_abort,                  tbb::user_abort                  );
                #undef CHECK_EXCEPTION
                {
                    bool ex_caught = false;
                    __TBB_TRY {
                        tbb::internal::handle_perror( EAGAIN, "apple" );
                    } __TBB_CATCH ( std::runtime_error const & ) {
                        SAY( "Expected exception caught." );
                        ex_caught = true;
                    } __TBB_CATCH ( ... ) {
                        SAY( "Unexpected exception caught." );
                    }; // try
                    CHECK( ex_caught );
                }
            }; // if
        }

        {
            SAY( "Test multiple proxies." );
            char const * path[] = { ".", NULL };
            tbb::runtime_loader rtl0( tbb::runtime_loader::em_status );
            tbb::runtime_loader rtl1( tbb::runtime_loader::em_status );
            CHECK( rtl0.status() == tbb::runtime_loader::ec_ok );
            CHECK( rtl1.status() == tbb::runtime_loader::ec_ok );
            SAY( "Load current version with the first rtl." );
            rtl0.load( path );
            CHECK( rtl0.status() == tbb::runtime_loader::ec_ok );
            CHECK_TBB_IS_LOADED();
            SAY( "Load another version with the second proxy, it should return a error." );
            rtl1.load( path, TBB_INTERFACE_VERSION + 1 );
            CHECK( rtl1.status() == tbb::runtime_loader::ec_bad_ver );
            SAY( "Load the same version with the second proxy, it should return ok." );
            rtl1.load( path );
            CHECK( rtl1.status() == tbb::runtime_loader::ec_ok );
            CHECK_TBB_IS_LOADED();
        }

    } __TBB_CATCH( ... ) {

        ASSERT( 0, "unexpected exception" );

    }; // __TBB_TRY

    if ( errors > 0 ) {
        REPORT( "Some tests failed.\n" );
        exit( 1 );
    }; // if

    return Harness::Done;

} // main

#endif // !(_WIN32||_WIN64)

// end of file //
