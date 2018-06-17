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

// Program for basic correctness of handle_perror, which is internal
// to the TBB shared library.

#include <cerrno>
#include <stdexcept>

#include "../tbb/tbb_misc.h"
#include "harness.h"

#if TBB_USE_EXCEPTIONS

static void TestHandlePerror() {
    bool caught = false;
    try {
        tbb::internal::handle_perror( EAGAIN, "apple" );
    } catch( std::runtime_error& e ) {
#if TBB_USE_EXCEPTIONS
        REMARK("caught runtime_exception('%s')\n",e.what());
        ASSERT( memcmp(e.what(),"apple: ",7)==0, NULL );
        ASSERT( strlen(strstr(e.what(), strerror(EAGAIN))), "bad error message?" );
#endif /* TBB_USE_EXCEPTIONS */
        caught = true;
    }
    ASSERT( caught, NULL );
}

int TestMain () {
    TestHandlePerror();
    return Harness::Done;
}

#else /* !TBB_USE_EXCEPTIONS */

int TestMain () {
    return Harness::Skipped;
}

#endif /* TBB_USE_EXCEPTIONS */
