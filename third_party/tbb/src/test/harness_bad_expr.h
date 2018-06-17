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

// Declarations for checking __TBB_ASSERT checks inside TBB.
// This header is an optional part of the test harness.
// It assumes that "harness.h" has already been included.

#define TRY_BAD_EXPR_ENABLED (TBB_USE_ASSERT && TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN)

#if TRY_BAD_EXPR_ENABLED

//! Check that expression x raises assertion failure with message containing given substring.
/** Assumes that tbb::set_assertion_handler( AssertionFailureHandler ) was called earlier. */
#define TRY_BAD_EXPR(x,substr)          \
    {                                   \
        const char* message = NULL;     \
        bool okay = false;              \
        try {                           \
            x;                          \
        } catch( AssertionFailure a ) { \
            okay = true;                \
            message = a.message;        \
        }                               \
        CheckAssertionFailure(__LINE__,#x,okay,message,substr); \
    }

//! Exception object that holds a message.
struct AssertionFailure {
    const char* message;
    AssertionFailure( const char* filename, int line, const char* expression, const char* comment );
};

AssertionFailure::AssertionFailure( const char* filename, int line, const char* expression, const char* comment ) :
    message(comment)
{
    ASSERT(filename,"missing filename");
    ASSERT(0<line,"line number must be positive");
    // All of our current files have fewer than 4000 lines.
    ASSERT(line<5000,"dubiously high line number");
    ASSERT(expression,"missing expression");
}

void AssertionFailureHandler( const char* filename, int line, const char* expression, const char* comment ) {
    throw AssertionFailure(filename,line,expression,comment);
}

void CheckAssertionFailure( int line, const char* expression, bool okay, const char* message, const char* substr ) {
    if( !okay ) {
        REPORT("Line %d, %s failed to fail\n", line, expression );
        abort();
    } else if( !message ) {
        REPORT("Line %d, %s failed without a message\n", line, expression );
        abort();
    } else if( strstr(message,substr)==0 ) {
        REPORT("Line %d, %s failed with message '%s' missing substring '%s'\n", __LINE__, expression, message, substr );
        abort();
    }
}

#endif /* TRY_BAD_EXPR_ENABLED */
