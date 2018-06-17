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

// Just the assertion portion of the harness.
// This is useful for writing portions of tests that include
// the minimal number of necessary header files.
//
// The full "harness.h" must be included later.

#ifndef harness_assert_H
#define harness_assert_H

void ReportError( const char* filename, int line, const char* expression, const char* message);
void ReportWarning( const char* filename, int line, const char* expression, const char* message);

#define ASSERT_CUSTOM(p,message,file,line)  ((p)?(void)0:ReportError(file,line,#p,message))
#define ASSERT(p,message)                   ASSERT_CUSTOM(p,message,__FILE__,__LINE__)
#define ASSERT_WARNING(p,message)           ((p)?(void)0:ReportWarning(__FILE__,__LINE__,#p,message))

//! Compile-time error if x and y have different types
template<typename T>
void AssertSameType( const T& /*x*/, const T& /*y*/ ) {}

#endif /* harness_assert_H */
