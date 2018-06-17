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

#include "tbb/tbb_stddef.h"

void TestInsideFunction(){
    __TBB_STATIC_ASSERT(sizeof(char)>=1,"");
}
void TestTwiceAtTheSameLine(){
//    for current implementation it is not possible to use
//    two __TBB_STATIC_ASSERT on a same line
//    __TBB_STATIC_ASSERT(true,""); __TBB_STATIC_ASSERT(true,"");
}

void TestInsideStructure(){
    struct helper{
        __TBB_STATIC_ASSERT(true,"");
    };
}

void TestTwiceInsideStructure(){
    struct helper{
        //for current implementation it is not possible to use
        //two __TBB_STATIC_ASSERT on a same line inside a class definition
        //__TBB_STATIC_ASSERT(true,"");__TBB_STATIC_ASSERT(true,"");

        __TBB_STATIC_ASSERT(true,"");
        __TBB_STATIC_ASSERT(true,"");
    };
}

namespace TestTwiceInsideNamespaceHelper{
    __TBB_STATIC_ASSERT(true,"");
    __TBB_STATIC_ASSERT(true,"");
}

namespace TestTwiceInsideClassTemplateHelper{
    template <typename T>
    struct template_struct{
        __TBB_STATIC_ASSERT(true,"");
        __TBB_STATIC_ASSERT(true,"");
    };
}

void TestTwiceInsideTemplateClass(){
    using namespace TestTwiceInsideClassTemplateHelper;
    typedef template_struct<int> template_struct_int_typedef;
    typedef template_struct<char> template_struct_char_typedef;
    tbb::internal::suppress_unused_warning(template_struct_int_typedef(), template_struct_char_typedef());
}

template<typename T>
void TestTwiceInsideTemplateFunction(){
    __TBB_STATIC_ASSERT(sizeof(T)>=1,"");
    __TBB_STATIC_ASSERT(true,"");
}

#include "harness.h"
int TestMain() {
    #if __TBB_STATIC_ASSERT_PRESENT
        REPORT("Known issue: %s\n", "no need to test ad-hoc implementation as native feature of C++11 is used");
        return Harness::Skipped;
    #else
        TestInsideFunction();
        TestInsideStructure();
        TestTwiceAtTheSameLine();
        TestTwiceInsideStructure();
        TestTwiceInsideTemplateClass();
        TestTwiceInsideTemplateFunction<char>();
        return Harness::Done;
    #endif
}
