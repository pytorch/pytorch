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

#ifndef __TIME_FRAMEWORK_H__
#error time_framework.h must be included
#endif

#define INJECT_TBB namespace tbb { using namespace ::tbb; namespace internal { using namespace ::tbb::internal; } }
#define INJECT_TBB5 namespace tbb { namespace interface5 { using namespace ::tbb::interface5; namespace internal { using namespace ::tbb::interface5::internal; } } }

#ifndef INJECT_BOX_NAMES
#if defined(__TBB_task_H) || defined(__TBB_concurrent_unordered_internal_H) || defined(__TBB_reader_writer_lock_H) || defined(__TBB__concurrent_unordered_impl_H)
#define INJECT_BOX_NAMES INJECT_TBB INJECT_TBB5
#else
#define INJECT_BOX_NAMES INJECT_TBB
#endif
#endif

#ifdef BOX1
namespace sandbox1 {
    INJECT_BOX_NAMES
#   ifdef BOX1HEADER
#   include BOX1HEADER
#   endif
    typedef ::BOX1TEST testbox;
}
#endif
#ifdef BOX2
namespace sandbox2 {
    INJECT_BOX_NAMES
#   ifdef BOX2HEADER
#   include BOX2HEADER
#   endif
    typedef ::BOX2TEST testbox;
}
#endif
#ifdef BOX3
namespace sandbox3 {
    INJECT_BOX_NAMES
#   ifdef BOX3HEADER
#   include BOX3HEADER
#   endif
    typedef ::BOX3TEST testbox;
}
#endif
#ifdef BOX4
namespace sandbox4 {
    INJECT_BOX_NAMES
#   ifdef BOX4HEADER
#   include BOX4HEADER
#   endif
    typedef ::BOX4TEST testbox;
}
#endif
#ifdef BOX5
namespace sandbox5 {
    INJECT_BOX_NAMES
#   ifdef BOX5HEADER
#   include BOX5HEADER
#   endif
    typedef ::BOX5TEST testbox;
}
#endif
#ifdef BOX6
namespace sandbox6 {
    INJECT_BOX_NAMES
#   ifdef BOX6HEADER
#   include BOX6HEADER
#   endif
    typedef ::BOX6TEST testbox;
}
#endif
#ifdef BOX7
namespace sandbox7 {
    INJECT_BOX_NAMES
#   ifdef BOX7HEADER
#   include BOX7HEADER
#   endif
    typedef ::BOX7TEST testbox;
}
#endif
#ifdef BOX8
namespace sandbox8 {
    INJECT_BOX_NAMES
#   ifdef BOX8HEADER
#   include BOX8HEADER
#   endif
    typedef ::BOX8TEST testbox;
}
#endif
#ifdef BOX9
namespace sandbox9 {
    INJECT_BOX_NAMES
#   ifdef BOX9HEADER
#   include BOX9HEADER
#   endif
    typedef ::BOX9TEST testbox;
}
#endif

//if harness.h included
#if defined(ASSERT) && !HARNESS_NO_PARSE_COMMAND_LINE
#ifndef TEST_PREFIX
#define TEST_PREFIX if(Verbose) printf("Processing with %d threads: %ld...\n", threads, long(value));
#endif
#endif//harness included

#ifndef TEST_PROCESSOR_NAME
#define TEST_PROCESSOR_NAME test_sandbox
#endif

class TEST_PROCESSOR_NAME : public TestProcessor {
public:
    TEST_PROCESSOR_NAME(const char *name, StatisticsCollector::Sorting sort_by = StatisticsCollector::ByAlg)
        : TestProcessor(name, sort_by) {}
    void factory(arg_t value, int threads) {
#ifdef TEST_PREFIX
        TEST_PREFIX
#endif
        process( value, threads,
#define RUNBOX(n) run(#n"."BOX##n, new sandbox##n::testbox() )
#ifdef BOX1
        RUNBOX(1),
#endif
#ifdef BOX2
        RUNBOX(2),
#endif
#ifdef BOX3
        RUNBOX(3),
#endif
#ifdef BOX4
        RUNBOX(4),
#endif
#ifdef BOX5
        RUNBOX(5),
#endif
#ifdef BOX6
        RUNBOX(6),
#endif
#ifdef BOX7
        RUNBOX(7),
#endif
#ifdef BOX8
        RUNBOX(8),
#endif
#ifdef BOX9
        RUNBOX(9),
#endif
        end );
#ifdef TEST_POSTFIX
        TEST_POSTFIX
#endif
    }
};
