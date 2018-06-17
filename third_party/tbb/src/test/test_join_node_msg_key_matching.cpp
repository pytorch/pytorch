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

#include "test_join_node.h"

int TestMain() {
#if __TBB_USE_TBB_TUPLE
    REMARK("  Using TBB tuple\n");
#else
    REMARK("  Using platform tuple\n");
#endif

#if !__TBB_MIC_OFFLOAD_TEST_COMPILATION_BROKEN
#if MAX_TUPLE_TEST_SIZE >= 3
    generate_test<serial_test, tbb::flow::tuple<MyMessageKeyWithoutKey<std::string, double>, MyMessageKeyWithoutKeyMethod<std::string, float>, MyMessageKeyWithBrokenKey<std::string, int> >, message_based_key_matching<std::string&> >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 7
    generate_test<serial_test, tbb::flow::tuple<
        MyMessageKeyWithoutKey<std::string, double>,
        MyMessageKeyWithoutKeyMethod<std::string, int>,
        MyMessageKeyWithBrokenKey<std::string, int>,
        MyMessageKeyWithoutKey<std::string, size_t>,
        MyMessageKeyWithoutKeyMethod<std::string, int>,
        MyMessageKeyWithBrokenKey<std::string, short>,
        MyMessageKeyWithoutKey<std::string, threebyte>
    >, message_based_key_matching<std::string&> >::do_test();
#endif

    generate_test<parallel_test, tbb::flow::tuple<MyMessageKeyWithBrokenKey<int, double>, MyMessageKeyWithoutKey<int, float> >, message_based_key_matching<int> >::do_test();
    generate_test<parallel_test, tbb::flow::tuple<MyMessageKeyWithoutKeyMethod<int, double>, MyMessageKeyWithBrokenKey<int, float> >, message_based_key_matching<int&> >::do_test();
    generate_test<parallel_test, tbb::flow::tuple<MyMessageKeyWithoutKey<std::string, double>, MyMessageKeyWithoutKeyMethod<std::string, float> >, message_based_key_matching<std::string&> >::do_test();

#if MAX_TUPLE_TEST_SIZE >= 10
    generate_test<parallel_test, tbb::flow::tuple<
        MyMessageKeyWithoutKeyMethod<std::string, double>,
        MyMessageKeyWithBrokenKey<std::string, int>,
        MyMessageKeyWithoutKey<std::string, int>,
        MyMessageKeyWithoutKeyMethod<std::string, size_t>,
        MyMessageKeyWithBrokenKey<std::string, int>,
        MyMessageKeyWithoutKeyMethod<std::string, short>,
        MyMessageKeyWithoutKeyMethod<std::string, threebyte>,
        MyMessageKeyWithBrokenKey<std::string, int>,
        MyMessageKeyWithoutKeyMethod<std::string, threebyte>,
        MyMessageKeyWithBrokenKey<std::string, size_t>
    >, message_based_key_matching<std::string&> >::do_test();
#endif
#endif /* __TBB_MIC_OFFLOAD_TEST_COMPILATION_BROKEN */

    return Harness::Done;
}
