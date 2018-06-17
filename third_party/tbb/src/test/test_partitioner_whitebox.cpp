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

#include "harness_assert.h"
#include "test_partitioner_whitebox.h"

using uniform_iterations_distribution::ParallelTestBody;

template<typename Partitioner>
class ParallelBody: public ParallelTestBody {
public:
    ParallelBody(size_t parallel_group_thread_starting_index)
        : ParallelTestBody(parallel_group_thread_starting_index) { }

    void operator()(size_t relative_thread_index) const {
        use_case_settings_t settings = {
            m_parallel_group_thread_starting_index + relative_thread_index, // thread_num
            0,                                                              // factors_array_len
            0,                                                              // range_begin
            false,                                                          // provide_feedback (disabled)
            true,                                                           // ensure_non_empty_size
            0,                                                              // above_threads_size_tolerance
            0,                                                              // below_threads_size_tolerance
            0,                                                              // between_min_max_ranges_tolerance
            &ParallelTestBody::uniform_distribution_checker                 // checker function for a particular test case
        };
        g_threadNums.local() = settings.thread_num;
        using namespace test_partitioner_utils::TestRanges;
        {
            size_t factors[] = { 1, 2, 3, 4, 5, 7, 9, 13, 27, 29, 30, 31, 32 };
            settings.factors_array_len = sizeof(factors) / sizeof(factors[0]);

            settings.between_min_max_ranges_tolerance = 0; // it should be equal to zero for blocked_range
            test<BlockedRange, Partitioner>(settings, factors);

            settings.checker = &ParallelTestBody::nonuniform_distribution_checker;
            test<InvertedProportionRange, Partitioner>(settings, factors);
            test<RoundedDownRange, Partitioner>(settings, factors);
            test<RoundedUpRange, Partitioner>(settings, factors);

            test<Range1_2, Partitioner>(settings, factors);
            test<Range1_999, Partitioner>(settings, factors);
            test<Range999_1, Partitioner>(settings, factors);
        }

        {
            // iterations might not be distributed uniformly
            float factors[] = { 1.2f, 2.5f, 3.7f, 4.2f, 5.1f, 8.9f, 27.8f };
            settings.factors_array_len = sizeof(factors) / sizeof(factors[0]);

            settings.between_min_max_ranges_tolerance = 1; // it should be equal to one for blocked_range
            settings.checker = &ParallelTestBody::uniform_distribution_checker;
            test<BlockedRange, Partitioner>(settings, factors);

            settings.checker = &ParallelTestBody::nonuniform_distribution_checker;
            test<InvertedProportionRange, Partitioner>(settings, factors);
            test<RoundedDownRange, Partitioner>(settings, factors);
            test<RoundedUpRange, Partitioner>(settings, factors);

            test<Range1_2, Partitioner>(settings, factors);
            test<Range1_999, Partitioner>(settings, factors);
            test<Range999_1, Partitioner>(settings, factors);
        }

        {
            // iterations might not be distributed uniformly
            size_t factors[] = { 1, 2, 3, 4, 5, 7, 9, 11, 13, 27, 29, 30, 31, 32 };
            settings.factors_array_len = sizeof(factors) / sizeof(factors[0]);

            settings.checker = &ParallelTestBody::uniform_distribution_checker;
            test<BlockedRange, Partitioner>(settings, factors, &shifted_left_range_size_generator);
            test<BlockedRange, Partitioner>(settings, factors, &shifted_right_range_size_generator);

            settings.checker = &ParallelTestBody::nonuniform_distribution_checker;
            test<InvertedProportionRange, Partitioner>(settings, factors, &shifted_left_range_size_generator);
            test<InvertedProportionRange, Partitioner>(settings, factors, &shifted_right_range_size_generator);

            test<RoundedDownRange, Partitioner>(settings, factors, &shifted_left_range_size_generator);
            test<RoundedDownRange, Partitioner>(settings, factors, &shifted_right_range_size_generator);

            test<RoundedUpRange, Partitioner>(settings, factors, &shifted_left_range_size_generator);
            test<RoundedUpRange, Partitioner>(settings, factors, &shifted_right_range_size_generator);

            test<Range1_2, Partitioner>(settings, factors, &shifted_left_range_size_generator);
            test<Range1_2, Partitioner>(settings, factors, &shifted_right_range_size_generator);

            test<Range1_999, Partitioner>(settings, factors, &shifted_left_range_size_generator);
            test<Range1_999, Partitioner>(settings, factors, &shifted_right_range_size_generator);

            test<Range999_1, Partitioner>(settings, factors, &shifted_left_range_size_generator);
            test<Range999_1, Partitioner>(settings, factors, &shifted_right_range_size_generator);
        }

        {
            settings.factors_array_len = 1;
            settings.between_min_max_ranges_tolerance = 1; // since range iterations are not divided without remainder
            settings.checker = &ParallelTestBody::uniform_distribution_checker;
            test<ExactSplitRange, Partitioner, size_t>(settings, NULL, &max_range_size_generator);
            settings.range_begin = size_t(-1) - 10000;
            test<ExactSplitRange, Partitioner, size_t>(settings, NULL, &max_range_size_generator);
        }

        {
            settings.range_begin = 0;
            settings.factors_array_len = 2 * unsigned(settings.thread_num);
            settings.checker = &ParallelTestBody::nonuniform_distribution_checker;

            test<RoundedUpRange, Partitioner, size_t>(settings, NULL, &simple_size_generator);
            test<RoundedDownRange, Partitioner, size_t>(settings, NULL, &simple_size_generator);

            test<InvertedProportionRange, Partitioner, size_t>(settings, NULL, &simple_size_generator);
            test<Range1_2, Partitioner, size_t>(settings, NULL, &simple_size_generator);
            test<Range1_999, Partitioner, size_t>(settings, NULL, &simple_size_generator);
            test<Range999_1, Partitioner, size_t>(settings, NULL, &simple_size_generator);

            settings.ensure_non_empty_size = false;
            test<RoundedUpRange, Partitioner, size_t>(settings, NULL, &simple_size_generator);
            test<RoundedDownRange, Partitioner, size_t>(settings, NULL, &simple_size_generator);

            test<InvertedProportionRange, Partitioner, size_t>(settings, NULL, &simple_size_generator);
            test<Range1_2, Partitioner, size_t>(settings, NULL, &simple_size_generator);
            test<Range1_999, Partitioner, size_t>(settings, NULL, &simple_size_generator);
            test<Range999_1, Partitioner, size_t>(settings, NULL, &simple_size_generator);
        }
    }
};

int TestMain() {
    uniform_iterations_distribution::test<ParallelBody <tbb::affinity_partitioner> >();
    uniform_iterations_distribution::test<ParallelBody <tbb::static_partitioner> >();
    uniform_iterations_distribution::test_task_affinity<tbb::affinity_partitioner>();
    uniform_iterations_distribution::test_task_affinity<tbb::static_partitioner>();
    return Harness::Done;
}
