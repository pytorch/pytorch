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

#ifndef __TBB_test_range_based_for_H
#define __TBB_test_range_based_for_H

#include <utility> //for std::pair
namespace range_based_for_support_tests{

    template<typename value_type, typename container, typename binary_op_type, typename init_value_type>
    inline init_value_type range_based_for_accumulate(container const& c, binary_op_type accumulator, init_value_type init )
    {
        init_value_type range_for_accumulated = init;
        #if __TBB_RANGE_BASED_FOR_PRESENT
        for (value_type  x : c) {
            range_for_accumulated = accumulator(range_for_accumulated, x);
        }
        #else
        for (typename container::const_iterator x =c.begin(); x != c.end(); ++x) {
            range_for_accumulated = accumulator(range_for_accumulated, *x);
        }
        #endif
        return range_for_accumulated;
    }

    template<typename container, typename binary_op_type, typename init_value_type>
    inline init_value_type range_based_for_accumulate(container const& c, binary_op_type accumulator, init_value_type init )
    {
        typedef typename container::value_type value_type;
        return range_based_for_accumulate<value_type>(c,accumulator,init);
    }

    template <typename integral_type >
    integral_type gauss_summ_of_int_sequence(integral_type sequence_length){
        return (sequence_length +1)* sequence_length /2;
    }

    struct pair_second_summer{
        template<typename first_type, typename second_type>
        second_type operator() (second_type const& lhs, std::pair<first_type, second_type> const& rhs) const
        {
            return lhs + rhs.second;
        }
    };
}

#endif /* __TBB_test_range_based_for_H */
