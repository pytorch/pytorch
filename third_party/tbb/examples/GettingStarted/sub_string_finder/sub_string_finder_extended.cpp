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

#if __TBB_MIC_OFFLOAD
#pragma offload_attribute (push,target(mic))
#endif // __TBB_MIC_OFFLOAD

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>    //std::max

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/tick_count.h"

#if __TBB_MIC_OFFLOAD
#pragma offload_attribute (pop)

class __declspec(target(mic)) SubStringFinder;
#endif // __TBB_MIC_OFFLOAD

static const std::size_t N = 22;

void SerialSubStringFinder ( const std::string &str, std::vector<std::size_t> &max_array, std::vector<std::size_t> &pos_array ) {
    for (std::size_t i = 0; i < str.size(); ++i) {
        std::size_t max_size = 0, max_pos = 0;
        for (std::size_t j = 0; j < str.size(); ++j)
            if (j != i) {
                std::size_t limit = str.size()-(std::max)(i,j);
                for (std::size_t k = 0; k < limit; ++k) {
                    if (str[i + k] != str[j + k])
                        break;
                    if (k > max_size) {
                        max_size = k;
                        max_pos = j;
                    }
                }
            }
        max_array[i] = max_size;
        pos_array[i] = max_pos;
    }
}

class SubStringFinder {
    const char *str;
    const std::size_t len;
    std::size_t *max_array;
    std::size_t *pos_array;
public:
    void operator() ( const tbb::blocked_range<std::size_t>& r ) const {
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
            std::size_t max_size = 0, max_pos = 0;
            for (std::size_t j = 0; j < len; ++j) {
                if (j != i) {
                    std::size_t limit = len-(std::max)(i,j);
                    for (std::size_t k = 0; k < limit; ++k) {
                        if (str[i + k] != str[j + k])
                            break;
                        if (k > max_size) {
                            max_size = k;
                            max_pos = j;
                        }
                    }
                }
            }
            max_array[i] = max_size;
            pos_array[i] = max_pos;
        }
    }
    // We do not use std::vector for compatibility with offload execution
    SubStringFinder( const char *s, const std::size_t s_len, std::size_t *m, std::size_t *p ) :
        str(s), len(s_len), max_array(m), pos_array(p) { }
};

int main() {
    using namespace tbb;

    std::string str[N] = { std::string("a"), std::string("b") };
    for (std::size_t i = 2; i < N; ++i)
        str[i] = str[i-1]+str[i-2];
    std::string &to_scan = str[N-1];
    const std::size_t num_elem = to_scan.size();

    std::vector<std::size_t> max1(num_elem);
    std::vector<std::size_t> pos1(num_elem);
    std::vector<std::size_t> max2(num_elem);
    std::vector<std::size_t> pos2(num_elem);

    std::cout << " Done building string." << std::endl;

    tick_count serial_t0 = tick_count::now();
    SerialSubStringFinder( to_scan, max2, pos2 );
    tick_count serial_t1 = tick_count::now();
    std::cout << " Done with serial version." << std::endl;

    tick_count parallel_t0 = tick_count::now();
    parallel_for(blocked_range<std::size_t>(0, num_elem, 100),
            SubStringFinder( to_scan.c_str(), num_elem, &max1[0], &pos1[0] ) );
    tick_count parallel_t1 = tick_count::now();
    std::cout << " Done with parallel version." << std::endl;

    for (std::size_t i = 0; i < num_elem; ++i) {
        if (max1[i] != max2[i] || pos1[i] != pos2[i]) {
            std::cout << "ERROR: Serial and Parallel Results are Different!" << std::endl;
            break;
        }
    }
    std::cout << " Done validating results." << std::endl;

    std::cout << "Serial version ran in " << (serial_t1 - serial_t0).seconds() << " seconds" << std::endl
              << "Parallel version ran in " <<  (parallel_t1 - parallel_t0).seconds() << " seconds" << std::endl
              << "Resulting in a speedup of " << (serial_t1 - serial_t0).seconds() / (parallel_t1 - parallel_t0).seconds() << std::endl;

#if __TBB_MIC_OFFLOAD
    // Do offloadable version. Do the timing on host.

    std::vector<std::size_t> max3(num_elem);
    std::vector<std::size_t> pos3(num_elem);

    std::size_t *max3_array = &max3[0];   // method data() for vector is not available in C++03
    std::size_t *pos3_array = &pos3[0];
    tick_count parallel_tt0 = tick_count::now();
    const char *to_scan_str = to_scan.c_str();  // Offload the string as a char array.
    #pragma offload target(mic) in(num_elem) in(to_scan_str:length(num_elem)) out(max3_array,pos3_array:length(num_elem))
    {
        parallel_for(blocked_range<std::size_t>(0, num_elem, 100),
                SubStringFinder ( to_scan_str, num_elem, max3_array, pos3_array ) );
    }
    tick_count parallel_tt1 = tick_count::now();
    std::cout << " Done with offloadable version." << std::endl;

    // Do validation of offloadable results on host.
    for (std::size_t i = 0; i < num_elem; ++i) {
        if (max1[i] != max3[i] || pos1[i] != pos3[i]) {
            std::cout << "ERROR: Serial and Offloadable Results are Different!" << std::endl;
            break;
        }
    }
    std::cout << " Done validating offloadable results." << std::endl;

    std::cout << "Offloadable version ran in " << (parallel_tt1 - parallel_tt0).seconds() << " seconds" << std::endl
              << "Resulting in a speedup of " << (serial_t1 - serial_t0).seconds() / (parallel_tt1 - parallel_tt0).seconds()
              << " of offloadable version" << std::endl;

#endif // __TBB_MIC_OFFLOAD

    return 0;
}
