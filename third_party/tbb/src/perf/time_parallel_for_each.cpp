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

#include <vector>
#include <list>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <string>

#include "tbb/parallel_for_each.h"
#include "tbb/tick_count.h"

template <typename Type>
void foo( Type &f ) {
    f += 1.0f;
}

template <typename Container>
void test( std::string testName, const int N, const int numRepeats ) {
    typedef typename Container::value_type Type;
    Container v;

    for ( int i = 0; i < N; ++i ) {
        v.push_back( static_cast<Type>(std::rand()) );
    }

    std::vector<double> times;
    times.reserve( numRepeats );

    for ( int i = 0; i < numRepeats; ++i ) {
        tbb::tick_count t0 = tbb::tick_count::now();
        tbb::parallel_for_each( v.begin(), v.end(), foo<Type> );
        tbb::tick_count t1 = tbb::tick_count::now();
        times.push_back( (t1 - t0).seconds()*1e+3 );
    }

    std::sort( times.begin(), times.end() );
    std::cout << "Test " << testName << std::endl
        << "min " << times[times.size() / 20] << " ms " << std::endl
        << "med " << times[times.size() / 2] << " ms " << std::endl
        << "max " << times[times.size() - times.size() / 20 - 1] << " ms " << std::endl;
}

int main( int argc, char* argv[] ) {
    const int N = argc > 1 ? std::atoi( argv[1] ) : 10 * 1000;
    const int numRepeats = argc > 2 ? std::atoi( argv[2] ) : 10;

    test< std::vector<float> >( "std::vector<float>", N, numRepeats );
    test< std::list<float> >( "std::list<float>", N / 100, numRepeats );

    return 0;
}
