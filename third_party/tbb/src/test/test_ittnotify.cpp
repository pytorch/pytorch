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

#define HARNESS_DEFAULT_MIN_THREADS 2
#define HARNESS_DEFAULT_MAX_THREADS 2

#if !TBB_USE_THREADING_TOOLS
    #define TBB_USE_THREADING_TOOLS 1
#endif

#include "harness.h"

#if DO_ITT_NOTIFY

#include "tbb/spin_mutex.h"
#include "tbb/spin_rw_mutex.h"
#include "tbb/queuing_rw_mutex.h"
#include "tbb/queuing_mutex.h"
#include "tbb/mutex.h"
#include "tbb/recursive_mutex.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"


#include "../tbb/itt_notify.h"


template<typename M>
class WorkEmulator: NoAssign {
    M& m_mutex;
    static volatile size_t s_anchor;
public:
    void operator()( tbb::blocked_range<size_t>& range ) const {
        for( size_t i=range.begin(); i!=range.end(); ++i ) {
            typename M::scoped_lock lock(m_mutex);
            for ( size_t j = 0; j!=range.end(); ++j )
                s_anchor = (s_anchor - i) / 2 + (s_anchor + j) / 2;
        }
    }
    WorkEmulator( M& mutex ) : m_mutex(mutex) {}
};

template<typename M>
volatile size_t WorkEmulator<M>::s_anchor = 0;


template<class M>
void Test( const char * name ) {
    REMARK("Testing %s\n",name);
    M mtx;
    tbb::profiling::set_name(mtx, name);

    const int n = 10000;
    tbb::parallel_for( tbb::blocked_range<size_t>(0,n,n/100), WorkEmulator<M>(mtx) );
}

    #define TEST_MUTEX(type, name)  Test<tbb::type>( name )

#endif /* !DO_ITT_NOTIFY */

int TestMain () {
#if DO_ITT_NOTIFY
    for( int p=MinThread; p<=MaxThread; ++p ) {
        REMARK( "testing with %d workers\n", p );
        tbb::task_scheduler_init init( p );
        TEST_MUTEX( spin_mutex, "Spin Mutex" );
        TEST_MUTEX( queuing_mutex, "Queuing Mutex" );
        TEST_MUTEX( queuing_rw_mutex, "Queuing RW Mutex" );
        TEST_MUTEX( spin_rw_mutex, "Spin RW Mutex" );
    }
    return Harness::Done;
#else /* !DO_ITT_NOTIFY */
    return Harness::Skipped;
#endif /* !DO_ITT_NOTIFY */
}
