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

#define HARNESS_DEFAULT_MIN_THREADS 4
#define HARNESS_DEFAULT_MAX_THREADS 4

#include "tbb/queuing_rw_mutex.h"
#include "tbb/spin_rw_mutex.h"
#include "harness.h"

using namespace tbb;

volatile int Count;

template<typename RWMutex>
struct Hammer: NoAssign {
    RWMutex &MutexProtectingCount;
    mutable volatile int dummy;

    Hammer(RWMutex &m): MutexProtectingCount(m) {}
    void operator()( int /*thread_id*/ ) const {
        for( int j=0; j<100000; ++j ) {
            typename RWMutex::scoped_lock lock(MutexProtectingCount,false);
            int c = Count;
            for( int k=0; k<10; ++k ) {
                ++dummy;
            }
            if( lock.upgrade_to_writer() ) {
                // The upgrade succeeded without any intervening writers
                ASSERT( c==Count, "another thread modified Count while I held a read lock" );
            } else {
                c = Count;
            }
            for( int k=0; k<10; ++k ) {
                ++Count;
            }
            lock.downgrade_to_reader();
            for( int k=0; k<10; ++k ) {
                ++dummy;
            }
        }
    }
};

queuing_rw_mutex QRW_mutex;
spin_rw_mutex SRW_mutex;

int TestMain () {
    for( int p=MinThread; p<=MaxThread; ++p ) {
        REMARK("Testing on %d threads", p);
        Count = 0;
        NativeParallelFor( p, Hammer<queuing_rw_mutex>(QRW_mutex) );
        Count = 0;
        NativeParallelFor( p, Hammer<spin_rw_mutex>(SRW_mutex) );
    }
    return Harness::Done;
}
