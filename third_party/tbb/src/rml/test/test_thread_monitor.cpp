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

#include "harness.h"
#if __TBB_MIC_OFFLOAD
int TestMain () {
    return Harness::Skipped;
}
#else
#include "thread_monitor.h"
#include "harness_memory.h"
#include "tbb/semaphore.cpp"

class ThreadState {
    void loop();
public:
    static __RML_DECL_THREAD_ROUTINE routine( void* arg ) {
        static_cast<ThreadState*>(arg)->loop();
        return 0;
    }
    typedef rml::internal::thread_monitor thread_monitor;
    thread_monitor monitor;
    volatile int request;
    volatile int ack;
    volatile unsigned clock;
    volatile unsigned stamp;
    ThreadState() : request(-1), ack(-1), clock(0) {}
};

void ThreadState::loop() {
    for(;;) {
        ++clock;
        if( ack==request ) {
            thread_monitor::cookie c;
            monitor.prepare_wait(c);
            if( ack==request ) {
                REMARK("%p: request=%d ack=%d\n", this, request, ack );
                monitor.commit_wait(c);
            } else
                monitor.cancel_wait();
        } else {
            // Throw in delay occasionally
            switch( request%8 ) {
                case 0: 
                case 1:
                case 5:
                    rml::internal::thread_monitor::yield();
            }
            int r = request;
            ack = request;
            if( !r ) return;
        }
    }
}

// Linux on IA-64 architecture seems to require at least 1<<18 bytes per stack.
const size_t MinStackSize = 1<<18;
const size_t MaxStackSize = 1<<22;

int TestMain () {
    for( int p=MinThread; p<=MaxThread; ++p ) {
        ThreadState* t = new ThreadState[p];
        for( size_t stack_size = MinStackSize; stack_size<=MaxStackSize; stack_size*=2 ) {
            REMARK("launching %d threads\n",p);
            for( int i=0; i<p; ++i )
                rml::internal::thread_monitor::launch( ThreadState::routine, t+i, stack_size ); 
            for( int k=1000; k>=0; --k ) {
                if( k%8==0 ) {
                    // Wait for threads to wait.
                    for( int i=0; i<p; ++i ) {
                        unsigned count = 0;
                        do {
                            t[i].stamp = t[i].clock;
                            rml::internal::thread_monitor::yield();
                            if( ++count>=1000 ) {
                                REPORT("Warning: thread %d not waiting\n",i);
                                break;
                            }
                        } while( t[i].stamp!=t[i].clock );
                    }
                }
                REMARK("notifying threads\n");
                for( int i=0; i<p; ++i ) {
                    // Change state visible to launched thread
                    t[i].request = k;
                    t[i].monitor.notify();
                }
                REMARK("waiting for threads to respond\n");
                for( int i=0; i<p; ++i ) 
                    // Wait for thread to respond 
                    while( t[i].ack!=k ) 
                        rml::internal::thread_monitor::yield();
            }
        }
        delete[] t;
    }

    return Harness::Done;
}
#endif /* __TBB_MIC_OFFLOAD */
