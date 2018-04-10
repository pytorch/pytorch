/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THREAD_H_
#define THREAD_H_
#include <pthread.h>
#include <stdio.h>
#include <errno.h>
#include <assert.h>
#include <vector>

#define NUM_CPUS_MAX    48

/*
 * Abstract joinable thread class.
 * The only thing the implementer has to fill in is the run method.
 */
class Thread {
private:
    cpu_set_t *_cpu_set;
    pthread_attr_t _pthread_attr;
    pthread_t _threadID;
    bool _joinable, _startable;

    static void* start_pthread_func(void *obj) {
        void* retval = reinterpret_cast<Thread*>(obj)->run();
        pthread_exit(retval);
        return retval;
    }
protected:
    virtual void* run() = 0;
public:
    Thread(bool joinable) : _cpu_set(NULL), _joinable(joinable), _startable(true) {
        pthread_attr_init(&_pthread_attr);
    }

    Thread(bool joinable, std::vector<int>& cpus) : _cpu_set(NULL), _joinable(joinable), _startable(true) {
        pthread_attr_init(&_pthread_attr);
        setAffinity(cpus);
    }

    virtual ~Thread() {
        if (_cpu_set != NULL) {
            CPU_FREE(_cpu_set);
        }
        pthread_attr_destroy(&_pthread_attr);
    }

    void setAffinity(std::vector<int>& cpus) {
        assert(_startable);
        _cpu_set = CPU_ALLOC(NUM_CPUS_MAX);
        size_t size = CPU_ALLOC_SIZE(NUM_CPUS_MAX);
        if (cpus.size() > 0 && cpus[0] >= 0) {
            CPU_ZERO_S(size, _cpu_set);
            for (int i = 0; i < cpus.size(); i++) {
                assert(cpus[i] < NUM_CPUS_MAX);
                CPU_SET_S(cpus[i], size, _cpu_set);
//                printf("set cpu %d\n", cpus[i]);
            }
            pthread_attr_setaffinity_np(&_pthread_attr, size, _cpu_set);
        }
    }

    pthread_t start() {
        assert(_startable);
        _startable = false;
        pthread_attr_setdetachstate(&_pthread_attr, _joinable ? PTHREAD_CREATE_JOINABLE : PTHREAD_CREATE_DETACHED);
        int n;
        if ((n = pthread_create(&_threadID, &_pthread_attr, &Thread::start_pthread_func, (void*)this))) {
            errno = n;
            perror("pthread_create error");
        }
        return _threadID;
    }

    void join(void **status) {
        assert(_joinable);
        int n;
        if((n = pthread_join(_threadID, status))) {
            errno = n;
            perror("pthread_join error");
        }
    }

    void join() {
        join(NULL);
    }

    pthread_t getThreadID() const {
        return _threadID;
    }

    bool isStartable() const {
        return _startable;
    }
};

#endif /* THREAD_H_ */
