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

#ifndef PIPEDISPENSER_CUH_
#define PIPEDISPENSER_CUH_

#include <pthread.h>
#include <set>
#include <algorithm>
#include <iterator>
#include "../../util/include/thread.h"
#include "util.cuh"

/*
 * PipeDispenser interface
 */
class PipeDispenser {
protected:
    int _numPipes;
    seti _pipes;
    pthread_mutex_t *_mutex;

    void lock() {
        pthread_mutex_lock(_mutex);
    }

    void unlock() {
        pthread_mutex_unlock(_mutex);
    }

    virtual void init() {
        _mutex = (pthread_mutex_t*)(malloc(sizeof (pthread_mutex_t)));
        pthread_mutex_init(_mutex, NULL);
    }
public:
    PipeDispenser(const seti& pipes) {
        _pipes.insert(pipes.begin(), pipes.end());
        init();
    }

    PipeDispenser(int numPipes) {
        for (int i = 0; i < numPipes; ++i) {
            _pipes.insert(i);
        }
        init();
    }

    virtual ~PipeDispenser() {
        pthread_mutex_destroy(_mutex);
        free(_mutex);
    }

    virtual int getPipe(const seti& interested) = 0;

    int getPipe(int interested) {
        seti tmp;
        tmp.insert(interested);
        return getPipe(tmp);
    }

    virtual void freePipe(int pipe) = 0;
};

/*
 * This one blocks until there is a free pipe to return.
 */
class PipeDispenserBlocking : public PipeDispenser {
protected:
    pthread_cond_t *_cv;

    void wait() {
        pthread_cond_wait(_cv, _mutex);
    }

    void broadcast() {
        pthread_cond_broadcast(_cv);
    }

    int getAvailablePipes(const seti& interested, intv& available) {
        available.clear();
        std::set_intersection(_pipes.begin(), _pipes.end(), interested.begin(), interested.end(), std::back_inserter(available));
        return available.size();
    }

    virtual void init() {
        PipeDispenser::init();
        _cv = (pthread_cond_t*)(malloc(sizeof (pthread_cond_t)));
                pthread_cond_init(_cv, NULL);
    }
public:
    PipeDispenserBlocking(const seti& pipes) : PipeDispenser(pipes) {
        init();
    }

    PipeDispenserBlocking(int numPipes) : PipeDispenser(numPipes) {
        init();
    }

    ~PipeDispenserBlocking() {
        pthread_cond_destroy(_cv);
        free(_cv);
    }

    int getPipe(const seti& interested) {
        lock();
        intv avail;
        while (getAvailablePipes(interested, avail) == 0) {
            wait();
        }
        int pipe = avail[0];
        _pipes.erase(pipe);
        unlock();
        return pipe;
    }

    void freePipe(int pipe) {
        lock();
        _pipes.insert(pipe);
        broadcast();
        unlock();
    }
};

/*
 * This one returns the least-occupied pipe.
 */
class PipeDispenserNonBlocking : public PipeDispenser  {
protected:
    std::map<int,int> _pipeUsers;

public:
    PipeDispenserNonBlocking(const seti& pipes) : PipeDispenser(pipes) {
        for (seti::iterator it = pipes.begin(); it != pipes.end(); ++it) {
            _pipeUsers[*it] = 0;
        }
    }

    int getPipe(const seti& interested) {
        lock();
        int pipe = -1, users = 1 << 30;
        for (seti::iterator it = _pipes.begin(); it != _pipes.end(); ++it) {
            if (interested.count(*it) > 0 && _pipeUsers[*it] < users) {
                pipe = *it;
                users = _pipeUsers[*it];
            }
        }
        if (pipe >= 0) {
            _pipeUsers[pipe]++;
        }
        unlock();
        return pipe;
    }

    void freePipe(int pipe) {
        lock();
        _pipeUsers[pipe]--;
        unlock();
    }
};


#endif /* PIPEDISPENSER_CUH_ */
