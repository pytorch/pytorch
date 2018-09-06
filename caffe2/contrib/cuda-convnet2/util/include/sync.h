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

#ifndef SYNC_H_
#define SYNC_H_

#include <pthread.h>

class Lock {
private: 
    pthread_mutex_t _mutex;
public:
    Lock() {
        pthread_mutex_init(&_mutex, NULL);
    }
    ~Lock() {
        pthread_mutex_destroy(&_mutex);
    }
    
    void acquire() {
        pthread_mutex_lock(&_mutex);
    }
    
    void release() {
        pthread_mutex_unlock(&_mutex);
    }
};

class ThreadSynchronizer {
private:
    int _numThreads;
    int _numSynced;
    pthread_mutex_t *_syncMutex;
    pthread_cond_t *_syncThresholdCV;
public:
    ThreadSynchronizer(int numThreads) {
        _numThreads = numThreads;
        _numSynced = 0;
        _syncMutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
        _syncThresholdCV = (pthread_cond_t*) malloc(sizeof(pthread_cond_t));
        pthread_mutex_init(_syncMutex, NULL);
        pthread_cond_init(_syncThresholdCV, NULL);
    }

    ~ThreadSynchronizer() {
        pthread_mutex_destroy(_syncMutex);
        pthread_cond_destroy(_syncThresholdCV);
        free(_syncMutex);
        free(_syncThresholdCV);
    }

    void sync() {
        pthread_mutex_lock(_syncMutex);
        _numSynced++;

        if (_numSynced == _numThreads) {
            _numSynced = 0;
            pthread_cond_broadcast(_syncThresholdCV);
        } else {
            pthread_cond_wait(_syncThresholdCV, _syncMutex);
        }
        pthread_mutex_unlock(_syncMutex);
    }
};

#endif /* SYNC_H_ */
