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

#ifndef QUEUE_H_
#define QUEUE_H_
#include <pthread.h>
#include <stdlib.h>

/*
 * A thread-safe circular queue that automatically grows but never shrinks.
 */
template <class T>
class Queue {
private:
    T *_elements;
    int _numElements;
    int _head, _tail;
    int _maxSize;
    pthread_mutex_t *_queueMutex;
    pthread_cond_t *_queueCV;

    void _init(int initialSize) {
        _numElements = 0;
        _head = 0;
        _tail = 0;
        _maxSize = initialSize;
        _elements = new T[initialSize];
        _queueCV = (pthread_cond_t*)(malloc(sizeof (pthread_cond_t)));
        _queueMutex = (pthread_mutex_t*)(malloc(sizeof (pthread_mutex_t)));
        pthread_mutex_init(_queueMutex, NULL);
        pthread_cond_init(_queueCV, NULL);
    }

    void expand() {
        T *newStorage = new T[_maxSize * 2];
        memcpy(newStorage, _elements + _head, (_maxSize - _head) * sizeof(T));
        memcpy(newStorage + _maxSize - _head, _elements, _tail * sizeof(T));
        delete[] _elements;
        _elements = newStorage;
        _head = 0;
        _tail = _numElements;
        _maxSize *= 2;
    }
public:
    Queue(int initialSize) {
        _init(initialSize);
    }

    Queue()  {
        _init(1);
    }

    ~Queue() {
        pthread_mutex_destroy(_queueMutex);
        pthread_cond_destroy(_queueCV);
        delete[] _elements;
        free(_queueMutex);
        free(_queueCV);
    }

    void enqueue(T el) {
        pthread_mutex_lock(_queueMutex);
        if (_numElements == _maxSize) {
            expand();
        }
        _elements[_tail] = el;
        _tail = (_tail + 1) % _maxSize;
        _numElements++;

        pthread_cond_signal(_queueCV);
        pthread_mutex_unlock(_queueMutex);
    }

    /*
     * Blocks until not empty.
     */
    T dequeue() {
        pthread_mutex_lock(_queueMutex);
        // Apparently, pthread_cond_signal may actually unblock
        // multiple threads, so a while loop is needed here.
        while (_numElements == 0) {
            pthread_cond_wait(_queueCV, _queueMutex);
        }
        T el = _elements[_head];
        _head = (_head + 1) % _maxSize;
        _numElements--;
        pthread_mutex_unlock(_queueMutex);
        return el;
    }

    /*
     * Obviously this number can change by the time you actually look at it.
     */
    inline int getNumElements() const {
        return _numElements;
    }
};

#endif /* QUEUE_H_ */
