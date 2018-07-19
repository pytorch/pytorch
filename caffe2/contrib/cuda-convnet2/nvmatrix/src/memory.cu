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

#include "../include/memory.cuh"

Lock MemoryManager::_globalLock;
std::map<int, MemoryManager*> FastMemoryManager::_memoryManagers;

MemoryManager& FastMemoryManager::getInstance(int deviceID) {
    _globalLock.acquire();
    if (_memoryManagers.count(deviceID) == 0) {
        _memoryManagers[deviceID] = (new FastMemoryManager(deviceID))->init();
    }
    MemoryManager& ret = *_memoryManagers[deviceID];
    _globalLock.release();
    return ret;
}

MemoryManager* CUDAMemoryManager::_memoryManager = NULL;
MemoryManager& CUDAMemoryManager::getInstance(int deviceID) {
    _globalLock.acquire();
    if (_memoryManager == NULL) {
        _memoryManager = new CUDAMemoryManager();
    }
    _globalLock.release();
    return *_memoryManager;
}

MemoryManager* CUDAHostMemoryManager::_memoryManager = NULL;
MemoryManager& CUDAHostMemoryManager::getInstance() {
    _globalLock.acquire();
    if (_memoryManager == NULL) {
        _memoryManager = new CUDAHostMemoryManager();
    }
    _globalLock.release();
    return *_memoryManager;
}

MemoryManager* FastHostMemoryManager::_memoryManager = NULL;
MemoryManager& FastHostMemoryManager::getInstance() {
    _globalLock.acquire();
    if (_memoryManager == NULL) {
        _memoryManager = (new FastHostMemoryManager())->init();
    }
    _globalLock.release();
    return *_memoryManager;
}


void FastMemoryManager::destroyInstance(int deviceID) {
    _globalLock.acquire();
    if (_memoryManagers.count(deviceID) != 0) {
        delete _memoryManagers[deviceID];
        _memoryManagers.erase(deviceID);
    }
    _globalLock.release();
}

void FastHostMemoryManager::destroyInstance() {
    _globalLock.acquire();
    if (_memoryManager != NULL) {
        delete _memoryManager;
        _memoryManager = NULL;
    }
    _globalLock.release();
}

void CUDAMemoryManager::destroyInstance(int deviceID) {
    _globalLock.acquire();
    if (_memoryManager != NULL) {
        delete _memoryManager;
        _memoryManager = NULL;
    }
    _globalLock.release();
}

void CUDAHostMemoryManager::destroyInstance() {
    _globalLock.acquire();
    if (_memoryManager != NULL) {
        delete _memoryManager;
        _memoryManager = NULL;
    }
    _globalLock.release();
}
