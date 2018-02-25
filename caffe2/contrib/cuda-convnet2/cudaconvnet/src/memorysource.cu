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

#include "../include/memorysource.cuh"

using namespace std;

/*
 * =======================
 * MemoryView
 * =======================
 */
MemoryView::MemoryView(MemorySource& src, std::string& name) : _src(&src), _name(name) {
}

MemoryView::~MemoryView() {
//    if (_src->truncate(_name)) {
//        delete _src;
//    }
}

NVMatrix& MemoryView::getMemory(int numCases) {
    return _src->getMemory(_name, numCases);
}

NVMatrix& MemoryView::getMemory() {
    return _src->getMemory(_name);
}

MemorySource& MemoryView::getMemorySource() {
    return *_src;
}

bool MemoryView::isParent() {
    return _src->getRange(_name).first == 0 && _src->getRange(_name).second == _src->getSize();
}

std::string& MemoryView::getName() {
    return _name;
}

MemoryView& MemoryView::clone(std::string& name) {
    return _src->addUser(name, _src->getRange(_name));
}

/*
 * =======================
 * MemorySource
 * =======================
 */
MemorySource::MemorySource(int size, int deviceID) : _size(size), _deviceID(deviceID) {
}

MemorySource::~MemorySource() {
    // Each MemoryView is deleted by owner Layer, and the last one deletes the MemorySource.
    // So this is a no-op.
}

NVMatrix& MemorySource::getMemory(std::string& name) {
    return getMemory(name, _memory.getLeadingDim());
}

// Deletes old view when appropriate
NVMatrix& MemorySource::getMemory(std::string& name, int numCases) {
    numCases = numCases < 0 ? _memory.getLeadingDim() : numCases;
    _lock.acquire();
    if (_memory.getLeadingDim() != numCases || _memory.getFollowingDim() != _size) {
        int d = NVMatrix::getDeviceID();
        NVMatrix::setDeviceID(_deviceID);
        _memory.resize(_size, numCases, false);
        for (map<std::string,NVMatrix*>::const_iterator it = _memoryViews.begin(); it != _memoryViews.end(); ++it) {
            delete it->second;
        }
        _memoryViews.clear();
        if (d >= 0) {
            NVMatrix::setDeviceID(d);
        }
    }
    if (_memoryViews.count(name) == 0) {
        assert(!_memory.isTrans());
        _memoryViews[name] = &_memory.sliceRows(_viewRanges[name].first, _viewRanges[name].second);
    }
    NVMatrix& view = *_memoryViews[name];
    assert(view.isContiguous());
    _lock.release();
    return view;
}

MemoryView& MemorySource::addUser(std::string& name, std::pair<int,int> range) {
    assert(_viewRanges.count(name) == 0);
    _viewRanges[name] = range;
    return *new MemoryView(*this, name);
}

MemoryView& MemorySource::addUser(std::string& name) {
    return addUser(name, std::pair<int,int>(0, _size));
}

MemoryView& MemorySource::make(int size, int deviceID, std::string& parentUser) {
    return (new MemorySource(size, deviceID))->addUser(parentUser);
}

pair<int,int> MemorySource::getRange(std::string& name) {
    return _viewRanges[name];
}

int MemorySource::getSize() {
    return _size;
}

bool MemorySource::truncate(std::string& name) {
    bool truncated = false;
    _lock.acquire();
    _truncateRequests.insert(name);
    if (_truncateRequests.size() == _viewRanges.size()) {
        for (map<std::string,NVMatrix*>::const_iterator it = _memoryViews.begin(); it != _memoryViews.end(); ++it) {
            delete it->second;
        }
        _memoryViews.clear();
        _memory.truncate();
        _truncateRequests.clear();
        truncated = true;
    }
    _lock.release();
    return truncated;
}
