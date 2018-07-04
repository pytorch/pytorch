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

#include <map>
#include <set>
#include "../../nvmatrix/include/nvmatrix.cuh"

class MemorySource;

class MemoryView {
protected:
    MemorySource* _src;
    std::string _name;
public:
    MemoryView(MemorySource& src, std::string& name);
    ~MemoryView();
    NVMatrix& getMemory(int numCases);
    NVMatrix& getMemory();
    MemorySource& getMemorySource();
    bool isParent();
    std::string& getName();
    MemoryView& clone(std::string& name);
};

// Remember: PassThroughLayer, and therefore MemorySource, exists on a particular GPU.
class MemorySource {
protected:
//    int _inputIdx;
    NVMatrix _memory;
    int _deviceID;
    int _size;
    std::map<std::string, std::pair<int,int> > _viewRanges;
    std::map<std::string, NVMatrix*> _memoryViews; // input idx --> slice of _memory
    std::set<std::string> _truncateRequests;
    Lock _lock;
public:
    MemorySource(int size, int deviceID);
    ~MemorySource();
    NVMatrix& getMemory(std::string& name, int numCases);
    NVMatrix& getMemory(std::string& name);
    MemoryView& addUser(std::string& name, std::pair<int,int> range);
    MemoryView& addUser(std::string& name);
    std::pair<int,int> getRange(std::string& name);
    int getSize();
    bool truncate(std::string& name);
    static MemoryView& make(int size, int deviceID, std::string& parentUser);
};

