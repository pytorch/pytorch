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

#ifndef STREAMBROADCAST_CUH_
#define STREAMBROADCAST_CUH_

#include <iostream>
#include "../../util/include/queue.h"
#include "../../nvmatrix/include/nvmatrix.cuh"
#include "util.cuh"

class Layer;

//#define NUM_STREAM_COPY_PARTS       4
// This is in 4-byte words, not bytes
#define SB_MIN_CHUNK_SIZE              (1<<17)
#define SB_MAX_CHUNKS                  16

class StreamBroadcast {
protected:
    std::map<int,cudaStream_t> _streams;
    std::set<int> _ownedStreams;
    HostNVMatrix _hostMem;
    void toHostMem(NVMatrix& src, NVMatrix& hostmem, int srcDevice);
    void toTarget(NVMatrix& hostmem, NVMatrix& tgt, int tgtDevice, float scaleTarget, float scaleOutput);
    void init(std::map<int,cudaStream_t>& streams);
    void init(std::map<int,NVMatrix*>& mats);
public:
    StreamBroadcast(std::map<int,cudaStream_t>& streams);
    StreamBroadcast();
    virtual ~StreamBroadcast();

    void transfer(std::map<int,NVMatrix*>& mats, HostNVMatrix& hostmem, int srcDevice, float scaleTarget, float scaleOutput);
    void transfer(std::map<int,NVMatrix*>& mats, int srcDevice, float scaleTarget, float scaleOutput);
    void transfer(std::map<int,NVMatrix*>& mats, int srcDevice);
    void sync(int deviceID);
    cudaStream_t getStream(int deviceID);
};

#endif /* STREAMBROADCAST_CUH_ */
