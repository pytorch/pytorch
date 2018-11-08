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

#ifndef REDUCEPIPELINE_CUH_H_
#define REDUCEPIPELINE_CUH_H_

#include "../../util/include/thread.h"
#include "../../util/include/queue.h"
#include <helper_cuda.h>
#include "../../nvmatrix/include/nvmatrix.cuh"
#include "util.cuh"

#define REDUCE_MIN_CHUNK_SIZE               (1<<18) // 256k
#define REDUCE_MAX_CHUNKS                   16
#define REDUCE_MIN_CHUNKS                   2

enum REDUCE_MESSAGE_TYPE {
    REDUCE_CHUNK,
    REDUCE_START,
    EXIT
};

class ReducePeer;
class ReducerSource;
class IReduceSegment;
class IEightGPUReducer;

class ReduceMessage {
protected:
    REDUCE_MESSAGE_TYPE _msgType;
    float _scaleIntermediates, _scaleTarget;
    std::map<int,NVMatrix*>* _mats;
public:
    ReduceMessage(REDUCE_MESSAGE_TYPE msgType, float scaleIntermediates, float scaleTarget, std::map<int,NVMatrix*>& mats)
        : _msgType(msgType), _scaleIntermediates(scaleIntermediates), _scaleTarget(scaleTarget), _mats(&mats) {
    }
    ReduceMessage(REDUCE_MESSAGE_TYPE msgType)
        : _msgType(msgType), _scaleIntermediates(0), _scaleTarget(0), _mats(NULL) {
    }
    inline REDUCE_MESSAGE_TYPE getType() const {
        return _msgType;
    }
    inline float getScaleIntermediates() const {
        return _scaleIntermediates;
    }
    inline float getScaleTarget() const {
        return _scaleTarget;
    }
    inline NVMatrix& getMatrix(int deviceID) const {
        return *_mats->at(deviceID);
    }
    inline std::map<int,NVMatrix*>& getMatrices() const {
        return *_mats;
    }
};

class ReduceChunkMessage : public ReduceMessage {
protected:
    int _chunkIdx;
    int _chunkSize;
    int _numChunks;

    IReduceSegment* _src;
public:
    ReduceChunkMessage(IReduceSegment& src, int chunkIdx, int chunkSize, int numChunks, float scaleIntermediates, float scaleTarget, std::map<int,NVMatrix*>& mats)
        : _src(&src), _chunkIdx(chunkIdx), _chunkSize(chunkSize), _numChunks(numChunks),
          ReduceMessage(REDUCE_CHUNK, scaleIntermediates, scaleTarget, mats) {
    }

    inline int getChunkIdx() const {
        return _chunkIdx;
    }

    inline int getChunkSize() const {
        return _chunkSize;
    }

    inline int getNumChunks() const {
        return _numChunks;
    }

    inline IReduceSegment& getSource() const {
        return *_src;
    }
};

class ReduceStartMessage : public ReduceMessage {
public:
    ReduceStartMessage(float scaleIntermediates, float scaleTarget, std::map<int,NVMatrix*>& mats)
        : ReduceMessage(REDUCE_START, scaleIntermediates, scaleTarget, mats) {
    }
};

class IReduceSegment : public Thread {
protected:
    int _deviceID;
    std::vector<IReduceSegment*> _prev;
    ReducePeer* _next;
    Queue<ReduceMessage*> _queue;
    Queue<int>* _finishQueue;

    NVMatrix& getChunk(const NVMatrix& mat, int chunkSize, int chunkIdx);
    void* run();
    virtual bool processMessage(ReduceMessage& msg) = 0;

public:
    IReduceSegment(IEightGPUReducer& parent, int deviceID, Queue<int>* finishQueue);
    virtual ~IReduceSegment();
    inline virtual NVMatrix& getMatrix(ReduceMessage& msg);
    Queue<ReduceMessage*>& getQueue();
    int getDeviceID() const;
    void addPrev(IReduceSegment& c);
    void addNext(ReducePeer& c);
    bool isTerminal() const;
};

class ReducerSource : public IReduceSegment {
protected:
    bool processMessage(ReduceMessage& msg);
public:
    ReducerSource(IEightGPUReducer& parent, int deviceID);
};

class ReducePeer : public IReduceSegment {
protected:
    std::map<int,cudaStream_t> _streams;  // device id -> stream
    std::map<int,int> _numInputsReceived; // chunk idx -> num inputs
    int _numInputsFinished;
    HostNVMatrix _mat;
    bool _add;
    bool processMessage(ReduceMessage& msg);
    inline cudaStream_t getStream(int deviceID);
    inline NVMatrix& getMatrix(ReduceMessage& msg);
    void hostAdd(const float* src, float* tgt, const int n, const float scaleTgt);
public:
    ReducePeer(IEightGPUReducer& parent, int deviceID, Queue<int>* finishQueue);
    ReducePeer(IEightGPUReducer& parent);
    ~ReducePeer();
};

class IEightGPUReducer {
protected:
    std::vector<ReducerSource*> _sources;
    std::vector<ReducePeer*> _peers;
    Queue<int> _finishQueue;
    int _tgtDeviceID;
    virtual void makeConnections(std::vector<int>& same, std::vector<int>&other) = 0;
public:
    IEightGPUReducer(int tgtDeviceID);
    virtual ~IEightGPUReducer();
    IEightGPUReducer& construct();
    void reduce(std::map<int, NVMatrix*>& mats, float scaleIntermediates, float scaleTarget);
    void reduce(std::map<int, NVMatrix*>& mats, float scaleIntermediates);
    void reduce(std::map<int, NVMatrix*>& mats);
    int getTgtDeviceID() const;
};

class EightGPUReducer1 : public IEightGPUReducer {
protected:
    void makeConnections(std::vector<int>& same, std::vector<int>&other);
public:
    EightGPUReducer1(int tgtDeviceID);
};

class EightGPUReducer2 : public IEightGPUReducer {
protected:
    void makeConnections(std::vector<int>& same, std::vector<int>&other);
public:
    EightGPUReducer2(int tgtDeviceID);
};

#endif /* REDUCEPIPELINE_CUH_H_ */
