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

#ifndef COPYPIPELINE_CUH_
#define COPYPIPELINE_CUH_

#include <set>
#include "../../util/include/thread.h"
#include "../../util/include/queue.h"
#include <helper_cuda.h>
#include "../../nvmatrix/include/nvmatrix.cuh"
#include "util.cuh"

#define COPY_MIN_CHUNK_SIZE                 (1<<18) // 256k
#define COPY_MAX_CHUNKS                     16
#define COPY_MIN_CHUNKS                     2

class CopyPeer;
class CopySource;
class ICopySegment;
class IBroadcastNetwork;

class CopyMessage {
protected:
    std::map<int,NVMatrix*>* _mats;
    float _scaleSource, _scaleTargets;
public:
    enum COPY_MESSAGE_TYPE {
        COPY_CHUNK,
        COPY_START,
        EXIT
    };
    CopyMessage(COPY_MESSAGE_TYPE msgType, float scaleSource, float scaleTargets, std::map<int, NVMatrix*>& mats)
        : _msgType(msgType), _scaleSource(scaleSource), _scaleTargets(scaleTargets), _mats(&mats) {
    }
    CopyMessage(COPY_MESSAGE_TYPE msgType)
        : _msgType(msgType), _scaleSource(0), _scaleTargets(0), _mats(NULL) {
    }
    inline COPY_MESSAGE_TYPE getType() const {
        return _msgType;
    }
    inline NVMatrix& getMatrix(int deviceID) const {
        return *_mats->at(deviceID);
    }
    inline std::map<int,NVMatrix*>& getMatrices() const {
        return *_mats;
    }
    inline float getScaleSource() const {
        return _scaleSource;
    }
    inline float getScaleTargets() const {
        return _scaleTargets;
    }
protected:
    COPY_MESSAGE_TYPE _msgType;
};

class CopyChunkMessage : public CopyMessage {
protected:
    int _chunkIdx;
    int _chunkSize;
    int _numChunks;
public:
    CopyChunkMessage(int chunkIdx, int chunkSize, int numChunks, float scaleSource, float scaleTargets, std::map<int, NVMatrix*>& mats)
        : _chunkIdx(chunkIdx), _chunkSize(chunkSize), _numChunks(numChunks), CopyMessage(COPY_CHUNK, scaleSource, scaleTargets, mats) {
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
};

class CopyStartMessage : public CopyMessage {
public:
    CopyStartMessage(float scaleSource, float scaleTargets, std::map<int,NVMatrix*>& mats) : CopyMessage(COPY_START, scaleSource, scaleTargets, mats) {
    }
};

class ICopySegment : public Thread {
protected:
    int _deviceID, _execDeviceID;
    cudaStream_t _stream;
    ICopySegment* _prev;
    std::vector<CopyPeer*> _next;
    Queue<CopyMessage*> _queue;
    Queue<int>* _finishQueue;
    HostNVMatrix _hmat;
    IBroadcastNetwork* _parent;

    NVMatrix& getChunk(NVMatrix& mat, int chunkSize, int chunkIdx);
    void* run();
    virtual bool processMessage(CopyMessage& msg) = 0;

public:
    ICopySegment(IBroadcastNetwork& parent, int deviceID, Queue<int>* finishQueue);
    virtual ~ICopySegment();
    inline NVMatrix& getMatrix(CopyMessage& msg);
    Queue<CopyMessage*>& getQueue();
    inline int getDeviceID();
    void addPrev(ICopySegment& c);
    void addNext(CopyPeer& c);
    bool isTerminal() const;
    virtual bool isSource() const = 0;
};

class CopySource : public ICopySegment {
protected:
    bool processMessage(CopyMessage& msg);
public:
    CopySource(IBroadcastNetwork& parent, int deviceID);
    inline bool isSource() const;
};

class CopyPeer : public ICopySegment {
protected:
    bool processMessage(CopyMessage& msg);
public:
    CopyPeer(IBroadcastNetwork& parent, int deviceID, Queue<int>* finishQueue);
    inline bool isSource() const;
};

class IBroadcastNetwork {
protected:
    Queue<int> _finishQueue;
    CopySource* _src;
    std::vector<CopyPeer*> _peers;
    int _srcDeviceID, _numTerminal;
    bool _constructed;
    std::set<int> _devices;
    std::pair<std::vector<int>,std::vector<int> > makeGPULists();

    void makePeers(std::pair<std::vector<int>,std::vector<int> >& gpus);
    virtual void makeConnections() = 0;
    virtual void _broadcast(std::map<int, NVMatrix*>& mats, float scaleSource, float scaleTargets);
    IBroadcastNetwork(std::set<int>& devices, int srcDeviceID, int numTerminal);
public:
    virtual IBroadcastNetwork& construct();
    virtual ~IBroadcastNetwork();

    virtual void broadcast(std::map<int, NVMatrix*>& mats);
    int getSourceDeviceID() const;
    static IBroadcastNetwork& make(std::set<int> devices, int srcDeviceID);
};

class ISafeBroadcastNetwork : public IBroadcastNetwork {
protected:
    ISafeBroadcastNetwork(std::set<int>& devices, int srcDeviceID, int numTerminal);
public:
    virtual void broadcast(std::map<int, NVMatrix*>& mats, float scaleSource, float scaleTargets);
    virtual ISafeBroadcastNetwork& construct();
    static ISafeBroadcastNetwork& make(std::set<int> devices, int srcDeviceID);
};

class NullBroadcaster : public ISafeBroadcastNetwork {
protected:
    NullBroadcaster(std::set<int>& devices, int srcDeviceID);
    void makeConnections();
public:
    NullBroadcaster& construct();
    void broadcast(std::map<int, NVMatrix*>& mats, float scaleSource, float scaleTargets);
    void broadcast(std::map<int, NVMatrix*>& mats);
    friend class IBroadcastNetwork;
    friend class ISafeBroadcastNetwork;
};

/*
 * This one goes to host and then to targets.
 */
class NaiveBroadcaster : public ISafeBroadcastNetwork {
protected:
    NaiveBroadcaster(std::set<int>& devices, int srcDeviceID);
    void makeConnections();
    friend class IBroadcastNetwork;
    friend class ISafeBroadcastNetwork;
};

class EightGPUBroadcaster1 : public IBroadcastNetwork {
protected:
    EightGPUBroadcaster1(std::set<int>& devices, int srcDeviceID);
    void makeConnections();
    friend class IBroadcastNetwork;
};

class TwoPeeringGPUsBroadcaster : public ISafeBroadcastNetwork {
protected:
    int _tgtDeviceID;
    cudaStream_t _tgtStream;
    void makeConnections();
    void resetDeviceID(int d);
    void _broadcast(std::map<int, NVMatrix*>& mats, float scaleSource, float scaleTargets);
public:
    TwoPeeringGPUsBroadcaster(std::set<int>& devices, int srcDeviceID);
    ~TwoPeeringGPUsBroadcaster();
    ISafeBroadcastNetwork& construct();
    friend class IBroadcastNetwork;
    friend class ISafeBroadcastNetwork;
};

#endif /* COPYPIPELINE_CUH_ */
