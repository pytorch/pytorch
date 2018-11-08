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

#include <vector>
#include <map>
#include "../include/reducepipeline.cuh"

using namespace std;

/* =========================
 * IReducerSegment
 * =========================
 */
// Null mat --> reducer on host
IReduceSegment::IReduceSegment(IEightGPUReducer& parent, int deviceID, Queue<int>* finishQueue)
: _deviceID(deviceID), _next(NULL), _finishQueue(finishQueue), Thread(true, getDeviceCPUs(parent.getTgtDeviceID())) {
}

IReduceSegment::~IReduceSegment() {
}

NVMatrix& IReduceSegment::getChunk(const NVMatrix& mat, int chunkSize, int chunkIdx) {
        NVMatrix& line = mat.reshaped(1, mat.getNumElements());
    int start = chunkIdx * chunkSize;
    int end = min((chunkIdx+1) * chunkSize, mat.getNumElements());
//        _mat->printShape("_mat");
    NVMatrix& chunk = line.sliceCols(start, end);
    delete &line;
//        chunk.printShape("chunk");
    return chunk;
}

void* IReduceSegment::run() {
    bool exit = false;
    while (!exit) {
        ReduceMessage& msg = *_queue.dequeue();
        if (msg.getType() == EXIT) {
            exit = true;
        } else {
            bool term = processMessage(msg);
            if (term) {
                assert(_finishQueue);
                _finishQueue->enqueue(1);
            }
        }
        delete &msg;
    }
    return NULL;
}

inline NVMatrix& IReduceSegment::getMatrix(ReduceMessage& msg) {
    return msg.getMatrix(getDeviceID());
}

Queue<ReduceMessage*>& IReduceSegment::getQueue() {
    return _queue;
}

inline int IReduceSegment::getDeviceID() const {
    return _deviceID;
}

void IReduceSegment::addPrev(IReduceSegment& c) {
    _prev.push_back(&c);
}

void IReduceSegment::addNext(ReducePeer& c) {
    assert(_next == NULL);
    _next = &c;
    c.addPrev(*this);
}

bool IReduceSegment::isTerminal() const {
    return _next == NULL;
}

/* =========================
 * ReducerSource
 * =========================
 */
ReducerSource::ReducerSource(IEightGPUReducer& parent, int deviceID) : IReduceSegment(parent, deviceID, NULL) {
}

bool ReducerSource::processMessage(ReduceMessage& msg) {
    assert(msg.getType() == REDUCE_START);
    int numChunks = min(getMatrix(msg).getNumElements(), max(REDUCE_MIN_CHUNKS, min(REDUCE_MAX_CHUNKS, DIVUP(getMatrix(msg).getNumElements(), REDUCE_MIN_CHUNK_SIZE))));
    int chunkSize = DIVUP(getMatrix(msg).getNumElements(), numChunks);
    //printf("num chunks: %d\n", numChunks);
    for (int c = 0; c <= numChunks; ++c) {
        _next->getQueue().enqueue(new ReduceChunkMessage(*this, c, chunkSize, numChunks, msg.getScaleIntermediates(), msg.getScaleTarget(), msg.getMatrices()));
    }
    return false;
}

/* =========================
 * ReducerPeer
 * =========================
 */
ReducePeer::ReducePeer(IEightGPUReducer& parent,int deviceID, Queue<int>* finishQueue) : IReduceSegment(parent, deviceID,  finishQueue), _numInputsFinished(0) {
    _add = deviceID != DEVICE_HOST;
}

ReducePeer::ReducePeer(IEightGPUReducer& parent) : IReduceSegment(parent, DEVICE_HOST, NULL), _numInputsFinished(0), _add(false) {
}

ReducePeer::~ReducePeer() {
    for(std::map<int,cudaStream_t>::iterator it = _streams.begin(); it != _streams.end(); ++it) {
        checkCudaErrors(cudaStreamDestroy(it->second));
    }
    _streams.clear();
}

inline cudaStream_t ReducePeer::getStream(int deviceID) {
    if (deviceID < 0) {
        return NULL;
    }
    if (_streams.count(deviceID) == 0) {
        NVMatrix::setDeviceID(deviceID);
        checkCudaErrors(cudaStreamCreateWithFlags(&_streams[deviceID], cudaStreamNonBlocking));
    }
    return _streams[deviceID];
}

bool ReducePeer::processMessage(ReduceMessage& msg) {
    assert(msg.getType() == REDUCE_CHUNK);

    ReduceChunkMessage& cmsg = *static_cast<ReduceChunkMessage*>(&msg);
//    if (_numInputsReceived.count(cmsg.getChunkIdx()) == 0) {
//        _numInputsReceived[cmsg.getChunkIdx()] = 0;
//    }
    int& inputsRcvd = ++_numInputsReceived[cmsg.getChunkIdx()];
//    printf("reducer on device %d got msg chunk idx %d of %d, inputs rcvd for this chunk idx: %d/%d\n",
//            getDeviceID(), cmsg.getChunkIdx(), cmsg.getNumChunks(),_numInputsReceived[cmsg.getChunkIdx()], _prev.size());
    if (cmsg.getChunkIdx() < cmsg.getNumChunks()) {
        IReduceSegment& src = cmsg.getSource();
        float scalePrev = isTerminal() ? cmsg.getScaleIntermediates() : 1;
        float scaleSelf = inputsRcvd == 1 ? _add * (isTerminal() ? cmsg.getScaleTarget() : 1): 1;
        if (scaleSelf == 0 || isTerminal()) {
            if (getDeviceID() >= 0) {
                NVMatrix::setDeviceID(getDeviceID());
            }
            getMatrix(msg).resize(src.getMatrix(msg));
        }
        assert(getMatrix(msg).isSameDims(src.getMatrix(msg)));
        NVMatrix& prevChunk = getChunk(src.getMatrix(msg), cmsg.getChunkSize(), cmsg.getChunkIdx());
        NVMatrix& myChunk = getChunk(getMatrix(msg), cmsg.getChunkSize(), cmsg.getChunkIdx());
        int execDeviceID = getDeviceID() >= 0 ? getDeviceID() : src.getDeviceID();
        if (execDeviceID >= 0) {
            NVMatrix::setDeviceID(execDeviceID);
            prevChunk.add(myChunk, scalePrev, scaleSelf, myChunk, getStream(execDeviceID));
            NVMatrix::syncStream(getStream(execDeviceID));
        } else {
            assert(!isTerminal());
            hostAdd(prevChunk.getDevData(), myChunk.getDevData(), prevChunk.getNumElements(), scaleSelf);
        }

        delete &prevChunk;
        delete &myChunk;

    } else {
        _numInputsFinished++;
    }
    if (!isTerminal() && inputsRcvd == _prev.size()) {
//        printf("    device %d enqueueing msg for next on device %d\n", getDeviceID(), _next->getDeviceID());
        _next->getQueue().enqueue(
                new ReduceChunkMessage(*this, cmsg.getChunkIdx(), cmsg.getChunkSize(), cmsg.getNumChunks(),
                                        cmsg.getScaleIntermediates(), cmsg.getScaleTarget(), cmsg.getMatrices()));
    }

    bool finished = _numInputsFinished == _prev.size();
    if (finished) {
        _numInputsFinished = 0;
        _numInputsReceived.clear();
    }
    return finished && isTerminal();
}

void ReducePeer::hostAdd(const float* src, float* tgt, const int n, const float scaleTgt) {
    if (scaleTgt != 0) {
        for (int i = 0; i < n; ++i) {
            tgt[i] = scaleTgt * tgt[i] + src[i];
        }
    } else {
        for (int i = 0; i < n; ++i) {
            tgt[i] = src[i];
        }
    }
}

inline NVMatrix& ReducePeer::getMatrix(ReduceMessage& msg) {
    if (getDeviceID() != DEVICE_HOST) {
        return IReduceSegment::getMatrix(msg);
    }
    return _mat;
}

/* =========================
 * EightGPUReducer
 * =========================
 */
IEightGPUReducer::IEightGPUReducer(int tgtDeviceID) : _tgtDeviceID(tgtDeviceID) {
}

IEightGPUReducer::~IEightGPUReducer() {
    vector<IReduceSegment*> v;
    v.insert(v.end(), _sources.begin(), _sources.end());
    v.insert(v.end(), _peers.begin(), _peers.end());
    for (vector<IReduceSegment*>::iterator it = v.begin(); it != v.end(); ++it) {
        (*it)->getQueue().enqueue(new ReduceMessage(EXIT));
        (*it)->join();
        delete *it;
    }
}

IEightGPUReducer& IEightGPUReducer::construct() {
    vector<int> same, other;
    for (int i = 0; i < 8; ++i) {
        if (i != _tgtDeviceID) {
            if (NVMatrix::canAccessPeer(_tgtDeviceID, i)) {
                same.insert(same.begin() + rand() % (1 + same.size()), i);
            } else {
                other.insert(other.begin() + rand() % (1 + other.size()), i);
            }
        }
    }
    assert(same.size() == 3);
    assert(other.size() == 4);
    makeConnections(same, other);
    for (vector<ReducerSource*>::const_iterator it = _sources.begin(); it != _sources.end(); ++it) {
        (*it)->start();
    }
    for (vector<ReducePeer*>::const_iterator it = _peers.begin(); it != _peers.end(); ++it) {
        (*it)->start();
    }
    return *this;
}

void IEightGPUReducer::reduce(std::map<int, NVMatrix*>& mats, float scaleIntermediates, float scaleTarget) {
    assert(mats.size() == 8);
    // Check if source matrices are 0-sized
    bool zero = true;
    for (map<int,NVMatrix*>::const_iterator it = mats.begin(); it != mats.end(); ++it) {
        if (it->first != _tgtDeviceID && it->second->getNumElements() != 0) {
            zero = false;
            break;
        }
    }
    if (zero) {
        mats[_tgtDeviceID]->resize(*mats[(_tgtDeviceID + 1) % 8]);
    } else {
        for (vector<ReducerSource*>::const_iterator it = _sources.begin(); it != _sources.end(); ++it) {
            (*it)->getQueue().enqueue(new ReduceStartMessage(scaleIntermediates, scaleTarget, mats));
        }
        _finishQueue.dequeue();
    }
    assert(_finishQueue.getNumElements() == 0);
}

void IEightGPUReducer::reduce(std::map<int, NVMatrix*>& mats, float scaleIntermediates) {
    reduce(mats, scaleIntermediates, 1);
}

void IEightGPUReducer::reduce(std::map<int, NVMatrix*>& mats) {
    reduce(mats, 1, 1);
}

int IEightGPUReducer::getTgtDeviceID() const {
    return _tgtDeviceID;
}

/* =========================
 * EightGPUReducer1
 * =========================
 */
EightGPUReducer1::EightGPUReducer1(int tgtDeviceID) : IEightGPUReducer(tgtDeviceID) {
}

void EightGPUReducer1::makeConnections(vector<int>& same, vector<int>&other) {
    // Setup segments on same truck
    _peers.push_back(new ReducePeer(*this, _tgtDeviceID, &_finishQueue));         // peers[0] = tgt
    _peers.push_back(new ReducePeer(*this,same[0], &_finishQueue));               // peers[1] = same truck 1
    _peers.push_back(new ReducePeer(*this,same[1], &_finishQueue));               // peers[2] = same truck 2
    _sources.push_back(new ReducerSource(*this,same[2]));                         // sources[0] = same truck 3
 
    _sources[0]->addNext(*_peers[2]);
    _peers[2]->addNext(*_peers[1]);
    _peers[1]->addNext(*_peers[0]);

    // Setup segments on other truck
    _sources.push_back(new ReducerSource(*this,other[0]));                        // sources[1] = other truck 1
    _peers.push_back(new ReducePeer(*this,other[1], &_finishQueue));              // peers[3] = other truck 2
    _peers.push_back(new ReducePeer(*this,other[2], &_finishQueue));              // peers[4] = other truck 3
    _sources.push_back(new ReducerSource(*this,other[3]));                        // sources[2] = other truck 4
    _peers.push_back(new ReducePeer(*this));                                      // peers[5] = host 1
    _peers.push_back(new ReducePeer(*this));                                      // peers[6] = host 2
    _peers.push_back(new ReducePeer(*this));                                      // peers[7] = host 3

    _sources[1]->addNext(*_peers[3]);
    _peers[3]->addNext(*_peers[5]);
    _peers[5]->addNext(*_peers[7]);
    _peers[7]->addNext(*_peers[0]);
    _peers[4]->addNext(*_peers[6]);
    _peers[6]->addNext(*_peers[7]);
    _sources[2]->addNext(*_peers[4]);
}

/* =========================
 * EightGPUReducer2
 * =========================
 */
EightGPUReducer2::EightGPUReducer2(int tgtDeviceID) : IEightGPUReducer(tgtDeviceID) {
}

void EightGPUReducer2::makeConnections(vector<int>& same, vector<int>&other) {
    // Setup segments on same truck
    _peers.push_back(new ReducePeer(*this,_tgtDeviceID, &_finishQueue));          // peers[0] = tgt
    _peers.push_back(new ReducePeer(*this,same[0], &_finishQueue));               // peers[1] = same truck 1
    _peers.push_back(new ReducePeer(*this,same[1], &_finishQueue));               // peers[2] = same truck 2
    _sources.push_back(new ReducerSource(*this,same[2]));                         // sources[0] = same truck 3

    _sources[0]->addNext(*_peers[2]);
    _peers[2]->addNext(*_peers[1]);
    _peers[1]->addNext(*_peers[0]);

    // Setup segments on other truck
    _sources.push_back(new ReducerSource(*this,other[0]));                        // sources[1] = other truck 1
    _peers.push_back(new ReducePeer(*this,other[1], &_finishQueue));              // peers[3] = other truck 2
    _peers.push_back(new ReducePeer(*this,other[2], &_finishQueue));              // peers[4] = other truck 3
    _peers.push_back(new ReducePeer(*this,other[3], &_finishQueue));              // peers[5] = other truck 4
    _peers.push_back(new ReducePeer(*this));                                      // peers[6] = host 1

    _sources[1]->addNext(*_peers[3]);
    _peers[3]->addNext(*_peers[4]);
    _peers[4]->addNext(*_peers[5]);
    _peers[5]->addNext(*_peers[6]);
    _peers[6]->addNext(*_peers[0]);
}
