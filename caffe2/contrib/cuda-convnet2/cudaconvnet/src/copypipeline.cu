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

#include "../include/copypipeline.cuh"
//#include "gpu_util.cuh"

using namespace std;

/* =========================
 * ICopySegment
 * =========================
 */
ICopySegment::ICopySegment(IBroadcastNetwork& parent, int deviceID, Queue<int>* finishQueue)
    : _parent(&parent), _prev(NULL), _stream(NULL), _deviceID(deviceID), _finishQueue(finishQueue), Thread(true, getDeviceCPUs(parent.getSourceDeviceID())) {
    _execDeviceID = _deviceID;
}

ICopySegment::~ICopySegment() {
    if (_stream != NULL) {
        checkCudaErrors(cudaStreamDestroy(_stream));
    }
}

void* ICopySegment::run() {
    assert(_execDeviceID != DEVICE_HOST);
    NVMatrix::setDeviceID(_execDeviceID);
    checkCudaErrors(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
    bool exit = false;
    while (!exit) {
        CopyMessage& msg = *_queue.dequeue();
        if (msg.getType() == CopyMessage::EXIT) {
            exit = true;
        } else {
            bool term = processMessage(msg);
            if (term) {
                assert(_finishQueue != NULL);
                _finishQueue->enqueue(1);
            }
        }
        delete &msg;
    }
    return NULL;
}

NVMatrix& ICopySegment::getChunk(NVMatrix& mat, int chunkSize, int chunkIdx) {
    NVMatrix& line = mat.reshaped(1, mat.getNumElements());
    int start = chunkIdx * chunkSize;
    int end = min((chunkIdx+1) * chunkSize, mat.getNumElements());
    NVMatrix& chunk = line.sliceCols(start, end);
    delete &line;
    return chunk;
}

inline NVMatrix& ICopySegment::getMatrix(CopyMessage& msg) {
    if (getDeviceID() == DEVICE_HOST) {
        return _hmat;
    }
    return msg.getMatrix(getDeviceID());
}

Queue<CopyMessage*>& ICopySegment::getQueue() {
    return _queue;
}

inline int ICopySegment::getDeviceID() {
    return _deviceID;
}

void ICopySegment::addPrev(ICopySegment& c) {
    _prev = &c;
    if (_deviceID == DEVICE_HOST) {
        _execDeviceID = c.getDeviceID();
    }
}

void ICopySegment::addNext(CopyPeer& c) {
    _next.push_back(&c);
    c.addPrev(*this);
}

bool ICopySegment::isTerminal() const {
    return _next.size() == 0;
}

/* =========================
 * CopySource
 * =========================
 */
CopySource::CopySource(IBroadcastNetwork& parent, int deviceID) : ICopySegment(parent, deviceID, NULL) {
}

bool CopySource::processMessage(CopyMessage& msg) {
    assert(msg.getType() == CopyMessage::COPY_START);
    int numChunks = min(getMatrix(msg).getNumElements(), max(COPY_MIN_CHUNKS, min(COPY_MAX_CHUNKS, DIVUP(getMatrix(msg).getNumElements(), COPY_MIN_CHUNK_SIZE))));
    int chunkSize = DIVUP(getMatrix(msg).getNumElements(), numChunks);
//                printf("num chunks: %d\n", numChunks);
    for (int c = 0; c <= numChunks; ++c) {
        for (vector<CopyPeer*>::const_iterator it = _next.begin(); it != _next.end(); ++it) {
            (*it)->getQueue().enqueue(new CopyChunkMessage(c, chunkSize, numChunks, msg.getScaleSource(), msg.getScaleTargets(), msg.getMatrices()));
        }
    }
    return false;
}

inline bool CopySource::isSource() const {
    return true;
}

/* =========================
 * CopyPeer
 * =========================
 */
CopyPeer::CopyPeer(IBroadcastNetwork& parent, int deviceID, Queue<int>* finishQueue) : ICopySegment(parent, deviceID, finishQueue) {
}

bool CopyPeer::processMessage(CopyMessage& msg) {
    assert(msg.getType() == CopyMessage::COPY_CHUNK);
    CopyChunkMessage& cmsg = *static_cast<CopyChunkMessage*>(&msg);
    if (cmsg.getChunkIdx() < cmsg.getNumChunks()) {
        if (!isTerminal() || (isTerminal() && msg.getScaleTargets() == 0)) {
            getMatrix(msg).resize(_prev->getMatrix(msg));
        }
//        getMatrix(msg).printShape("getMatrix(msg)");
//        _prev->getMatrix(msg).printShape("_prev->getMatrix(msg)");
        assert(getMatrix(msg).isSameDims(_prev->getMatrix(msg)));
        const float scaleSelf = isTerminal() ? msg.getScaleTargets() : 0;
        const float scalePrev = _prev->isSource() ? msg.getScaleSource() : 1;
        NVMatrix& prevChunk = getChunk(_prev->getMatrix(msg), cmsg.getChunkSize(), cmsg.getChunkIdx());
        NVMatrix& myChunk = getChunk(getMatrix(msg), cmsg.getChunkSize(), cmsg.getChunkIdx());
        prevChunk.add(myChunk, scalePrev, scaleSelf, myChunk, _stream);
        NVMatrix::syncStream(_stream);
        delete &prevChunk;
        delete &myChunk;
    }
    for (vector<CopyPeer*>::const_iterator it = _next.begin(); it != _next.end(); ++it) {
        (*it)->getQueue().enqueue(new CopyChunkMessage(cmsg));
    }
    return cmsg.getChunkIdx() >= cmsg.getNumChunks() && isTerminal();
}

inline bool CopyPeer::isSource() const {
    return false;
}

/* =========================
 * IBroadcastNetwork
 * =========================
 */
IBroadcastNetwork& IBroadcastNetwork::make(set<int> devices, int srcDevice) {
    if (devices.size() == 8) {
        return (new EightGPUBroadcaster1(devices, srcDevice))->construct();
    } else if (devices.size() == 1) {
        return (new NullBroadcaster(devices, srcDevice))->construct();
    } else if (devices.size() == 2 && NVMatrix::canAccessPeer(*devices.begin(), *(++devices.begin()))) {
        return (new TwoPeeringGPUsBroadcaster(devices, srcDevice))->construct();
    }
    return (new NaiveBroadcaster(devices, srcDevice))->construct();
}

IBroadcastNetwork::IBroadcastNetwork(set<int>& devices, int srcDeviceID, int numTerminal)
    : _devices(devices), _srcDeviceID(srcDeviceID), _numTerminal(numTerminal), _constructed(false), _src(NULL) {
}

IBroadcastNetwork::~IBroadcastNetwork() {
    vector<ICopySegment*> v;
    v.insert(v.end(), _peers.begin(), _peers.end());
    v.insert(v.end(), _src);
    for (vector<ICopySegment*>::const_iterator it = v.begin(); it != v.end(); ++it) {
        (*it)->getQueue().enqueue(new CopyMessage(CopyMessage::EXIT));
        (*it)->join();
        delete *it;
    }
}

IBroadcastNetwork& IBroadcastNetwork::construct() {
    assert(!_constructed);
    pair<vector<int>,vector<int> > gpus = makeGPULists();
    _src = new CopySource(*this, _srcDeviceID);
    makePeers(gpus);
    makeConnections();
    _src->start();
    for (vector<CopyPeer*>::const_iterator it = _peers.begin(); it != _peers.end(); ++it) {
        (*it)->start();
    }
    _constructed = true;
    return *this;
}

pair<vector<int>,vector<int> > IBroadcastNetwork::makeGPULists() {
    vector<int> same, other;
    for (set<int>::const_iterator it = _devices.begin(); it != _devices.end(); ++it) {
        if (*it != _srcDeviceID) {
            if (NVMatrix::canAccessPeer(_srcDeviceID, *it)) {
                same.insert(same.begin() + rand() % (1 + same.size()), *it);
            } else {
                other.insert(other.begin() + rand() % (1 + other.size()), *it);
            }
        }
    }
    return pair<vector<int>,vector<int> >(same, other);
}

void IBroadcastNetwork::broadcast(std::map<int, NVMatrix*>& mats) {
    _broadcast(mats, 1, 0);
}

void IBroadcastNetwork::_broadcast(std::map<int, NVMatrix*>& mats, float scaleSource, float scaleTargets) {
    assert(_constructed);
    assert(_finishQueue.getNumElements() == 0);
    assert(mats.size() == _devices.size());
    assert(mats.size() > 1);
    if (mats[_srcDeviceID]->getNumElements() == 0) {
        for (map<int,NVMatrix*>::const_iterator it = mats.begin(); it != mats.end(); ++it) {
            it->second->resize(*mats[_srcDeviceID]);
        }
    } else {
        _src->getQueue().enqueue(new CopyStartMessage(scaleSource, scaleTargets, mats));
        for (int i = 0; i < _numTerminal; ++i) {
            _finishQueue.dequeue();
        }
    }
    assert(_finishQueue.getNumElements() == 0);
}

int IBroadcastNetwork::getSourceDeviceID() const {
    return _srcDeviceID;
}

void IBroadcastNetwork::makePeers(pair<vector<int>,vector<int> >& gpus) {
    vector<int>& same = gpus.first, &other = gpus.second;
    for (int i = 0; i < same.size(); ++i) {
        _peers.push_back(new CopyPeer(*this, same[i], &_finishQueue));
    }
    for (int i = 0; i < other.size(); ++i) {
        _peers.push_back(new CopyPeer(*this, other[i], &_finishQueue));
    }
    _peers.push_back(new CopyPeer(*this, DEVICE_HOST, &_finishQueue)); // peers[7]
}

/* =========================
 * ISafeBroadcastNetwork
 * =========================
 */
ISafeBroadcastNetwork& ISafeBroadcastNetwork::make(set<int> devices, int srcDevice) {
    if (devices.size() == 1) {
        return (new NullBroadcaster(devices, srcDevice))->construct();
    } else if (devices.size() == 2 && NVMatrix::canAccessPeer(*devices.begin(), *(++devices.begin()))) {
        return (new TwoPeeringGPUsBroadcaster(devices, srcDevice))->construct();
    }
    return (new NaiveBroadcaster(devices, srcDevice))->construct();
}

ISafeBroadcastNetwork::ISafeBroadcastNetwork(std::set<int>& devices, int srcDeviceID, int numTerminal) : IBroadcastNetwork(devices, srcDeviceID, numTerminal) {
}

void ISafeBroadcastNetwork::broadcast(std::map<int, NVMatrix*>& mats, float scaleSource, float scaleTargets) {
    _broadcast(mats, scaleSource, scaleTargets);
}

ISafeBroadcastNetwork& ISafeBroadcastNetwork::construct() {
    IBroadcastNetwork::construct();
    return *this;
}

/* =========================
 * NullBroadcaster
 * =========================
 */
NullBroadcaster::NullBroadcaster(std::set<int>& devices, int srcDeviceID) : ISafeBroadcastNetwork(devices, srcDeviceID, 0) {
}

void NullBroadcaster::makeConnections() {
}

NullBroadcaster& NullBroadcaster::construct() {
    _constructed = true;
    return *this;
}

void NullBroadcaster::broadcast(std::map<int, NVMatrix*>& mats, float scaleSource, float scaleTargets) {
}

void NullBroadcaster::broadcast(std::map<int, NVMatrix*>& mats) {
}

/* =========================
 * NaiveBroadcaster
 * =========================
 *
 * This one does src -> host -> all
 */
NaiveBroadcaster::NaiveBroadcaster(std::set<int>& devices, int srcDeviceID) : ISafeBroadcastNetwork(devices, srcDeviceID, devices.size()-1) {
}

void NaiveBroadcaster::makeConnections() {
    _src->addNext(*_peers.back()); // Make connection src -> host
    for (int i = 0; i < _peers.size() - 1; ++i) {
        if (_peers[i]->getDeviceID() != _src->getDeviceID()) {
            _peers.back()->addNext(*_peers[i]); // Make connection host -> peer
        }
    }
}

/* =========================
 * EightGPUBroadcaster1
 * =========================
 *
 * This one does a fancy graph
 */
EightGPUBroadcaster1::EightGPUBroadcaster1(set<int>& devices, int srcDeviceID) : IBroadcastNetwork(devices, srcDeviceID, 4) {
}

void EightGPUBroadcaster1::makeConnections() {
    _src->addNext(*_peers[7]);
    _peers[7]->addNext(*_peers[0]);
    _peers[7]->addNext(*_peers[1]);
    _peers[7]->addNext(*_peers[3]);
    _peers[7]->addNext(*_peers[4]);

    _peers[1]->addNext(*_peers[2]);
    _peers[3]->addNext(*_peers[5]);
    _peers[4]->addNext(*_peers[6]);
}

/* =========================
 * TwoPeeringGPUsBroadcaster
 * =========================
 */
TwoPeeringGPUsBroadcaster::TwoPeeringGPUsBroadcaster(std::set<int>& devices, int srcDeviceID) : ISafeBroadcastNetwork(devices, srcDeviceID, 0) {
    _tgtDeviceID = *devices.begin() == srcDeviceID ? *(++devices.begin()) : *devices.begin();
}

TwoPeeringGPUsBroadcaster::~TwoPeeringGPUsBroadcaster() {
    if (_constructed) {
        checkCudaErrors(cudaStreamDestroy(_tgtStream));
    }
}

void TwoPeeringGPUsBroadcaster::makeConnections() {
}

void TwoPeeringGPUsBroadcaster::resetDeviceID(int d) {
    if (d >= 0) {
        NVMatrix::setDeviceID(d);
    }
}

ISafeBroadcastNetwork& TwoPeeringGPUsBroadcaster::construct() {
    assert(!_constructed);
    int d = NVMatrix::getDeviceID();
    NVMatrix::setDeviceID(_tgtDeviceID);
    checkCudaErrors(cudaStreamCreateWithFlags(&_tgtStream, cudaStreamNonBlocking));
    resetDeviceID(d);
    _constructed = true;
    return *this;
}

void TwoPeeringGPUsBroadcaster::_broadcast(std::map<int, NVMatrix*>& mats, float scaleSource, float scaleTargets) {
    int d = NVMatrix::getDeviceID();
    NVMatrix::setDeviceID(_tgtDeviceID);
    mats[_tgtDeviceID]->add(*mats[_srcDeviceID], scaleTargets, scaleSource, *mats[_tgtDeviceID], _tgtStream);
    NVMatrix::syncStream(_tgtStream);
    resetDeviceID(d);
}

