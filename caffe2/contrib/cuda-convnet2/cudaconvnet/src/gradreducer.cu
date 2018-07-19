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

#include "../include/util.cuh"
#include "../include/gradreducer.cuh"

using namespace std;

/* =====================
 * IGradReducer
 * =====================
 */
IActGradReducer::IActGradReducer(Layer& parent, map<int, int> numExpectedMsgs)
    : Thread(true, getDeviceCPUs(parent.getDeviceID())), _parent(&parent), _numExpectedMsgs(numExpectedMsgs) {
    _numExpectedMsgsTotal = 0;
    for (map<int,int>::const_iterator it = numExpectedMsgs.begin(); it != numExpectedMsgs.end(); ++it) {
        _numExpectedMsgsTotal += it->second;
    }
//    printf("%s[%d] expected %d backward msgs\n", parent.getName().c_str(), parent.getReplicaID(), _numExpectedMsgsTotal);
}

IActGradReducer::~IActGradReducer() {

}

void* IActGradReducer::run() {
    while (true) {
        reset();
        if (reduce()) {
            break;
        }
        _finishQueue.enqueue(0);
    }
    return NULL;
}

// Cost layer will have nothing to dequeue, so just return immediately.
int IActGradReducer::waitForFinish() {
    if (_numExpectedMsgsTotal > 0) {
        int i = _finishQueue.dequeue();
        assert(_finishQueue.getNumElements() == 0);
        return i;
    }
//    printf("%s not waiting for finish\n", _name.c_str());
    return 0;
}

IActGradReducer& IActGradReducer::makeGradReducer(Layer& parent, map<int, int> numExpectedMsgs) {
    int tgtDeviceID = parent.getDeviceID();
    if (numExpectedMsgs.count(tgtDeviceID) == 0) {
        numExpectedMsgs[tgtDeviceID] = 0;
    }
    if (numExpectedMsgs.size() == 8) {
        return *new ParallelActGradReducer(parent, numExpectedMsgs);
    }
    return *new SequentialActGradReducer(parent, numExpectedMsgs);
}

/* =====================
 * SequentialGradReducer
 * =====================
 */
SequentialActGradReducer::SequentialActGradReducer(Layer& parent, map<int, int> numExpectedMsgs)
    : IActGradReducer(parent, numExpectedMsgs) {
    intv deviceIDs;
    int tgtDeviceID = parent.getDeviceID();
    for (map<int, int>::const_iterator it = numExpectedMsgs.begin(); it != numExpectedMsgs.end(); ++it) {
        if (it->first != tgtDeviceID) {
            deviceIDs.push_back(it->first);
        }
    }
    if (numExpectedMsgs[tgtDeviceID] > 0) {
        deviceIDs.push_back(tgtDeviceID);
    }

    sort(deviceIDs.begin(), deviceIDs.end());

    int firstDeviceIdx = 0, firstDeviceID = 1 << 16;
    for (int i = 0; i < deviceIDs.size(); ++i) {
        if (deviceIDs[i] >= tgtDeviceID && deviceIDs[i] < firstDeviceID) {
            firstDeviceIdx = i;
            firstDeviceID = deviceIDs[i];
        }
    }

    // This is the order in which we process devices.
    for (int i = firstDeviceIdx; _deviceIDs.size() < deviceIDs.size(); i = (i + 1) % deviceIDs.size()) {
        int d = deviceIDs[i];
        _deviceIDs.push_back(d);
        _messageQueues[d] = new Queue<int>();
    }
    //shuffleVector(_deviceIDs, 1, _deviceIDs.size()); 
    _broadcaster = new StreamBroadcast();

    // Note that we MUST process the tgtDeviceID first because
    // we write to it at every iteration, and the computation
    // thread writes to it too. By processing it first we ensure
    // that there's no race condition.
    assert(numExpectedMsgs[tgtDeviceID] == 0 || _deviceIDs[0] == tgtDeviceID);
    reset();
}

SequentialActGradReducer::~SequentialActGradReducer() {
    for(map<int,Queue<int>* >::const_iterator it = _messageQueues.begin(); it != _messageQueues.end(); ++it) {
        delete it->second;
    }
    delete _broadcaster;
}

void SequentialActGradReducer::reset() {
    for (map<int,int>::iterator it = _numReceivedMsgs.begin(); it != _numReceivedMsgs.end(); ++it) {
        _numReceivedMsgs[it->first] = 0;
    }
}

bool SequentialActGradReducer::reduce() {
    int tgtDeviceID = _parent->getDeviceID();
    for (int didx = 0; didx < _deviceIDs.size(); ) {
        int d = _deviceIDs[didx];
        _numReceivedMsgs[d] += _messageQueues[d]->dequeue();
        if (_numReceivedMsgs[d] == _numExpectedMsgs[d]) {
            if (d != tgtDeviceID) {
                NVMatrix::setDeviceID(tgtDeviceID);

                _parent->getActsGrad().resize(_parent->getActsGrad(d));
                map<int, NVMatrix*> mats;
                mats[d] = &_parent->getActsGrad(d);
                mats[tgtDeviceID] = &_parent->getActsGrad(tgtDeviceID);

                _broadcaster->transfer(mats, d, didx > 0, 1);
            }
            didx++;
            assert(_messageQueues[d]->getNumElements() == 0);
        } else if (_numReceivedMsgs[d] >= _numExpectedMsgs[d]) { // exit
            return true;
        }
    }
    return false;
}

void SequentialActGradReducer::enqueueReduction(int deviceID) {
    _messageQueues[deviceID]->enqueue(1);
}

void SequentialActGradReducer::stop() {
    for(map<int,Queue<int>* >::const_iterator it = _messageQueues.begin(); it != _messageQueues.end(); ++it) {
        it->second->enqueue(ACT_GRAD_REDUCER_EXIT);
    }
    join();
}

/* =====================
 * ParallelActGradReducer
 * =====================
 */
ParallelActGradReducer::ParallelActGradReducer(Layer& parent, map<int, int> numExpectedMsgs)
    : IActGradReducer(parent, numExpectedMsgs), _numReceivedMsgs(0) {
    _reducer = &(new EightGPUReducer1(parent.getDeviceID()))->construct();

    _scaleTarget = numExpectedMsgs.count(parent.getDeviceID()) > 0 && numExpectedMsgs[parent.getDeviceID()] > 0;
}

bool ParallelActGradReducer::reduce() {
    // TODO: make it so that you can start the reduction before you've received all the messages.
    while(_numReceivedMsgs < _numExpectedMsgsTotal) {
        _numReceivedMsgs += _messageQueue.dequeue();
    }
    if (_numReceivedMsgs > _numExpectedMsgsTotal) {
        return true; // exit
    }
    map<int,NVMatrix*> mats = _parent->getAllActsGrads();
    _reducer->reduce(mats, 1, _scaleTarget);
    assert(_messageQueue.getNumElements() == 0);
    return false;

}

void ParallelActGradReducer::enqueueReduction(int deviceID) {
    _messageQueue.enqueue(1);
}

void ParallelActGradReducer::stop() {
    _messageQueue.enqueue(ACT_GRAD_REDUCER_EXIT);
    join();
}

void ParallelActGradReducer::reset() {
    _numReceivedMsgs = 0;
}
