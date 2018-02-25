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

#ifndef GRADREDUCER_CUH_
#define GRADREDUCER_CUH_

#include <set>
#include <algorithm>
#include "streambroadcast.cuh"
#include "reducepipeline.cuh"
#include "layer.cuh"
#include "util.cuh"

class StreamBroadcast;
class Layer;

#define ACT_GRAD_REDUCER_EXIT       (1 << 16)

//class ReduceMessage {
//    ReduceMessage();
//    ReduceMessage(bool exit);
//};

class IActGradReducer : public Thread {
protected:
    Layer* _parent;
    Queue<int> _finishQueue;
    int _numExpectedMsgsTotal;
    std::map<int,int> _numExpectedMsgs; // map from device id -> num expected msgs

    void* run();
    virtual bool reduce() = 0;
    virtual void reset() = 0;
public:
    IActGradReducer(Layer& parent, std::map<int, int> numExpectedMsgs);
    virtual ~IActGradReducer();
    int waitForFinish();
    virtual void enqueueReduction(int deviceID) = 0;
    virtual void stop() = 0;
    static IActGradReducer& makeGradReducer(Layer& parent, std::map<int, int> numExpectedMsgs);
};

class SequentialActGradReducer : public IActGradReducer {
protected:

    std::map<int,int> _numReceivedMsgs; // map from device id -> num received msgs

    std::map<int,Queue<int>* > _messageQueues;
    intv _deviceIDs;
    StreamBroadcast* _broadcaster;
    bool reduce();
    void reset();
public:
    SequentialActGradReducer(Layer& parent, std::map<int, int> numExpectedMsgs);
    ~SequentialActGradReducer();
    void enqueueReduction(int deviceID);
    void stop();
};

class ParallelActGradReducer : public IActGradReducer {
protected:
    IEightGPUReducer* _reducer;
    int _numReceivedMsgs;
    float _scaleTarget;
    Queue<int> _messageQueue;
    bool reduce();
    void reset();
public:
    ParallelActGradReducer(Layer& parent, std::map<int, int> numExpectedMsgs);
    void enqueueReduction(int deviceID);
    void stop();
};


#endif /* GRADREDUCER_CUH_ */
