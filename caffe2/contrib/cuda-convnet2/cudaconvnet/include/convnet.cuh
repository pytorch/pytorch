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

#ifndef CONVNET3
#define	CONVNET3

#include <vector>
#include <string>
#include <set>
#include <map>
#include <helper_cuda.h>
#include <time.h>
#include "../../util/include/queue.h"
#include "../../util/include/thread.h"
#include <math.h>
#include "../../util/include/sync.h"
#include "messages.cuh"
#include "streambroadcast.cuh"

#include "layer.cuh"
#include "data.cuh"
#include "worker.cuh"
#include "weights.cuh"
#include "pipedispenser.cuh"
#include "timer.cuh"

class Worker;
class WorkResult;
class Layer;
class DataLayer;
class CostLayer;
class ConvNetThread;
class StreamBroadcast;
class Weights;

// name -> device id -> layer*
typedef std::map<std::string,std::map<int, Layer*> > NameReplicaLayerMap;
typedef std::map<std::string, Layer*> NameLayerMap;
// name -> ReplicaMap
//typedef std::map<int,NameLayerMap> ReplicaNameLayerMap;
typedef std::vector<ConvNetThread*> ConvNetThreadV;
typedef std::vector<DataLayer*> DataLayerVector;
//typedef std::map<int,ConvNetThreadV> ReplicaThreadsMap;

class ConvNet : public Thread {
private:
    void checkGradient_copyWeightsToGPU(Matrix& weightsCPU, Weights& weights);
protected:
    NameReplicaLayerMap _layerMap;
    DataLayerVector _dataLayers;
    // Vector of convnet threads (one thread == one GPU)
    ConvNetThreadV _convNetThreads;

    DataProvider* _dp;
    CPUData* _data, *_bufferData;
    int _bufferMinibatchIdx, _bufferPassIdx;
    ThreadSynchronizer* _sync;
    intv _deviceIDs;
    
    Queue<Worker*> _workerQueue;
    Queue<WorkResult*> _resultQueue;
    Queue<Message*> _msgQueue;
    
    int _numFwdTerminal;
    std::map<int, int> _numBwdTerminal; // pass idx -> #terminal
    int _totalPassesDone;
    int _numReplicasMin, _numReplicasMax;
    // For gradient checking
    int _numFailures;
    int _numTests;

    // Training progress (between 0 and 1).
    // Used to determine learning rate based on ParameterSchedule.
    double _trainingProgress;
    double _baseErr;
    bool _conserveMem;
    PipeDispenser *_dataCopyPD;

    void waitForTerminals(int numMsgs, MESSAGES msg);
    void sendMessage(MESSAGES msg, bool sync);
    void sendMessage(Message* msg, bool sync);
    void findBwdTerminal(Layer& l, std::set<Layer*>& visited, int& terminal, int passIdx);
    void connectReplicas();
    void initDataLayers(PyObjectV* layerList);
    void initGPUThreads(PyObjectV* layerList);
    void connectChildren(PyObject* layerParams);
    void* run();
    void setData(CPUData& data, int passIdx);
    void setDataFromBuffer();
    void setBuffer(CPUData* bufferData, int bufferMinibatchIdx, int bufferPassIdx);
public:
    ConvNet(PyObject* layerParams, intv& deviceIDs,
            int minibatchSize, bool conserveMem);
    ~ConvNet();
    void stop();
    
    Queue<Message*>& getMessageQueue();
    Queue<Worker*>& getWorkerQueue();
    Queue<WorkResult*>& getResultQueue();
    DataProvider& getDataProvider();
    
    Layer& getLayer(std::string& name, int replicaID);
    void copyToCPU();
    void copyToGPU();
    void updateWeights(int passIdx);
    void reset(int passIdx);
    void reset();

    void bprop(int passIdx, PASS_TYPE passType);
    void fprop(int miniIdx, int passIdx, PASS_TYPE passType);
    void fprop(CPUData& data, int passIdx, PASS_TYPE passType);

    void setTrainingProgress(double progress);
    double getTrainingProgress() const;

    bool checkGradient(const std::string& name, float eps, Weights& weights); 
    void checkGradients();
    Cost& getCost();
    Cost& getCost(Cost& cost);
    CPUData& getData(); // Returns last minibatch fpropped
    double getCostValue();
    intv& getDeviceIDs();
    ThreadSynchronizer& getSync();
    void syncWithChildren();
    int getMinibatchSize();
    bool isConserveMemory();
    int getNumReplicasMax();
    int getNumReplicasMin();
    int getNumPasses();
    int getTotalPassesDone();
    PipeDispenser& getDataCopyPD();
};

class ConvNetThread : public Thread {
protected:
    NameLayerMap _nameLayerMap;
    std::vector<CostLayer*> _costs;
    ConvNet* _convNet;
    int _deviceID;
    Queue<Message*> _msgQueue;
    Timer _timer;
//    StreamBroadcast* _weightSynchronizer;
    
    void initCuda();
    virtual void initLayer(PyObject* paramsDict, int replicaID);
    void* run();
public:
    ConvNetThread(PyObjectV* layerList, int deviceID, int deviceIdx, ConvNet* convNet);
    ~ConvNetThread();
    
    NameLayerMap& getLayerMap();
    int getDeviceID();
    
    ConvNet& getConvNet();
    
    Queue<Message*>& getMessageQueue();
    std::vector<CostLayer*>& getCostLayers();
//    StreamBroadcast& getWeightSynchronizer();
    
    Cost& getCost();
    Layer& getLayer(std::string& name);
    void startTimer();
    double stopTimer();
};

#endif	/* CONVNET */

