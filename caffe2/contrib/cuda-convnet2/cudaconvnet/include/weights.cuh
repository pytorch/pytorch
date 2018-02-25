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

#ifndef WEIGHTS_CUH
#define	WEIGHTS_CUH

#include <string>
#include <vector>
#include <iostream>
#include <helper_cuda.h>
#include <assert.h>
#include "../../nvmatrix/include/nvmatrix.cuh"
#include "../../util/include/matrix.h"
#include "util.cuh"
#include "lr.cuh"
#include "layer.cuh"
#include "copypipeline.cuh"
#include "reducepipeline.cuh"
#include "streambroadcast.cuh"

class Layer;
class Weights;
class StreamBroadcast;

class IWeightReducer {
protected:
    int _tgtReplicaID;
    std::map<int,Weights*> _replicas;

    int getDeviceID();
public:
    IWeightReducer(std::map<int,Weights*>& replicas, int srcReplicaID);
    virtual ~IWeightReducer();
    static IWeightReducer& make(std::map<int,Weights*>& replicas, int srcReplicaID);
    virtual void reduce(std::map<int, NVMatrix*> gradShards, float gradScale, bool toInc) = 0;
};

class SequentialWeightReducer : public IWeightReducer {
protected:
    StreamBroadcast* _sb;
public:
    SequentialWeightReducer(std::map<int,Weights*>& replicas, int srcReplicaID);
    ~SequentialWeightReducer();
    void reduce(std::map<int, NVMatrix*> gradShards, float gradScale, bool toInc);
};

class ParallelWeightReducer : public IWeightReducer {
protected:
    IEightGPUReducer* _reducer;
public:
    ParallelWeightReducer(std::map<int,Weights*>& replicas, int srcReplicaID);
    ~ParallelWeightReducer();
    void reduce(std::map<int, NVMatrix*> gradShards, float gradScale, bool toInc);
};

class Weights {
protected:
    Matrix* _hWeights, *_hWeightsInc;
    NVMatrix* _weights, *_weightsInc, *_weightsGrad;
    
    ParameterSchedule* _lrs;

    float _wc, _mom, _wball;
    bool _onGPU, _useGrad, _cleanup;
    int _numUpdates;

    // Note: every layer is its own sibling too
    std::map<int,Weights*> _replicas;
    
    // Non-NULL if these weights are really shared from some other layer
    Weights* _srcWeights;
    Layer* _parent;
    int _shardSize;
    IWeightReducer* _reducer;
    ISafeBroadcastNetwork* _broadcaster;

    void aggregateReplicaGradients(float progress);

    // TODO: assert that these retrun contiguous views
    template<class T> T& getShard(T& mat, int replicaID);
    template<class T> T& getShard(T& mat);
    void init(Matrix& hWeights, Matrix& hWeightsInc, ParameterSchedule& lrs, Layer& parent, float wc, float wball, float mom, bool useGrad, bool cleanup);

public:
    NVMatrix& operator*() const;
    
    Weights(Weights& srcWeights, ParameterSchedule& lrs, Layer& parent);
    Weights(Matrix& hWeights, Matrix& hWeightsInc, ParameterSchedule& lrs, Layer& parent,
            float wc, float wball, float mom, bool useGrad);
        
    virtual ~Weights();

    virtual NVMatrix& getW() const;
    virtual NVMatrix& getInc() const;
    virtual NVMatrix& getGrad() const;
    virtual Matrix& getCPUW() const;
    virtual Matrix& getCPUWInc() const;
    virtual ParameterSchedule& getLearningRateSchedule() const;
    virtual int getNumRows() const;
    virtual int getNumCols() const;
    virtual void copyToCPU();
    
    // This function is assumed to be called in the order in which the layers
    // were defined
    virtual void copyToGPU();
    
    virtual void update(float progress);
    virtual void addReplica(Weights& sibling);
    int incNumUpdates();
    
    // Returns the number of times a gradient has been computed for this
    // weight matrix during the current pass (interval between two calls of update())
    // through the net. This number will only be greater than 1 if this weight matrix
    // is *shared* by multiple layers in the net.
    int getNumUpdates() const;
    float getEps(float progress) const;
    float getMom() const;
    float getWC() const;
    float getWBall() const;
    bool isUseGrad() const;
    bool isOwner() const;
    int getReplicaID();
    int getDeviceID();
    Layer& getParent();
    std::map<int,Weights*>& getReplicas();
    ISafeBroadcastNetwork& getBroadcaster();
    IWeightReducer& getReducer();
};

class WeightList {
private:
    std::vector<Weights*> _weightList;
public:
    Weights& operator[](const int idx) const;
    ~WeightList();
    WeightList();
    Weights& at(const int i) const;
    void addWeights(Weights& w);
    void addReplica(WeightList& sibling);
    void update(float progress);
    void copyToCPU();
    void copyToGPU();
    int getSize() const;
};

#endif	/* WEIGHTS_CUH */
