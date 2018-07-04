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
#include <algorithm>
#include "../include/weights.cuh"
#include "../include/lr.cuh"
#include "../include/worker.cuh"

using namespace std;

/* ========================
 * IWeightReducer
 * ========================
 */
int IWeightReducer::getDeviceID() {
    return _replicas[_tgtReplicaID]->getDeviceID();
}

IWeightReducer::IWeightReducer(std::map<int,Weights*>& replicas, int tgtReplicaID) : _replicas(replicas), _tgtReplicaID(tgtReplicaID) {
}

IWeightReducer::~IWeightReducer() {
}

IWeightReducer& IWeightReducer::make(std::map<int,Weights*>& replicas, int tgtReplicaID) {
    if (replicas.size() == 8) {
        return *new ParallelWeightReducer(replicas, tgtReplicaID);
    }
    return *new SequentialWeightReducer(replicas, tgtReplicaID);
}

/* ========================
 * SequentialWeightReducer
 * ========================
 */
SequentialWeightReducer::SequentialWeightReducer(std::map<int,Weights*>& replicas, int tgtReplicaID) : IWeightReducer(replicas, tgtReplicaID) {
    _sb = new StreamBroadcast();
}

SequentialWeightReducer::~SequentialWeightReducer() {
    delete _sb;
}

void SequentialWeightReducer::reduce(std::map<int, NVMatrix*> gradShards, float gradScale, bool toInc) {
    std::map<int, NVMatrix*> mats; // device id -> grad
    mats[getDeviceID()] = toInc ? &_replicas[_tgtReplicaID]->getInc() : &_replicas[_tgtReplicaID]->getGrad();
    for (int i = 0, r = _tgtReplicaID; i < _replicas.size(); ++i, r = (r + 1) % _replicas.size()) {
        if (r != _tgtReplicaID) {
            mats[_replicas[r]->getDeviceID()] = gradShards[r];
            _sb->transfer(mats, _replicas[r]->getDeviceID(), 1, gradScale);
            mats.erase(_replicas[r]->getDeviceID());
        }
    }
}

/* ========================
 * ParallelWeightReducer
 * ========================
 */
ParallelWeightReducer::ParallelWeightReducer(std::map<int,Weights*>& replicas, int tgtReplicaID) : IWeightReducer(replicas, tgtReplicaID) {
    _reducer = &(new EightGPUReducer1(getDeviceID()))->construct();
}

ParallelWeightReducer::~ParallelWeightReducer() {
    delete _reducer;
}

void ParallelWeightReducer::reduce(std::map<int, NVMatrix*> gradShards, float gradScale, bool toInc) {
    std::map<int, NVMatrix*> mats; // device id -> grad
    mats[getDeviceID()] = toInc ? &_replicas[_tgtReplicaID]->getInc() : &_replicas[_tgtReplicaID]->getGrad();
    for (std::map<int,Weights*>::const_iterator it = _replicas.begin(); it != _replicas.end(); ++it) {
        if (it->first != _tgtReplicaID) {
            mats[it->second->getDeviceID()] = gradShards[it->first];
        }
    }
    _reducer->reduce(mats, gradScale, 1);
}

// weights has pointer to layer, layer pointer to thread
// thread has sync (copy) object for every other thread
// weights uses copy object to sum grad contributions into inc matrix slice (phase 1)
// weights broadcasts inc matrix slice to other inc matrix replicas (phase 2)

NVMatrix& Weights::operator*() const {
    return getW();
}

/*
 * TODO: get rid of this constructor duplication.
 */
Weights::Weights(Weights& srcWeights, ParameterSchedule& lrs, Layer& parent) {
    init(srcWeights.getCPUW(), srcWeights.getCPUWInc(), lrs, parent, 0, 0, srcWeights.getMom(), srcWeights.isUseGrad(), false);
    _srcWeights = &srcWeights;
}

Weights::Weights(Matrix& hWeights, Matrix& hWeightsInc, ParameterSchedule& lrs, Layer& parent, float wc,
                 float wball, float mom, bool useGrad) {
    init(hWeights, hWeightsInc, lrs, parent, wc, wball, mom, useGrad, true);
}

void Weights::init(Matrix& hWeights, Matrix& hWeightsInc, ParameterSchedule& lrs, Layer& parent, float wc,
              float wball, float mom, bool useGrad, bool cleanup) {
    _srcWeights = NULL;
    _hWeights = &hWeights;
    _hWeightsInc = &hWeightsInc;
    _numUpdates = 0;
    _lrs = &lrs;
    _parent = &parent;
    _wc = wc;
    _wball = wball;
    _mom = mom;
    _useGrad = useGrad;
    _onGPU = false;
    _weights = NULL;
    _weightsInc = NULL;
    _weightsGrad = NULL;
    _cleanup = cleanup;
    _reducer = NULL;
    _broadcaster = NULL;
}

Weights::~Weights() {
	delete _lrs;
	delete _reducer;
	delete _broadcaster;
    if (_cleanup) {
        delete _hWeights;
        delete _hWeightsInc;
        if (_srcWeights == NULL) {
            delete _weights;
            delete _weightsInc;
            delete _weightsGrad;
        }
    }
}

NVMatrix& Weights::getW() const {
    assert(_onGPU);
    return *_weights;
}

NVMatrix& Weights::getInc() const {
    assert(_onGPU);
    return *_weightsInc;
}

/*
 * TODO: This seems like pretty nasty behavior, I should change this.
 */
NVMatrix& Weights::getGrad() const {
    assert(_onGPU);
    return _useGrad ? *_weightsGrad : *_weightsInc;
}

Matrix& Weights::getCPUW() const {
    return *_hWeights;
}

Matrix& Weights::getCPUWInc() const {
    return *_hWeightsInc;
}

int Weights::getNumRows() const {
    return _hWeights->getNumRows();
}

int Weights::getNumCols() const {
    return _hWeights->getNumCols();
}

map<int,Weights*>& Weights::getReplicas() {
    return _replicas;
}

template<class T> T& Weights::getShard(T& mat, int replicaID) {
    const int n = mat.getNumElements();
    T& line = mat.reshaped(1, n);
    const int shardStart = min(n, replicaID * _shardSize);
    const int shardEnd = min(n, (replicaID + 1) * _shardSize);
    T& slice = line.sliceCols(shardStart, shardEnd);
    assert(slice.isView());
    delete &line;
    return slice;
}

template<class T> T& Weights::getShard(T& mat) {
    return getShard(mat, getReplicaID());
}

ISafeBroadcastNetwork& Weights::getBroadcaster() {
    if (_broadcaster == NULL) {
        set<int> devices;
        for (map<int, Weights*>::const_iterator it = _replicas.begin(); it != _replicas.end(); ++it) {
            devices.insert(it->second->getDeviceID());
        }
        // NOTE: we must use safe broadcaster becasue we want to *add* our value to everyone else
        _broadcaster = &ISafeBroadcastNetwork::make(devices, getDeviceID()); //&(new NaiveBroadcaster(devices, getDeviceID()))->construct();
    }
    return *_broadcaster;
}

IWeightReducer& Weights::getReducer() {
    if (_reducer == NULL) {
        _reducer = &IWeightReducer::make(_replicas, getReplicaID());
    }
    return *_reducer;
}

void Weights::copyToCPU() {
    if (_srcWeights == NULL) {
        assert(_onGPU);
        NVMatrix::syncStream(); // for safety
        if (getReplicaID() == 0) {
            _weights->copyToHost(*_hWeights);

            // Synchronize weights amongst replicas while we're at it.
            map<int,NVMatrix*> weights;
            for (map<int,Weights*>::const_iterator it = _replicas.begin(); it != _replicas.end(); ++it) {
                weights[it->second->getDeviceID()] = &it->second->getW();
            }
            // These things sync before returning.
            getBroadcaster().broadcast(weights, 1, 0);
        }
        if (_useGrad) {
            Matrix& hIncShard = getShard(*_hWeightsInc);
            _weightsInc->copyToHost(hIncShard);
            delete &hIncShard;
        } else { // In this case there's definitely only one replica
            _weightsInc->copyToHost(*_hWeightsInc);
        }
    }
}

// This function is assumed to be called in the order in which the layers
// were defined
void Weights::copyToGPU() {
    assert(!_onGPU);
    // Copies are performed on the default (computation) stream, so that's fine.
    if (_srcWeights == NULL) {
        _weights = _weights == NULL ? new NVMatrix() : _weights;
        _weightsInc = _weightsInc == NULL ? new NVMatrix() : _weightsInc;
        _weights->copyFromHost(*_hWeights, true);

        if (_useGrad) {
            // In this case there is no need to store the entire inc matrix.
            // Just this replica's shard (for synchronization purposes) will do.
            Matrix& hIncShard = getShard(*_hWeightsInc);
            _weightsInc->copyFromHost(hIncShard, true);
            delete &hIncShard;
        } else {
            _weightsInc->copyFromHost(*_hWeightsInc, true);
        }

        _weightsGrad = _useGrad ? (_weightsGrad == NULL ? new NVMatrix(*_weights) : _weightsGrad) : NULL;
    } else {
        _weights = _srcWeights->_weights;
        _weightsInc = _srcWeights->_weightsInc;
        _weightsGrad = _srcWeights->_weightsGrad;
    }
    _onGPU = true;
}

void Weights::aggregateReplicaGradients(float progress) {
    map<int, NVMatrix*> gradShards;
    map<int, NVMatrix*> wShards;
    for (map<int,Weights*>::const_iterator it = _replicas.begin(); it != _replicas.end(); ++it) {
        gradShards[it->first] = &getShard(it->second->getGrad(), getReplicaID());
        wShards[it->first] = &getShard(it->second->getW(), getReplicaID());
        assert(wShards[it->first]->isContiguous() && gradShards[it->first]->isContiguous());
    }

    float gradScale = _lrs->getValue(progress);
    NVMatrix::setDeviceID(getDeviceID());

    if (_wc > 0) {
        NVMatrixTernaryOps::WeightedAdd wadd = NVMatrixTernaryOps::WeightedAdd(_mom, gradScale, -_wc * _lrs->getValue(progress));
        _weightsInc->applyTernary(wadd, *gradShards[getReplicaID()], *wShards[getReplicaID()], *_weightsInc);
    } else {
        _weightsInc->add(*gradShards[getReplicaID()], _mom, gradScale);
    }

    // Reduce everyone's gradient into my inc shard
    NVMatrix::syncStream(); // Crucial since the reducer does everything in its own streams!!
    getReducer().reduce(gradShards, gradScale, true);

    // Broadcast my inc -> all replicas
    map<int, NVMatrix*> mats; // device id -> grad
    mats[getDeviceID()] = _weightsInc;
    for (map<int, Weights*>::const_iterator it = _replicas.begin(); it != _replicas.end(); ++it) {
        if (it->first != getReplicaID()) {
            mats[it->second->getDeviceID()] = wShards[it->first];
        }
    }
    getBroadcaster().broadcast(mats, 1, 1);

    NVMatrix::setDeviceID(getDeviceID());
    wShards[getReplicaID()]->add(*_weightsInc);

    // Cleanup
    for (map<int,Weights*>::const_iterator it = _replicas.begin(); it != _replicas.end(); ++it) {
        delete gradShards[it->first];
        delete wShards[it->first];
    }
}


// When _useGrad is false, weightsInc is assumed to contain the 
// entire, properly scaled weight increment.
// OTHERWISE, scale your gradient by 1 / numCases only.
// The scaling by epsW will be done in this routine.
void Weights::update(float progress) {
    // Only true owner of weights updates
//    printf("%s update weights\n", _parent->getName().c_str());
    if (_srcWeights == NULL && _lrs->getBaseValue() > 0) {
        assert(_onGPU);
        if (_useGrad) {
            aggregateReplicaGradients(progress);
        } else { // Definitely no replicas in this case
            if (_wc > 0) {
                _weightsInc->add(*_weights, -_wc * _lrs->getValue(progress));
            }
            _weights->add(*_weightsInc);
        }
        _numUpdates = 0;
    }
}

int Weights::incNumUpdates() {
    if (_srcWeights != NULL) {
        return _srcWeights->incNumUpdates();
    }
    return _numUpdates++;
}

// Returns the number of times a gradient has been computed for this
// weight matrix during the current pass (interval between two calls of update())
// through the net. This number will only be greater than 1 if this weight matrix
// is *shared* by multiple layers in the net.
int Weights::getNumUpdates() const {
    if (_srcWeights != NULL) {
        return _srcWeights->getNumUpdates();
    }
    return _numUpdates;
}

float Weights::getEps(float progress) const {
    return _lrs->getValue(progress);
}

float Weights::getMom() const {
    return _mom;
}

float Weights::getWC() const {
    return _wc;
}

float Weights::getWBall() const {
    return _wball;
}

bool Weights::isUseGrad() const { // is good grammar
    return _useGrad;
}

bool Weights::isOwner() const {
    return _srcWeights == NULL;
}

ParameterSchedule& Weights::getLearningRateSchedule() const {
	return *_lrs;
}

void Weights::addReplica(Weights& replica) {
    _replicas[replica.getReplicaID()] = &replica;

    const int n = _hWeights->getNumElements();
    _shardSize = DIVUP(n, _replicas.size());
}

int Weights::getReplicaID() {
    return _parent->getReplicaID();
}

int Weights::getDeviceID() {
    return _parent->getDeviceID();
}

Layer& Weights::getParent() {
    return *_parent;
}

/* 
 * ===============
 * WeightList
 * ===============
 */
Weights& WeightList::operator[](const int i) const {
    return *_weightList[i];
}

Weights& WeightList::at(const int i) const {
    return *_weightList[i];
}

WeightList::~WeightList() {
    for (int i = 0; i < _weightList.size(); i++) {
        delete _weightList[i];
    }
}

WeightList::WeightList() {
}

void WeightList::addWeights(Weights& w) {
    _weightList.push_back(&w);
}


void WeightList::update(float progress) {
    for (int i = 0; i < getSize(); i++) {
        _weightList[i]->update(progress);
    }
}

void WeightList::copyToCPU() {
    for (int i = 0; i < getSize(); i++) {
        _weightList[i]->copyToCPU();
    }
}

void WeightList::copyToGPU() {
    for (int i = 0; i < getSize(); i++) {
        _weightList[i]->copyToGPU();
    }
}

int WeightList::getSize() const {
    return _weightList.size();
}

void WeightList::addReplica(WeightList& replica) {
    for (int i = 0; i < getSize(); i++) {
        _weightList[i]->addReplica(replica[i]);
    }
}
