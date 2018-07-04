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

#include <helper_cuda.h>
#include <iostream>
#include <set>
#include "../../cudaconv3/include/cudaconv2.cuh"
#include "../../util/include/matrix.h"
#include "../include/layer_kernels.cuh"
#include "../include/layer.cuh"
#include "../include/data.cuh"
#include "../include/util.cuh"
#include "../include/weights.cuh"

using namespace std;

/*
 * =======================
 * Layer
 * =======================
 */
Layer::Layer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool trans) :
             _convNetThread(convNetThread),  _replicaID(replicaID), _trans(trans) {
    _name = pyDictGetString(paramsDict, "name");
    _type = pyDictGetString(paramsDict, "type");
   
    _foundGradConsumers = false;
    _gradConsumer = pyDictGetInt(paramsDict, "gradConsumer");
    _actsTarget = pyDictGetInt(paramsDict, "actsTarget");
    _actsGradTarget = pyDictGetInt(paramsDict, "actsGradTarget");
    _numOutputs = pyDictGetInt(paramsDict, "outputs");
    _numReplicas = pyDictGetInt(paramsDict, "numReplicas");
    _numReplicasPrev = 1;
    _rcvdBInputMsgs = 0;

    _actBroadcaster = NULL;
    _gradReducer = NULL;
    _initialized = false;
}

Layer::~Layer() {
    if (_actBroadcaster != NULL) {
        _actBroadcaster->stop();
        delete _actBroadcaster;
    }
    if (_gradReducer != NULL) {
        _gradReducer->stop();
        delete _gradReducer;
    }
    // For now, gradReducer doesn't have a destructor
//    delete _gradReducer;
    for (std::map<int, MemoryView*>::iterator it = _memSrcActs.begin(); it != _memSrcActs.end(); ++it) {
        if (it->second->getMemorySource().truncate(_name)) {
            delete &it->second->getMemorySource();
        }
    }
    for (std::map<int, MemoryView*>::iterator it = _memSrcActsGrad.begin(); it != _memSrcActsGrad.end(); ++it) {
        if (it->second->getMemorySource().truncate(_name)) {
            delete &it->second->getMemorySource();
        }
    }
}

cudaStream_t Layer::getStream() {
    assert(getDeviceID() >= 0);
    return NVMatrix::getDefaultStream(getDeviceID());
}

void Layer::syncStream() {
    NVMatrix::syncStream(getStream());
}

void Layer::fpropNext(PASS_TYPE passType, int passIdx) {
    if (_next.size() > 0) {
        if (getFwdActiveReplicaIdx(passIdx) == 0/*getReplicaIdx()*/) { // 0 turns on pipelining
            if (_nextDeviceIDs.size() > 1 || (_nextDeviceIDs.size() == 1 && _nextDeviceIDs[0] != getDeviceID())) {
                syncStream(); // Make sure I've finished computing before broadcasting
            }
            getActBroadcaster().getMessageQueue().enqueue(new BroadcastMessage(getAllActs(), getDeviceID(), getReplicaIdx(), _broadcastFinishQueue));
        }
        if (getFwdActiveReplicaIdx(passIdx) == getReplicaIdx()) {
            _broadcastFinishQueue.dequeue();
            assert(_broadcastFinishQueue.getNumElements() == 0);
        }
    }

    for (int i = 0; i < _next.size(); i++) {
        _next[i]->getConvNetThread().getMessageQueue().enqueue(new FpropMessage(*_next[i], passType, passIdx));
    }
}

bool Layer::fprop(PASS_TYPE passType, int passIdx) {
    _rcvdFInputMsgs++;
    // I require messages from *all* input replicas because it makes the propagation easier to think about.
    // Without this requirement, when all fprop terminal msgs arrive to ConvNet, the forward propagation
    // might not actually be finished yet.
    if (_rcvdFInputMsgs == getNumExpectedFwdMsgs()) {
//        printf("Layer %s[%d] fprop\n", _name.c_str(), getReplicaID());
        int ridx = getFwdActiveInputReplicaIdx(passIdx);
        assert(getDeviceID() == NVMatrix::getDeviceID());
        map<int, NVMatrix*> v;
        if (ridx >= 0) {
            for (int i = 0; i < getNumLayersPrev(); i++) {
                v[i] = &_prev[ridx][i]->getActs(getDeviceID());
            }
        }
        fprop(v, passType, passIdx);
        return true;
    }
    return false;
}

void Layer::fprop(map<int,NVMatrix*>& v, PASS_TYPE passType, int passIdx) {
    if (getFwdActiveInputReplicaIdx(passIdx) >= 0) {
        assert(v.size() == getNumLayersPrev());
        _inputs.clear();
        _inputs.insert(v.begin(), v.end());

        int numCases = _inputs[0]->getLeadingDim();
        for (map<int,MemoryView*>::iterator it = _memSrcActs.begin(); it != _memSrcActs.end(); ++it) {
            it->second->getMemory(numCases);
        }

        if (numCases > 0) {
            //printf("layer %s fprop, numcases: %d\n", _name.c_str(), numCases);
            _rcvdFInputMsgs = getNumExpectedFwdMsgs();
            for (map<int,NVMatrix*>::iterator it = v.begin(); it != v.end(); ++it) {
                it->second->transpose(_trans);
            }
            getActs().transpose(_trans);
   
            fpropCommon(passType);

            // First do fprop on the input whose acts matrix I'm sharing, if any
            if (_actsTarget >= 0) {
                fpropActs(_actsTarget, 0, passType, passIdx);
            }
            // Then add the rest of the inputs to that
            for (int i = 0; i < getNumLayersPrev(); i++) {
                if (i != _actsTarget) {
                    fpropActs(i, _actsTarget >= 0 || i > 0, passType, passIdx);
                }
            }
        }
    }
    fpropNext(passType, passIdx);
}

void Layer::truncBwdActs() {
    // Only truncate actsGrad if I own it
    if (_actsGradTarget < 0) {
        for (map<int,MemoryView*>::iterator it = _memSrcActsGrad.begin(); it != _memSrcActsGrad.end(); ++it) {
            it->second->getMemorySource().truncate(getName());
        }
    }
    if (_actsTarget < 0) {
        for (map<int,MemoryView*>::iterator it = _memSrcActs.begin(); it != _memSrcActs.end(); ++it) {
            it->second->getMemorySource().truncate(getName());
        }
    }
}

int Layer::getNumGradProducersNext() {
    return _numGradProducersNext;
}

int Layer::getNumExpectedBwdMsgs() {
    return _numGradProducersNext * getNumSiblingReplicas();
}

int Layer::getNumExpectedFwdMsgs() {
    return getNumLayersPrev() * getNumInputReplicas();
}

void Layer::bprop(PASS_TYPE passType, int passIdx) {
    if (getBwdActiveInputReplicaIdx(passIdx) >= 0 && _rcvdBInputMsgs == getNumExpectedBwdMsgs()) {
//        printf("Layer %s[%d] bprop\n", _name.c_str(), getReplicaID());
        if (_gradReducer != NULL) {
            _gradReducer->waitForFinish();
        }

        // This does sync, but only if it has grad consumers below! so we must sync again before sending bprop terminal messages
        bprop(getActsGrad(), passType, passIdx);
       
        if (_bwdTerminal[passIdx]) {
            syncStream();
            getConvNet().getMessageQueue().enqueue(new Message(BPROP_TERMINAL));
        }
    }
}

void Layer::bpropActsCall(NVMatrix& v, PASS_TYPE passType, int replicaIdx, int inputIdx) {
    Layer& prev = *_prev[replicaIdx][inputIdx];
    if (prev.isGradConsumer() && isGradProducer(prev.getName())) {
        if (v.getLeadingDim() > 0) { // Only do computation if #cases > 0
            bpropActs(v, replicaIdx, inputIdx, prev.getNumComputedActsGrads(getDeviceID()) > 0, passType);
        }
        prev.getNumComputedActsGrads(getDeviceID())++;
        // Synchronize if the previous layer is going to actually do a reduction.
        // If the previous layer is on the same GPU as us and has no next layers
        // on other GPUs then it won't need to do a reduction.
        if (prev.getNextDeviceIDs().size() > 1 || (prev.getNextDeviceIDs().size() == 1 && getDeviceID() != prev.getDeviceID())) {
            syncStream();
        }
        prev.getGradReducer().enqueueReduction(getDeviceID());
    }
}

void Layer::bprop(NVMatrix& v, PASS_TYPE passType, int passIdx) {

    v.transpose(_trans);
    assert(getDeviceID() == NVMatrix::getDeviceID());
    int ridx = getBwdActiveInputReplicaIdx(passIdx);
    LayerV& prev = _prev[ridx];
    map<int, set<Layer*> > prevByDevice = _prevByDevice[ridx];

    for (int i = 0; i < prev.size(); i++) {
        _inputs[i]->transpose(_trans);
        prev[i]->getActsGrad().transpose(_trans);
    }
    getActs().transpose(_trans);
    // NOTE: this should be here (before the bpropActs) because if you have a layer
    // that has a weight matrix AND actsGradTarget >= 0, then the stuff below will overwrite
    // v which is used in bpropCommon. So bpropCommon must come first.
    bpropCommon(v, ridx, passType);

    if (isGradProducer()) {
        // First propagate activity gradient to all layers whose activity
        // gradient matrix I'm definitely not sharing.
        for (map<int, set<Layer*> >::const_iterator it = prevByDevice.begin(); it != prevByDevice.end(); ++it) {
            const set<Layer*>& deviceLayers = it->second;
            for (set<Layer*>::const_iterator it2 = deviceLayers.begin(); it2 != deviceLayers.end(); ++it2) {
                if (_actsGradTarget != (*it2)->getInputIdx(_name)) {
                    bpropActsCall(v, passType, ridx, (*it2)->getInputIdx(_name));
                }
            }
        }

        // Then propagate activity gradient to the layer whose activity gradient
        // matrix I'm sharing, if any.
        if (_actsGradTarget >= 0) {
            bpropActsCall(v, passType, ridx, _actsGradTarget);
        }
    }

    // Synchronization is necessary because the kernel calls that compute my backward acts
    // execute asynchronously. Therefore I don't want to tell other threads that I've
    // computed bprop activities for them when in fact I've only called a function which
    // will eventually compute them.
    if (_prevDeviceIDs.size() > 1 || (_prevDeviceIDs.size() == 1 && _prevDeviceIDs[0] != getDeviceID())) {
        syncStream();
    }

    if (getConvNet().isConserveMemory()) {
        truncBwdActs();
    }

    if (isGradProducer()) {
        /*for (int i = 0; i < prev.size(); i++) {
            if (prev[i]->isGradConsumer() && isGradProducer(prev[i]->getName())) {
                prev[i]->getGradReducer().enqueueReduction(getDeviceID());
            }
        }*/

        // Send backward messages to *all* replicas.
        // Note that the messages will be dismissed unless the passIdx indicates
        // that the previous layer should do some work.
        for (int r = 0; r < getNumInputReplicas(); r++) {
            for (int i = 0; i < _prev[r].size(); i++) {
                if (_prev[r][i]->isGradConsumer() && isGradProducer(_prev[r][i]->getName())) {
                    _prev[r][i]->getConvNetThread().getMessageQueue().enqueue(new BpropMessage(*_prev[r][i], passType, passIdx));
                }
            }
        }
    }
}

IActGradReducer& Layer::getGradReducer() {
    return *_gradReducer;
}

// This is called between minibatches
void Layer::reset() {
    _rcvdFInputMsgs = 0;
    _rcvdBInputMsgs = 0;
    for (map<int,int>::iterator it = _numComputedActsGrads.begin(); it != _numComputedActsGrads.end(); ++it) {
        it->second = 0;
    }
}

// This is called between microbatches
void Layer::resetPassIdx() {
    _rcvdFInputMsgs = 0;
    if (_rcvdBInputMsgs >= getNumExpectedBwdMsgs()) {
        reset();
    }
}

/*
 * Returns number of cases in given matrix.
 */
int Layer::getNumCases(NVMatrix& v) {
    return v.getLeadingDim();
}

int Layer::incRcvdBInputMsgs() {
    return ++_rcvdBInputMsgs;
}

std::string& Layer::getName() {
    return _name;
}

std::string& Layer::getType() {
    return _type;
}

int& Layer::getNumComputedActsGrads(int deviceID) {
    return _numComputedActsGrads[deviceID];
}

void Layer::addNext(Layer& l) {
    _next.push_back(&l);
    _numReplicasNext = l.getNumReplicas();
    if (count(_nextDeviceIDs.begin(), _nextDeviceIDs.end(), l.getDeviceID()) == 0) {
        int pos = rand() % (_nextDeviceIDs.size() + 1);
        _nextDeviceIDs.insert(_nextDeviceIDs.begin() + pos, l.getDeviceID());
    }
}

void Layer::addPrev(Layer& l, int replicaIdx) {
    _prev[replicaIdx].push_back(&l);
    _numReplicasPrev = l.getNumReplicas();
    l.setInputIdx(getName(), _prev[replicaIdx].size() - 1);
    if (l.getDeviceID() >= 0 && count(_prevDeviceIDs.begin(), _prevDeviceIDs.end(), l.getDeviceID()) == 0) {
        int pos = rand() % (_prevDeviceIDs.size() + 1);
        _prevDeviceIDs.insert(_prevDeviceIDs.begin() + pos, l.getDeviceID());
    }
}

void Layer::addReplica(Layer& l) {
    assert(_replicas.count(l.getReplicaID()) == 0);
    _replicas[l.getReplicaID()] = &l;
}

bool Layer::hasGradProducerNext(std::string& layerName) {
    bool b = _next.size() == 0;
    for (int i = 0; i < _next.size(); i++) {
        b |= _next[i]->hasGradProducerNext(_name);
    }
    return b && isGradProducer(layerName);
}

bool Layer::postInit() {
    // We choose not to populate _outputs[getDeviceID()] here because we do it instead in fprop().
    // In fprop(), we can populate it from the _inputs vector, which is a bit more general than populating
    // it from _prev->getActs()
//    _outputs = _actsTarget < 0 ? new NVMatrix() : &_prev[_actsTarget]->getActs();
    if (!_initialized) {
        _initialized = true;
        map<int,int> numGradProducersNext;
        _numGradProducersNext = 0;
        for (int r = 0; r < getNumInputReplicas(); ++r) {
            for (vector<Layer*>::const_iterator it = _prev[r].begin(); it != _prev[r].end(); ++it) {
                (*it)->postInit();
            }
        }

        _memSrcActs[getDeviceID()] = _actsTarget < 0 ? &MemorySource::make(_numOutputs, getDeviceID(), getName())
                                                     : &_prev[0][_actsTarget]->getMemorySourceActs(getDeviceID()).clone(_name);

        // _actsGradTarget will only be >= 0 when the number of replicas is the same in both layers, so this justifies the use of _prev[0]

        _memSrcActsGrad[getDeviceID()] = _actsGradTarget < 0 ? &MemorySource::make(_numOutputs, getDeviceID(), getName())
                                                             : &_prev[0][_actsGradTarget]->getMemorySourceActsGrad(getDeviceID()).clone(_name);
        for (int i = 0; i < _next.size(); ++i) {
            int d = _next[i]->getDeviceID();
            _numComputedActsGrads[d] = 0;
            if (_next[i]->hasGradProducerNext(_name)) {
                if (numGradProducersNext.count(d) == 0) {
                    numGradProducersNext[d] = 0;
                }
                numGradProducersNext[d]++;
                _numGradProducersNext++;
                if (_memSrcActsGrad.count(d) == 0) {
                    _memSrcActsGrad[d] = &MemorySource::make(_numOutputs, d, getName());
                }
            }
            if (_memSrcActs.count(d) == 0) {
                _memSrcActs[d] = &MemorySource::make(_numOutputs, d, getName());
            }
        }

        if (_next.size() == 0) {
            _numReplicasNext = getNumReplicas();
        }

        /*
         * Initialize forward broadcaster. First sibling owns it.
         */
        if (getReplicaIdx() == 0 && _convNetThread != NULL) {
            _actBroadcaster = new ActBroadcaster(getNumSiblingReplicas(), getDeviceCPUs(_convNetThread->getDeviceID()));
            _actBroadcaster->start();
        }

        /*
         * Initialize backward reducer.
         */
        if (isGradConsumer() && _numGradProducersNext > 0) {
            _gradReducer = &IActGradReducer::makeGradReducer(*this, numGradProducersNext);
            _gradReducer->start();
        }

        /*
         * Initialize specially sorted previous array
         */
        for (int r = 0; r < _prev.size(); ++r) {
            for (int i = 0; i < _prev[r].size(); ++i) {
                // Previous devices in reverse order of processing by (sequential) GradReducer
                _prevByDevice[r][getDeviceID() - _prev[r][i]->getDeviceID()
                                 + 16 * (_prev[r][i]->getDeviceID() > getDeviceID())].insert(_prev[r][i]);

            }
        }
        return true;
    }
    return false;
}

ActBroadcaster& Layer::getActBroadcaster() {
    return getReplicaIdx() == 0 ? *_actBroadcaster : _replicas[getReplicaID() - getReplicaIdx()]->getActBroadcaster();
}

// Does this layer, or some layer below it, need the gradient
// for parameter updates?
// Only weight layers should be grad consumers themselves.
bool Layer::isGradConsumer() {
    if (!_foundGradConsumers && _prev.size() > 0) {
        for (int i = 0; i < _prev[0].size(); i++) {
            _gradConsumer |= _prev[0][i]->isGradConsumer();
        }
        _foundGradConsumers = true;
    }
    return _gradConsumer;
}

// Does this layer produce gradient for layers below?
bool Layer::isGradProducer() {
    return true;
}

bool Layer::isGradProducer(std::string& layerName) {
    return isGradProducer();
}

map<int,vector<Layer*> >& Layer::getPrev() {
    return _prev;
}

vector<Layer*>& Layer::getNext() {
    return _next;
}

NVMatrix& Layer::getActs() {
    return getActs(getDeviceID());
}

NVMatrix& Layer::getActs(int deviceID) {
    assert(_memSrcActs.count(deviceID) > 0);
    return _memSrcActs[deviceID]->getMemory();
}

NVMatrix& Layer::getActs(int deviceID, int numCases) {
    assert(_memSrcActs.count(deviceID) > 0);
    return _memSrcActs[deviceID]->getMemory(numCases);
}

NVMatrix& Layer::getActsGrad(int deviceID) {
    assert(_memSrcActsGrad.count(deviceID) > 0);
    return _memSrcActsGrad[deviceID]->getMemory(getActs(deviceID).getLeadingDim());
}

NVMatrix& Layer::getActsGrad() {
    return getActsGrad(NVMatrix::getDeviceID());
}

map<int, NVMatrix*> Layer::getAllActs() {
    map<int, NVMatrix*> m;
    for (map<int, MemoryView*>::const_iterator it = _memSrcActs.begin(); it != _memSrcActs.end(); ++it) {
        m[it->first] = &it->second->getMemory();
    }
    return m;
}

map<int, NVMatrix*> Layer::getAllActsGrads() {
    map<int, NVMatrix*> m;
    for (map<int, MemoryView*>::const_iterator it = _memSrcActsGrad.begin(); it != _memSrcActsGrad.end(); ++it) {
        m[it->first] = &it->second->getMemory();
    }
    return m;
}

int Layer::getDeviceID() {
    return _convNetThread == NULL ? -1 : _convNetThread->getDeviceID();
}

ConvNetThread& Layer::getConvNetThread() {
    assert(_convNetThread != NULL);
    return *_convNetThread;
}

ConvNet& Layer::getConvNet() {
    return getConvNetThread().getConvNet();
}

void Layer::setBwdTerminal(int passIdx) {
    _bwdTerminal[passIdx] = true;
}

int Layer::getReplicaID() {
    return  _replicaID;
}

int Layer::getActivePassPeriod() {
    return getNumReplicas() / getConvNet().getNumReplicasMin();
}

int Layer::getFwdActiveInputReplicaIdx(int passIdx) {
    const int edge = (passIdx / getActivePassPeriod()) % getNumInputReplicas();
    return passIdx % getActivePassPeriod() == 0 ? edge : -1;
}

int Layer::getBwdActiveInputReplicaIdx(int passIdx) {
    const int edge = (passIdx / getActivePassPeriod()) % getNumInputReplicas();
    return (passIdx + 1) % getActivePassPeriod() == 0 ? edge : -1;
}

int Layer::getFwdActiveReplicaIdx(int passIdx) {
    assert(_next.size() > 0);
    return _next[0]->getFwdActiveInputReplicaIdx(passIdx);
}

int Layer::getNumReplicas() {
    return _replicas.size();
}

int Layer::getNumSiblingReplicas() {
    return getNumReplicas() / getNumReplicasNext();
}

int Layer::getNumReplicasPrev() {
    return _numReplicasPrev;
}

int Layer::getNumReplicasNext() {
    return _numReplicasNext;
}

int Layer::getNumInputReplicas() {
    return _numReplicasPrev / getNumReplicas();
}

int Layer::getReplicaIdx() {
    return getReplicaID() % getNumSiblingReplicas();
}

int Layer::getNumLayersPrev() {
    return _prev.size() > 0 ? _prev[0].size() : 0;
}

void Layer::setMemorySourceActs(int deviceID, MemoryView& mem) {
    assert(_memSrcActs[deviceID]->isParent());
    delete _memSrcActs[deviceID];
    _memSrcActs[deviceID] = &mem;
    if (_actsTarget >= 0 && deviceID == getDeviceID()) {
        assert(getNumInputReplicas() == 1);
        _prev[0][_actsTarget]->setMemorySourceActs(deviceID, mem.clone(_prev[0][_actsTarget]->getName()));
    }
}

void Layer::setMemorySourceActsGrad(int deviceID, MemoryView& mem) {
    assert(_memSrcActsGrad[deviceID]->isParent());
    delete _memSrcActsGrad[deviceID];
    _memSrcActsGrad[deviceID] = &mem;
    if (_actsGradTarget >= 0 && deviceID == getDeviceID()) {
        assert(getNumInputReplicas() == 1);
        _prev[0][_actsGradTarget]->setMemorySourceActsGrad(deviceID, mem.clone(_prev[0][_actsGradTarget]->getName()));
    }
}

MemoryView& Layer::getMemorySourceActs(int deviceID) {
    return *_memSrcActs[deviceID];
}

MemoryView& Layer::getMemorySourceActsGrad(int deviceID) {
    return *_memSrcActsGrad[deviceID];
}

int Layer::getNumOutputs() {
    return _numOutputs;
}

void Layer::setInputIdx(std::string& parentName, int idx) {
    _inputIndices[parentName] = idx;
}

int Layer::getInputIdx(std::string& parentName) {
    return _inputIndices[parentName];
}

/*
 * =======================
 * NeuronLayer
 * =======================
 */
NeuronLayer::NeuronLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID)
    : Layer(convNetThread, paramsDict, replicaID, true) {
    PyObject* neuronDict = PyDict_GetItemString(paramsDict, "neuron");
    _neuronType = pyDictGetString(neuronDict, "type");
    _neuron = &Neuron::makeNeuron(neuronDict);
}

NeuronLayer::~NeuronLayer() {
    delete _neuron;
}

void NeuronLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 0);
    if (!bpropSpecial(v, replicaIdx, inpIdx, scaleTargets, passType)) {
        _neuron->computeInputGrad(v, _prev[replicaIdx][0]->getActsGrad(), scaleTargets > 0);
    }
}

bool NeuronLayer::bpropSpecial(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // Special optimization for cross-entropy objective with logistic units.
    // Better to just compute the input gradient in one go to avoid division by small numbers.
    bool doCrossEntGrad = _neuronType == "logistic" && _next.size() == 1
                        && (_next[0]->getType() == "cost.bce" || _next[0]->getType() == "cost.dce")
                        && _next[0]->getDeviceID() == getDeviceID()
                        && _next[0]->getNumReplicas() == getNumReplicas();
    LayerV& prev = _prev[replicaIdx];
    if (doCrossEntGrad) {
        NVMatrix& labels = _next[0]->getPrev()[replicaIdx][0]->getActs(getDeviceID());
        BinomialCrossEntropyCostLayer& cost = *static_cast<BinomialCrossEntropyCostLayer*>(_next[0]);
        float gradCoeff = cost.getCoeff();
        labels.transpose(_trans);
        if (cost.getPosWeight() == 1) {
            if (scaleTargets == 0) {
                getActs().add(labels, -gradCoeff, gradCoeff, prev[0]->getActsGrad());
            } else {
                getActs().applyTernary(AddGradientBinaryOperator<NVMatrixBinaryOps::WeightedAdd>(NVMatrixBinaryOps::WeightedAdd(-gradCoeff, gradCoeff)),
                                       labels, prev[0]->getActsGrad(), prev[0]->getActsGrad());
            }
        } else {
            if (scaleTargets == 0) {
                getActs().applyBinary(CrossEntLogisticGradientOperator(gradCoeff, cost.getPosWeight()), labels, prev[0]->getActsGrad());
            } else {
                getActs().applyTernary(AddGradientBinaryOperator<CrossEntLogisticGradientOperator>(CrossEntLogisticGradientOperator(gradCoeff, cost.getPosWeight())),
                                       labels, prev[0]->getActsGrad(), prev[0]->getActsGrad());
            }
        }
    }
    return doCrossEntGrad;
}

void NeuronLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    _neuron->activate(*_inputs[0], getActs());
}

std::string& NeuronLayer::getNeuronType() {
    return _neuronType;
}

/*
 * =======================
 * WeightLayer
 * =======================
 *
 * The useGrad parameter here merely expresses a preference by the subclass. It may
 * be overridden by the superclass (WeightLayer) and in that case the subclass must follow its wishes.
 * So when computing gradient updates, the subclass must always first check weights.isUseGrad().
 *
 * Note: biases always useGrad.
 */
WeightLayer::WeightLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool trans, bool useGrad) :
    Layer(convNetThread, paramsDict, replicaID, trans) {
    _weightUpdatePassPeriod = pyDictGetInt(paramsDict, "updatePeriod");

    MatrixV& hWeights = *pyDictGetMatrixV(paramsDict, "weights");
    MatrixV& hWeightsInc = *pyDictGetMatrixV(paramsDict, "weightsInc");
    Matrix& hBiases = *pyDictGetMatrix(paramsDict, "biases");
    Matrix& hBiasesInc = *pyDictGetMatrix(paramsDict, "biasesInc");
    PyObject* pyEpsWList = PyDict_GetItemString(paramsDict, "epsW");
    PyObject* pyEpsB = PyDict_GetItemString(paramsDict, "epsB");
    floatv& momW = *pyDictGetFloatV(paramsDict, "momW");
    float momB = pyDictGetFloat(paramsDict, "momB");
    floatv& wc = *pyDictGetFloatV(paramsDict, "wc");
    floatv& wball = *pyDictGetFloatV(paramsDict, "wballNormed");

    /*
     * When there are multiple replicas, the present implementation
     * requires that useGrad is true. This is because weights.update()
     * performs a simultaneous write to both replicas' weightsInc matrix,
     * which means that the read should come from somewhere else (i.e. a
     * grads matrix).
     */
    useGrad |= _numReplicas > 1;

    // Source layers for shared weights
    stringv& weightSourceLayers = *pyDictGetStringV(paramsDict, "weightSourceLayers");

    // Weight matrix indices (inside the above source layers) for shared weights
    intv& weightSourceMatrixIndices = *pyDictGetIntV(paramsDict, "weightSourceMatrixIndices");
    _weights = new WeightList();
    for (int i = 0; i < weightSourceLayers.size(); i++) {
        std::string& srcLayerName = weightSourceLayers[i];
        int matrixIdx = weightSourceMatrixIndices[i];
        PyObject* pyEpsW = PyList_GetItem(pyEpsWList, i);
        ParameterSchedule& lrs = ParameterSchedule::make(pyEpsW); // Learning rate schedule
        if (srcLayerName == _name) { // Current layer
            _weights->addWeights(*new Weights(_weights->at(matrixIdx), lrs, *this));
        } else if (srcLayerName != "") {
            WeightLayer& srcLayer = *static_cast<WeightLayer*>(&convNetThread->getLayer(srcLayerName));
            Weights* srcWeights = &srcLayer.getWeights(matrixIdx);
            _weights->addWeights(*new Weights(*srcWeights, lrs, *this));
        } else {
            _weights->addWeights(*new Weights(*hWeights[i], *hWeightsInc[i], lrs, *this, wc[i], wball[i], momW[i], useGrad));
        }
    }
    _biases = new Weights(hBiases, hBiasesInc, ParameterSchedule::make(pyEpsB), *this, 0, 0, momB, true);

    delete &weightSourceLayers;
    delete &weightSourceMatrixIndices;
    delete &hWeights;
    delete &hWeightsInc;
    delete &momW;
    delete &wc;
    delete &wball;

    _wStep = 0.02;
    _bStep = 0.005;
}

WeightLayer::~WeightLayer() {
    delete _weights;
    delete _biases;
}

bool WeightLayer::postInit() {
    if (Layer::postInit()) {
        _weightUpdatePassPeriod = max(_weightUpdatePassPeriod, getActivePassPeriod());
        assert(_weightUpdatePassPeriod % getActivePassPeriod() == 0);
        return true;
    }
    return false;
}

void WeightLayer::fpropCommon(PASS_TYPE passType) {
}

void WeightLayer::bpropCommon(NVMatrix& v, int replicaIdx, PASS_TYPE passType) {
    if (_biases->getLearningRateSchedule().getBaseValue() > 0) {
        if (v.getNumElements() > 0) {
            bpropBiases(v, passType);
        } else {
            _biases->getGrad().resize(_biases->getW());
            _biases->getGrad().scale(getBIncScale());
        }
        _biases->incNumUpdates();
    }
    for (int i = 0; i < _weights->getSize(); i++) {
        if (_weights->at(i).getLearningRateSchedule().getBaseValue() > 0) {
            if (v.getNumElements() > 0) {
                bpropWeights(v, replicaIdx, i, passType);
            } else {
                _weights->at(i).getGrad().resize(_weights->at(i).getW());
                // This will cause it to forget momentum when shown 0 training cases
                // and _useGrad = false but it's not too important.
                _weights->at(i).getGrad().scale(getIncScale(i, passType));
            }
            // Increment its number of updates
            _weights->at(i).incNumUpdates();
        }
    }
}

bool WeightLayer::updateWeights() {
     if (getConvNet().getTotalPassesDone() % _weightUpdatePassPeriod == 0) {
        _weights->update(getConvNet().getTrainingProgress());
        _biases->update(getConvNet().getTrainingProgress());
//        constrainWeights();
        return true;
    }
    return false;
}

bool WeightLayer::constrainWeights() {
    if (getConvNet().getTotalPassesDone() % _weightUpdatePassPeriod == 0) {
        _constrainWeights();
        return true;
    }
    return false;
}

void WeightLayer::_constrainWeights() {
}

void WeightLayer::copyToCPU() {
    _weights->copyToCPU();
    _biases->copyToCPU();
}

void WeightLayer::copyToGPU() {
    _weights->copyToGPU();
    _biases->copyToGPU();
}

void WeightLayer::checkGradient() {
    for (int i = 0; i < _weights->getSize(); i++) {
        getConvNet().checkGradient(_name + " weights[" + tostr(i) + "]", _wStep, _weights->at(i));
    }
    getConvNet().checkGradient(_name + " biases", _bStep, *_biases);
}

void WeightLayer::addReplica(Layer& l) {
    Layer::addReplica(l);
    _weights->addReplica(*static_cast<WeightLayer*>(&l)->_weights);
    _biases->addReplica(*static_cast<WeightLayer*>(&l)->_biases);
}

Weights& WeightLayer::getWeights(int idx) {
    return _weights->at(idx);
}

float WeightLayer::getGradScale(int inpIdx, PASS_TYPE passType) {
    // weight update period must be multiple of activation period
    // TODO: simply accumulate # of cases seen between weight updates. simpler and more accurate.
    double numCases = _weightUpdatePassPeriod * (getConvNet().getMinibatchSize() / double(getConvNet().getNumPasses()));
    if (_weights->at(inpIdx).isUseGrad()) {
        return passType == PASS_GC ? 1.0f : 1.0f / numCases;
    }
    return passType == PASS_GC ? 1.0f : _weights->at(inpIdx).getEps(getConvNet().getTrainingProgress()) / numCases;
}

float WeightLayer::getIncScale(int inpIdx, PASS_TYPE passType) {
    if (_weights->at(inpIdx).isUseGrad()) {
        return _weights->at(inpIdx).getNumUpdates() > 0;
    }
    return  (passType == PASS_GC ? _weights->at(inpIdx).getNumUpdates() > 0
                                 : (_weights->at(inpIdx).getNumUpdates() == 0 ? _weights->at(inpIdx).getMom() : 1.0f));
}

NVMatrix& WeightLayer::getGradTarget(int inpIdx) {
    return _weights->at(inpIdx).getGrad();
}

float WeightLayer::getBGradScale(PASS_TYPE passType) {
    int numCases = _weightUpdatePassPeriod * DIVUP(getConvNet().getMinibatchSize(), getConvNet().getNumPasses());
    return passType == PASS_GC ? 1.0f : 1.0f / numCases;
}

float WeightLayer::getBIncScale() {
    return _biases->getNumUpdates() > 0;
}

NVMatrix& WeightLayer::getWeightMatrix(PASS_TYPE passType, int inpIdx) {
    return _weights->at(inpIdx).getW();
}

NVMatrix& WeightLayer::getBiasMatrix(PASS_TYPE passType) {
    return _biases->getW();
}

/*
 * =======================
 * FCLayer
 * =======================
 */
FCLayer::FCLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool useGrad)
    : WeightLayer(convNetThread, paramsDict, replicaID, true, useGrad) {
    _wStep = 0.01;
    _bStep = 0.01;
}

void FCLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    getActs().addProduct(*_inputs[inpIdx], getWeightMatrix(passType, inpIdx), scaleTargets, 1);
    if (scaleTargets == 0) {
        getActs().addVector(getBiasMatrix(passType), 1, getActs());
    }
}

void FCLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& weights_T = getWeightMatrix(passType, inpIdx).getTranspose();
    _prev[replicaIdx][inpIdx]->getActsGrad().addProduct(v, weights_T, scaleTargets, 1);
    delete &weights_T;
}

void FCLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    _biases->getGrad().addSum(v, 0, getBIncScale(), getBGradScale(passType));
}

void FCLayer::bpropWeights(NVMatrix& v, int replicaIdx, int inpIdx, PASS_TYPE passType) {
    NVMatrix& prevActs_T = _inputs[inpIdx]->getTranspose();
    float scaleGrad = getGradScale(inpIdx, passType);
    float scaleInc = getIncScale(inpIdx, passType);
    getGradTarget(inpIdx).addProduct(prevActs_T, v, scaleInc, scaleGrad);
    delete &prevActs_T;
}

void FCLayer::_constrainWeights() {
    for (int i = 0; i < _weights->getSize(); i++) {
        if (_weights->at(i).getWBall() > 0 && _weights->at(i).isOwner() && _weights->at(i).getLearningRateSchedule().getBaseValue() > 0) {
//            NVMatrix norm2; // Unfortunate extra weight matrix...
            _weights->at(i).getW().sumOfSquares(0, _norm2);
//            norm2.apply(MaxWeightConstraintOperator(_weights->at(i).getWBall()));
            _norm2.apply(HardWeightConstraintOperator(_weights->at(i).getWBall()));
            _weights->at(i).getW().eltwiseMultByVector(_norm2);
        }
    }
}

/*
 * =======================
 * SplitFCLayer
 * =======================
 */
SplitFCLayer::SplitFCLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool useGrad)
    : FCLayer(convNetThread, paramsDict, replicaID, useGrad) {
    _numParts = pyDictGetInt(paramsDict, "parts");
}

void SplitFCLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    getActs().resize(_inputs[inpIdx]->getNumRows(), _numOutputs, true);
    NVMatrixV& splitInput = _inputs[inpIdx]->splitCols(_numParts);
    NVMatrixV& splitWeights = getWeightMatrix(passType, inpIdx).splitRows(_numParts);
    NVMatrixV& splitTarget = getActs().splitCols(_numParts);

    NVMatrix::batchedMatrixMultiply(splitInput, splitWeights, splitTarget, scaleTargets, 1);
    if (scaleTargets == 0) {
        getActs().addVector(getBiasMatrix(passType), 1, getActs());
    }

    deleteElements(splitInput, true);
    deleteElements(splitWeights, true);
    deleteElements(splitTarget, true);
}

void SplitFCLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& weights_T = getWeightMatrix(passType, inpIdx).getTranspose();
    _prev[replicaIdx][inpIdx]->getActsGrad().resize(*_inputs[inpIdx]);

    NVMatrixV& splitV = v.splitCols(_numParts);
    NVMatrixV& splitWeights_T = weights_T.splitCols(_numParts);
    NVMatrixV& splitTarget = _prev[replicaIdx][inpIdx]->getActsGrad().splitCols(_numParts);

    NVMatrix::batchedMatrixMultiply(splitV, splitWeights_T, splitTarget, scaleTargets, 1);

    delete &weights_T;
    deleteElements(splitV, true);
    deleteElements(splitWeights_T, true);
    deleteElements(splitTarget, true);
}

void SplitFCLayer::bpropWeights(NVMatrix& v, int replicaIdx, int inpIdx, PASS_TYPE passType) {
    NVMatrix& prevActs_T = _inputs[inpIdx]->getTranspose();
    NVMatrixV& splitPrevActs_T = prevActs_T.splitRows(_numParts);
    NVMatrixV& splitV = v.splitCols(_numParts);
    NVMatrixV& splitGradTarget = getGradTarget(inpIdx).splitRows(_numParts);

    NVMatrix::batchedMatrixMultiply(splitPrevActs_T, splitV, splitGradTarget, getIncScale(inpIdx, passType), getGradScale(inpIdx, passType));

    delete &prevActs_T;
    deleteElements(splitPrevActs_T, true);
    deleteElements(splitV, true);
    deleteElements(splitGradTarget, true);
}

/*
 * =======================
 * TwoDLayerInterface
 * =======================
 */
TwoDLayerInterface::TwoDLayerInterface(PyObject* paramsDict) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
    _imgPixels = _imgSize * _imgSize;
}

/*
 * =======================
 * LocalLayer
 * =======================
 */
LocalLayer::LocalLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool useGrad)
    : WeightLayer(convNetThread, paramsDict, replicaID, false, useGrad) {
    _padding = pyDictGetIntV(paramsDict, "padding");
    _stride = pyDictGetIntV(paramsDict, "stride");
    _filterSize = pyDictGetIntV(paramsDict, "filterSize");
    _channels = pyDictGetIntV(paramsDict, "channels");
    _imgSize = pyDictGetIntV(paramsDict, "imgSize");
    _numFilters = pyDictGetInt(paramsDict, "filters");
    _groups = pyDictGetIntV(paramsDict, "groups");
    _filterChannels = pyDictGetIntV(paramsDict, "filterChannels");
    _filterPixels = pyDictGetIntV(paramsDict, "filterPixels");
    _imgPixels = pyDictGetIntV(paramsDict, "imgPixels");
   
    _modulesX = pyDictGetInt(paramsDict, "modulesX");
    _modules = pyDictGetInt(paramsDict, "modules");
}

LocalLayer::~LocalLayer() {
    delete _padding;
    delete _stride;
    delete _filterSize;
    delete _channels;
    delete _imgSize;
    delete _groups;
    delete _filterChannels;
    delete _filterPixels;
    delete _imgPixels;
}

/*
 * =======================
 * ConvLayer
 * =======================
 */
ConvLayer::ConvLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID)
    : LocalLayer(convNetThread, paramsDict, replicaID, true) {
    _sumWidth = pyDictGetInt(paramsDict, "sumWidth");
    _sharedBiases = pyDictGetInt(paramsDict, "sharedBiases");
    _weightContrastNormMin = pyDictGetFloatV(paramsDict, "wcNormMin");
    _weightContrastNormMax = pyDictGetFloatV(paramsDict, "wcNormMax");
}

ConvLayer::~ConvLayer() {
    delete _weightContrastNormMin;
    delete _weightContrastNormMax;
}

void ConvLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    convFilterActs(*_inputs[inpIdx], getWeightMatrix(passType, inpIdx), getActs(), _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx),
                   _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);

    if (scaleTargets == 0) {
        if (_sharedBiases) {
            getActs().reshape(_numFilters, getActs().getNumElements() / _numFilters);
            getActs().addVector(getBiasMatrix(passType));
            getActs().reshape(_numFilters * _modules, getActs().getNumElements() / (_numFilters * _modules));
        } else {
            getActs().addVector(getBiasMatrix(passType));
        }
    }
}

void ConvLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    float scaleBGrad = getBGradScale(passType);
    float scaleInc = getBIncScale();
    if (_sharedBiases) {
        v.reshape(_numFilters, v.getNumElements() / _numFilters);
        _biases->getGrad().addSum(v, 1, scaleInc, scaleBGrad);
        v.reshape(_numFilters * _modules, v.getNumElements() / (_numFilters * _modules));
    } else {
        _biases->getGrad().addSum(v, 1, scaleInc, scaleBGrad);
    }
}

void ConvLayer::bpropWeights(NVMatrix& v, int replicaIdx, int inpIdx, PASS_TYPE passType) {
    assert(_weights->at(inpIdx).isUseGrad());
    bool doPartialSum = _sumWidth < _modulesX;
    NVMatrix& tgt = doPartialSum ? _weightGradTmp : _weights->at(inpIdx).getGrad();

    float scaleWGrad = getGradScale(inpIdx, passType);
    float scaleTargets = getIncScale(inpIdx, passType) * !doPartialSum;

    convWeightActs(*_inputs[inpIdx], v, tgt, _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx), _padding->at(inpIdx),
                   _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), _sumWidth, scaleTargets, scaleWGrad);

    if (doPartialSum) {
        scaleTargets = _weights->at(inpIdx).getNumUpdates() > 0;
        int outWidth = DIVUP(_modulesX, _sumWidth);
        _weightGradTmp.reshape(outWidth*outWidth, _filterChannels->at(inpIdx) * _filterPixels->at(inpIdx) * _numFilters);
        _weights->at(inpIdx).getGrad().addSum(_weightGradTmp, 0, scaleTargets, 1);
        _weights->at(inpIdx).getGrad().reshape(_filterChannels->at(inpIdx) * _filterPixels->at(inpIdx), _numFilters);
    }
}

void ConvLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convImgActs(v, getWeightMatrix(passType, inpIdx), _prev[replicaIdx][inpIdx]->getActsGrad(), _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX,
                _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
}

void ConvLayer::truncBwdActs() {
    LocalLayer::truncBwdActs();
    _weightGradTmp.truncate();
}

void ConvLayer::_constrainWeights() {
    for (int i = 0; i < _weights->getSize(); i++) {
        if (_weightContrastNormMax->at(i) > 0 && _weights->at(i).isOwner() && _weights->at(i).getLearningRateSchedule().getBaseValue() > 0) {
            float fz = _weights->at(i).getW().getNumRows();
            NVMatrix tmp;
            _weights->at(i).getW().sum(0, tmp);
            _weights->at(i).getW().addVector(tmp, -1.0f / fz, _weights->at(i).getGrad());
            // Now _weights->at(i).getGrad() contains zero-mean filters
            _weights->at(i).getGrad().apply(NVMatrixOps::Square());
            _weights->at(i).getGrad().sum(0, tmp);

            tmp.apply(WeightContrastNormOperator(_weightContrastNormMin->at(i), _weightContrastNormMax->at(i), 1.0f / fz));
            // Now tmp has the stdev
            _weights->at(i).getW().eltwiseMultByVector(tmp);
        }
        // It's pretty silly to do both these things but whatever
        if (_weights->at(i).getWBall() > 0 && _weights->at(i).isOwner() && _weights->at(i).getLearningRateSchedule().getBaseValue() > 0) {
//            NVMatrix norm2;
            _weights->at(i).getW().sumOfSquares(0, _norm2);

//            norm.apply(MaxWeightConstraintOperator(_weights->at(i).getWBall()));
            _norm2.apply(HardWeightConstraintOperator(_weights->at(i).getWBall()));
            _weights->at(i).getW().eltwiseMultByVector(_norm2);
        }
    }
}

/*
 * =======================
 * LocalUnsharedLayer
 * =======================
 */
LocalUnsharedLayer::LocalUnsharedLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID)
    : LocalLayer(convNetThread, paramsDict, replicaID, false) {
}

void LocalUnsharedLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    localFilterActs(*_inputs[inpIdx], getWeightMatrix(passType, inpIdx), getActs(), _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx),
                    _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    if (scaleTargets == 0) {
        getActs().addVector(getBiasMatrix(passType));
    }
}

void LocalUnsharedLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    _biases->getGrad().addSum(v, 1, getBIncScale(), getBGradScale(passType));
}

void LocalUnsharedLayer::bpropWeights(NVMatrix& v, int replicaIdx, int inpIdx, PASS_TYPE passType) {
    float scaleWGrad = getGradScale(inpIdx, passType);
    float scaleInc = getIncScale(inpIdx, passType);
    localWeightActs(*_inputs[inpIdx], v, getGradTarget(inpIdx), _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx), _padding->at(inpIdx),
                    _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleInc, scaleWGrad);
}

void LocalUnsharedLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    localImgActs(v, getWeightMatrix(passType, inpIdx), _prev[replicaIdx][inpIdx]->getActsGrad(),_imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX,
                 _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
}

void LocalUnsharedLayer::_constrainWeights() {
    for (int i = 0; i < _weights->getSize(); i++) {
        if (_weights->at(i).getWBall() > 0  && _weights->at(i).isOwner() && _weights->at(i).getLearningRateSchedule().getBaseValue() > 0) {
            normalizeLocalWeights(*_weights->at(i), _modules, _weights->at(i).getWBall());
        }
    }
}

/*
 * =======================
 * SoftmaxLayer
 * =======================
 */
SoftmaxLayer::SoftmaxLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID)
    : Layer(convNetThread, paramsDict, replicaID, true), _doUpperGrad(false) {
}

void SoftmaxLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    NVMatrix& input = *_inputs[0];
    input.max(1, _max);
    input.addVector(_max, -1, getActs());
    getActs().apply(NVMatrixOps::Exp());
    getActs().sum(1, _sum);
    getActs().eltwiseDivideByVector(_sum);
}

void SoftmaxLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 0);
    LayerV& prev = _prev[replicaIdx];
    if (_doUpperGrad) {
        // Todo: rethink replica IDs or idxes... this here doesn't make a huge amount of sense
        for (int i = 0; i < _next.size(); ++i) {
            if (_next[i]->isGradProducer(getName())) {
                NVMatrix& labels = _next[i]->getPrev()[replicaIdx][0]->getActs(getDeviceID()); // Get cost's labels
                float gradCoeff = dynamic_cast<CostLayer*>(_next[i])->getCoeff();

                computeLogregSoftmaxGrad(labels, getActs(), prev[0]->getActsGrad(), scaleTargets == 1, gradCoeff);
                break;
            }
        }

    } else {
        computeSoftmaxGrad(getActs(), v, prev[0]->getActsGrad(), scaleTargets, 1);
    }
}

void SoftmaxLayer::setDoUpperGrad(bool b) {
    _doUpperGrad = b;
}

/*
 * =======================
 * ConcatenationLayer
 * =======================
 */
ConcatenationLayer::ConcatenationLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID)
    : Layer(convNetThread, paramsDict, replicaID, false) {
    _copyOffsets = pyDictGetIntV(paramsDict, "copyOffsets");
    _copyOffsets->push_back(_numOutputs);
}

ConcatenationLayer::~ConcatenationLayer() {
    delete _copyOffsets;
}

void ConcatenationLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    getActs().resize(_numOutputs, _inputs[inpIdx]->getNumCols());
    _inputs[inpIdx]->copy(getActs(), 0, -1, 0, -1, _copyOffsets->at(inpIdx), 0);
}

void ConcatenationLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& copySrc = v.sliceRows(_copyOffsets->at(inpIdx), _copyOffsets->at(inpIdx + 1)); // view
    _prev[replicaIdx][inpIdx]->getActsGrad().add(copySrc, scaleTargets, 1);
    delete &copySrc;
}

/*
 * =======================
 * PassThroughLayer
 * =======================
 */
PassThroughLayer::PassThroughLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID)
    : Layer(convNetThread, paramsDict, replicaID, false) {
}

void PassThroughLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    // No-op
}

void PassThroughLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // No-op
}

bool PassThroughLayer::postInit() {
    if (Layer::postInit()) {
        assert(getNumInputReplicas() == 1);
        for (int i = 0, offset = 0; i < _prev[0].size(); offset += _prev[0][i]->getNumOutputs(), i++) {
            MemoryView& vActs = _memSrcActs[getDeviceID()]->getMemorySource().addUser(_prev[0][i]->getName(), pair<int,int>(offset, offset + _prev[0][i]->getNumOutputs()));
            MemoryView& vActsGrad = _memSrcActsGrad[getDeviceID()]->getMemorySource().addUser(_prev[0][i]->getName(), pair<int,int>(offset, offset + _prev[0][i]->getNumOutputs()));
            _prev[0][i]->setMemorySourceActs(getDeviceID(), vActs);
            _prev[0][i]->setMemorySourceActsGrad(getDeviceID(), vActsGrad);
        }
        return true;
    }
    return false;
}


/*
 * =======================
 * EltwiseSumLayer
 * =======================
 */
EltwiseSumLayer::EltwiseSumLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : Layer(convNetThread, paramsDict, replicaID, false) {
    _coeffs = pyDictGetFloatV(paramsDict, "coeffs");
}

EltwiseSumLayer::~EltwiseSumLayer() {
    delete _coeffs;
}

void EltwiseSumLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    getActs().add(*_inputs[inpIdx], scaleTargets, _coeffs->at(inpIdx));
}

void EltwiseSumLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _prev[replicaIdx][inpIdx]->getActsGrad().add(v, scaleTargets, _coeffs->at(inpIdx));
}

/*
 * =======================
 * EltwiseMaxLayer
 * =======================
 */
EltwiseMaxLayer::EltwiseMaxLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : Layer(convNetThread, paramsDict, replicaID, false) {
}

void EltwiseMaxLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    if (inpIdx == 1) { // First input, do nothing
        _inputs[inpIdx]->applyBinary(NVMatrixAggs::Max(), *_inputs[0], getActs());
    } else if (inpIdx > 1) {
        getActs().applyBinary(NVMatrixAggs::Max(), *_inputs[inpIdx]);
    }
}

void EltwiseMaxLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    computeEltwiseMaxGrad(v, *_inputs[inpIdx], getActs(), _prev[replicaIdx][inpIdx]->getActsGrad(), scaleTargets != 0);
}


/*
 * =======================
 * DropoutLayer
 * =======================
 *
 * TODO: optimize away the case when using dopout over relus. Don't need the keepmask.
 */
DropoutLayer::DropoutLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : Layer(convNetThread, paramsDict, replicaID, false) {
    _enable = pyDictGetInt(paramsDict, "enable");
    _keep = pyDictGetFloat(paramsDict, "keep");
}

void DropoutLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    if (_enable && passType == PASS_TRAIN) {
        _keepMask.resize(*_inputs[inpIdx]);
        _keepMask.randomizeUniform();
        _keepMask.apply(DropoutSmallerThanOperator(_keep));
        _inputs[inpIdx]->eltwiseMult(_keepMask, getActs());
    } else {
        _inputs[inpIdx]->copy(getActs());
    }
}

void DropoutLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    LayerV& prev = _prev[replicaIdx];
    if (_enable && passType == PASS_TRAIN) {
        if (scaleTargets != 0) {
            v.applyTernary(AddGradientBinaryOperator<NVMatrixBinaryOps::Multiply>(NVMatrixBinaryOps::Multiply()),
                           _keepMask, prev[inpIdx]->getActsGrad(), prev[inpIdx]->getActsGrad());
        } else {
            v.eltwiseMult(_keepMask, prev[inpIdx]->getActsGrad());
        }
    } else {
         prev[inpIdx]->getActsGrad().add(v, scaleTargets, 1);
    }
}

void DropoutLayer::truncBwdActs() {
    Layer::truncBwdActs();
    _keepMask.truncate();
}


/*
 * =======================
 * Dropout2Layer
 * =======================
 *
 * TODO: optimize away the case when using dopout over relus. Don't need the keepmask.
 */
Dropout2Layer::Dropout2Layer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : DropoutLayer(convNetThread, paramsDict, replicaID) {
}

void Dropout2Layer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    if (_enable && passType == PASS_TRAIN) {
        _keepMask.resize(*_inputs[inpIdx]);
        _keepMask.randomizeUniform();
        _keepMask.smallerThanScalar(_keep);
        _inputs[inpIdx]->eltwiseMult(_keepMask, getActs());
    } else {
        _inputs[inpIdx]->scale(_keep, getActs());
    }
}

void Dropout2Layer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    LayerV& prev = _prev[replicaIdx];
    if (_enable && passType == PASS_TRAIN) {
        if (scaleTargets != 0) {
            v.applyTernary(AddGradientBinaryOperator<NVMatrixBinaryOps::Multiply>(NVMatrixBinaryOps::Multiply()),
                           _keepMask, prev[inpIdx]->getActsGrad(), prev[inpIdx]->getActsGrad());
        } else {
            v.eltwiseMult(_keepMask, prev[inpIdx]->getActsGrad());
        }
    } else {
        if (scaleTargets != 0) {
             v.applyBinary(AddGradientOperator<NVMatrixOps::MultByScalar>(NVMatrixOps::MultByScalar(_keep)),
                           prev[inpIdx]->getActsGrad(), prev[inpIdx]->getActsGrad());
        } else {
            v.scale(_keep, prev[inpIdx]->getActsGrad());
        }
    }
}

/*
 * =======================
 * DataLayer
 * =======================
 */
DataLayer::DataLayer(ConvNet* convNet, PyObject* paramsDict, int replicaID) : Layer(NULL, paramsDict, replicaID, false) {
    _dataIdx = pyDictGetInt(paramsDict, "dataIdx");
    _start = pyDictGetInt(paramsDict, "start");
    _end = pyDictGetInt(paramsDict, "end");
    _useBuffer = false;
    _outstandingCopyRequest = false;
    _convNet = convNet;
}

DataLayer::~DataLayer() {
    for (map<int,cudaStream_t>::const_iterator it = _copyStreams.begin(); it != _copyStreams.end(); ++it) {
        checkCudaErrors(cudaStreamDestroy(it->second));
    }
    for (std::map<int, MemoryView*>::iterator it = _memSrcActs2.begin(); it != _memSrcActs2.end(); ++it) {
        if (it->second->getMemorySource().truncate(_name)) {
            delete &it->second->getMemorySource();
        }
    }
    _copier->stop();
    delete _copier;
}

void DataLayer::fprop(PASS_TYPE passType, int passIdx, bool fromBuffer) {
    waitForCopyFinish();
    if (fromBuffer && getFwdActiveInputReplicaIdx(passIdx) >= 0) {
        _useBuffer = !_useBuffer;
    }

    for (int i = 0; i < _next.size(); i++) {
        _next[i]->getConvNetThread().getMessageQueue().enqueue(new FpropMessage(*_next[i], passType, passIdx));
    }
}

void DataLayer::waitForCopyFinish() {
    if (_outstandingCopyRequest) {
        _copyFinishQueue.dequeue();
        assert(_copyFinishQueue.getNumElements() == 0);
        _outstandingCopyRequest = false;
    }
}

cudaStream_t DataLayer::getCopyStream(int deviceID) {
    if (_copyStreams.count(deviceID) == 0) {
        NVMatrix::setDeviceID(deviceID);
        checkCudaErrors(cudaStreamCreateWithFlags(&_copyStreams[deviceID], cudaStreamNonBlocking));
    }
    return _copyStreams[deviceID];
}

void DataLayer::copyData(CPUData& data, bool other, int passIdx) {
    assert(!_outstandingCopyRequest);
    assert(_copyFinishQueue.getNumElements() == 0);
    _copier->getQueue().enqueue(new DataCopyMessage(data, other, passIdx));
    _outstandingCopyRequest = true;
}

int DataLayer::getNumInputReplicas() {
    return _convNet->getNumReplicasMax() / getNumReplicas();
}

void DataLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
   
}

NVMatrix& DataLayer::getActs(int deviceID) {
    return getActs(deviceID, false, -1);
}

NVMatrix& DataLayer::getActs(int deviceID, bool other, int numCases) {
//    printf("%s[%d] getActs(%d, %d, %d)\n", _name.c_str(), getReplicaID(), deviceID, other, numCases);
    assert(_memSrcActs.count(deviceID) > 0);
    assert(_memSrcActs2.count(deviceID) > 0);
    return (_useBuffer != other ? _memSrcActs2[deviceID]->getMemory(numCases) : _memSrcActs[deviceID]->getMemory(numCases));
}

ConvNet& DataLayer::getConvNet() {
    return *_convNet;
}

bool DataLayer::postInit() {
    if (Layer::postInit()) {
        for (int i = 0; i < _next.size(); ++i) {
            int d = _next[i]->getDeviceID();
            if (_memSrcActs2.count(d) == 0) {
                _memSrcActs2[d] = &MemorySource::make(_numOutputs, d, getName());
            }
        }
        intv cpus = getDeviceCPUs(_next[0]->getDeviceID());
        _copier = new DataCopyThread(*this, cpus);
        _copier->start();
        return true;
    }
    return false;
}

bool DataLayer::isGradProducer() {
    return false;
}

/*
 * =======================
 * DataCopyThread
 * =======================
 */
DataCopyThread::DataCopyThread(DataLayer& parent, intv& cpus) : _parent(&parent), _sleepUsec(0), Thread(true, cpus) {
}

Queue<DataCopyMessage*>& DataCopyThread::getQueue() {
    return _queue;
}

void DataCopyThread::stop() {
    getQueue().enqueue(new DataCopyExitMessage());
    join();
}

void* DataCopyThread::run() {
    NVMatrix::setDeviceID(*_parent->getNextDeviceIDs().begin());
    bool exit = false;
    while(!exit) {
        DataCopyMessage& msg = *_queue.dequeue();
        exit = msg.getType() == DataCopyMessage::EXIT;
        if (!exit) {
            CPUData& data = msg.getData();
            int passIdx = msg.getPassIdx();
            bool other = msg.isOther();

            Matrix& dataMatrix = data.getData(_parent->getDataIdx());
            // How many times is this layer going to process microbatches from this minibatch?
            assert(_parent->getNumReplicasNext() == _parent->getNumReplicas());
            int microIdx = _parent->getFwdActiveInputReplicaIdx(passIdx);

            if (microIdx >= 0) {
                if (_requestTimer.isStarted()) {
                    double requestIntervalMsec = _requestTimer.stop();
                    // Sleep for up to 1/20th the average request interval
                    _sleepUsec = int(round(0.95 * _sleepUsec + 0.05 * (_parent->getReplicaID() / double(_parent->getNumReplicas())) * requestIntervalMsec * 1000.0 / 20.0));
                }
                _requestTimer.start();
                if (other) {
                    // Sleeping a bit is helpful because in typical nets, copying input data
                    // as soon as it's available will produce contention with other communications
                    // that are happening at the time. This is very much a hack, so in the future
                    // it might be good to replace it with something smarter which schedules access
                    // to communication links.
                    usleep(_sleepUsec);
                }
                microIdx += _parent->getReplicaID() * _parent->getNumInputReplicas();
                // Safer to divup because this way you won't get a minibatch size of 0
                int microbatchSize = DIVUP(data.getNumCases(), _parent->getConvNet().getNumReplicasMax());
                int microStart = microIdx * microbatchSize;
                int microEnd = min(data.getNumCases(), (microIdx + 1) * microbatchSize);
                // Check that this replica has some data. This can be false when, for example,
                // there are only 7 examples in the minibatch but 8 replicas.
                if (microStart < microEnd) {
                    assert(dataMatrix.isView() == dataMatrix.isTrans());
                    int pipe = _parent->getConvNet().getDataCopyPD().getPipe(_parent->getReplicaID()/2);
                    if (dataMatrix.isTrans()) {
                        Matrix& replicaDataMatrix = dataMatrix.sliceCols(microStart, microEnd);
                        // In this case, dataMatrix is a view on memory allocated by Python.
                        //_hostMemFwd.copyFromHost(replicaDataMatrix, true);
                        _hostMemFwd.resize(replicaDataMatrix.getNumRows(), replicaDataMatrix.getNumCols(), true);
                        memcpy(_hostMemFwd.getDevData(), replicaDataMatrix.getData(), replicaDataMatrix.getNumDataBytes());
                        delete &replicaDataMatrix; // view
                        NVMatrix& hostMemFwdSlice = _hostMemFwd.sliceRows(_parent->getStart(), _parent->getEnd());
                        for (intv::iterator it = _parent->getNextDeviceIDs().begin(); it != _parent->getNextDeviceIDs().end(); ++it) {
                            int deviceID = *it;
                            // Copy my output to this guy's GPU
                            NVMatrix::setDeviceID(deviceID);
                            // Note to self: this is the path that gets executed in practice
                            // in my models. It does a transpose & copy simultaneously.
                            hostMemFwdSlice.flipTrans(_parent->getActs(deviceID, other, microEnd - microStart), _parent->getCopyStream(deviceID));
                        }
                        delete &hostMemFwdSlice;
                    } else {
                        // Hacky way to copy a slice to _hostMemFwd
                        _hostMemFwd.resize(dataMatrix.getNumRows(), microEnd - microStart);
                        Matrix tmp(_hostMemFwd.getDevData(), _hostMemFwd.getNumRows(), _hostMemFwd.getNumCols(), _hostMemFwd.isTrans());
                        dataMatrix.sliceCols(microStart, microEnd, tmp);
                        NVMatrix& hostMemFwdSlice = _hostMemFwd.sliceRows(_parent->getStart(), _parent->getEnd());
                        for (intv::iterator it = _parent->getNextDeviceIDs().begin(); it != _parent->getNextDeviceIDs().end(); ++it) {
                            int deviceID = *it;
                            // Copy my output to this guy's GPU
                            NVMatrix::setDeviceID(deviceID);
                            hostMemFwdSlice.copy(_parent->getActs(deviceID, other, microEnd - microStart), _parent->getCopyStream(deviceID));
                        }
                        delete &hostMemFwdSlice;
                    }

                    for (intv::iterator it = _parent->getNextDeviceIDs().begin(); it != _parent->getNextDeviceIDs().end(); ++it) {
                        int deviceID = *it;
                        NVMatrix::setDeviceID(deviceID);
                        NVMatrix::syncStream(_parent->getCopyStream(deviceID));
                    }
                    _parent->getConvNet().getDataCopyPD().freePipe(pipe);
                } else {
                    for (intv::iterator it = _parent->getNextDeviceIDs().begin(); it != _parent->getNextDeviceIDs().end(); ++it) {
                        int deviceID = *it;
                        _parent->getActs(deviceID, other, 0);
                    }
                }
            }
            _parent->getCopyFinishQueue().enqueue(1);
        }
        delete &msg;
    }
    return NULL;
}

/*
 * =====================
 * PoolLayer
 * =====================
 */
PoolLayer::PoolLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool trans)
    : Layer(convNetThread, paramsDict, replicaID, trans), TwoDLayerInterface(paramsDict) {
    _sizeX = pyDictGetInt(paramsDict, "sizeX");
    _start = pyDictGetInt(paramsDict, "start");
    _stride = pyDictGetInt(paramsDict, "stride");
    _outputsX = pyDictGetInt(paramsDict, "outputsX");
    _pool = pyDictGetString(paramsDict, "pool");
}

PoolLayer& PoolLayer::make(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) {
    std::string _pool = pyDictGetString(paramsDict, "pool");
    if (_pool == "max") {
        return *new MaxPoolLayer(convNetThread, paramsDict, replicaID, false);
    } else if(_pool == "maxabs") {
        return *new MaxPoolLayer(convNetThread, paramsDict, replicaID, true);
    } else if(_pool == "avg") {
        return *new AvgPoolLayer(convNetThread, paramsDict, replicaID);
    }
    throw std::string("Unknown pooling layer type ") + _pool;
}

/*
 * =====================
 * AvgPoolLayer
 * =====================
 */
AvgPoolLayer::AvgPoolLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : PoolLayer(convNetThread, paramsDict, replicaID, false) {
    _sum = pyDictGetInt(paramsDict, "sum");
}

void AvgPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    if (_sum) {
        convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride, _outputsX, AvgPooler<true>());
    } else {
        convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride, _outputsX, AvgPooler<false>());
    }
}

void AvgPoolLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalAvgUndo(v, _prev[replicaIdx][0]->getActsGrad(), _sizeX, _start, _stride, _outputsX, _imgSize, _sum, scaleTargets, 1);
}

/*
 * =====================
 * MaxPoolLayer
 * =====================
 */
MaxPoolLayer::MaxPoolLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool abs) : PoolLayer(convNetThread, paramsDict, replicaID, false), _abs(abs) {
}

void MaxPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    if (_abs) {
        convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride, _outputsX, MaxAbsPooler());
    } else {
        convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride, _outputsX, MaxPooler());
    }
}

void MaxPoolLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 0);
    convLocalMaxUndo(*_inputs[0], v, getActs(), _prev[replicaIdx][inpIdx]->getActsGrad(), _sizeX, _start, _stride, _outputsX, scaleTargets, 1);
}

/*
 * =====================
 * CrossMapPoolLayer
 * =====================
 */
CrossMapPoolLayer::CrossMapPoolLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool trans)
    : Layer(convNetThread, paramsDict, replicaID, trans), TwoDLayerInterface(paramsDict) {
    _size = pyDictGetInt(paramsDict, "size");
    _start = pyDictGetInt(paramsDict, "start");
    _stride = pyDictGetInt(paramsDict, "stride");
    _outputs = pyDictGetInt(paramsDict, "outputChannels");
    _pool = pyDictGetString(paramsDict, "pool");
}

CrossMapPoolLayer& CrossMapPoolLayer::make(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) {
    std::string _pool = pyDictGetString(paramsDict, "pool");
    if (_pool == "max") {
        return *new CrossMapMaxPoolLayer(convNetThread, paramsDict, replicaID);
    }
    throw std::string("Unknown pooling layer type ") + _pool;
}

/*
 * =====================
 * CrossMapMaxPoolLayer
 * =====================
 */
CrossMapMaxPoolLayer::CrossMapMaxPoolLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : CrossMapPoolLayer(convNetThread, paramsDict, replicaID, false) {
}

void CrossMapMaxPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    convPoolCrossMap(*_inputs[0], getActs(), _start, _size, _outputs, _stride, _imgSize, MaxPooler());
}

void CrossMapMaxPoolLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 0);
    convCrossMapMaxPoolUndo(*_inputs[0], v, getActs(), _prev[replicaIdx][0]->getActsGrad(), _imgSize, _start, _size, _stride, scaleTargets, 1);
}

/*
 * =====================
 * RandomScaleLayer
 * =====================
 */
RandomScaleLayer::RandomScaleLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : Layer(convNetThread, paramsDict, replicaID, false), TwoDLayerInterface(paramsDict) {
    _maxScale = pyDictGetFloat(paramsDict, "maxScale");
    _tgtSize = pyDictGetInt(paramsDict, "tgtSize");
    // The smallest size the image could be after rescaling
    _minScaledSize = _imgSize / _maxScale;
   
    // The number of discrete scales we're considering
    int numScales = _imgSize - _minScaledSize + 1;
   
    // The total number of squares of size _tgtSize that we can extract
    // from all these scales
    double numCrops = numScales * (numScales + 1) * (2 * numScales + 1) / 6;
   
    // For each scale, record the fraction of the squares that it has.
    // This will be the probability of sampling this scale.
    _scaleProbs.push_back(1.0 / numCrops);
    for (int s = 1; s < numScales; ++s) {
        _scaleProbs.push_back(_scaleProbs[s-1] + (s + 1) * (s + 1) / numCrops);
    }
}

void RandomScaleLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    if (IS_TRAIN(passType)) {
        // _maxScale is in the range [1, 2)
        float r = randf;
        int rescaledSize = _tgtSize;
        float scaleFactor = _maxScale;
        // Find which scale we have sampled
        for (int s = 0; s < _scaleProbs.size(); ++s) {
            if (r <= _scaleProbs[s]) {
                rescaledSize += s;
                float scaleFactorEnd = _imgSize / float(rescaledSize);
                float scaleFactorStart = max(1.0, _imgSize / (1.0 + rescaledSize));
                scaleFactor = scaleFactorStart + randf * (scaleFactorEnd - scaleFactorStart);
                break;
            }
        }
        assert(rescaledSize >= _tgtSize);
        int maxStart = rescaledSize - _tgtSize;
        int startY = rand() % (1 + maxStart), startX = rand() % (1 + maxStart);
        if (rescaledSize  == _imgSize) {
            convCrop(*_inputs[0], getActs(), rescaledSize, _tgtSize, startY, startX);
        } else {
            convResizeBilinear(*_inputs[0], _rescaledActs, _imgSize, rescaledSize, scaleFactor);
            convCrop(_rescaledActs, getActs(), rescaledSize, _tgtSize, startY, startX);
        }
        _rescaledActs.truncate(); // this'll have a different size each time so may as well truncate it.
    } else if (IS_MULTIVIEW_TEST(passType)) { // for now...
        _inputs[0]->copy(getActs());
    } else if (IS_TEST(passType)) { // Test on center patch
        convResizeBilinear(*_inputs[0], getActs(), _imgSize, _tgtSize, _maxScale);
    }
}

void RandomScaleLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/*
 * =====================
 * CropLayer
 * =====================
 */
CropLayer::CropLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : Layer(convNetThread, paramsDict, replicaID, false), TwoDLayerInterface(paramsDict) {
    _startX = pyDictGetInt(paramsDict, "startX");
    _startY = pyDictGetInt(paramsDict, "startY");
    _tgtSize = pyDictGetInt(paramsDict, "sizeX");
}

void CropLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    convCrop(*_inputs[0], getActs(), _imgSize, _tgtSize, _startY, _startX);
}

void CropLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/*
 * =====================
 * NailbedLayer
 * =====================
 */
NailbedLayer::NailbedLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : Layer(convNetThread, paramsDict, replicaID, false), TwoDLayerInterface(paramsDict) {
    _start = pyDictGetInt(paramsDict, "start");
    _stride = pyDictGetInt(paramsDict, "stride");
    _outputsX = pyDictGetInt(paramsDict, "outputsX");
}

void NailbedLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    convBedOfNails(*_inputs[0], getActs(), _channels, _imgSize, _start, _stride, 0, 1);
}

void NailbedLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convBedOfNailsUndo(v, _prev[replicaIdx][0]->getActsGrad(), _channels, _imgSize, _start, _stride, scaleTargets, 1);
}

/*
 * =====================
 * GaussianBlurLayer
 * =====================
 */
GaussianBlurLayer::GaussianBlurLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : Layer(convNetThread, paramsDict, replicaID, false), TwoDLayerInterface(paramsDict) {
    _hFilter = pyDictGetMatrix(paramsDict, "filter");
}

GaussianBlurLayer::~GaussianBlurLayer() {
    delete _hFilter;
}

void GaussianBlurLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    convGaussianBlur(*_inputs[0], _filter, getActs(), true, _channels, 0, 1);
    convGaussianBlur(getActs(), _filter, getActs(), false, _channels, 0, 1);
}

void GaussianBlurLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& tgt = _prev[replicaIdx][0]->getNumComputedActsGrads(getDeviceID()) > 0 ? _actGradsTmp : _prev[replicaIdx][0]->getActsGrad();
    convGaussianBlur(v, _filter, tgt, true, _channels, 0, 1);
    convGaussianBlur(tgt, _filter, _prev[replicaIdx][0]->getActsGrad(), false, _channels, scaleTargets, 1);
}

void GaussianBlurLayer::copyToGPU() {
    _filter.copyFromHost(*_hFilter, true);
}

 /*
 * =====================
 * HorizontalReflectionLayer
 * =====================
 */
HorizontalReflectionLayer::HorizontalReflectionLayer(ConvNetThread* convNet, PyObject* paramsDict, int replicaID) : Layer(convNet, paramsDict, replicaID, false), TwoDLayerInterface(paramsDict) {
    assert(_channels >= 1 && _channels <= 3);
}

void HorizontalReflectionLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    convReflectHorizontal(*_inputs[0], getActs(), _imgSize);
}

void HorizontalReflectionLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convReflectHorizontal(v, _prev[replicaIdx][0]->getActsGrad(), _imgSize);
}

/*
 * =====================
 * ResizeLayer
 * =====================
 */
ResizeLayer::ResizeLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : Layer(convNetThread, paramsDict, replicaID, false), TwoDLayerInterface(paramsDict) {
    _tgtSize = pyDictGetInt(paramsDict, "tgtSize");
    _scale = pyDictGetFloat(paramsDict, "scale");
}

void ResizeLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    convResizeBilinear(*_inputs[0], getActs(), _imgSize, _tgtSize, _scale);
}

// Can't do this
void ResizeLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/*
 * =====================
 * RGBToYUVLayer
 * =====================
 */
RGBToYUVLayer::RGBToYUVLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : Layer(convNetThread, paramsDict, replicaID, false) {
}

void RGBToYUVLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    convRGBToYUV(*_inputs[0], getActs());
}

// Can't do this
void RGBToYUVLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/*
 * =====================
 * RGBToLABLayer
 * =====================
 */
RGBToLABLayer::RGBToLABLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : Layer(convNetThread, paramsDict, replicaID, false) {
    _center = pyDictGetInt(paramsDict, "center");
}

void RGBToLABLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    convRGBToLAB(*_inputs[0], getActs(), _center);
}

// Can't do this
void RGBToLABLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/*
 * =====================
 * ResponseNormLayer
 * =====================
 */
ResponseNormLayer::ResponseNormLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID)
: Layer(convNetThread, paramsDict, replicaID, false), TwoDLayerInterface(paramsDict) {
    _size = pyDictGetInt(paramsDict, "size");
    _scale = pyDictGetFloat(paramsDict, "scale");
    _pow = pyDictGetFloat(paramsDict, "pow");
    _minDiv = pyDictGetFloat(paramsDict, "minDiv");
}

void ResponseNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    convResponseNorm(*_inputs[0], _denoms, getActs(), _channels, _size, _scale, _pow, _minDiv);
}

void ResponseNormLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNormUndo(v, _denoms, *_inputs[0], getActs(), _prev[replicaIdx][0]->getActsGrad(), _channels, _size, _scale, _pow, scaleTargets, 1);
}

void ResponseNormLayer::truncBwdActs() {
    Layer::truncBwdActs();
    _denoms.truncate();
}

/*
 * =====================
 * CrossMapResponseNormLayer
 * =====================
 */
CrossMapResponseNormLayer::CrossMapResponseNormLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID)
: ResponseNormLayer(convNetThread, paramsDict, replicaID) {
    _blocked = pyDictGetInt(paramsDict, "blocked");
}

void CrossMapResponseNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    assert(inpIdx == 0);
    convResponseNormCrossMap(*_inputs[0], getActs(), _channels, _size, _scale, _pow, _minDiv, _blocked);
}

void CrossMapResponseNormLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNormCrossMapUndo(v, *_inputs[0], getActs(), _prev[replicaIdx][0]->getActsGrad(), _channels, _size, _scale, _pow, _minDiv, _blocked, scaleTargets, 1);
}

/*
 * =====================
 * ContrastNormLayer
 * =====================
 */
ContrastNormLayer::ContrastNormLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : ResponseNormLayer(convNetThread, paramsDict, replicaID) {
}

void ContrastNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    NVMatrix& images = *_inputs[0];
    convLocalPool(images, _meanDiffs, _channels, _size, -_size/2, 1, _imgSize, AvgPooler<false>());
    _meanDiffs.add(images, -1, 1);
    convContrastNorm(images, _meanDiffs, _denoms, getActs(), _channels, _size, _scale, _pow, _minDiv);
}

void ContrastNormLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convContrastNormUndo(v, _denoms, _meanDiffs, getActs(), _prev[replicaIdx][inpIdx]->getActsGrad(), _channels, _size, _scale, _pow, scaleTargets, 1);
}

void ContrastNormLayer::truncBwdActs() {
    ResponseNormLayer::truncBwdActs();
    _meanDiffs.truncate();
}

/*
 * =====================
 * CostLayer
 * =====================
 */
CostLayer::CostLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool trans)
    : Layer(convNetThread, paramsDict, replicaID, trans) {
    _coeff = pyDictGetFloat(paramsDict, "coeff");
    _numCases = 0;
    _aggregated = pyDictGetInt(paramsDict, "aggregated") != 0;
}

float CostLayer::getCoeff() {
    return _coeff;
}

void CostLayer::bprop(NVMatrix& v, PASS_TYPE passType, int passIdx) {
    if (_coeff != 0) {
        Layer::bprop(v, passType, passIdx);
    }
}

bool CostLayer::fprop(PASS_TYPE passType, int passIdx) {
    if (Layer::fprop(passType, passIdx)) {
        syncStream();
        getConvNet().getMessageQueue().enqueue(new Message(FPROP_TERMINAL));
        return true;
    }
    return false;
}

void CostLayer::fpropCommon(PASS_TYPE passType) {
    _numCases = Layer::getNumCases(*_inputs[0]);
}

int CostLayer::getNumCases() {
    return _numCases;
}

bool CostLayer::isGradProducer() {
    return _coeff != 0;
}

doublev& CostLayer::getCost() {
    return *new doublev(_costv);
}

// This is called between microbatches
void CostLayer::resetPassIdx() {
    Layer::resetPassIdx();
    _costv.clear();
}

CostLayer& CostLayer::make(ConvNetThread* convNetThread, PyObject* paramsDict, std::string& type, int replicaID) {
    if (type == "cost.crossent") {
        return *new CrossEntCostLayer(convNetThread, paramsDict, replicaID);
    } else if (type == "cost.bce") {
        return *new BinomialCrossEntropyCostLayer(convNetThread, paramsDict, replicaID);
    } else if (type == "cost.dce") {
        return *new DetectionCrossEntropyCostLayer(convNetThread, paramsDict, replicaID);
    } else if (type == "cost.logreg") {
        return *new LogregCostLayer(convNetThread, paramsDict, replicaID);
    } else if (type == "cost.sum2") {
        return *new SumOfSquaresCostLayer(convNetThread, paramsDict, replicaID);
    }
    throw std::string("Unknown cost layer type ") + type;
}

/*
 * =====================
 * CrossEntCostLayer
 * =====================
 */
CrossEntCostLayer::CrossEntCostLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : CostLayer(convNetThread, paramsDict, replicaID, false) {
}

void CrossEntCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& probs = *_inputs[1];
        int numCases = labels.getLeadingDim();
        computeCrossEntCost(labels, probs, _trueLabelLogProbs, _correctProbs);
        _costv.clear();
        _costv.push_back(-_trueLabelLogProbs.sum());
        _costv.push_back(numCases - _correctProbs.sum());
    }
}

void CrossEntCostLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 1);
    LayerV& prev = _prev[replicaIdx];
    NVMatrix& labels = *_inputs[0];
    NVMatrix& probs = *_inputs[1];
    NVMatrix& target = prev[1]->getActsGrad();
    // Numerical stability optimization: if the layer below me is a softmax layer, let it handle
    // the entire gradient computation to avoid multiplying and dividing by a near-zero quantity.
    bool doWork = prev[1]->getNext().size() > 1 || prev[1]->getType() != "softmax" || prev[1]->getDeviceID() != getDeviceID();
    if (doWork) {
        computeCrossEntGrad(labels, probs, target, scaleTargets == 1, _coeff);
    }
}

/*
 * =====================
 * BinomialCrossEntropyCostLayer
 * =====================
 */
BinomialCrossEntropyCostLayer::BinomialCrossEntropyCostLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : CostLayer(convNetThread, paramsDict, replicaID, false) {
    _computeSoftmaxErrorRate = pyDictGetInt(paramsDict, "computeSoftmaxErrorRate");
    _posWeight = pyDictGetFloat(paramsDict, "posWeight");
}

void BinomialCrossEntropyCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& probs = *_inputs[1];
        int numCases = labels.getLeadingDim();
        labels.applyBinary(BinomialCrossEntOperator(_posWeight), probs, _tmpProbs);
        _costv.clear();
        // Cross-entropy cost
        _costv.push_back(-_tmpProbs.sum(_tmpbuf));// / labels.getFollowingDim());

        // If aggregated, we don't produce these outputs because they're not additive.
        // They have no meaning if this is just a partial cost.
        if (!_aggregated) {
            // "Correct" classifications. To compute these we threshold probs
            // and just count the number of entries that agree with labels.
            probs.biggerThanScalar(0.5, _tmpProbs);
            _tmpProbs.equals(labels);
            _costv.push_back((_tmpProbs.getNumElements() - _tmpProbs.sum(_tmpbuf)) / double(labels.getFollowingDim()));

            if (_computeSoftmaxErrorRate) {
                // Also compute top-1 error as if this is softmax and there's only one correct class
                probs.max(0, _tmpVec);
                assert(_tmpVec.getNumElements() == numCases); // Make sure we did max on correct axis
                probs.equalsVector(_tmpVec, _correctProbs);
                _correctProbs.sum(0, _tmpVec); // Divide by the # of labels that we predict as being present
                float m = _tmpVec.max();

                _correctProbs.eltwiseDivideByVector(_tmpVec);
                _correctProbs.eltwiseMult(labels);

                _costv.push_back(numCases - _correctProbs.sum(_tmpbuf));
            }
        }
    }
}

void BinomialCrossEntropyCostLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 1);
    LayerV& prev = _prev[replicaIdx];
    NVMatrix& labels = *_inputs[0];
    NVMatrix& probs = *_inputs[1];
    NVMatrix& target = prev[1]->getActsGrad();
    // Numerical stability optimization: if the layer below me is a logistic neuron layer, let it handle
    // the entire gradient computation to avoid multiplying and dividing by a near-zero quantity.
    bool doWork =   prev[1]->getNext().size() > 1
                    || prev[1]->getType() != "neuron"
                    || static_cast<NeuronLayer*>(prev[1])->getNeuronType() != "logistic"
                    ||  prev[1]->getDeviceID() != getDeviceID()
                    || prev[1]->getNumReplicas() != getNumReplicas();
    if (doWork) {
        printf("Computing cross-entropy gradient the stupid way\n");
        if (scaleTargets == 0) {
            labels.applyBinary(BinomialCrossEntGradientOperator(_coeff, _posWeight), probs, target);
        } else {
            labels.applyTernary(AddGradientBinaryOperator<BinomialCrossEntGradientOperator>(BinomialCrossEntGradientOperator(_coeff, _posWeight)), probs, target, target);
        }
    }
}

float BinomialCrossEntropyCostLayer::getPosWeight() {
    return _posWeight;
}
/*
 * =====================
 * DetectionCrossEntropyCostLayer
 * =====================
 */
DetectionCrossEntropyCostLayer::DetectionCrossEntropyCostLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID)
    : BinomialCrossEntropyCostLayer(convNetThread, paramsDict, replicaID) {
    assert(!_aggregated);
}

void DetectionCrossEntropyCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    BinomialCrossEntropyCostLayer::fpropActs(inpIdx, scaleTargets, passType, passIdx);
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& probs = *_inputs[1];
        int numCases = labels.getLeadingDim();

        /*
         * Add information sufficient to compute precision and recall for each class.
         */
        // NOTE: _tmpProbs contains ((probs > 0.5) == labels)
        labels.sum(1, _numPositive);      // sum(labels, 1)

        _tmpProbs.eltwiseMult(labels); // labels * ((probs > 0.5) == labels)
        _tmpProbs.sum(1, _numTruePositive);

        probs.biggerThanScalar(0.5, _tmpProbs);
        _tmpProbs.sum(1, _numDeclaredPositive);

        _numDeclaredPositive.copyToHost(_hNumDeclaredPositive, true);
        _numPositive.copyToHost(_hNumPositive, true);
        _numTruePositive.copyToHost(_hNumTruePositive, true);

        for (int i = 0; i < labels.getFollowingDim(); ++i) {
            _costv.push_back(_hNumDeclaredPositive(i, 0));                  // 2
            _costv.push_back(_hNumPositive(i, 0));                          // 3
            _costv.push_back(_hNumTruePositive(i, 0));                      // 4
        }

    }
}

/*
 * =====================
 * LogregCostLayer
 * =====================
 */
LogregCostLayer::LogregCostLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : CostLayer(convNetThread, paramsDict, replicaID, false) {
    _topk = pyDictGetInt(paramsDict, "topk");
//    _numAccumed = 0;
}

void LogregCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix* probs = _inputs[1];

        _doCompute = !IS_MULTIVIEW_TEST(passType);
        if (!_doCompute) {
            if (IS_MULTIVIEW_TEST_START(passType)) {
                if (_probsAccum.count(passIdx) == 0) {
                    _probsAccum[passIdx] = new NVMatrix(*probs);
                }
                probs->copy(*_probsAccum[passIdx]);
                _numAccumed[passIdx] = 1;
            } else {
                _probsAccum[passIdx]->add(*probs);
                _numAccumed[passIdx] += 1;
            }
            if (IS_MULTIVIEW_TEST_END(passType)) {
                probs = _probsAccum[passIdx];
                probs->scale(1.0 / _numAccumed[passIdx]);
                _doCompute = true;
            }
        }
        if (_doCompute) {
            int numCases = labels.getNumElements();
            probs->max(0,_maxProbs);
            if (_topk == 1) {
                computeLogregCost(labels, *probs, _maxProbs, _trueLabelLogProbs, _correctProbs);
            } else {
                computeMultiSoftmaxCost(labels, *probs, _maxProbs, _trueLabelLogProbs, _correctProbs, _topkProbs, _topk);
            }
            _costv.clear();
            double top1 = _correctProbs.sum(_tmpbuf);

            _costv.push_back(-_trueLabelLogProbs.sum(_tmpbuf));
            _costv.push_back(numCases - top1);
            _costv.push_back(numCases - (_topk == 1 ? top1 : _topkProbs.sum(_tmpbuf)));

        }
    }
}

NVMatrix& LogregCostLayer::getProbsAccum(int replicaIdx) {
    return *_probsAccum[replicaIdx];
}

void LogregCostLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (inpIdx == 1) {
        LayerV& prev = _prev[replicaIdx];
        NVMatrix& labels = *_inputs[0];
        NVMatrix& probs = *_inputs[1];
        NVMatrix& target = prev[1]->getActsGrad();
        // Numerical stability optimization: if the layer below me is a softmax layer, let it handle
        // the entire gradient computation to avoid multiplying and dividing by a near-zero quantity.
        bool doWork = prev[1]->getNext().size() > 1 || prev[1]->getType() != "softmax"
                    || prev[1]->getDeviceID() != getDeviceID() || prev[1]->getNumReplicas() != getNumReplicas();
        if (prev[1]->getType() == "softmax") {
            static_cast<SoftmaxLayer*>(prev[1])->setDoUpperGrad(!doWork);
        }
        if (doWork) {
            computeLogregGrad(labels, probs, target, scaleTargets == 1, _coeff);
        }
    }
}

/*
 * =====================
 * SumOfSquaresCostLayer
 * =====================
 */
SumOfSquaresCostLayer::SumOfSquaresCostLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID) : CostLayer(convNetThread, paramsDict, replicaID, false) {
}

void SumOfSquaresCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) {
    _inputs[0]->apply(NVMatrixOps::Square(), _tmp);
    _costv.clear();
    _costv.push_back(_tmp.sum());
}

void SumOfSquaresCostLayer::bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _prev[replicaIdx][inpIdx]->getActsGrad().add(*_inputs[0], scaleTargets, -2 * _coeff);
}

