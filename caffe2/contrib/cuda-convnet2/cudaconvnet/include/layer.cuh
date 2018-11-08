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

#ifndef LAYER_CUH
#define    LAYER_CUH

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <assert.h>
#include <helper_timer.h>
#include "../../nvmatrix/include/nvmatrix.cuh"
//#include "experimental/akrizhevsky/g3/mactruck-gpu-tests/gpu_util.cuh"

#include "weights.cuh"
#include "convnet.cuh"
#include "cost.cuh"
#include "neuron.cuh"
#include "data.cuh"
#include "layer_kernels.cuh"
#include "streambroadcast.cuh"
#include "actbroadcaster.cuh"
#include "gradreducer.cuh"
#include "util.cuh"
#include "timer.cuh"
#include "memorysource.cuh"

class Cost;
class ConvNet;
class ConvNetThread;
class CostLayer;
class DataLayer;
class Layer;
class ActBroadcaster;
class BroadcastMessage;
class IActGradReducer;
class Weights;
class WeightList;
typedef std::vector<Layer*> LayerV;

class BinomialCrossEntOperator {
protected:
    float _posWeight;
public:
    BinomialCrossEntOperator(float posWeight) : _posWeight(posWeight) {
    }
    __device__ inline float operator()(const float t, const float y) const {
        return _posWeight * t * safelog(y) + (1.0f - t) * safelog(1.0f - y);
    }
};

class CrossEntOperator {
protected:
    float _posWeight;
public:
    CrossEntOperator(float posWeight) : _posWeight(posWeight) {
    }
    __device__ inline float operator()(const float t, const float y) const {
        return _posWeight * t * safelog(y);
    }
};

/*
 * Abstract layer.
 */
class Layer {
protected:
    ConvNetThread* _convNetThread;

    // This is a vector[#layers_next]
    std::vector<Layer*> _next;
    // This is a vector[#replicas_prev][#layers_prev]
    std::map<int, std::vector<Layer*> > _prev;

    int _rcvdFInputMsgs;
    std::map<int, int> _numComputedActsGrads;
    int _rcvdBInputMsgs;
    int _numOutputs;
    std::map<int, NVMatrix*> _inputs;                // input idx -> matrix
    std::map<int, MemoryView*> _memSrcActs;        // device id -> memory source
    std::map<int, MemoryView*> _memSrcActsGrad;    // device id -> memory source

    bool _gradConsumer, _foundGradConsumers, _trans;
    std::map<int,bool> _bwdTerminal; // One bool per pass
    int _numGradProducersNext;
    int _actsTarget, _actsGradTarget;
    std::string _name, _type;
    intv _nextDeviceIDs, _prevDeviceIDs;
    HostNVMatrix _hostMemFwd;

    // New replica-related stuff:
    std::map<int,Layer*> _replicas; // NOTE: a layer is its own sibling, too
    // Previous layers sorted by device ID, in reverse order in which they are procesed by
    // sequential grad reducer. map from replica -> device id -> layers
    std::map<int,std::map<int,std::set<Layer*> > > _prevByDevice;
    std::map<std::string, int> _inputIndices;
    int _replicaID;
    int _numReplicas;
    int _numReplicasPrev, _numReplicasNext;

    Queue<int> _broadcastFinishQueue;
    Queue<int> _reductionFinishQueue;
    ActBroadcaster* _actBroadcaster;
    IActGradReducer* _gradReducer;
    Timer _timer;
    bool _initialized;

    virtual void fpropNext(PASS_TYPE passType, int passIdx);
    virtual void truncBwdActs(); 
    virtual void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx) = 0;
    
    virtual void bpropCommon(NVMatrix& v, int replicaIdx, PASS_TYPE passType) {
        // Do nothing by default
    }
    virtual void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType) {
        assert(!isGradProducer()); // Only do nothing if not grad producer
    }
    virtual void fpropCommon(PASS_TYPE passType) {

    }
    void bpropActsCall(NVMatrix& v, PASS_TYPE passType, int replicaIdx, int inputIdx);

    ActBroadcaster& getActBroadcaster();
    IActGradReducer& getGradReducer();
    int getInputIdx(std::string& parentName);
    void setInputIdx(std::string& parentName, int idx);

public:
    static bool _saveActsGrad, _saveActs;
    
    Layer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool trans);
    virtual ~Layer();
    
    virtual bool fprop(PASS_TYPE passType, int passIdx);
    void fprop(NVMatrix& v, int inpIdx, PASS_TYPE passType, int passIdx);
    virtual void fprop(std::map<int,NVMatrix*>& v, PASS_TYPE passType, int passIdx);
    virtual void bprop(PASS_TYPE passType, int passIdx);
    virtual void bprop(NVMatrix& v, PASS_TYPE passType, int passIdx);
    virtual void reset();
    virtual void resetPassIdx();
    int getNumCases(NVMatrix& v);
    int& getNumComputedActsGrads(int deviceID);
    int incRcvdBInputMsgs();
    bool isGradConsumer();
    bool hasGradProducerNext(std::string& layerName);
    // Does this layer produce a gradient for any layer?
    virtual bool isGradProducer();
    // Does this layer produce a gradient for layer of given name?
    virtual bool isGradProducer(std::string& layerName);
    std::string& getName();
    std::string& getType();
    virtual void addNext(Layer& l);
    virtual void addPrev(Layer& l, int replicaIdx);
    virtual void addReplica(Layer& l);
    std::map<int,std::vector<Layer*> >& getPrev();
    std::vector<Layer*>& getNext();
    virtual NVMatrix& getActs();
    virtual NVMatrix& getActs(int deviceID);
    virtual NVMatrix& getActs(int deviceID, int numCases);
    virtual NVMatrix& getActsGrad();
    virtual NVMatrix& getActsGrad(int deviceID);
    virtual std::map<int,NVMatrix*> getAllActs();
    virtual std::map<int, NVMatrix*> getAllActsGrads();
    virtual bool postInit();
    int getDeviceID();
    ConvNetThread& getConvNetThread();
    cudaStream_t getStream();
    void syncStream();
    void setBwdTerminal(int passIdx);
    // Do nothing if this layer has no weights
    virtual bool updateWeights() {
        return false;
    }
    virtual bool constrainWeights() {
        return false;
    }
    virtual void checkGradient() {
    }
    virtual void copyToCPU() {
    }
    virtual void copyToGPU()  {
    }
    intv& getNextDeviceIDs() {
        return _nextDeviceIDs;
    }

    int getReplicaID();
    int getNumReplicas();
    int getNumSiblingReplicas();
    int getNumReplicasPrev();
    int getNumReplicasNext();
    int getNumOutputs();
    void setMemorySourceActs(int deviceID, MemoryView& mem);
    void setMemorySourceActsGrad(int deviceID, MemoryView& mem);
    MemoryView& getMemorySourceActs(int deviceID);
    MemoryView& getMemorySourceActsGrad(int deviceID);
    int getFwdActiveInputReplicaIdx(int passIdx);
    int getBwdActiveInputReplicaIdx(int passIdx);
    int getFwdActiveReplicaIdx(int passIdx);
    int getNumLayersPrev();
    virtual int getNumInputReplicas();
    int getNumExpectedBwdMsgs();
    int getNumExpectedFwdMsgs();
    int getReplicaIdx();
    int getActivePassPeriod();
    int getNumGradProducersNext();
    virtual ConvNet& getConvNet();
};

class TwoDLayerInterface {
protected:
    int _channels, _imgSize, _imgPixels;
public:
    TwoDLayerInterface(PyObject* paramsDict);
};

class NeuronLayer : public Layer {
protected:
    Neuron* _neuron;
    std::string _neuronType;
    
    virtual void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    virtual void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
    virtual bool bpropSpecial(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    class CrossEntLogisticGradientOperator {
    private:
        float _coeff, _posWeight;
    public:
        CrossEntLogisticGradientOperator(float coeff, float posWeight) : _coeff(coeff), _posWeight(posWeight) {
        }
        __device__ inline float operator()(const float y, const float t) const {
            return _coeff * (_posWeight * t * (1.0f - y) + (t - 1.0f) * y);
        }
    };
    NeuronLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
    ~NeuronLayer();
    std::string& getNeuronType();
};

class WeightLayer : public Layer {
protected:
    WeightList* _weights;
    Weights *_biases;
    NVMatrix _norm2;
    float _wStep, _bStep;
    int _weightUpdatePassPeriod;
    void fpropCommon(PASS_TYPE passType);
    void bpropCommon(NVMatrix& v, int replicaIdx, PASS_TYPE passType);
    virtual void bpropBiases(NVMatrix& v, PASS_TYPE passType) = 0;
    virtual void bpropWeights(NVMatrix& v, int replicaIdx, int inpIdx, PASS_TYPE passType) = 0;
    virtual void _constrainWeights();
    virtual float getGradScale(int inpIdx, PASS_TYPE passType);
    virtual float getIncScale(int inpIdx, PASS_TYPE passType);
    virtual float getBGradScale(PASS_TYPE passType);
    virtual float getBIncScale();
    virtual NVMatrix& getGradTarget(int inpIdx);
    NVMatrix& getWeightMatrix(PASS_TYPE passType, int inpIdx);
    NVMatrix& getBiasMatrix(PASS_TYPE passType);
public:
    WeightLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool trans, bool useGrad);
    virtual ~WeightLayer();
    virtual bool updateWeights();
    virtual bool constrainWeights();
    virtual void copyToCPU();
    virtual void copyToGPU();
    virtual void checkGradient();
    Weights& getWeights(int idx);
    void addReplica(Layer& l);
    virtual bool postInit();
};

class FCLayer : public WeightLayer {
protected:
    virtual void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    virtual void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
    virtual void bpropBiases(NVMatrix& v, PASS_TYPE passType);
    virtual void bpropWeights(NVMatrix& v, int replicaIdx, int inpIdx, PASS_TYPE passType);
    virtual void _constrainWeights();
public:
    FCLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool useGrad);
    FCLayer();
};

class SplitFCLayer : public FCLayer {
protected:
    int _numParts;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
//    void bpropBiases(NVMatrix& v, PASS_TYPE passType);
    void bpropWeights(NVMatrix& v, int replicaIdx, int inpIdx, PASS_TYPE passType);
    void splitWeights();
public:
    SplitFCLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool useGrad);
};

class SoftmaxLayer : public Layer {
protected:
    bool _doUpperGrad;
    NVMatrix _max, _sum;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    SoftmaxLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
    void setDoUpperGrad(bool b);
};

class ConcatenationLayer : public Layer {
protected:
    intv* _copyOffsets;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    ConcatenationLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
    virtual ~ConcatenationLayer();
};

class PassThroughLayer : public Layer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    PassThroughLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
    virtual bool postInit();
};

class EltwiseSumLayer : public Layer {
protected:
    floatv* _coeffs;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    EltwiseSumLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
    ~EltwiseSumLayer();
};

class EltwiseMaxLayer : public Layer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    EltwiseMaxLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
};

class SumLayer : public Layer {
protected:
    int _stride;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    SumLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
};

class DataCopyMessage {
public:
    enum MESSAGE_TYPE {
        COPY,
        EXIT
    };
protected:
    CPUData* _cpuData;
    int _passIdx;
    bool _other;
    DataCopyMessage::MESSAGE_TYPE _type;
    DataCopyMessage(DataCopyMessage::MESSAGE_TYPE type) : _cpuData(NULL), _other(false), _passIdx(0), _type(type) {
    }
public:
    DataCopyMessage(CPUData& cpuData, bool other, int passIdx) : _cpuData(&cpuData), _other(other), _passIdx(passIdx), _type(DataCopyMessage::COPY) {
    }
    
    CPUData& getData() const {
        return *_cpuData;
    }
    
    int getPassIdx() const {
        return _passIdx;
    }
    
    bool isOther() const {
        return _other;
    }

    DataCopyMessage::MESSAGE_TYPE getType() {
        return _type;
    }
};

class DataCopyExitMessage : public DataCopyMessage {
public:
    DataCopyExitMessage() : DataCopyMessage(DataCopyMessage::EXIT) {
    }
};

class DataCopyThread;

class DataLayer : public Layer {
protected:
    bool _useBuffer;
    int _dataIdx;
    ConvNet* _convNet;
//    std::map<int, NVMatrix*> _outputs2; // Buffer for copying data during computation
    std::map<int, MemoryView*> _memSrcActs2;        // // Buffer for copying data during computation
    std::map<int, cudaStream_t> _copyStreams;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    Queue<int> _copyFinishQueue;
    DataCopyThread* _copier;
    bool _outstandingCopyRequest;
    int _start, _end;
    
public:
    void fprop(PASS_TYPE passType, int passIdx, bool fromBuffer);
    DataLayer(ConvNet* convNet, PyObject* paramsDict, int replicaID);
    ~DataLayer();
    NVMatrix& getActs(int deviceID);
//    NVMatrix& getActs(int deviceID, bool other);
    NVMatrix& getActs(int deviceID, bool other, int numCases);
    bool isGradProducer();
    void toggleBuffer(int passIdx);
    void copyData(CPUData& data, bool other, int passIdx);
    bool postInit();
    ConvNet& getConvNet();
    int getNumInputReplicas();
    cudaStream_t getCopyStream(int deviceID);
    Queue<int>& getCopyFinishQueue() {
        return _copyFinishQueue;
    }
    void waitForCopyFinish();
    int getDataIdx() const {
        return _dataIdx;
    }
    int getStart() const {
        return _start;
    }
    int getEnd() const {
        return _end;
    }
};


class DataCopyThread : public Thread {
protected:
    DataLayer* _parent;
    Queue<DataCopyMessage*> _queue;
    HostNVMatrix _hostMemFwd;
    Timer _requestTimer;
    int _sleepUsec;
    virtual void* run();
    
public:
    DataCopyThread(DataLayer& parent, intv& cpus);
    Queue<DataCopyMessage*>& getQueue();
    void stop();
};


class LocalLayer : public WeightLayer {
protected:
    intv* _padding, *_stride, *_filterSize, *_channels, *_imgSize, *_groups;
    intv* _imgPixels, *_filterPixels, *_filterChannels;
    int _modulesX, _modules, _numFilters;
    
public:
    LocalLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool useGrad);
    virtual ~LocalLayer();
};

class ConvLayer : public LocalLayer {
protected:
    int _sumWidth;
    bool _sharedBiases;
    floatv* _weightContrastNormMin, *_weightContrastNormMax;
    NVMatrix _weightGradTmp;

    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropBiases(NVMatrix& v, PASS_TYPE passType);
    void bpropWeights(NVMatrix& v, int replicaIdx, int inpIdx, PASS_TYPE passType);
    void truncBwdActs();
    void _constrainWeights();

public:
    ConvLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
    virtual ~ConvLayer();
}; 

class LocalUnsharedLayer : public LocalLayer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropBiases(NVMatrix& v, PASS_TYPE passType);
    void bpropWeights(NVMatrix& v, int replicaIdx, int inpIdx, PASS_TYPE passType);
    void _constrainWeights();
public:
    LocalUnsharedLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
}; 

class PoolLayer : public Layer, public TwoDLayerInterface {
protected:
    int _sizeX, _start, _stride, _outputsX;
    std::string _pool;
public:
    PoolLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool trans);
    
    static PoolLayer& make(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
}; 

class AvgPoolLayer : public PoolLayer {
protected:
    bool _sum;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    AvgPoolLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
}; 

class MaxPoolLayer : public PoolLayer {
protected:
    bool _abs;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    MaxPoolLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool abs);
};

class CrossMapPoolLayer : public Layer, public TwoDLayerInterface {
protected:
    int _size, _start, _stride, _outputs;
    std::string _pool;
public:
    CrossMapPoolLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool trans);

    static CrossMapPoolLayer& make(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
};

class CrossMapMaxPoolLayer : public CrossMapPoolLayer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    CrossMapMaxPoolLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
};

class RandomScaleLayer : public Layer, public TwoDLayerInterface {
protected:
    int _tgtSize, _minScaledSize;
    float _maxScale; // should be >= 1
    NVMatrix _rescaledActs;
    std::vector<double> _scaleProbs;
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
    
    RandomScaleLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
};

class CropLayer : public Layer, public TwoDLayerInterface {
protected:
    int _tgtSize, _startX, _startY;
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);

    CropLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
};

class NailbedLayer : public Layer, public TwoDLayerInterface {
protected:
    int _start, _stride, _outputsX;
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
    
    NailbedLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
};

class GaussianBlurLayer : public Layer, public TwoDLayerInterface {
protected:
    Matrix* _hFilter;
    NVMatrix _filter;
    NVMatrix _actGradsTmp;
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void copyToGPU();
    
    GaussianBlurLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
    ~GaussianBlurLayer();
};

class HorizontalReflectionLayer : public Layer, public TwoDLayerInterface {
protected:
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
    
    HorizontalReflectionLayer(ConvNetThread* convNet, PyObject* paramsDict, int replicaID);
};

class ResizeLayer : public Layer, public TwoDLayerInterface {
protected:
    float _scale;
    int _tgtSize;
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);

    ResizeLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
};

class DropoutLayer : public Layer {
protected:
    bool _enable;
    float _keep;
    NVMatrix _keepMask;
public:
    virtual void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    virtual void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void truncBwdActs();
    DropoutLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
    class DropoutSmallerThanOperator {
    private:
        float _keep, _scale;
    public:
        DropoutSmallerThanOperator(float keep) : _keep(keep), _scale(1.0f/keep) {
        }
        __device__ inline float operator()(const float x) const {
            return (x < _keep) * _scale;
        }
    };
};

class Dropout2Layer : public DropoutLayer {
protected:
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
    Dropout2Layer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
};

class RGBToYUVLayer : public Layer {
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);

    RGBToYUVLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
};

class RGBToLABLayer : public Layer {
protected:
    bool _center;
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);

    RGBToLABLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
};

class ResponseNormLayer : public Layer, public TwoDLayerInterface {
protected:
    int _size;
    float _scale, _pow;
    float _minDiv;
    NVMatrix _denoms;

    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void truncBwdActs();
public:
    ResponseNormLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
}; 

class CrossMapResponseNormLayer : public ResponseNormLayer {
protected:
    bool _blocked;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    CrossMapResponseNormLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
}; 

class ContrastNormLayer : public ResponseNormLayer {
protected:
    NVMatrix _meanDiffs;
    
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void truncBwdActs();
public:
    ContrastNormLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
};

class CostLayer : public Layer {
protected:
    float _coeff;
    doublev _costv;
    NVMatrix _tmpbuf; // For error accumulation
    int _numCases; // number of cases that the values in _costv were computed on
    bool _aggregated;
    void fpropCommon(PASS_TYPE passType);
public:
    CostLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID, bool trans);
    void bprop(NVMatrix& v, PASS_TYPE passType, int passIdx);
    bool fprop(PASS_TYPE passType, int passIdx);
    
    int getNumCases();
    virtual doublev& getCost();
    float getCoeff();
    bool isGradProducer();
    void setSendTerminalMessages(bool send);
    void resetPassIdx();
    
    static CostLayer& make(ConvNetThread* convNetThread, PyObject* paramsDict, std::string& type, int replicaID);
};

/*
 * Input 0: labels
 * Input 1: softmax outputs
 */
class CrossEntCostLayer : public CostLayer {
protected:
    NVMatrix _trueLabelLogProbs, _correctProbs;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    CrossEntCostLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
};

/*
 * Input 0: labels
 * Input 1: softmax outputs
 */
class LogregCostLayer : public CostLayer {
protected:
    NVMatrix _trueLabelLogProbs, _correctProbs, _topkProbs;
    std::map<int,NVMatrix*> _probsAccum; // input replica idx -> nvmatrix
    NVMatrix _maxProbs;
    std::map<int,int> _numAccumed; // input replica idx -> int
    int _topk;
    bool _doCompute;
    virtual void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    LogregCostLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
    NVMatrix& getProbsAccum(int replicaIdx);
};

/*
 * Input 0: labels
 * Input 1: logistic outputs
 */
class BinomialCrossEntropyCostLayer : public CostLayer {
protected:
    bool _computeSoftmaxErrorRate;
    NVMatrix _tmpProbs, _tmpVec, _correctProbs;
    float _posWeight;
    virtual void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    BinomialCrossEntropyCostLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
    float getPosWeight();

    // Only for use with non-logistic units
    class BinomialCrossEntGradientOperator {
    private:
        float _coeff, _posWeight;
    public:
        BinomialCrossEntGradientOperator(float coeff, float posWeight) : _coeff(coeff), _posWeight(posWeight) {
        }
        __device__ inline float operator()(const float t, const float y) const {
            return _coeff * (_posWeight * __fdividef(t, y) + __fdividef(t - 1.0f, 1.0f - y));
        }
    };
};

/*
 * Input 0: labels
 * Input 1: logistic outputs
 */
class DetectionCrossEntropyCostLayer : public BinomialCrossEntropyCostLayer {
protected:
    Matrix _hNumPositive, _hNumTruePositive, _hNumDeclaredPositive;
    NVMatrix _numPositive, _numTrueNegative, _numTruePositive, _numDeclaredPositive;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
public:
    DetectionCrossEntropyCostLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
};

class SumOfSquaresCostLayer : public CostLayer {
protected:
    NVMatrix _tmp;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType, int passIdx);
    void bpropActs(NVMatrix& v, int replicaIdx, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    SumOfSquaresCostLayer(ConvNetThread* convNetThread, PyObject* paramsDict, int replicaID);
};

#endif    /* LAYER_CUH */

