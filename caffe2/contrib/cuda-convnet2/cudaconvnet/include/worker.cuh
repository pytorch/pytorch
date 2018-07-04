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

#ifndef WORKER_CUH
#define WORKER_CUH

#include "convnet.cuh"
#include "cost.cuh"
#include "data.cuh"

class ConvNet;
class Cost;

class WorkResult {
public:
    enum RESULTS {BATCH_DONE, SYNC_DONE};
protected:
    WorkResult::RESULTS _resultType;
    Cost* _results;
public:
    WorkResult(WorkResult::RESULTS resultType, Cost& results);
    WorkResult(WorkResult::RESULTS resultType);
    virtual ~WorkResult();
    Cost& getResults() const;
    WorkResult::RESULTS getResultType() const;
};

class Worker {
protected:
    ConvNet* _convNet;
public:
    Worker(ConvNet& convNet);
    virtual ~Worker();
    virtual bool run() = 0;
};

class DataWorker : public Worker {
protected:
    CPUData* _data;
    DataProvider* _dp;
public:
    DataWorker(ConvNet& convNet, CPUData& data);
    virtual ~DataWorker();
    bool run();
    virtual void _run() = 0;
};

class TrainingWorker : public DataWorker {
protected:
    bool _test;
    double _progress;
public:
    TrainingWorker(ConvNet& convNet, CPUData& data, double progress, bool test);
    void _run();
};

class SyncWorker : public Worker {
public:
    SyncWorker(ConvNet& convNet);
    bool run();
};

class ExitWorker : public Worker {
public:
    ExitWorker(ConvNet& convNet);
    bool run();
};

class GradCheckWorker : public DataWorker {
public:
    GradCheckWorker(ConvNet& convNet, CPUData& data);
    void _run();
};

class MultiviewTestWorker : public DataWorker {
protected:
    int _numViews;
    Matrix* _cpuProbs;
    std::string _logregName;
    CPUData& getMinibatch(int v, int i);
public:
    MultiviewTestWorker(ConvNet& convNet, CPUData& data, int numViews, Matrix& cpuProbs, const char* softmaxName);
    MultiviewTestWorker(ConvNet& convNet, CPUData& data, int numViews);
    ~MultiviewTestWorker();
    void _run();
};

class FeatureWorker : public DataWorker {
protected:
    MatrixV *_ftrs;
    stringv *_layerNames;
    bool _deleteFeatures;
public:
    FeatureWorker(ConvNet& convNet, CPUData& data, MatrixV& ftrs, stringv& layerNames, bool deleteFeatures=true);
    ~FeatureWorker();
    void _run();
};

class DataGradWorker : public DataWorker {
protected:
    Matrix* _dataGrads;
    int _dataLayerIdx, _softmaxLayerIdx;
public:
    DataGradWorker(ConvNet& convNet, CPUData& data, Matrix& dataGrads, int dataLayerIdx, int softmaxLayerIdx);
    ~DataGradWorker();
    void _run();
};

#endif/* WORKER_CUH */

