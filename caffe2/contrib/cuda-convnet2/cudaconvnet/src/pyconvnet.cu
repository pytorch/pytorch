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

#include <Python.h>
#include <arrayobject.h>
#include <assert.h>
#include <helper_cuda.h>
#include <cublas.h>
#include <time.h>
#include <vector>
#include <execinfo.h>
#include <signal.h>

#include "../../util/include/matrix.h"
#include "../../util/include/queue.h"
#include "../include/worker.cuh"
#include "../include/util.cuh"
#include "../include/cost.cuh"

#include "../include/pyconvnet.cuh"
#include "../include/convnet.cuh"

#include "../include/jpeg.h"

using namespace std;
static ConvNet* model = NULL;

static PyMethodDef _ConvNetMethods[] = {{ "initModel",          initModel,              METH_VARARGS },
                                        { "startBatch",         startBatch,             METH_VARARGS },
                                        { "finishBatch",        finishBatch,            METH_VARARGS },
                                        { "checkGradients",     checkGradients,         METH_VARARGS },
                                        { "startMultiviewTest", startMultiviewTest,     METH_VARARGS },
                                        { "startFeatureWriter", startFeatureWriter,     METH_VARARGS },
                                        { "startDataGrad",      startDataGrad,          METH_VARARGS },
                                        { "syncWithHost",       syncWithHost,           METH_VARARGS },
                                        { "decodeJpeg",         decodeJpeg,             METH_VARARGS },
                                        { NULL, NULL }
};

void init_ConvNet() {
    (void) Py_InitModule("_ConvNet", _ConvNetMethods);
    import_array();
}

void signalHandler(int sig) {
    const size_t max_trace_size = 40;
    void *array[max_trace_size];
    size_t trace_size = backtrace(array, max_trace_size);
    fprintf(stderr, "Error signal %d:\n", sig);
    backtrace_symbols_fd(array, trace_size, STDERR_FILENO);
    exit(1);
}

PyObject* initModel(PyObject *self, PyObject *args) {
    assert(model == NULL);
    signal(SIGSEGV, signalHandler);
    signal(SIGABRT, signalHandler);

    PyDictObject* pyLayerParams;
    PyListObject* pyDeviceIDs;
    int pyMinibatchSize;
    int conserveMem;

    if (!PyArg_ParseTuple(args, "O!O!ii",
                          &PyDict_Type, &pyLayerParams,
                          &PyList_Type, &pyDeviceIDs,
                          &pyMinibatchSize,
                          &conserveMem)) {
        return NULL;
    }
    intv& deviceIDs = *getIntV((PyObject*)pyDeviceIDs);

    model = new ConvNet((PyObject*)pyLayerParams,
                        deviceIDs,
                        pyMinibatchSize,
                        conserveMem);

    model->start();
    return Py_BuildValue("i", 0);
}

/*
 * Starts training/testing on the given batch (asynchronous -- returns immediately).
 */
PyObject* startBatch(PyObject *self, PyObject *args) {
    assert(model != NULL);
//    printf("starting next batch\n");
    PyListObject* data;
    double progress;
    int test = 0;
    if (!PyArg_ParseTuple(args, "O!d|i",
        &PyList_Type, &data,
        &progress,
        &test)) {
        return NULL;
    }
    CPUData* cpuData = new CPUData((PyObject*)data);
    
    TrainingWorker* wr = new TrainingWorker(*model, *cpuData, progress, test);
    model->getWorkerQueue().enqueue(wr);
    return Py_BuildValue("i", 0);
}

/*
 * Starts testing on the given batch (asynchronous -- returns immediately).
 */
PyObject* startMultiviewTest(PyObject *self, PyObject *args) {
    assert(model != NULL);
    PyListObject* data;
    int numViews;
    PyArrayObject* pyProbs = NULL;
    char* logregName = NULL;
    if (!PyArg_ParseTuple(args, "O!i|O!s",
        &PyList_Type, &data,
        &numViews,
        &PyArray_Type, &pyProbs,
        &logregName)) {
        return NULL;
    }
    CPUData* cpuData = new CPUData((PyObject*)data);
    MultiviewTestWorker* wr = pyProbs == NULL ? new MultiviewTestWorker(*model, *cpuData, numViews)
                                              : new MultiviewTestWorker(*model, *cpuData, numViews, *new Matrix(pyProbs), logregName);
    model->getWorkerQueue().enqueue(wr);
    return Py_BuildValue("i", 0);
}

PyObject* startFeatureWriter(PyObject *self, PyObject *args) {
    assert(model != NULL);
    PyListObject* data;
    PyListObject* pyFtrs;
    PyListObject* pyLayerNames;
    if (!PyArg_ParseTuple(args, "O!O!O!",
        &PyList_Type, &data,
        &PyList_Type, &pyFtrs,
        &PyList_Type, &pyLayerNames)) {
        return NULL;
    }
    stringv* layerNames = getStringV((PyObject*)pyLayerNames);
    CPUData* cpuData = new CPUData((PyObject*)data);
    MatrixV* ftrs = getMatrixV((PyObject*)pyFtrs);
    
    FeatureWorker* wr = new FeatureWorker(*model, *cpuData, *ftrs, *layerNames);
    model->getWorkerQueue().enqueue(wr);
    return Py_BuildValue("i", 0);
}

PyObject* startDataGrad(PyObject *self, PyObject *args) {
//    assert(model != NULL);
//    PyListObject* data;
//    int dataLayerIdx, softmaxLayerIdx;
//    if (!PyArg_ParseTuple(args, "O!ii",
//        &PyList_Type, &data,
//        &dataLayerIdx, &softmaxLayerIdx)) {
//        return NULL;
//    }
//    CPUData* cpuData = new CPUData((PyObject*)data);
//    Matrix& ftrs = *mvec.back();
//    mvec.pop_back();
//    
//    DataGradWorker* wr = new DataGradWorker(*model, *cpuData, ftrs, dataLayerIdx, softmaxLayerIdx);
//    model->getWorkerQueue().enqueue(wr);
    return Py_BuildValue("i", 0);
}

/*
 * Waits for the trainer to finish training on the batch given to startBatch.
 * This is a blocking call so lets release the GIL.
 */
PyObject* finishBatch(PyObject *self, PyObject *args) {
    assert(model != NULL);
    WorkResult* res = model->getResultQueue().dequeue();
    assert(res != NULL);
    assert(res->getResultType() == WorkResult::BATCH_DONE);
    
    Cost& cost = res->getResults();
    PyObject* dict = PyDict_New();
    CostMap& costMap = cost.getCostMap();
    for (CostMap::const_iterator it = costMap.begin(); it != costMap.end(); ++it) {
        PyObject* v = PyList_New(0);
        for (vector<double>::const_iterator iv = it->second->begin(); iv != it->second->end(); ++iv) {
            PyObject* f = PyFloat_FromDouble(*iv);
            PyList_Append(v, f);
        }
        PyDict_SetItemString(dict, it->first.c_str(), v);
    }
    PyObject* retVal = Py_BuildValue("Ni", dict, cost.getNumCases());
    delete res; // Deletes cost too
    
    return retVal;
}

PyObject* checkGradients(PyObject *self, PyObject *args) {
    assert(model != NULL);
    PyListObject* data;
    if (!PyArg_ParseTuple(args, "O!",
        &PyList_Type, &data)) {
        return NULL;
    }
    CPUData* cpuData = new CPUData((PyObject*)data);
    
    GradCheckWorker* wr = new GradCheckWorker(*model, *cpuData);
    model->getWorkerQueue().enqueue(wr);
    WorkResult* res = model->getResultQueue().dequeue();
    assert(res != NULL);
    assert(res->getResultType() == WorkResult::BATCH_DONE);
    delete res;
    return Py_BuildValue("i", 0);
}

/*
 * Copies weight matrices from GPU to system memory.
 */
PyObject* syncWithHost(PyObject *self, PyObject *args) {
    assert(model != NULL);
    SyncWorker* wr = new SyncWorker(*model);
    model->getWorkerQueue().enqueue(wr);
    WorkResult* res = model->getResultQueue().dequeue();
    assert(res != NULL);
    assert(res->getResultType() == WorkResult::SYNC_DONE);
    
    delete res;
    return Py_BuildValue("i", 0);
}

PyObject* decodeJpeg(PyObject *self, PyObject *args) {
    PyListObject* pyJpegStrings;
    PyArrayObject* pyTarget;
    int img_size, inner_size, test, multiview;
    if (!PyArg_ParseTuple(args, "O!O!iiii",
        &PyList_Type, &pyJpegStrings,
        &PyArray_Type, &pyTarget,
        &img_size,
        &inner_size,
        &test,
        &multiview)) {
        return NULL;
    }

    Thread* threads[NUM_JPEG_DECODER_THREADS];
    int num_imgs = PyList_GET_SIZE(pyJpegStrings);
    int num_imgs_per_thread = DIVUP(num_imgs, NUM_JPEG_DECODER_THREADS);
    Matrix& dstMatrix = *new Matrix(pyTarget);
    for (int t = 0; t < NUM_JPEG_DECODER_THREADS; ++t) {
        int start_img = t * num_imgs_per_thread;
        int end_img = min(num_imgs, (t+1) * num_imgs_per_thread);

        threads[t] = new DecoderThread((PyObject*)pyJpegStrings, dstMatrix, start_img, end_img, img_size, inner_size, test, multiview);
        threads[t]->start();
    }

    for (int t = 0; t < NUM_JPEG_DECODER_THREADS; ++t) {
        threads[t]->join();
        delete threads[t];
    }
    assert(dstMatrix.isView());
    delete &dstMatrix;
    return Py_BuildValue("i", 0);
}
