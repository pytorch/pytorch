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

#include <set>
#include <vector>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <typeinfo>
#include <map>
#include <cuda.h>
#include <signal.h>
#include "../include/nvmatrix.cuh"
#include "../include/nvmatrix_operators.cuh"

using namespace std;

/*
 * Device random number generator pointers.
 */
//map<int,curandGenerator_t> NVMatrix::rndGen;
map<int,MemorySegment*> NVMatrix::_rndDevStates;
map<int,int> NVMatrix::_rndDevThreads;
pthread_mutex_t* NVMatrix::_rndMutex = makeMutex();
pthread_mutex_t* NVMatrix::_cublasMutex = makeMutex();
pthread_mutex_t* NVMatrix::_streamMutex = makeMutex();
std::map<int,cublasHandle_t> NVMatrix::_cublasHandles;
std::map<int,cudaStream_t> NVMatrix::_defaultStreams;

pthread_mutex_t* NVMatrix::makeMutex() {
    pthread_mutex_t* m = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(m, NULL);
    return m;
}
/*
   Do not call resize in _init because resize is a virtual function
   which is overridden in base class. Since C++ is retarded and unable
   to call overridden functions from constructors, we shall call resize
   separately from every constructor after calling _init.
*/
void NVMatrix::_init(bool isTrans) {
    _numRows = 0;
    _numCols = 0;
    _numElements = 0;
    _ownsData = true;

    _isTrans = isTrans;
    _memSegment = NULL;

    _stride = 0;
    _texObj = 0;
}

NVMatrix::NVMatrix() : _deleted(false) {
    _init(false);
}

NVMatrix::NVMatrix(bool isTrans) : _deleted(false) {
    _init(isTrans);
}

NVMatrix::NVMatrix(int numRows, int numCols, bool isTrans) : _deleted(false) {
    _init(isTrans);
    resize(numRows, numCols);
}

NVMatrix::NVMatrix(const Matrix& like, bool copy) : _deleted(false) {
    _init(like.isTrans());
    resize(like.getNumRows(), like.getNumCols());
    if (copy) {
        copyFromHost(like);
    }
}

NVMatrix::NVMatrix(const NVMatrix& like, bool copy) : _deleted(false) {
    _init(like.isTrans());
    resize(like.getNumRows(), like.getNumCols());
    if (copy) {
        like.copy(*this);
    }
}

/*
 * Initializes NVMatrix with same dimensions as given matrix but
 * does not copy any data.
 */
NVMatrix::NVMatrix(const NVMatrix& like) : _deleted(false) {
    _init(like.isTrans());
    resize(like.getNumRows(), like.getNumCols());
}

/*
 * Initializes NVMatrix with same dimensions as given matrix but
 * does not copy any data.
 */
NVMatrix::NVMatrix(const Matrix& like) : _deleted(false) {
    _init(false);
    resize(like.getNumRows(), like.getNumCols());
}

NVMatrix::NVMatrix(MemorySegment* mem, int numRows, int numCols, int stride, bool isTrans) :
    _numRows(numRows),
    _numCols(numCols),
    _numElements(numRows*numCols),
    _ownsData(false),
    _memSegment(mem),
    _isTrans(isTrans),
    _deleted(false),
    _texObj(0) {
    _stride = stride < 0 ? getLeadingDim() : stride;
}

NVMatrix::~NVMatrix() {
    if (!_deleted) {
        deallocTexture();
        if(_ownsData && _numElements > 0) {
            dealloc();
        } else {
            // dealloc deletes the mem segment. But if this is a view,
            // then we still need to delete the mem segment object.
//            assert(_memSegment == NULL || _memSegment->getSize() == 0);
            delete _memSegment;
        }
    }
}

void NVMatrix::copyFromHost(const Matrix& hostMatrix) {
    copyFromHost(hostMatrix, false, getDefaultStream());
}

void NVMatrix::copyFromHost(const Matrix& hostMatrix, bool resizeTarget) {
    copyFromHost(hostMatrix, resizeTarget, getDefaultStream());
}

void NVMatrix::copyFromHost(const Matrix& hostMatrix, bool resizeTarget, cudaStream_t stream) {
    if (resizeTarget) {
        resize(hostMatrix);
    } else {
        assert(isSameDims(hostMatrix));
    }
    setTrans(hostMatrix.isTrans());

    if (getNumElements() > 0) {
        CUBLAS_CALL(cublasSetMatrixAsync(hostMatrix.getLeadingDim(), hostMatrix.getFollowingDim(), sizeof(float),
                                    hostMatrix.getData(), hostMatrix.getLeadingDim(), getDevData(), _stride, stream));
        syncStream(stream);
    }
}

void NVMatrix::copyToHost(Matrix& hostMatrix) const {
    copyToHost(hostMatrix, false, getDefaultStream());
}

void NVMatrix::copyToHost(Matrix& hostMatrix, bool resizeTarget) const {
    copyToHost(hostMatrix, resizeTarget, getDefaultStream());
}

void NVMatrix::copyToHost(Matrix& hostMatrix, bool resizeTarget, cudaStream_t stream) const {
    if (resizeTarget) {
        hostMatrix.resize(_numRows, _numCols);
    } else {
        assert(isSameDims(hostMatrix));
    }
    hostMatrix.setTrans(_isTrans);

    if (getNumElements() > 0) {
        CUBLAS_CALL(cublasGetMatrixAsync(getLeadingDim(),getFollowingDim(), sizeof(float),
                                         getDevData(), getStride(), hostMatrix.getData(), hostMatrix.getLeadingDim(), stream));
        syncStream(stream);
    }
}

void NVMatrix::copy(NVMatrix& dest) const {
    copy(dest, getDefaultStream());
}

void NVMatrix::copy(NVMatrix& dest, cudaStream_t stream) const {
    if (&dest != this) {
        if (!isSameDims(dest)) {
            dest.resize(*this);
        }
        copy(dest, 0, -1, 0, -1, 0, 0, stream);
    }
}

NVMatrix& NVMatrix::copy() const {
    NVMatrix& c = construct();
    copy(c);
    return c;
}

void NVMatrix::rightMult(NVMatrix &b, float scaleAB, NVMatrix &target) {
    rightMult(b, scaleAB, target, getDefaultStream());
}

void NVMatrix::rightMult(NVMatrix &b, float scaleAB, NVMatrix &target, cudaStream_t stream) {
//    if(&target != this && &target != &b) {
//        target.resize(_numRows, b.getNumCols());
//        target.setTrans(true);
//    }
    target.addProduct(*this, b, 0, scaleAB, stream);
}

void NVMatrix::rightMult(NVMatrix &b, float scaleAB) {
    rightMult(b, scaleAB, *this);
}

void NVMatrix::rightMult(NVMatrix &b, NVMatrix& target) {
    rightMult(b, 1, target);
}

void NVMatrix::addProduct(NVMatrix& a, NVMatrix &b, float scaleThis, float scaleAB) {
    addProduct(a, b, scaleThis, scaleAB, getDefaultStream());
}

/*
 * This will only work if this matrix is in column-major order! In other words,
 * if isTrans() returns true.
 */
void NVMatrix::addProduct(NVMatrix& a, NVMatrix &b, float scaleThis, float scaleAB, cudaStream_t stream) {
    assert(a.getNumCols() == b.getNumRows());

    if (scaleThis == 0) {
        resize(a.getNumRows(), b.getNumCols());
        setTrans(true);
    }

    assert(this->getNumRows() == a.getNumRows());
    assert(this->getNumCols() == b.getNumCols());
    assert(_isTrans);
    CUBLAS_CALL(cublasSetStream_v2(getCublasHandle(), stream));
    CUBLAS_CALL(cublasSgemm_v2(getCublasHandle(), a.getTransChar(), b.getTransChar(), a.getNumRows(), b.getNumCols(), a.getNumCols(),
                               &scaleAB, a.getDevData(), a.getStride(), b.getDevData(), b.getStride(),
                               &scaleThis, getDevData(), getStride()));
}

void NVMatrix::addProduct(NVMatrix& a, NVMatrix &b) {
    addProduct(a, b, 1, 1);
}

void NVMatrix::assertSame(NVMatrixV& a) {
    for (int i = 1; i < a.size(); ++i) {
        assert(a[i]->isSameDims(*a[0]));
        assert(a[i]->isTrans() == a[0]->isTrans());
        assert(a[i]->getStride() == a[0]->getStride());
        assert(a[i]->getDataDeviceID() == a[0]->getDataDeviceID());
    }
}

void NVMatrix::batchedMatrixMultiply(NVMatrixV& a, NVMatrixV& b, NVMatrixV& target, float scaleTarget, float scaleAB,
                                     const float** aPtrsDev, const float** bPtrsDev, float** tgtPtrsDev) {
    batchedMatrixMultiply(a, b, target, scaleTarget, scaleAB, getDefaultStream(), aPtrsDev, bPtrsDev, tgtPtrsDev);
}

void NVMatrix::batchedMatrixMultiply(NVMatrixV& a, NVMatrixV& b, NVMatrixV& target, float scaleTarget, float scaleAB) {
    batchedMatrixMultiply(a, b, target, scaleTarget, scaleAB, getDefaultStream());
}

void NVMatrix::batchedMatrixMultiply(NVMatrixV& a, NVMatrixV& b, NVMatrixV& target, float scaleTarget, float scaleAB, cudaStream_t stream,
                                     const float** aPtrsDev, const float** bPtrsDev, float** tgtPtrsDev) {
    assert(a.size() == b.size());
    assert(a.size() == target.size());
    assertSame(a);
    assertSame(b);
    assertSame(target);

    const int batch = a.size();
    if (batch > 0) {
        const int rows = a[0]->getNumRows(), inner = a[0]->getNumCols(), cols = b[0]->getNumCols();

        assert(inner == b[0]->getNumRows());
        assert(target[0]->getNumRows() == rows);
        assert(target[0]->getNumCols() == cols);

        const int lda = a[0]->getStride(), ldb = b[0]->getStride(), ldc = target[0]->getStride();
        cublasOperation_t atrans = a[0]->getTransChar(), btrans = b[0]->getTransChar();

        CUBLAS_CALL(cublasSetStream_v2(getCublasHandle(), stream));
        CUBLAS_CALL(cublasSgemmBatched(getCublasHandle(), atrans, btrans, rows, cols, inner, &scaleAB, aPtrsDev, lda, bPtrsDev, ldb, &scaleTarget, tgtPtrsDev, ldc, batch));
    }
}

void NVMatrix::batchedMatrixMultiply(NVMatrixV& a, NVMatrixV& b, NVMatrixV& target, float scaleTarget, float scaleAB, cudaStream_t stream) {
    assert(a.size() == b.size());
    assert(a.size() == target.size() || target.size() == 0);

    const int batch = a.size();
    if (batch > 0) {
        const int rows = a[0]->getNumRows(), cols = b[0]->getNumCols();

        const float* aPtrs[batch], *bPtrs[batch], *tgtPtrs[batch];
        for (int i = 0; i < batch; ++i) {
            if (target.size() <= i) {
                target.push_back(new NVMatrix(rows, cols, true));
            }
            aPtrs[i] = a[i]->getDevData();
            bPtrs[i] = b[i]->getDevData();
            tgtPtrs[i] = target[i]->getDevData();
        }

//        const float** aPtrsDev, **bPtrsDev;
//        float **tgtPtrsDev;
//        checkCudaErrors(cudaMalloc(&aPtrsDev, batch * sizeof(float*)));
//        checkCudaErrors(cudaMalloc(&bPtrsDev, batch * sizeof(float*)));
//        checkCudaErrors(cudaMalloc(&tgtPtrsDev, batch * sizeof(float*)));
        MemorySegment* aPtrsDev = DEVICE_MEMORY_MANAGER::getInstance(getDeviceID()).malloc(batch * sizeof(float*));
        MemorySegment* bPtrsDev = DEVICE_MEMORY_MANAGER::getInstance(getDeviceID()).malloc(batch * sizeof(float*));
        MemorySegment* tgtPtrsDev = DEVICE_MEMORY_MANAGER::getInstance(getDeviceID()).malloc(batch * sizeof(float*));

        checkCudaErrors(cudaMemcpyAsync(aPtrsDev, aPtrs, batch * sizeof(float*), cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(bPtrsDev, bPtrs, batch * sizeof(float*), cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(tgtPtrsDev, tgtPtrs, batch * sizeof(float*), cudaMemcpyHostToDevice, stream));

        batchedMatrixMultiply(a, b, target, scaleTarget, scaleAB, stream, const_cast<const float**>(aPtrsDev->getData<float*>()),
                                                                          const_cast<const float**>(bPtrsDev->getData<float*>()),
                                                                          tgtPtrsDev->getData<float*>());

//        checkCudaErrors(cudaFree(aPtrsDev));
//        checkCudaErrors(cudaFree(bPtrsDev));
//        checkCudaErrors(cudaFree(tgtPtrsDev));
        DEVICE_MEMORY_MANAGER::getInstance(getDeviceID()).free(aPtrsDev);
        DEVICE_MEMORY_MANAGER::getInstance(getDeviceID()).free(bPtrsDev);
        DEVICE_MEMORY_MANAGER::getInstance(getDeviceID()).free(tgtPtrsDev);
    }
}

template <class Randomizer>
void NVMatrix::_unaryRandomize(NVMatrix& target, Randomizer rnd) {
    _unaryRandomize(target, rnd, getDefaultStream());
}

template <class Randomizer>
void NVMatrix::_unaryRandomize(NVMatrix& target, Randomizer rnd, cudaStream_t stream) {
    assert(isRndInitialized());
    assert(isContiguous() && target.isContiguous());
    if (!isSameDims(target)) {
        target.resize(*this);
    }
    assert(isTrans() == target.isTrans());
    kUnaryRandomize<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK, 0, stream>>>(getDevData(), target.getDevData(), getCurandState(), getNumElements(), rnd);
    getLastCudaError("kUnaryRandomize: Kernel execution failed");
}

template <class Randomizer>
void NVMatrix::_binaryRandomize(NVMatrix& data2, NVMatrix& target, Randomizer rnd) {
    _binaryRandomize(data2, target, rnd, getDefaultStream());
}

template <class Randomizer>
void NVMatrix::_binaryRandomize(NVMatrix& data2, NVMatrix& target, Randomizer rnd, cudaStream_t stream) {
    assert(isRndInitialized());
    assert(isContiguous() && data2.isContiguous() && target.isContiguous());
    assert(isSameDims(data2));
    assert(isTrans() == data2.isTrans());
    if (!isSameDims(target)) {
        target.resize(*this);
    }
    assert(isTrans() == target.isTrans());
    kBinaryRandomize<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK, 0, stream>>>(getDevData(), data2.getDevData(), target.getDevData(), getCurandState(), getNumElements(), rnd);
    getLastCudaError("kBinaryRandomize: Kernel execution failed");
}

void NVMatrix::initRandom(unsigned long long seed, int numStreams) {
    NVMatrix::initRandom(seed, numStreams, NVMatrix::getDefaultStream());
}

void NVMatrix::initRandom(unsigned long long seed, int numStreams, cudaStream_t stream) {
//    printf("init random on device %d\n", getDeviceID());
    pthread_mutex_lock(_rndMutex);
    assert(!isRndInitialized(true));
    int d = getDeviceID();
//    _rndDevStates[d] = NULL;
    _rndDevThreads[d] = numStreams;
    _rndDevStates[d] = DEVICE_MEMORY_MANAGER::getInstance(d).malloc(numStreams * sizeof(curandState));
//    checkCudaErrors(cudaMalloc((void **)&_rndDevStates[d], numStreams * sizeof(curandState)));
    pthread_mutex_unlock(_rndMutex);
    kSetupCurand<<<NUM_RND_BLOCKS, NUM_RND_THREADS_PER_BLOCK, 0, stream>>>(getCurandState(), 1 + seed*2); // so there's no chance it'll be correlated with the other one
    getLastCudaError("kSetupCurand: Kernel execution failed");
}

void NVMatrix::initRandom(unsigned long long seed) {
    initRandom(seed, NUM_RND_STREAMS);
}

void NVMatrix::initRandom() {
    NVMatrix::initRandom(time(0));
}

void NVMatrix::initCublas() {
    int d = getDeviceID();
    pthread_mutex_lock(_cublasMutex);
    assert(_cublasHandles.count(d) == 0);
    CUBLAS_CALL(cublasCreate(&_cublasHandles[d]));
    // It appears that cublasCreate causes a host -> device copy on stream 0,
    // so we synchronize with it because we run everything else on other
    // streams.
    syncDevice();
    pthread_mutex_unlock(_cublasMutex);
}

void NVMatrix::destroyCublas() {
    int d = getDeviceID();
    pthread_mutex_lock(_cublasMutex);
    assert(_cublasHandles.count(d) > 0);
    CUBLAS_CALL(cublasDestroy(_cublasHandles[d]));
    _cublasHandles.erase(d);
    pthread_mutex_unlock(_cublasMutex);
}

cublasHandle_t NVMatrix::getCublasHandle() {
    return getCublasHandle(getDeviceID());
}

cublasHandle_t NVMatrix::getCublasHandle(int deviceID) {
    pthread_mutex_lock(_cublasMutex);
    assert(_cublasHandles.count(deviceID) > 0);
    cublasHandle_t h = _cublasHandles[deviceID];
    pthread_mutex_unlock(_cublasMutex);
    return h;
}

cudaStream_t NVMatrix::getDefaultStream() {
    return getDefaultStream(NVMatrix::getDeviceID());
}

cudaStream_t NVMatrix::getDefaultStream(int deviceID) {
    if (deviceID >= 0) {
        pthread_mutex_lock(_streamMutex);
        if (_defaultStreams.count(deviceID) == 0) {
            int oldDeviceID = getDeviceID();
            NVMatrix::setDeviceID(deviceID);
            checkCudaErrors(cudaStreamCreateWithFlags(&_defaultStreams[deviceID], cudaStreamNonBlocking));
            NVMatrix::setDeviceID(oldDeviceID);
        }
        cudaStream_t s = _defaultStreams[deviceID];
        pthread_mutex_unlock(_streamMutex);
        return s;
    }
    return 0;
}

void NVMatrix::syncDevice() {
    checkCudaErrors(cudaDeviceSynchronize());
}

void NVMatrix::syncStream(cudaStream_t stream) {
    checkCudaErrors(cudaStreamSynchronize(stream));
}

void NVMatrix::syncStream() {
    syncStream(getDefaultStream());
}

curandState* NVMatrix::getCurandState() {
    /*
     * Even though we're only reading from the map here, it's important to grab
     * the mutex because another thread may be writing to it.
     */
    pthread_mutex_lock(_rndMutex);
    int d = getDeviceID();
    assert(isRndInitialized(true));
    curandState* r = _rndDevStates[d]->getData<curandState>();
    pthread_mutex_unlock(_rndMutex);
    return r;
}

curandState* NVMatrix::getCurandState(int numStreams) {
    int d = getDeviceID();
    pthread_mutex_lock(_rndMutex);
    assert(isRndInitialized(true));
    bool realloc = numStreams >  _rndDevThreads[d];
    pthread_mutex_unlock(_rndMutex);

    if (realloc) {
        destroyRandom();
        initRandom(time(0), numStreams);
    }
    return getCurandState();
}

int NVMatrix::getDataDeviceID() const {
    if (getDevData() == NULL) {
        return DEVICE_NULL;
    }
    struct cudaPointerAttributes atts;
    checkCudaErrors(cudaPointerGetAttributes(&atts, getDevData()));
    return atts.memoryType == cudaMemoryTypeDevice ? atts.device : DEVICE_HOST;
}


int NVMatrix::getDeviceID() {
    int d;
    checkCudaErrors(cudaGetDevice(&d));
//    if (d == 0) {
//        raise(SIGABRT);
//    }
    return d;
}

void NVMatrix::setDeviceID(int d) {
    assert(d >= 0);
//    printf("Setting device to %d\n", d);
//    if (d == 0) {
//        raise(SIGABRT);
//    }
    checkCudaErrors(cudaSetDevice(d));
}

bool NVMatrix::canAccessPeer(int srcDevice, int tgtDevice) {
    if (srcDevice == tgtDevice) {
        return true;
    }
    int canAccess;
    checkCudaErrors(cudaDeviceCanAccessPeer(&canAccess, srcDevice, tgtDevice));
    return canAccess;
}

bool NVMatrix::isRndInitialized(bool haveLock) {
    if (!haveLock) {
        pthread_mutex_lock(_rndMutex);
    }
    bool b = _rndDevStates.count(getDeviceID()) != 0;
    if (!haveLock) {
        pthread_mutex_unlock(_rndMutex);
    }
    return b;
}

bool NVMatrix::isRndInitialized() {
    return isRndInitialized(false);
}

void NVMatrix::destroyRandom() {
    int d = getDeviceID();
    pthread_mutex_lock(_rndMutex);
    assert(isRndInitialized(true));
//    checkCudaErrors(cudaFree(_rndDevStates[d]));
    DEVICE_MEMORY_MANAGER::getInstance(d).free(_rndDevStates[d]);
    _rndDevStates.erase(d);
    _rndDevThreads.erase(d);
    pthread_mutex_unlock(_rndMutex);
}

void NVMatrix::binarizeProbs() {
    binarizeProbs(*this);
}

void NVMatrix::binarizeProbs(NVMatrix& target) {
    _unaryRandomize(target, BinarizeUnaryRandomizer());
}

void NVMatrix::randomizeUniform() {
    assert(isContiguous());
    assert(isRndInitialized());
//    CURAND_CALL(curandGenerateUniform(rndGen, _devData, getNumElements()));
    _unaryRandomize(*this, UniformUnaryRandomizer());
}

void NVMatrix::randomizeGaussian() {
    randomizeGaussian(1);
}

void NVMatrix::randomizeGaussian(float stdev) {
    randomizeGaussian(0, stdev);
}

void NVMatrix::randomizeGaussian(float mean, float stdev) {
    assert(isContiguous());
    assert(isRndInitialized());
//    CURAND_CALL(curandGenerateNormal(rndGen, _devData, getNumElements(), mean, stdev));
    _unaryRandomize(*this, GaussianUnaryRandomizer(mean, stdev));
}

/*
 * Kind of a hack since we don't actually need the contents of this matrix for it,
 * so we don't really need a binary randomizer.
 */
void NVMatrix::randomizeGaussian(NVMatrix& stdevs) {
    randomizeGaussian(0, stdevs);
}

void NVMatrix::randomizeGaussian(float mean, NVMatrix& stdevs) {
    _binaryRandomize(stdevs, *this, GaussianBinaryRandomizer(mean));
}

void NVMatrix::randomizeGaussian(float mean, float stdevMult, NVMatrix& stdevs) {
    _binaryRandomize(stdevs, *this, ScaledGaussianBinaryRandomizer(mean, stdevMult));
}

void NVMatrix::addGaussianNoise() {
    addGaussianNoise(1);
}

void NVMatrix::addGaussianNoise(float stdev) {
    addGaussianNoise(stdev, *this);
}

void NVMatrix::addGaussianNoise(float stdev, NVMatrix& target) {
    _unaryRandomize(target, AddGaussianUnaryRandomizer(stdev));
}

void NVMatrix::addGaussianNoise(NVMatrix& stdevs, bool var) {
    addGaussianNoise(stdevs, var, *this);
}

void NVMatrix::addGaussianNoise(NVMatrix& stdevs) {
    addGaussianNoise(stdevs, false, *this);
}

void NVMatrix::addGaussianNoise(NVMatrix& stdevs, bool var, NVMatrix& target) {
    if (var) {
        _binaryRandomize(stdevs, target, AddGaussianBinaryRandomizer<true>());
    } else {
        _binaryRandomize(stdevs, target, AddGaussianBinaryRandomizer<false>());
    }
}

void NVMatrix::biggerThan(NVMatrix& b, NVMatrix& target) {
    applyBinary(NVMatrixBinaryOps::BiggerThan(), b, target);
}

void NVMatrix::biggerThan(NVMatrix& b) {
    biggerThan(b, *this);
}

void NVMatrix::equals(NVMatrix& b, NVMatrix& target) {
    applyBinary(NVMatrixBinaryOps::Equals(), b, target);
}

void NVMatrix::equals(NVMatrix& m) {
    equals(m, *this);
}

void NVMatrix::biggerThanVector(NVMatrix& vec, NVMatrix& target) {
    applyBinaryV(NVMatrixBinaryOps::BiggerThan(), vec, target);
}

void NVMatrix::biggerThanVector(NVMatrix& vec) {
    biggerThanVector(vec, *this);
}

void NVMatrix::_checkBounds(int startRow, int endRow, int startCol, int endCol) const {
    assert(startRow >= 0 && startRow <= _numRows);
    assert(endRow >= startRow && endRow <= _numRows);

    assert(startCol >= 0 && startCol <= _numCols);
    assert(endCol >= startCol && endCol <= _numCols);
}

/*
 * The only place where stride is supported for now!
 * Will ALWAYS return a view of the original data, sometimes non-contiguous.
 */
NVMatrix& NVMatrix::slice(int startRow, int endRow, int startCol, int endCol) const {
    endRow = endRow < 0 ? this->_numRows : endRow;
    endCol = endCol < 0 ? this->_numCols : endCol;
    _checkBounds(startRow, endRow, startCol, endCol);

    if (!isTrans()) {
        return construct(new MemorySegment(this->getDevData() + startRow * _stride + startCol), endRow - startRow, endCol - startCol, _stride, false);
    }
    return construct(new MemorySegment(this->getDevData() + startCol * _stride + startRow), endRow - startRow, endCol - startCol, _stride, true);
}

/* this will NEVER return a view */
void NVMatrix::slice(int startRow, int endRow, int startCol, int endCol, NVMatrix& target) const {
    endRow = endRow < 0 ? this->_numRows : endRow;
    endCol = endCol < 0 ? this->_numCols : endCol;
    _checkBounds(startRow, endRow, startCol, endCol);

    int sliceRows = endRow - startRow, sliceCols = endCol - startCol;
    if (target.getNumRows() != sliceRows || target.getNumCols() != sliceCols) {
        target.resize(sliceRows, sliceCols);
    }
    this->copy(target, startRow, endRow, startCol, endCol, 0, 0);
}

NVMatrix& NVMatrix::sliceRows(int startRow, int endRow) const {
    return slice(startRow, endRow, 0, -1);
}

void NVMatrix::sliceRows(int startRow, int endRow, NVMatrix& target) const {
    slice(startRow, endRow, 0, -1, target);
}

NVMatrix& NVMatrix::sliceCols(int startCol, int endCol) const {
    return slice(0, -1, startCol, endCol);
}

void NVMatrix::sliceCols(int startCol, int endCol, NVMatrix& target) const {
    slice(0, -1, startCol, endCol, target);
}

NVMatrixV& NVMatrix::splitRows(int numParts) {
    assert(getNumRows() % numParts == 0);
    NVMatrixV& v = *new NVMatrixV();
    int partSize = getNumRows() / numParts;
    for (int p = 0; p < numParts; ++p) {
        v.push_back(&sliceRows(p * partSize, (p+1) * partSize));
    }
    return v;
}

NVMatrixV& NVMatrix::splitCols(int numParts) {
    assert(getNumCols() % numParts == 0);
    NVMatrixV& v = *new NVMatrixV();
    int partSize = getNumCols() / numParts;
    for (int p = 0; p < numParts; ++p) {
        v.push_back(&sliceCols(p * partSize, (p+1) * partSize));
    }
    return v;
}

/*
 * Guaranteed to not change the data if the number of elements doesn't change.
 * So you can use this to "reshape" a matrix.
 */
bool NVMatrix::resize(int numRows, int numCols, bool trans) {
    setTrans(trans);
    bool reallocated = false;
    if (numRows != _numRows || numCols != _numCols) {
        assert(_ownsData || (_numElements == numRows * numCols && isContiguous()));
        if (_numElements != numRows * numCols) {
            if (_numElements > 0) { // free old memory
                dealloc();
            }
            if (numRows * numCols > 0) { // allocate new memory
                alloc(numCols * numRows);
            } else {
                _memSegment = NULL;
            }
            reallocated = true;
        }
        _numRows = numRows;
        _numCols = numCols;
        _numElements = numRows * numCols;
        _stride = getLeadingDim();
    }
    return reallocated;
}

bool NVMatrix::resize(int numRows, int numCols) {
    return resize(numRows, numCols, isTrans());
}

bool NVMatrix::resize(const NVMatrix& like) {
    setTrans(like.isTrans());
    return resize(like.getNumRows(), like.getNumCols());
}

bool NVMatrix::resize(const Matrix& like) {
    setTrans(like.isTrans());
    return resize(like.getNumRows(), like.getNumCols());
}

void NVMatrix::reshape(int numRows, int numCols) {
    assert(isContiguous());
    assert(_numElements == numRows*numCols);
    _numRows = numRows;
    _numCols = numCols;
    _stride = getLeadingDim();
}

NVMatrix& NVMatrix::reshaped(int numRows, int numCols) const {
    assert(isContiguous());
    assert(_numElements == numRows*numCols);
    return construct(new MemorySegment(*_memSegment), numRows, numCols, -1, _isTrans);
}

void NVMatrix::copy(NVMatrix &dest, int srcStartRow, int srcEndRow,
                    int srcStartCol, int srcEndCol,
                    int destStartRow, int destStartCol) const {
    copy(dest, srcStartRow, srcEndRow, srcStartCol, srcEndCol, destStartRow, destStartCol, getDefaultStream());
}

void NVMatrix::copy(NVMatrix &dest, int srcStartRow, int srcEndRow,
                    int srcStartCol, int srcEndCol,
                    int destStartRow, int destStartCol, cudaStream_t stream) const {
    srcEndRow = srcEndRow < 0 ? _numRows : srcEndRow;
    srcEndCol = srcEndCol < 0 ? _numCols : srcEndCol;
    NVMatrix* srcSlice = &slice(srcStartRow, srcEndRow, srcStartCol, srcEndCol);
    NVMatrix* destSlice = &dest.slice(destStartRow, destStartRow + srcEndRow - srcStartRow, destStartCol, destStartCol + srcEndCol - srcStartCol);
    if (srcSlice->isContiguous() && destSlice->isContiguous() && srcSlice->isSameDims(*destSlice) && srcSlice->isTrans() == destSlice->isTrans()) {
        // The commonest case.
        checkCudaErrors(cudaMemcpyAsync(destSlice->getDevData(), srcSlice->getDevData(), srcSlice->getNumDataBytes(), cudaMemcpyDefault, stream));
    } else {
        srcSlice->apply(NVMatrixOps::Identity(), *destSlice, stream);
    }
    delete srcSlice;
    delete destSlice;
}


NVMatrix& NVMatrix::getTranspose() {
    return construct(new MemorySegment(*_memSegment), _numCols, _numRows, _stride, !_isTrans);
}

NVMatrix& NVMatrix::getClone() {
    return construct(new MemorySegment(*_memSegment), _numRows, _numCols, _stride, _isTrans);
}

void NVMatrix::transpose(NVMatrix& target) {
    flipTrans(target);
    target.setTrans(!target.isTrans());
    target.reshape(target.getNumCols(), target.getNumRows());
}

void NVMatrix::transpose() {
    int tmp = _numCols;
    _numCols = _numRows;
    _numRows = tmp;
    _isTrans = !_isTrans;
}

bool NVMatrix::transpose(bool trans) {
    bool oldTrans = _isTrans;
    if (oldTrans != trans) {
        transpose();
    }
    return oldTrans;
}

/*
 * Flips the ordering of the matrix from row-major to column-major and vice versa.
 * This creates temporary storage -- not a cheap operation.
 *
 * This is not equivalent to a "hard transpose". The resultant matrix still has
 * the same dimensions, its layout in memory just changes.
 */
NVMatrix& NVMatrix::flipTrans() {
    NVMatrix& meTrans = construct(*this);
    flipTrans(meTrans);
    return meTrans;
}

void NVMatrix::flipTrans(NVMatrix& target) {
    flipTrans(target, getDefaultStream());
}

void NVMatrix::flipTrans(NVMatrix& target, cudaStream_t stream) {
    assert(&target != this);
    target.resize(_numRows, _numCols);
    target.setTrans(!isTrans());
//    target.printShape("target");
//    this->printShape("this");
    apply(NVMatrixOps::Identity(), target, stream);
}

void NVMatrix::squaredDiff(NVMatrix& b) {
    squaredDiff(b, *this);
}

void NVMatrix::squaredDiff(NVMatrix& b, NVMatrix& target) {
    applyBinary(NVMatrixBinaryOps::SquaredDiff(), b, target);
}

void NVMatrix::add(NVMatrix& b, float scaleA, float scaleB, NVMatrix& target) {
    add(b, scaleA, scaleB, target, NVMatrix::getDefaultStream());
}

void NVMatrix::add(NVMatrix& b, float scaleA, float scaleB, NVMatrix& target, cudaStream_t stream) {
    if (scaleA == 0) {
        b.scale(scaleB, target, stream);
    } else if (scaleB == 0) {
        scale(scaleA, target, stream);
    } else if (scaleA == 1 && scaleB == 1) { // slight optimization
        applyBinary(NVMatrixBinaryOps::Add(), b, target, stream);
    } else if (scaleA == 1) {
        applyBinary(NVMatrixBinaryOps::WeightedAdd1(scaleB), b, target, stream);
    } else {
        applyBinary(NVMatrixBinaryOps::WeightedAdd(scaleA, scaleB), b, target, stream);
    }
}

void NVMatrix::add(NVMatrix& b, float scaleB, NVMatrix& target) {
    add(b, 1, scaleB, target);
}

void NVMatrix::add(NVMatrix& b, NVMatrix& target) {
    add(b, 1, target);
}

void NVMatrix::add(NVMatrix& b, float scaleB) {
    add(b, scaleB, *this);
}

void NVMatrix::add(NVMatrix& b, float scaleA, float scaleB) {
    add(b, scaleA, scaleB, *this);
}

void NVMatrix::add(NVMatrix& b) {
    add(b, 1, *this);
}

void NVMatrix::subtract(NVMatrix& b, NVMatrix& target) {
    add(b, -1, target);
}

void NVMatrix::subtract(NVMatrix& b) {
    add(b, -1);
}

void NVMatrix::eltwiseMult(NVMatrix& b, NVMatrix& target) {
    applyBinary(NVMatrixBinaryOps::Multiply(), b, target);
}

void NVMatrix::eltwiseMult(NVMatrix& b) {
    eltwiseMult(b, *this);
}

void NVMatrix::eltwiseDivide(NVMatrix& b, NVMatrix& target) {
    applyBinary(NVMatrixBinaryOps::Divide(), b, target);
}

void NVMatrix::eltwiseDivide(NVMatrix& b) {
    eltwiseDivide(b, *this);
}

void NVMatrix::tile(int timesY, int timesX, NVMatrix& target) {
    tile(timesY, timesX, target, getDefaultStream());
}

void NVMatrix::tile(int timesY, int timesX, NVMatrix& target, cudaStream_t stream) {
    assert(isContiguous() && target.isContiguous());
    assert(timesX > 0 && timesY > 0);
    target.resize(_numRows*timesY, _numCols*timesX);
    target.setTrans(_isTrans);
    if(!isTrans()) {
        kTile<<<NUM_TILE_BLOCKS,NUM_TILE_THREADS_PER_BLOCK, 0, stream>>>(getDevData(), target.getDevData(), _numCols, _numRows, target._numCols, target._numRows);
    } else {
        kTile<<<NUM_TILE_BLOCKS,NUM_TILE_THREADS_PER_BLOCK, 0, stream>>>(getDevData(), target.getDevData(), _numRows, _numCols, target._numRows, target._numCols);
    }
    getLastCudaError("Kernel execution failed");
}

void NVMatrix::addVector(NVMatrix& vec, float scaleVec, NVMatrix& target) {
    addVector(vec, scaleVec, target, getDefaultStream());
}

void NVMatrix::addVector(NVMatrix& vec, float scaleVec, NVMatrix& target, cudaStream_t stream) {
    applyBinaryV(NVMatrixBinaryOps::ScaledAdd(scaleVec), vec, target, stream);
}

void NVMatrix::addVector(NVMatrix& vec) {
    addVector(vec, 1);
}

void NVMatrix::addVector(NVMatrix& vec, float scaleVec) {
    addVector(vec, scaleVec, *this);
}

void NVMatrix::addVector(NVMatrix& vec, NVMatrix& target) {
    addVector(vec, 1, target);
}

void NVMatrix::equalsVector(NVMatrix& vec, NVMatrix& target) {
    applyBinaryV(NVMatrixBinaryOps::Equals(), vec, target);
}

void NVMatrix::equalsVector(NVMatrix& vec) {
    equalsVector(vec, *this);
}

void NVMatrix::eltwiseMultByVector(NVMatrix& vec, NVMatrix& target) {
    eltwiseMultByVector(vec, target, getDefaultStream());
}

void NVMatrix::eltwiseMultByVector(NVMatrix& vec, NVMatrix& target, cudaStream_t stream) {
    applyBinaryV(NVMatrixBinaryOps::Multiply(), vec, target, stream);
}

void NVMatrix::eltwiseMultByVector(NVMatrix& vec, cudaStream_t stream) {
    eltwiseMultByVector(vec, *this, stream);
}

void NVMatrix::eltwiseMultByVector(NVMatrix& vec) {
    eltwiseMultByVector(vec, *this);
}

void NVMatrix::eltwiseDivideByVector(NVMatrix& vec) {
    eltwiseDivideByVector(vec,  *this);
}

void NVMatrix::eltwiseDivideByVector(NVMatrix& vec, NVMatrix& target) {
    applyBinaryV(NVMatrixBinaryOps::Divide(), vec, target);
}

template<class Agg, class UnaryOp, class BinaryOp>
void NVMatrix::_aggregate(int axis, NVMatrix& target, Agg agg, UnaryOp uop, BinaryOp bop, cudaStream_t stream) {
    _aggregate(axis, target, agg, uop, bop, stream, NULL);
}

/*
 * TODO: this is a mess, fix it. it works pretty fast but it's too ugly.
 * TODO: this function is _really_ bad for very long aggregations of few columns.
 */
template<class Agg, class UnaryOp, class BinaryOp>
void NVMatrix::_aggregate(int axis, NVMatrix& target, Agg agg, UnaryOp uop, BinaryOp bop, cudaStream_t stream, NVMatrix* tmp) {
    assert(axis == 0 || axis == 1);
    assert(isContiguous()  && target.isContiguous());
    assert(&target != this);
    int width = _isTrans ? _numRows : _numCols;
    int height = _isTrans ? _numCols : _numRows;

    target.setTrans(_isTrans);
    assert(width > 0);
    assert(height > 0);
    if((axis == 0 && !_isTrans) || (axis == 1 && _isTrans)) { //col sum
        target.resize(!_isTrans ? 1 : _numRows, !_isTrans ? _numCols : 1);
//        int height = getFollowingDim();
        if ((height <= 2048 || width >= 4096)) {
            int numBlocks = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
            assert(numBlocks * NUM_SUM_COLS_THREADS_PER_BLOCK >= width);
            assert(numBlocks < NUM_BLOCKS_MAX);
            kDumbAggCols<Agg, UnaryOp, BinaryOp><<<numBlocks,NUM_SUM_COLS_THREADS_PER_BLOCK, 0, stream>>>(getTextureObject(), target.getDevData(), width, height, agg, uop, bop);
            getLastCudaError("kDumbAggCols: Kernel execution failed");
        } else { // Specialize the case when we have very long columns and few of them
            const int sumLength = 128;
            bool deltmp = tmp == NULL;
            if (tmp == NULL) {
                tmp = new NVMatrix(false);
            }

            int numBlocksX = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
            int numBlocksY = DIVUP(height, sumLength);
            tmp->resize(numBlocksY, width);

            dim3 blocks(numBlocksX, numBlocksY);
            dim3 threads(NUM_SUM_COLS_THREADS_PER_BLOCK);
            kAggCols<Agg, UnaryOp><<<blocks,threads, 0, stream>>>(getTextureObject(), tmp->getDevData(), width, height, sumLength, agg, uop);
            getLastCudaError("kAggCols: Kernel execution failed");

            int numBlocks = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
            kDumbAggCols<Agg, NVMatrixOps::Identity, BinaryOp><<<numBlocks,NUM_SUM_COLS_THREADS_PER_BLOCK, 0, stream>>>(tmp->getTextureObject(), target.getDevData(), width, numBlocksY, agg, NVMatrixOps::Identity(), bop);
            getLastCudaError("kDumbAggCols: Kernel execution failed");
            if (deltmp) {
                delete tmp;
            }
        }
    } else { // row sum
        target.resize(_isTrans ? 1 : _numRows, _isTrans ? _numCols : 1);
        if (width > 1) {
            if (height >= 16384) { // linear aggregation
                int numBlocksX = 1;
                int numBlocksY = DIVUP(height, AGG_SHORT_ROWS_THREADS_Y*AGG_SHORT_ROWS_LOOPS_Y);
                int numThreadsX = width <= 4 ? 4 : width <= 8 ? 8 : width <= 12 ? 12 : width <= 16 ? 16 : AGG_SHORT_ROWS_THREADS_X;
                int numThreadsY = AGG_SHORT_ROWS_THREADS_Y;
                while (numBlocksY > NUM_BLOCKS_MAX) {
                    numBlocksY = DIVUP(numBlocksY,2);
                    numBlocksX *= 2;
                }
                dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
                if(width <= 16) {
                    if(width <= 4) {
                        kAggShortRows<Agg, UnaryOp, BinaryOp, 1, 4><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),width, height, agg, uop, bop);
                    } else if(width <= 8) {
                        kAggShortRows<Agg, UnaryOp, BinaryOp, 1, 8><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),width, height, agg, uop, bop);
                    } else if(width <= 12) {
                        kAggShortRows<Agg, UnaryOp, BinaryOp, 1, 12><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),width, height, agg, uop, bop);
                    } else {
                        kAggShortRows<Agg, UnaryOp, BinaryOp, 1, 16><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),width, height, agg, uop, bop);
                    }
                } else if(width <= 32) {
                    kAggShortRows<Agg, UnaryOp, BinaryOp, 2, AGG_SHORT_ROWS_THREADS_X><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),width, height, agg, uop, bop);
                } else if(width <= 48){
                    kAggShortRows<Agg, UnaryOp, BinaryOp, 3, AGG_SHORT_ROWS_THREADS_X><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),width, height, agg, uop, bop);
                } else if(width <= 64){
                    kAggShortRows<Agg, UnaryOp, BinaryOp, 4, AGG_SHORT_ROWS_THREADS_X><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),width, height, agg, uop, bop);
                } else {
                    kAggShortRows2<Agg, UnaryOp, BinaryOp><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),width, height, agg, uop, bop);
                }
            } else {
                if (width >= 512) {
                    // NOTE: this is the only case which I bothered to try to optimize for Kepler
                    dim3 threads(AWR_NUM_THREADS);
                    dim3 blocks(1, height);
                    kAggRows_wholerow_nosync<<<blocks, threads, 0, stream>>>(getDevData(), target.getDevData(), width, height, agg, uop, bop);
                } else {

                    int numThreadsX = width <= 64 ? 32 : (width <= 128 ? 64 : (width <= 256 ? 128 : (width <= 512 ? 256 : 512)));
                    int numThreadsY = 1;
                    int numBlocksX = DIVUP(width, 2*numThreadsX);
                    int numBlocksY = std::min(height, NUM_BLOCKS_MAX);

                    dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
                    assert(numBlocksX <= NUM_BLOCKS_MAX);
                    assert(numBlocksY <= NUM_BLOCKS_MAX);

                    if(width <= 64) {
                        kAggRows<Agg, UnaryOp, BinaryOp, 32><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),
                                                   width, height, target.getLeadingDim(), agg, uop, bop);
                    } else if(width <= 128) {
                        kAggRows<Agg, UnaryOp, BinaryOp, 64><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),
                                                   width, height, target.getLeadingDim(), agg, uop, bop);
                    } else if(width <= 256) {
                        kAggRows<Agg, UnaryOp, BinaryOp, 128><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),
                                                   width, height, target.getLeadingDim(), agg, uop, bop);
                    } else if(width <= 512) {
                        kAggRows<Agg, UnaryOp, BinaryOp, 256><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),
                                                   width, height, target.getLeadingDim(), agg, uop, bop);
                    } else {
                        kAggRows<Agg, UnaryOp, BinaryOp, 512><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),
                                                   width, height, target.getLeadingDim(), agg, uop, bop);
                    }

                    getLastCudaError("agg rows: Kernel execution failed");
                }
            }
        } else {
            target.applyBinary(NVMatrixBinaryOps::CompositeSecond<UnaryOp, BinaryOp>(uop, bop), *this, target, stream);
//            copy(target, stream);
        }
    }
}

template<class Agg, class UnaryOp, class BinaryOp>
void NVMatrix::_aggregate(int axis, NVMatrix& target, Agg agg, UnaryOp uop, BinaryOp bop) {
    _aggregate(axis, target, agg, uop, bop, getDefaultStream());
}

template<class Agg, class BinaryOp>
void NVMatrix::_aggregate(int axis, NVMatrix& target, Agg agg, BinaryOp bop) {
    _aggregate(axis, target, agg, NVMatrixOps::Identity(), bop, getDefaultStream());
}

template<class Agg, class BinaryOp>
void NVMatrix::_aggregate(int axis, NVMatrix& target, Agg agg, BinaryOp bop, cudaStream_t stream) {
    _aggregate(axis, target, agg, NVMatrixOps::Identity(), bop, stream);
}

template<class Agg, class UnaryOp, class BinaryOp>
NVMatrix& NVMatrix::_aggregate(int axis, Agg agg, UnaryOp uop, BinaryOp bop) {
    NVMatrix &sumVec = construct();
    _aggregate(axis, sumVec, agg, uop, bop);
    return sumVec;
}

template<class Agg, class UnaryOp, class BinaryOp>
NVMatrix& NVMatrix::_aggregate(int axis, Agg agg, UnaryOp uop, BinaryOp bop, cudaStream_t stream) {
    NVMatrix &sumVec = construct();
    _aggregate(axis, sumVec, agg, uop, bop, stream);
    return sumVec;
}

template<class Agg, class BinaryOp>
NVMatrix& NVMatrix::_aggregate(int axis, Agg agg, BinaryOp bop) {
    return _aggregate(axis, agg, NVMatrixOps::Identity(), bop);
}

template<class Agg, class BinaryOp>
NVMatrix& NVMatrix::_aggregate(int axis, Agg agg, BinaryOp bop, cudaStream_t stream) {
    return _aggregate(axis, agg, NVMatrixOps::Identity(), bop, stream);
}



template<class Agg, class UnaryOp, class BinaryOp>
void NVMatrix::_aggregate(int axis, NVMatrix& target, Agg agg, UnaryOp uop, BinaryOp bop, NVMatrix& tmp) {
    _aggregate(axis, target, agg, uop, bop, getDefaultStream(), tmp);
}

template<class Agg, class BinaryOp>
void NVMatrix::_aggregate(int axis, NVMatrix& target, Agg agg, BinaryOp bop, NVMatrix& tmp) {
    _aggregate(axis, target, agg, NVMatrixOps::Identity(), bop, getDefaultStream(), &tmp);
}

template<class Agg, class BinaryOp>
void NVMatrix::_aggregate(int axis, NVMatrix& target, Agg agg, BinaryOp bop, cudaStream_t stream, NVMatrix& tmp) {
    _aggregate(axis, target, agg, NVMatrixOps::Identity(), bop, stream, &tmp);
}

template<class Agg, class UnaryOp, class BinaryOp>
NVMatrix& NVMatrix::_aggregate(int axis, Agg agg, UnaryOp uop, BinaryOp bop, NVMatrix& tmp) {
    NVMatrix &sumVec = construct();
    _aggregate(axis, sumVec, agg, uop, bop, tmp);
    return sumVec;
}

template<class Agg, class UnaryOp, class BinaryOp>
NVMatrix& NVMatrix::_aggregate(int axis, Agg agg, UnaryOp uop, BinaryOp bop, cudaStream_t stream, NVMatrix& tmp) {
    NVMatrix &sumVec = construct();
    _aggregate(axis, sumVec, agg, uop, bop, stream, tmp);
    return sumVec;
}

template<class Agg, class BinaryOp>
NVMatrix& NVMatrix::_aggregate(int axis, Agg agg, BinaryOp bop, NVMatrix& tmp) {
    return _aggregate(axis, agg, NVMatrixOps::Identity(), bop, tmp);
}

template<class Agg, class BinaryOp>
NVMatrix& NVMatrix::_aggregate(int axis, Agg agg, BinaryOp bop, cudaStream_t stream, NVMatrix& tmp) {
    return _aggregate(axis, agg, NVMatrixOps::Identity(), bop, stream, tmp);
}

void NVMatrix::inRangeInc(float lower, float upper) {
    inRangeInc(lower, upper, *this);
}
void NVMatrix::inRangeInc(float lower, float upper, NVMatrix& target) {
    apply(NVMatrixOps::InRange<false>(lower, upper), target);
}

void NVMatrix::inRangeExc(float lower, float upper) {
    inRangeExc(lower, upper, *this);
}

void NVMatrix::inRangeExc(float lower, float upper, NVMatrix& target) {
    apply(NVMatrixOps::InRange<true>(lower, upper), target);
}

void NVMatrix::biggerThanScalar(float scalar) {
    biggerThanScalar(scalar, *this);
}

void NVMatrix::biggerThanScalar(float scalar, NVMatrix& target) {
    apply(NVMatrixOps::BiggerThanScalar(scalar), target);
}

void NVMatrix::smallerThanScalar(float scalar) {
    smallerThanScalar(scalar, *this);
}

void NVMatrix::smallerThanScalar(float scalar, NVMatrix& target) {
    apply(NVMatrixOps::SmallerThanScalar(scalar), target);
}

void NVMatrix::addScalar(float scaleThis, float scalar, NVMatrix& target) {
    apply(NVMatrixOps::WeightedAddScalar(scaleThis, scalar), target);
}

void NVMatrix::addScalar(float scalar, NVMatrix& target) {
    apply(NVMatrixOps::AddScalar(scalar), target);
}

void NVMatrix::addScalar(float scalar) {
    addScalar(scalar, *this);
}

void NVMatrix::minWithScalar(float scalar, NVMatrix& target) {
    apply(NVMatrixOps::MinWithScalar(scalar), target);
}

void NVMatrix::minWithScalar(float scalar) {
    minWithScalar(scalar, *this);
}

void NVMatrix::maxWithScalar(float scalar, NVMatrix& target) {
    apply(NVMatrixOps::MaxWithScalar(scalar), target);
}

void NVMatrix::maxWithScalar(float scalar) {
    maxWithScalar(scalar, *this);
}

void NVMatrix::pow(float p, NVMatrix& target) {
    apply(NVMatrixOps::Pow(p), target);
}

void NVMatrix::pow(float p) {
    pow(p, *this);
}

void NVMatrix::scale(float _scale) {
    scale(_scale, *this);
}

void NVMatrix::scale(float _scale, cudaStream_t stream) {
    scale(_scale, *this, stream);
}

void NVMatrix::scale(float _scale, NVMatrix& target) {
    scale(_scale, target, NVMatrix::getDefaultStream());
}

void NVMatrix::scale(float _scale, NVMatrix& target, cudaStream_t stream) {
    if (_scale != 1 || &target != this) { // optimize away scale by 1
        if (_scale == 1) {
            copy(target, stream);
        } else {
            apply(NVMatrixOps::MultByScalar(_scale), target, stream);
        }
    }
}

void NVMatrix::zero() {
    apply(NVMatrixOps::Zero());
}

void NVMatrix::zero(NVMatrix& like) {
    resize(like);
    zero();
}

void NVMatrix::max(int axis, NVMatrix& target) {
    _aggregate(axis, target, NVMatrixAggs::Max(), NVMatrixBinaryOps::Second());
}

void NVMatrix::max(int axis, NVMatrix& target, NVMatrix& tmp) {
    _aggregate(axis, target, NVMatrixAggs::Max(), NVMatrixBinaryOps::Second(), tmp);
}

void NVMatrix::addSum(NVMatrix& a, int axis, float scaleThis, float scaleSum) {
    addSum(a, axis, scaleThis, scaleSum, getDefaultStream());
}

void NVMatrix::addSum(NVMatrix& a, int axis, float scaleThis, float scaleSum, cudaStream_t stream) {
    if (scaleThis != 0) {
        a._aggregate(axis, *this, NVMatrixAggs::Sum(), NVMatrixBinaryOps::WeightedAdd(scaleThis, scaleSum), stream);
    } else {
        a._aggregate(axis, *this, NVMatrixAggs::Sum(), NVMatrixBinaryOps::SecondScaled(scaleSum), stream);
    }
}

void NVMatrix::addMax(NVMatrix& a, int axis, float scaleThis, float scaleMax) {
    addMax(a, axis, scaleThis, scaleMax, getDefaultStream());
}

void NVMatrix::addMax(NVMatrix& a, int axis, float scaleThis, float scaleMax, cudaStream_t stream) {
    if (scaleThis != 0) {
        a._aggregate(axis, *this, NVMatrixAggs::Max(), NVMatrixBinaryOps::WeightedAdd(scaleThis, scaleMax), stream);
    } else {
        a._aggregate(axis, *this, NVMatrixAggs::Max(), NVMatrixBinaryOps::SecondScaled(scaleMax), stream);
    }
}

void NVMatrix::sum(int axis, NVMatrix& target) {
    sum(axis, target, getDefaultStream());
}

void NVMatrix::sum(int axis, NVMatrix& target, cudaStream_t stream) {
    _aggregate(axis, target, NVMatrixAggs::Sum(), NVMatrixBinaryOps::Second(), stream);
}

void NVMatrix::sum(int axis, NVMatrix& target, NVMatrix& tmp) {
    sum(axis, target, getDefaultStream(), tmp);
}

void NVMatrix::sum(int axis, NVMatrix& target, cudaStream_t stream, NVMatrix& tmp) {
    _aggregate(axis, target, NVMatrixAggs::Sum(), NVMatrixBinaryOps::Second(), stream, tmp);
}

void NVMatrix::sumOfSquares(int axis, NVMatrix& target) {
    sumOfSquares(axis, target, getDefaultStream());
}

void NVMatrix::sumOfSquares(int axis, NVMatrix& target, cudaStream_t stream) {
    _aggregate(axis, target, NVMatrixAggs::Sum(), NVMatrixOps::Square(), NVMatrixBinaryOps::Second(), stream);
}

void NVMatrix::min(int axis, NVMatrix& target) {
    _aggregate(axis, target, NVMatrixAggs::Min(), NVMatrixBinaryOps::Second());
}

NVMatrix& NVMatrix::max(int axis) {
    return _aggregate(axis, NVMatrixAggs::Max(), NVMatrixBinaryOps::Second());
}

NVMatrix& NVMatrix::sum(int axis) {
    return _aggregate(axis, NVMatrixAggs::Sum(), NVMatrixBinaryOps::Second());
}

NVMatrix& NVMatrix::min(int axis) {
    return _aggregate(axis, NVMatrixAggs::Min(), NVMatrixBinaryOps::Second());
}

NVMatrix& NVMatrix::sumOfSquares(int axis) {
    return _aggregate(axis, NVMatrixAggs::Sum(), NVMatrixOps::Square(), NVMatrixBinaryOps::Second());
}

void NVMatrix::_sum_setParams(int n, dim3* blocks, dim3* threads) {
    *threads = dim3(DP_BLOCKSIZE);
    *blocks = dim3(std::min(CPUSUM_MAX, DIVUP(n, DP_BLOCKSIZE)));
}

float NVMatrix::mean() {
    return sum() / getNumElements();
}

float NVMatrix::sum() {
    return _totalAgg(NVMatrixAggs::Sum());
}

float NVMatrix::sum(NVMatrix& tmpbuf) {
    return _totalAgg(NVMatrixAggs::Sum(), tmpbuf, getDefaultStream());
}

float NVMatrix::max() {
    return _totalAgg(NVMatrixAggs::Max());
}

float NVMatrix::min() {
    return _totalAgg(NVMatrixAggs::Min());
}

float NVMatrix::countNan() {
    return _totalAgg(NVMatrixAggs::CountNan());
}

float NVMatrix::countInf() {
    return _totalAgg(NVMatrixAggs::CountInf());
}

template<class Agg>
float NVMatrix::_totalAgg(Agg agg) {
    return _totalAgg(agg, getDefaultStream());
}

template<class Agg>
float NVMatrix::_totalAgg(Agg agg, cudaStream_t stream) {
    NVMatrix tmp;
    return _totalAgg(agg, tmp, stream);
}

template<class Agg>
float NVMatrix::_totalAgg(Agg agg, NVMatrix& tmpbuf, cudaStream_t stream) {
    assert(isContiguous());
    dim3 blocks, threads;
    // Sum most of it on GPU

    _sum_setParams(getNumElements(), &blocks, &threads);
    tmpbuf.resize(1, blocks.x);
    kTotalAgg<<<blocks, threads, 0, stream>>>(getDevData(), tmpbuf.getDevData(), getNumElements(), agg);
    getLastCudaError("kTotalAgg: Kernel execution failed");
    // Don't need to sync because we copyToHost in the same stream, so it's serialized
//    NVMatrix::syncStream(stream);
    return tmpbuf.cpuAgg(agg, stream);
}
template<class Agg>
float NVMatrix::cpuAgg(Agg agg, cudaStream_t stream) {
    Matrix bufCPU(getNumRows(), getNumCols());
    copyToHost(bufCPU, false, stream);
    if (getNumElements() > 1) { // Sum remainder on CPU
        if (typeid(Agg) == typeid(NVMatrixAggs::Sum)) {
            return bufCPU.sum();
        } else if (typeid(Agg) == typeid(NVMatrixAggs::Max)) {
            return bufCPU.max();
        } else if (typeid(Agg) == typeid(NVMatrixAggs::Min)) {
            return bufCPU.min();
        } else if (typeid(Agg) == typeid(NVMatrixAggs::CountNan)) {
            return bufCPU.hasNan(); //yea, it's not the same, who cares
        } else if (typeid(Agg) == typeid(NVMatrixAggs::CountInf)) {
            return bufCPU.hasInf();
        } else {
            assert(false);
        }
    }
    return bufCPU(0,0);
}

float NVMatrix::dotProduct(NVMatrix& b) {
    return dotProduct(b, getDefaultStream());
}

float NVMatrix::dotProduct(NVMatrix& b, cudaStream_t stream) {
    NVMatrix tmp;
    return dotProduct(b, tmp, stream);
}

/*
 * Fast dot product only for matrices with same transposedness.
 */
float NVMatrix::dotProduct(NVMatrix& b, NVMatrix& tmp, cudaStream_t stream) {
    assert(isContiguous() && b.isContiguous());
    assert(isSameDims(b));
    assert(isTrans() == b.isTrans()); // see?
    dim3 blocks, threads;
    _sum_setParams(getNumElements(), &blocks, &threads);
//    NVMatrix target(1, blocks.x);
    tmp.resize(1, blocks.x);
    kDotProduct_r<<<blocks, threads, 0, stream>>>(getDevData(), b.getDevData(), tmp.getDevData(), getNumElements());
    getLastCudaError("kDotProduct_r: Kernel execution failed");
//    cudaThreadSynchronize();
//    syncStream(stream);
//    return tmp._totalAgg(NVMatrixAggs::Sum(), stream);
    return tmp.cpuAgg(NVMatrixAggs::Sum(), stream);
}

float NVMatrix::norm2() {
    return dotProduct(*this);
}

float NVMatrix::norm() {
    return sqrt(norm2());
}

void NVMatrix::print(int startRow, int rows, int startCol, int cols) const {
//    cudaThreadSynchronize();
    syncDevice();
    Matrix hm = Matrix(_numRows, _numCols);
    copyToHost(hm);
    hm.print(startRow, rows, startCol, cols);
}

void NVMatrix::print(int rows, int cols) const {
    print(0, rows, 0, cols);
}

void NVMatrix::printShape(const char* name) const {
    printf("%s: %dx%d\n", name, _numRows, _numCols);
}

void NVMatrix::alloc(int numElements) {
    _memSegment = DEVICE_MEMORY_MANAGER::getInstance(getDeviceID()).malloc(numElements * sizeof(float));
}

void NVMatrix::dealloc() {
    DEVICE_MEMORY_MANAGER::getInstance(_memSegment->getDeviceID()).free(_memSegment);
    _memSegment = NULL;
    deallocTexture();
}

void NVMatrix::deallocTexture() {
    if (_texObj != 0) {
        checkCudaErrors(cudaDestroyTextureObject(_texObj));
        _texObj = 0;
    }
}

cudaTextureObject_t NVMatrix::getTextureObject() {
   if (_texObj == 0) {
       assert(isContiguous());
       //size_t memFree, memTotal;

       struct cudaResourceDesc resDesc;
       memset(&resDesc, 0, sizeof(resDesc));
       resDesc.resType = cudaResourceTypeLinear;
       resDesc.res.linear.devPtr = getDevData();
       resDesc.res.linear.sizeInBytes = getNumDataBytes();
       resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
       struct cudaTextureDesc texDesc;
       memset(&texDesc, 0, sizeof(texDesc));
       checkCudaErrors(cudaCreateTextureObject(&_texObj, &resDesc, &texDesc, NULL));
   }
   assert(_texObj != 0);
   return _texObj;
}

NVMatrix& NVMatrix::construct() const {
    return *new NVMatrix();
}
NVMatrix& NVMatrix::construct(bool isTrans) const {
    return *new NVMatrix(isTrans);
}
NVMatrix& NVMatrix::construct(int numRows, int numCols, bool isTrans) const {
    return *new NVMatrix(numRows, numCols, isTrans);
}
NVMatrix& NVMatrix::construct(const Matrix& like, bool copy) const {
    return *new NVMatrix(like, copy);
}
NVMatrix& NVMatrix::construct(const NVMatrix& like, bool copy) const {
    return *new NVMatrix(like, copy);
}
NVMatrix& NVMatrix::construct(const NVMatrix& like) const {
    return *new NVMatrix(like);
}
NVMatrix& NVMatrix::construct(const Matrix& like) const {
    return *new NVMatrix(like);
}
NVMatrix& NVMatrix::construct(MemorySegment* mem, int numRows, int numCols, int stride, bool isTrans) const {
    return *new NVMatrix(mem, numRows, numCols, stride, isTrans);
}

std::pair<size_t, size_t> NVMatrix::getCudaMemorySize() {
    size_t memFree, memTotal;
    checkCudaErrors(cudaMemGetInfo(&memFree, &memTotal));
    return std::pair<size_t,size_t>(memFree, memTotal);
}


/* ================
 * HostNVMatrix
 * ================
 */
HostNVMatrix::~HostNVMatrix() {
    if (_ownsData && _numElements > 0) {
        dealloc();
    } else {
        // dealloc frees the mem segment. But if this is a view,
        // then we need to delete the mem segment object.
//        assert(_memSegment == NULL || _memSegment->getSize() == 0);
        delete _memSegment;
    }
    _deleted = true;
}
HostNVMatrix::HostNVMatrix() : NVMatrix() {
    _init(false);
}
HostNVMatrix::HostNVMatrix(bool isTrans) {
    _init(isTrans);
}
HostNVMatrix::HostNVMatrix(int numRows, int numCols, bool isTrans)  {
    _init(isTrans);
    resize(numRows, numCols);
}
HostNVMatrix::HostNVMatrix(const Matrix& like, bool copy)  {
    _init(like.isTrans());
    resize(like.getNumRows(), like.getNumCols());
    if (copy) {
        copyFromHost(like);
    }
}
HostNVMatrix::HostNVMatrix(const NVMatrix& like, bool copy)  {
    _init(like.isTrans());
    resize(like.getNumRows(), like.getNumCols());
    if (copy) {
        like.copy(*this);
    }
}
HostNVMatrix::HostNVMatrix(const NVMatrix& like)  {
    _init(like.isTrans());
    resize(like.getNumRows(), like.getNumCols());
}
HostNVMatrix::HostNVMatrix(const Matrix& like) {
    _init(false);
    resize(like.getNumRows(), like.getNumCols());
}
HostNVMatrix::HostNVMatrix(MemorySegment* mem, int numRows, int numCols, int stride, bool isTrans)
    : NVMatrix(mem, numRows, numCols, stride, isTrans) {
}

NVMatrix& HostNVMatrix::construct() const {
    return *new HostNVMatrix();
}
NVMatrix& HostNVMatrix::construct(bool isTrans) const {
    return *new HostNVMatrix(isTrans);
}
NVMatrix& HostNVMatrix::construct(int numRows, int numCols, bool isTrans) const {
    return *new HostNVMatrix(numRows, numCols, isTrans);
}
NVMatrix& HostNVMatrix::construct(const Matrix& like, bool copy) const {
    return *new HostNVMatrix(like, copy);
}
NVMatrix& HostNVMatrix::construct(const NVMatrix& like, bool copy) const {
    return *new HostNVMatrix(like, copy);
}
NVMatrix& HostNVMatrix::construct(const NVMatrix& like) const {
    return *new HostNVMatrix(like);
}
NVMatrix& HostNVMatrix::construct(const Matrix& like) const {
    return *new HostNVMatrix(like);
}
NVMatrix& HostNVMatrix::construct(MemorySegment* mem, int numRows, int numCols, int stride, bool isTrans) const {
    return *new HostNVMatrix(mem, numRows, numCols, stride, isTrans);
}

void HostNVMatrix::copyFromHost(const Matrix& hostMatrix, bool resizeTarget, cudaStream_t stream) {
    if (resizeTarget) {
        resize(hostMatrix);
    } else {
        assert(isSameDims(hostMatrix));
    }
    setTrans(hostMatrix.isTrans());
    if (getNumElements() > 0) {
        checkCudaErrors(cudaMemcpy2D(getDevData(), _stride * sizeof(float), hostMatrix.getData(),
                                     hostMatrix.getLeadingDim() * sizeof(float), getLeadingDim() * sizeof(float),
                                     getFollowingDim(), cudaMemcpyHostToHost));
//        syncStream(stream);
    }
}

void HostNVMatrix::copyFromHost(const Matrix& hostMatrix, bool resizeTarget) {
    copyFromHost(hostMatrix, resizeTarget, 0);
}

void HostNVMatrix::copyFromHost(const Matrix& hostMatrix) {
    copyFromHost(hostMatrix, false, 0);
}

void HostNVMatrix::copyToHost(Matrix& hostMatrix, bool resizeTarget, cudaStream_t stream) const {
    if (resizeTarget) {
        hostMatrix.resize(getNumRows(), getNumCols());
    } else {
        assert(isSameDims(hostMatrix));
    }
    hostMatrix.setTrans(_isTrans);
    if (getNumElements() > 0) {
        checkCudaErrors(cudaMemcpy2D(hostMatrix.getData(), hostMatrix.getLeadingDim() * sizeof(float),
                                     getDevData(), _stride * sizeof(float), getLeadingDim() * sizeof(float),
                                     getFollowingDim(), cudaMemcpyHostToHost));
//        syncStream(stream);
    }
}

void HostNVMatrix::copyToHost(Matrix& hostMatrix, bool resizeTarget) const {
    copyToHost(hostMatrix, resizeTarget, 0);
}

void HostNVMatrix::copyToHost(Matrix& hostMatrix) const {
    copyToHost(hostMatrix, false, 0);
}

void HostNVMatrix::alloc(int numElements) {
//    checkCudaErrors(cudaHostAlloc(&_devData, numElements * sizeof(float), cudaHostAllocPortable));
    _memSegment = HOST_MEMORY_MANAGER::getInstance().malloc(numElements * sizeof(float));
//    _memSegment = FastHostMemoryManager::getInstance().malloc(numElements * sizeof(float));
}

void HostNVMatrix::dealloc() {
//    FastHostMemoryManager::getInstance().free(_memSegment);
    HOST_MEMORY_MANAGER::getInstance().free(_memSegment);
    _memSegment = NULL;
//    checkCudaErrors(cudaFreeHost(_devData));
}

cudaTextureObject_t HostNVMatrix::getTextureObject() {
    assert(false);
    return 0;
}
