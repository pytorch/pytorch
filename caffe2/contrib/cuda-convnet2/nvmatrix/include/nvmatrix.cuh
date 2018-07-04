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

#ifndef NVMATRIX_H_
#define NVMATRIX_H_

#include <map>
#include <vector>
#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>
#include <curand_kernel.h>

#include <helper_cuda.h>
#include "../../util/include/matrix.h"
#include "nvmatrix_kernels.cuh"
#include "nvmatrix_operators.cuh"
#include "memory.cuh"

#ifdef WARNINGS
#define WARN(msg) printf("WARN: File %s, line %d: %s\n", __FILE__, __LINE__, msg);
#else
#define WARN(msg) ;
#endif

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
                            printf("CURAND Error at %s:%d\n",__FILE__,__LINE__);\
                            exit(EXIT_FAILURE);}} while(0)

#define CUBLAS_CALL(x) do { if((x) != CUBLAS_STATUS_SUCCESS) { \
                            printf("CUBLAS Error at %s:%d\n",__FILE__,__LINE__);\
                            exit(EXIT_FAILURE);}} while(0)

/*
 * Memory manager to use for GPU memory allocations.
 *
 * CUDAMemoryManager: Default Nvidia memory manager; just calls cudaMalloc / cudaFree.
 *                    Allocating and freeing memory is slow.
 * FastMemoryManager: A GPU memory manager with very fast (constant time)
 *                    alloc / free, but possibly more wasteful of memory.
 */
#define DEVICE_MEMORY_MANAGER       CUDAMemoryManager

/*
 * Memory manager to use for host memory allocations.
 *
 * CUDAHostMemoryManager: Default Nvidia memory manager; just calls cudaHostAlloc / cudaFreeHost.
 *                        Allocating and freeing memory is slow.
 * FastHostMemoryManager: A host memory manager with very fast (constant time)
 *                        alloc / free, but possibly more wasteful of memory.
 */
#define HOST_MEMORY_MANAGER         CUDAHostMemoryManager

class NVMatrix;
typedef std::vector<NVMatrix*> NVMatrixV;

class NVMatrix {
protected:
    int _numCols, _numRows;
    int _numElements;
    int _stride;
//    float* getDevData();
    MemorySegment* _memSegment;
    bool _isTrans;
    bool _ownsData;
    // This flag makes sure that the NVMatrix destructor does nothing
    // when called on HostNVMatrix instance.
    bool _deleted;
    cudaTextureObject_t _texObj;

//    static std::map<int,curandGenerator_t> rndGen;
    static std::map<int,MemorySegment*> _rndDevStates;
    static std::map<int,cublasHandle_t> _cublasHandles;
    // Map from device id --> # of random streams initialized on that device
    static std::map<int,int> _rndDevThreads;
    static pthread_mutex_t *_rndMutex, *_cublasMutex, *_streamMutex;
    // Map from device id --> default stream
    static std::map<int,cudaStream_t> _defaultStreams;

    cublasOperation_t getTransChar() const {
        /*
         * not a typo! return opposite character because a
         * non-transposed nvmatrix is in row-major order while a non-transposed
         * cublas matrix is in column-major order.
         */
        return _isTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
    }

    void _init(bool isTrans);
    void _sum_setParams(int n, dim3* blocks, dim3* threads);
    template<class Agg> float cpuAgg(Agg agg, cudaStream_t stream);
    template<class Agg> float _totalAgg(Agg agg);
    template<class Agg> float _totalAgg(Agg agg, cudaStream_t stream);
    template<class Agg> float _totalAgg(Agg agg, NVMatrix& tmpbuf, cudaStream_t stream);
    template<class Agg, class UnaryOp, class BinaryOp> void _aggregate(int axis, NVMatrix& target, Agg agg, UnaryOp uop, BinaryOp bop, cudaStream_t stream, NVMatrix* tmp);
    template<class Agg, class UnaryOp, class BinaryOp> void _aggregate(int axis, NVMatrix& target, Agg agg, UnaryOp uop, BinaryOp bop, cudaStream_t stream);
    template<class Agg, class UnaryOp, class BinaryOp> void _aggregate(int axis, NVMatrix& target, Agg agg, UnaryOp uop, BinaryOp bop);
    template<class Agg, class BinaryOp> void _aggregate(int axis, NVMatrix& target, Agg agg, BinaryOp bop, cudaStream_t stream);
    template<class Agg, class BinaryOp> void _aggregate(int axis, NVMatrix& target, Agg agg, BinaryOp bop);
    template<class Agg, class BinaryOp> NVMatrix& _aggregate(int axis, Agg agg, BinaryOp bop, cudaStream_t stream);
    template<class Agg, class BinaryOp> NVMatrix& _aggregate(int axis, Agg agg, BinaryOp bop);
    template<class Agg, class UnaryOp, class BinaryOp> NVMatrix& _aggregate(int axis, Agg agg, UnaryOp, BinaryOp bop, cudaStream_t stream);
    template<class Agg, class UnaryOp, class BinaryOp> NVMatrix& _aggregate(int axis, Agg agg, UnaryOp, BinaryOp bop);

    template<class Agg, class UnaryOp, class BinaryOp> void _aggregate(int axis, NVMatrix& target, Agg agg, UnaryOp uop, BinaryOp bop, NVMatrix& tmp);
    template<class Agg, class BinaryOp> void _aggregate(int axis, NVMatrix& target, Agg agg, BinaryOp bop, cudaStream_t stream, NVMatrix& tmp);
    template<class Agg, class BinaryOp> void _aggregate(int axis, NVMatrix& target, Agg agg, BinaryOp bop, NVMatrix& tmp);
    template<class Agg, class BinaryOp> NVMatrix& _aggregate(int axis, Agg agg, BinaryOp bop, cudaStream_t stream, NVMatrix& tmp);
    template<class Agg, class BinaryOp> NVMatrix& _aggregate(int axis, Agg agg, BinaryOp bop, NVMatrix& tmp);
    template<class Agg, class UnaryOp, class BinaryOp> NVMatrix& _aggregate(int axis, Agg agg, UnaryOp, BinaryOp bop, cudaStream_t stream, NVMatrix& tmp);
    template<class Agg, class UnaryOp, class BinaryOp> NVMatrix& _aggregate(int axis, Agg agg, UnaryOp, BinaryOp bop, NVMatrix& tmp);

    template <class Randomizer> void _unaryRandomize(NVMatrix& target, Randomizer rnd, cudaStream_t stream);
    template <class Randomizer> void _unaryRandomize(NVMatrix& target, Randomizer rnd);
    template <class Randomizer> void _binaryRandomize(NVMatrix& data2, NVMatrix& target, Randomizer rnd);
    template <class Randomizer> void _binaryRandomize(NVMatrix& data2, NVMatrix& target, Randomizer rnd, cudaStream_t stream);

    virtual void alloc(int numElements);
    virtual void dealloc();
    void deallocTexture();
    virtual NVMatrix& construct() const;
    virtual NVMatrix& construct(bool isTrans) const;
    virtual NVMatrix& construct(int numRows, int numCols, bool isTrans=false) const;
    virtual NVMatrix& construct(const Matrix& like, bool copy) const;
    virtual NVMatrix& construct(const NVMatrix& like, bool copy) const;
    virtual NVMatrix& construct(const NVMatrix& like) const;
    virtual NVMatrix& construct(const Matrix& like) const;
    virtual NVMatrix& construct(MemorySegment* mem, int numRows, int numCols, int stride, bool isTrans) const;
    static cublasHandle_t getCublasHandle();
    static cublasHandle_t getCublasHandle(int deviceID);
public:
    NVMatrix();
    NVMatrix(bool isTrans);
    NVMatrix(int numRows, int numCols, bool isTrans=false);
    NVMatrix(const Matrix& like, bool copy);
    NVMatrix(const NVMatrix& like, bool copy);
    NVMatrix(const NVMatrix& like);
    NVMatrix(const Matrix& like);
    NVMatrix(MemorySegment* mem, int numRows, int numCols, int stride, bool isTrans);
    virtual ~NVMatrix();

    // Returns the device ID on which the data pointer is allocated
    int getDataDeviceID() const;
    static void initRandom(unsigned long long seed, int numStreams, cudaStream_t stream);
    static void initRandom(unsigned long long seed, int numStreams);
    static void initRandom(unsigned long long seed);
    static void initRandom();
    static void initCublas();
    static void destroyCublas();
    static std::pair<size_t, size_t> getCudaMemorySize();

    // Returns the currently-active device ID for calling thread
    static int getDeviceID();
    static void setDeviceID(int d);
    static bool canAccessPeer(int srcDevice, int tgtDevice);
    static bool isRndInitialized();
    static bool isRndInitialized(bool haveLock);
    static curandState* getCurandState();
    static curandState* getCurandState(int numStreams);
    static void destroyRandom();
    static pthread_mutex_t* makeMutex();
    static cudaStream_t getDefaultStream(int deviceID);
    static cudaStream_t getDefaultStream();
    static void syncDevice();
    static void syncStream();
    static void syncStream(cudaStream_t stream);

    /*
     * DO NOT DEREFERENCE IN HOST CODE! This is a device memory pointer.
     */
    float* getCellPtr(int i, int j) const {
        if (_isTrans) {
            return &getDevData()[j * _numRows + i];
        }
        return &getDevData()[i * _numCols + j];
    }

    bool isSameDims(const Matrix& m) const {
        return m.getNumRows() == _numRows && m.getNumCols() == _numCols;
    }

    bool isSameDims(const NVMatrix& m) const {
        return m.getNumRows() == _numRows && m.getNumCols() == _numCols;
    }

    int getNumRows() const {
        return _numRows;
    }

    int getNumCols() const {
        return _numCols;
    }

    int getStride() const {
        return _stride;
    }

    int getLeadingDim() const {
        return _isTrans ? _numRows : _numCols;
    }

    int getFollowingDim() const {
        return !_isTrans ? _numRows : _numCols;
    }

    /*
     * FALSE:    Row-major order.
     * TRUE:     Column-major order.
     */
    bool isTrans() const {
        return _isTrans;
    }

    bool isView() const {
        return !_ownsData;
    }

    float* getDevData() const {
        return _memSegment == NULL ? NULL : _memSegment->getData<float>();
    }

    MemorySegment& getMemorySegment() const {
        return *_memSegment;
    }

    int getNumElements() const {
        return _numElements;
    }

    size_t getNumDataBytes() const {
        return size_t(_numElements) * 4;
    }

    /*
     * Only use if you know what you're doing!
     * Does not actually transpose matrix.
     */
    void setTrans(bool trans) {
        if (trans != _isTrans) {
            assert(isContiguous());
            _isTrans = trans;
            _stride = getLeadingDim();
        }
    }

    /*
     * Only use if you know what you're doing!
     * This toggles whether this object will free its GPU memory when it's destroyed.
     */
    void setIsView(bool isView) {
        _ownsData = !isView;
    }

    bool isContiguous() const {
        return _stride == getLeadingDim() || getFollowingDim() == 1;
    }

    void truncate() {
        resize(0,0);
    }

    virtual cudaTextureObject_t getTextureObject();

    virtual void copyFromHost(const Matrix& hostMatrix);
    virtual void copyFromHost(const Matrix& hostMatrix, bool resizeTarget);
    virtual void copyFromHost(const Matrix& hostMatrix, bool resizeTarget, cudaStream_t stream);
    virtual void copyToHost(Matrix& hostMatrix) const;
    virtual void copyToHost(Matrix& hostMatrix, bool resizeTarget) const;
    virtual void copyToHost(Matrix& hostMatrix, bool resizeTarget, cudaStream_t stream) const;
    void copy(NVMatrix& dest) const;
    void copy(NVMatrix& dest, cudaStream_t stream) const;
    NVMatrix& copy() const;
    void addProduct(NVMatrix& a, NVMatrix &b, float scaleThis, float scaleAB, cudaStream_t stream);
    void addProduct(NVMatrix& a, NVMatrix &b, float scaleThis, float scaleAB);
    void addProduct(NVMatrix& a, NVMatrix &b);
    void rightMult(NVMatrix &b, float scaleAB, NVMatrix &target, cudaStream_t stream);
    void rightMult(NVMatrix &b, float scaleAB, NVMatrix &target);
    void rightMult(NVMatrix &b, NVMatrix &target);
    void rightMult(NVMatrix &b, float scaleAB);
    void randomizeUniform();
    void addGaussianNoise(NVMatrix& stdevs, bool var, NVMatrix& target);
    void addGaussianNoise(float stdev, NVMatrix& target);
    void addGaussianNoise(NVMatrix& stdevs, bool var);
    void addGaussianNoise(NVMatrix& stdevs);
    void addGaussianNoise(float stdev);
    void addGaussianNoise();
    void randomizeGaussian();
    void randomizeGaussian(float stdev);
    void randomizeGaussian(float mean, float stdev);
    void randomizeGaussian(float mean, NVMatrix& stdevs);
    void randomizeGaussian(float mean, float stdevMult, NVMatrix& stdevs);
    void randomizeGaussian(NVMatrix& stdevs);
    void randomizeGaussian(NVMatrix& stdevs, NVMatrix& target);
    void binarizeProbs();
    void binarizeProbs(NVMatrix& target);

    void biggerThan(NVMatrix& m, NVMatrix& target);
    void biggerThan(NVMatrix& m);
    void biggerThanVector(NVMatrix& vec, NVMatrix& target);
    void biggerThanVector(NVMatrix& vec);
    void equals(NVMatrix& m, NVMatrix& target);
    void equals(NVMatrix& m);

    void _checkBounds(int startRow, int endRow, int startCol, int endCol) const;
    NVMatrix& slice(int startRow, int endRow, int startCol, int endCol) const;
    void slice(int startRow, int endRow, int startCol, int endCol, NVMatrix& target) const;
    NVMatrix& sliceRows(int startRow, int endRow) const;
    void sliceRows(int startRow, int endRow, NVMatrix& target) const;
    NVMatrix& sliceCols(int startCol, int endCol) const;
    void sliceCols(int startCol, int endCol, NVMatrix& target) const;

    NVMatrixV& splitRows(int numParts);
    NVMatrixV& splitCols(int numParts);

    template <class Op> void apply(Op op, NVMatrix& target, cudaStream_t stream) {
        if (!target.isSameDims(*this)) {
            target.resize(*this);
        }
        if (getNumElements() > 0) {
            int height = target.getFollowingDim(), width = target.getLeadingDim();

            if (target.isTrans() == isTrans()) {
                if (!isContiguous() || !target.isContiguous()) {
                    dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ELTWISE_THREADS_X)),
                            std::min(NUM_BLOCKS_MAX, DIVUP(height, ELTWISE_THREADS_Y)));
                    dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
                    kEltwiseUnaryOp<Op><<<blocks, threads, 0, stream>>>(getDevData(), target.getDevData(), height, width, getStride(), target.getStride(), op);
                    getLastCudaError("kEltwiseUnaryOp: Kernel execution failed");
                } else {
                    dim3 threads = dim3(ELTWISE_FLAT_THREADS_X);
                    dim3 blocks = dim3(std::min(128, DIVUP(_numElements, ELTWISE_FLAT_THREADS_X)));
                    kEltwiseUnaryOpFlat<Op><<<blocks, threads, 0, stream>>>(getDevData(), target.getDevData(), _numElements, op);
                    getLastCudaError("kEltwiseUnaryOpFlat: Kernel execution failed");
                }
            } else {
                dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ELTWISE_THREADS_X)),
                        std::min(NUM_BLOCKS_MAX, DIVUP(height, ELTWISE_THREADS_Y)));
                dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
                bool checkBounds = !(width % ELTWISE_THREADS_X == 0 && height % ELTWISE_THREADS_X == 0);
    //            printf("height: %d, width: %d, stride: %d, target stride: %d, check bounds: %d, threads.x: %d, threads.y: %d, blocks.x: %d, blocks.y: %d\n",
    //                    height, width, getStride(), target.getStride(), checkBounds, threads.x, threads.y, blocks.x, blocks.y);
                if (checkBounds) {
                    kEltwiseUnaryOpTrans<Op, true><<<blocks, threads, 0, stream>>>(getDevData(), target.getDevData(), height, width, getStride(), target.getStride(), op);
                } else {
                    kEltwiseUnaryOpTrans<Op, false><<<blocks, threads, 0, stream>>>(getDevData(), target.getDevData(), height, width, getStride(), target.getStride(), op);
                }
                getLastCudaError("kEltwiseUnaryOpTrans: Kernel execution failed");
            }
        }
    }

    template <class Op> void apply(Op op, cudaStream_t stream) {
        apply(op, *this, stream);
    }

    template <class Op> void apply(Op op, NVMatrix& target) {
        apply(op, target, getDefaultStream());
    }

    template <class Op> void apply(Op op) {
        apply(op, *this);
    }

    template <class Op> void applyBinary(Op op, NVMatrix& b) {
        applyBinary(op, b, *this);
    }

    template <class Op> void applyBinary(Op op, NVMatrix& b, NVMatrix& target) {
        applyBinary(op, b, target, getDefaultStream());
    }

    template <class Op> void applyBinary(Op op, NVMatrix& b, NVMatrix& target, cudaStream_t stream) {
        assert(this->isSameDims(b));

        if (!target.isSameDims(*this)) {
            target.resize(*this);
        }

        if (getNumElements() > 0) {
            int height = target.getFollowingDim(), width = target.getLeadingDim();
            if (target.isTrans() == isTrans() && target.isTrans() == b.isTrans()) {
                if (!isContiguous() || !b.isContiguous() || !target.isContiguous()) {
                    dim3 blocks(std::min(128, DIVUP(width, ELTWISE_THREADS_X)),
                                std::min(128, DIVUP(height, ELTWISE_THREADS_Y)));
                    dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
                    kEltwiseBinaryOp<Op><<<blocks, threads, 0, stream>>>(getDevData(), b.getDevData(), target.getDevData(), height, width, getStride(),
                                                              b.getStride(), target.getStride(), op);
                } else {
                    dim3 threads = dim3(ELTWISE_FLAT_THREADS_X);
                    dim3 blocks = dim3(std::min(128, DIVUP(_numElements, ELTWISE_FLAT_THREADS_X)));
                    kEltwiseBinaryOpFlat<Op><<<blocks, threads, 0, stream>>>(getDevData(), b.getDevData(), target.getDevData(), _numElements, op);
                }
                getLastCudaError("kEltwiseBinaryOp: Kernel execution failed");
            } else {

                dim3 blocks(std::min(128, DIVUP(width, ELTWISE_THREADS_X)),
                            std::min(128, DIVUP(height, ELTWISE_THREADS_Y)));
                dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
                //  both x here since y divides x
                bool checkBounds = !(width % ELTWISE_THREADS_X == 0 && height % ELTWISE_THREADS_X == 0);
                if (target.isTrans() == isTrans() && target.isTrans() != b.isTrans()) {
                    if (checkBounds) {
                        kEltwiseBinaryOpTrans<Op,true,false,false><<<blocks, threads, 0, stream>>>(getDevData(), b.getDevData(), target.getDevData(), height, width,getStride(),
                                                                   b.getStride(), target.getStride(), op);
                    } else {
                        kEltwiseBinaryOpTrans<Op,false,false,false><<<blocks, threads, 0, stream>>>(getDevData(), b.getDevData(), target.getDevData(), height, width,getStride(),
                                                                   b.getStride(), target.getStride(), op);
                    }
                } else if (target.isTrans() != isTrans() && target.isTrans() != b.isTrans()) {
                    if (checkBounds) {
                        kEltwiseBinaryOpTrans<Op,true,true,false><<<blocks, threads, 0, stream>>>(getDevData(), b.getDevData(), target.getDevData(), height, width,getStride(),
                                                                   b.getStride(), target.getStride(), op);
                    } else {
                        kEltwiseBinaryOpTrans<Op,false,true,false><<<blocks, threads, 0, stream>>>(getDevData(), b.getDevData(), target.getDevData(), height, width,getStride(),
                                                                   b.getStride(), target.getStride(), op);
                    }
                } else if (target.isTrans() != isTrans() && target.isTrans() == b.isTrans()) {
                    if (checkBounds) {
                        kEltwiseBinaryOpTrans<Op,true,false,true><<<blocks, threads, 0, stream>>>(b.getDevData(), getDevData(), target.getDevData(), height, width,b.getStride(),
                                                                   getStride(), target.getStride(), op);
                    } else {
                        kEltwiseBinaryOpTrans<Op,false,false,true><<<blocks, threads, 0, stream>>>(b.getDevData(), getDevData(), target.getDevData(), height, width, b.getStride(),
                                                                   getStride(), target.getStride(), op);
                    }
                }
                getLastCudaError("kEltwiseBinaryOpTrans: Kernel execution failed");
            }
        }
    }

    template <class Op> void applyTernary(Op op, NVMatrix& b, NVMatrix& c, NVMatrix& target) {
        applyTernary(op, b, c, target, getDefaultStream());
    }

    template <class Op> void applyTernary(Op op, NVMatrix& b, NVMatrix& c, NVMatrix& target, cudaStream_t stream) {
        assert(isSameDims(b));
        assert(isSameDims(c));
        // For now ternary ops are only supported for matrices of same transposedness
        assert(isTrans() == b.isTrans());
        assert(isTrans() == c.isTrans());
        if (!target.isSameDims(*this) || target.isTrans() != isTrans()) {
            target.resize(*this);
        }
        if (getNumElements() > 0) {
            int height = target.getFollowingDim(), width = target.getLeadingDim();
            if (!isContiguous() || !b.isContiguous() || !c.isContiguous() || !target.isContiguous()) {
                dim3 blocks(std::min(512, DIVUP(width, ELTWISE_THREADS_X)),
                            std::min(512, DIVUP(height, ELTWISE_THREADS_Y)));
                dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
                kEltwiseTernaryOp<Op><<<blocks, threads, 0, stream>>>(getDevData(), b.getDevData(), c.getDevData(), target.getDevData(), height, width,
                                                                       getStride(), b.getStride(), c.getStride(), target.getStride(), op);
                getLastCudaError("kEltwiseTernaryOp: Kernel execution failed");
            } else {
                dim3 threads = dim3(ELTWISE_FLAT_THREADS_X);
                dim3 blocks = dim3(std::min(128, DIVUP(_numElements, ELTWISE_FLAT_THREADS_X)));
                kEltwiseTernaryOpFlat<Op><<<blocks, threads, 0, stream>>>(getDevData(), b.getDevData(), c.getDevData(), target.getDevData(), _numElements, op);
                getLastCudaError("kEltwiseTernaryOpFlat: Kernel execution failed");
            }
        }
    }

    bool resize(int numRows, int numCols, bool trans);
    bool resize(int numRows, int numCols);
    bool resize(const NVMatrix &like);
    bool resize(const Matrix &like);
    void reshape(int numRows, int numCols);
    NVMatrix& reshaped(int numRows, int numCols) const;
    void copy(NVMatrix &dest, int srcStartRow, int srcEndRow, int srcStartCol, int srcEndCol, int destStartRow, int destStartCol) const;
    void copy(NVMatrix &dest, int srcStartRow, int srcEndRow, int srcStartCol, int srcEndCol, int destStartRow, int destStartCol, cudaStream_t stream) const;
    void add(NVMatrix& b, float scaleA, float scaleB, NVMatrix& target, cudaStream_t stream);
    void add(NVMatrix& b, float scaleA, float scaleB, NVMatrix& target);
    void add(NVMatrix& b, float scaleB, NVMatrix& target);
    void add(NVMatrix& b, NVMatrix& target);
    void add(NVMatrix& b, float scaleB);
    void add(NVMatrix& b, float scaleA, float scaleB);
    void add(NVMatrix& b);
    void eltwiseMult(NVMatrix& b);
    void eltwiseMult(NVMatrix& b, NVMatrix& target);
    void eltwiseDivide(NVMatrix& b);
    void eltwiseDivide(NVMatrix& b, NVMatrix& target);
    void squaredDiff(NVMatrix& b);
    void squaredDiff(NVMatrix& b, NVMatrix& target);
    void subtract(NVMatrix& b, NVMatrix& target);
    void subtract(NVMatrix& b);
    void addVector(NVMatrix& vec, float scaleVec, NVMatrix& target, cudaStream_t stream);
    void addVector(NVMatrix& vec, float scaleVec, NVMatrix& target);
    void addVector(NVMatrix& vec);
    void addVector(NVMatrix& vec, float scaleVec);
    void addVector(NVMatrix& vec, NVMatrix& target);
    void equalsVector(NVMatrix& vec, NVMatrix& target);
    void equalsVector(NVMatrix& vec);
    void eltwiseMultByVector(NVMatrix& vec, NVMatrix& target, cudaStream_t stream);
    void eltwiseMultByVector(NVMatrix& vec, NVMatrix& target);
    void eltwiseMultByVector(NVMatrix& vec);
    void eltwiseMultByVector(NVMatrix& vec, cudaStream_t stream);
    void eltwiseDivideByVector(NVMatrix& vec, NVMatrix& target);
    void eltwiseDivideByVector(NVMatrix& vec);
    void tile(int timesY, int timesX, NVMatrix& target);
    void tile(int timesY, int timesX, NVMatrix& target, cudaStream_t stream);

    void addSum(NVMatrix& a, int axis, float scaleThis, float scaleSum);
    void addSum(NVMatrix& a, int axis, float scaleThis, float scaleSum, cudaStream_t stream);
    void addMax(NVMatrix& a, int axis, float scaleThis, float scaleMax);
    void addMax(NVMatrix& a, int axis, float scaleThis, float scaleMax, cudaStream_t stream);
    void sum(int axis, NVMatrix& target, cudaStream_t stream);
    void sum(int axis, NVMatrix& target);
    void sum(int axis, NVMatrix& target, cudaStream_t stream, NVMatrix& tmp);
    void sum(int axis, NVMatrix& target, NVMatrix& tmp);
    NVMatrix& sum(int axis);
    void max(int axis, NVMatrix& target);
    void max(int axis, NVMatrix& target, NVMatrix& tmp);
    NVMatrix& max(int axis);
    void min(int axis, NVMatrix& target);
    NVMatrix& min(int axis);
    void sumOfSquares(int axis, NVMatrix& target, cudaStream_t stream);
    void sumOfSquares(int axis, NVMatrix& target);
    NVMatrix& sumOfSquares(int axis);
    float mean();
    float sum();
    float sum(NVMatrix& tmpbuf);
    float max();
    float min();
    float countInf();
    float countNan();
    float norm2();
    float norm();

    void inRangeInc(float lower, float upper);
    void inRangeInc(float lower, float upper, NVMatrix& target);
    void inRangeExc(float lower, float upper);
    void inRangeExc(float lower, float upper, NVMatrix& target);
    void biggerThanScalar(float scalar);
    void biggerThanScalar(float scalar, NVMatrix& target);
    void smallerThanScalar(float scalar);
    void smallerThanScalar(float scalar, NVMatrix& target);
    void addScalar(float scaleThis, float scalar, NVMatrix& target);
    void addScalar(float scalar, NVMatrix& target);
    void addScalar(float scalar);
    void minWithScalar(float scalar, NVMatrix& target);
    void minWithScalar(float scalar);
    void maxWithScalar(float scalar, NVMatrix& target);
    void maxWithScalar(float scalar);
    void pow(float p, NVMatrix& target);
    void pow(float p);
    void scale(float _scale);
    void scale(float _scale, NVMatrix& target);
    void scale(float _scale, NVMatrix& target, cudaStream_t stream);
    void scale(float _scale, cudaStream_t stream);
    void zero();
    void zero(NVMatrix& like);

    float dotProduct(NVMatrix& b, NVMatrix& tmp, cudaStream_t stream);
    float dotProduct(NVMatrix& b, cudaStream_t stream);
    float dotProduct(NVMatrix& b);

    /*
     * Does SOFT transpose and returns result, leaving this matrix unchanged
     */
    NVMatrix& getTranspose();
    NVMatrix& getClone();

    /*
     * Does HARD transpose and puts result in target
     */
    void transpose(NVMatrix& target);

    /*
     * Does SOFT transpose
     */
    void transpose();
    bool transpose(bool trans);

    void flipTrans(NVMatrix& target, cudaStream_t stream);
    void flipTrans(NVMatrix& target);
    NVMatrix& flipTrans();

    void print(int startRow, int rows, int startCol, int cols) const;
    void print(int rows, int cols) const;
    void printShape(const char* name) const;

    template <class Op> void applyBinaryV(Op op, NVMatrix& vec, NVMatrix& target) {
        applyBinaryV(op, vec, target, getDefaultStream());
    }

    template <class Op> void applyBinaryV(Op op, NVMatrix& vec, NVMatrix& target, cudaStream_t stream) {
        assert(&target != &vec); // for now
        if (isSameDims(vec)) {
            applyBinary(op, vec, target, stream);
            return;
        }
        assert(vec.getNumRows() == 1 || vec.getNumCols() == 1);
        assert(vec.getNumRows() == _numRows || vec.getNumCols() == _numCols);
        assert(vec.isContiguous());

        target.resize(*this); // target must be same orientation as me for now
        int width = getLeadingDim(); //_isTrans ? _numRows : _numCols;
        int height = getFollowingDim(); //_isTrans ? _numCols : _numRows;
        dim3 threads(ADD_VEC_THREADS_X, ADD_VEC_THREADS_Y);

        if ((vec.getNumRows() == _numRows && !isTrans()) || (vec.getNumCols() == _numCols && isTrans())) {
            dim3 blocks(std::min(512, DIVUP(width, ADD_VEC_THREADS_X)), std::min(NUM_BLOCKS_MAX, DIVUP(height, ADD_VEC_THREADS_Y)));
            kColVectorOp<Op><<<blocks, threads, 0, stream>>>(getDevData(), vec.getDevData(), target.getDevData(), width, height, getStride(), target.getStride(), op);
        } else {
            dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ADD_VEC_THREADS_X)), std::min(NUM_BLOCKS_MAX, DIVUP(height, ADD_VEC_THREADS_Y)));
            kRowVectorOp<Op><<<blocks, threads, 0, stream>>>(getDevData(), vec.getDevData(), target.getDevData(), width, height, getStride(), target.getStride(), op);
        }
        getLastCudaError("Kernel execution failed");
    //    cudaThreadSynchronize();
    }

    template<class UnaryOperator> float argMax(UnaryOperator u) {
       return _totalAgg(NVMatrixAggs::ArgMax<UnaryOperator>(u));
    }
    static void batchedMatrixMultiply(NVMatrixV& a, NVMatrixV& b, NVMatrixV& target, float scaleTarget, float scaleAB, cudaStream_t stream, const float** aPtrsDev, const float** bPtrsDev, float** tgtPtrsDev);
    static void batchedMatrixMultiply(NVMatrixV& a, NVMatrixV& b, NVMatrixV& target, float scaleTarget, float scaleAB, cudaStream_t stream);
    static void batchedMatrixMultiply(NVMatrixV& a, NVMatrixV& b, NVMatrixV& target, float scaleTarget, float scaleAB, const float** aPtrsDev, const float** bPtrsDev, float** tgtPtrsDev);
    static void batchedMatrixMultiply(NVMatrixV& a, NVMatrixV& b, NVMatrixV& target, float scaleTarget, float scaleAB);

    static void assertSame(NVMatrixV& a);
};

class HostNVMatrix : public NVMatrix {
protected:
    void alloc(int numElements);
    void dealloc();
    NVMatrix& construct() const;
    NVMatrix& construct(bool isTrans) const;
    NVMatrix& construct(int numRows, int numCols, bool isTrans=false) const;
    NVMatrix& construct(const Matrix& like, bool copy) const;
    NVMatrix& construct(const NVMatrix& like, bool copy) const;
    NVMatrix& construct(const NVMatrix& like) const;
    NVMatrix& construct(const Matrix& like) const;
    NVMatrix& construct(MemorySegment* mem, int numRows, int numCols, int stride, bool isTrans) const;
public:
    ~HostNVMatrix();
    HostNVMatrix();
    HostNVMatrix(bool isTrans);
    HostNVMatrix(int numRows, int numCols, bool isTrans=false);
    HostNVMatrix(const Matrix& like, bool copy);
    HostNVMatrix(const NVMatrix& like, bool copy);
    HostNVMatrix(const NVMatrix& like);
    HostNVMatrix(const Matrix& like);
    HostNVMatrix(MemorySegment* mem, int numRows, int numCols, int stride, bool isTrans);
    void copyFromHost(const Matrix& hostMatrix);
    void copyFromHost(const Matrix& hostMatrix, bool resizeTarget);
    void copyFromHost(const Matrix& hostMatrix, bool resizeTarget, cudaStream_t stream);
    void copyToHost(Matrix& hostMatrix) const;
    void copyToHost(Matrix& hostMatrix, bool resizeTarget) const;
    void copyToHost(Matrix& hostMatrix, bool resizeTarget, cudaStream_t stream) const;
    cudaTextureObject_t getTextureObject();
};

#endif /* NVMATRIX_H_ */
