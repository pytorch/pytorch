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

#ifndef MEMORY_CUH_H_
#define MEMORY_CUH_H_
#include <map>
#include <cuda.h>
#include <string.h>
#include <vector>
#include <assert.h>

#include <helper_cuda.h>
#include "../../util/include/sync.h"
#include "nvmatrix_kernels.cuh"

#define GPU_ALLOC_FRACTION                  0.95 // Take 95% of available GPU memory
#define HOST_ALLOC_CHUNK                    (1UL << 32)
#define SYNC_ON_FREE                        true
#define BUCKET_TYPE                         unsigned int

// Allocte memory from up to this many buckets higher than desired without subdividing
#define BUCKET_DIVISION_THRESHOLD           1
#define NUM_BUCKETS                         static_cast<int>(sizeof(BUCKET_TYPE) * 8)
#define CLZ(x)                              ((x) == 0 ? (NUM_BUCKETS) : __builtin_clz(x))
#define CEIL_LOG2(x)                        (NUM_BUCKETS - CLZ(x))                      // Ceiling of log base 2 of (x + 1)
#define LOG_FIRST_BUCKET_SIZE               12
#define FIRST_BUCKET_SIZE                   (1 << LOG_FIRST_BUCKET_SIZE)                // First bucket is for 4K bytes
#define GET_ALLOC_BUCKET(size)              (CEIL_LOG2(((size) - 1) >> LOG_FIRST_BUCKET_SIZE))
#define GET_DEALLOC_BUCKET(size)            (CEIL_LOG2((size) >> (1 + LOG_FIRST_BUCKET_SIZE)))
#define GET_BUCKET_SIZE(b)                  (1UL << (LOG_FIRST_BUCKET_SIZE + b))

#define BUCKET_MASK(b)                      (1UL << (b))
#define PREV_BUCKETS_MASK(b)                (BUCKET_MASK(b) - 1)
#define AVAILABLE_NEXT_MASK(b, buckets)     ((buckets) & ~PREV_BUCKETS_MASK(b))

/*
 * Returns the "best-matching" available bucket as defined by policy.
 * The two policies are:
 *
 *      TAKE_FROM_BIGGEST = true: If a bucket in the range
 *      b...{b + BUCKET_DIVISION_THRESHOLD} is available, return the smallest
 *      available bucket in that range. Otherwise return the *biggest* available
 *      bucket greater than or equal to b.
 *
 *      TAKE_FROM_BIGGEST = false: Return the *smallest* available bucket greater
 *      than or equal to b.
 *
 * Returns -1 when no satisfactory bucket is available.
 */
#define TAKE_FROM_BIGGEST                   true
#if TAKE_FROM_BIGGEST
#define GET_AVAILABLE_BUCKET(b, buckets)                                                                 \
                                    (-1 + (((AVAILABLE_NEXT_MASK(b, buckets))                            \
                                             & (PREV_BUCKETS_MASK((b) + 1 + BUCKET_DIVISION_THRESHOLD))) \
        /* Smallest bucket >= b */         ? __builtin_ffs(AVAILABLE_NEXT_MASK(b, buckets))              \
        /* Biggest bucket >= b */          : CEIL_LOG2(AVAILABLE_NEXT_MASK(b, buckets))))
#else
#define GET_AVAILABLE_BUCKET(b, buckets)    __builtin_ffs(AVAILABLE_NEXT_MASK(b, buckets))
#endif

/*
 * Bit get/set/clear.
 */
#define GET_BIT(x, bit)             ((x) & (1 << (bit)))
#define SET_BIT(x, bit)             ((x) |= (1 << (bit)))
#define CLEAR_BIT(x, bit)           ((x) &= ~(1 << (bit)))

typedef struct __align__(512) {
    char data;
} DataType;

#define SIZE_ROUNDUP(size) (sizeof(DataType) * DIVUP((size), sizeof(DataType)))

class MemorySegment {
    friend class FastMemoryManager;
protected:
    DataType* _data;
    size_t _size;
    int _deviceID;
    // Resizes itself to _size - size and
    // returns pointer to new memory segment
    MemorySegment* subdivide(size_t size) {
        assert(size < _size);
//        assert(size % sizeof(DataType) == 0);
        _size -= size;
        return new MemorySegment(_data + _size / sizeof(DataType), size, _deviceID);
    }

    inline size_t getSize() const {
        return _size;
    }
public:
    MemorySegment(DataType* data, size_t size, int deviceID) : _data(data), _size(size), _deviceID(deviceID) {
        assert(size % sizeof(DataType) == 0);
    }
    // In some cases size is irrelevant
    template<typename T> MemorySegment(T* data) : _data(reinterpret_cast<DataType*>(data)), _size(0), _deviceID(-1) {

    }

    template <class T /*= DataType*/>
    inline T* getData() const {
        return reinterpret_cast<T*>(_data);
    }

    template <class T /*= DataType*/>
    inline T** getDataPtr() {
        return reinterpret_cast<T**>(&_data);
    }

    inline int getDeviceID() const {
        return _deviceID;
    }
};

class MemoryManager {
protected:
    static Lock _globalLock;
public:
    virtual MemoryManager* init() = 0;
    virtual MemorySegment* malloc(size_t size) = 0;
    virtual void free(MemorySegment* mem) = 0;
    virtual ~MemoryManager() {

    }
};

class FastMemoryManager : public MemoryManager {
protected:
    int _deviceID;
    Lock _lock;
    DataType* _data;
    size_t _size;
    BUCKET_TYPE _buckets; // Bucket availability bit vector
    std::vector<std::vector<MemorySegment*> > _freeSegments; // bucket idx -> vector of segments

    static std::map<int, MemoryManager*> _memoryManagers;

    virtual void allocateInitialSegment() {
        assert(_deviceID >= 0);
        assert(FIRST_BUCKET_SIZE % sizeof(DataType) == 0);
        checkCudaErrors(cudaSetDevice(_deviceID));
        size_t memFree, memTotal;
        checkCudaErrors(cudaMemGetInfo(&memFree, &memTotal));
        _size = sizeof(DataType) * (size_t(round(double(memFree) * GPU_ALLOC_FRACTION)) / sizeof(DataType));
        printf("FastMemoryManager[%d] allocating %lu-byte initial segment\n", _deviceID, _size);
        checkCudaErrors(cudaMalloc(&_data, _size));
    }

    virtual void freeInitialSegment() {
        checkCudaErrors(cudaFree(_data));
    }

public:
    static MemoryManager& getInstance(int deviceID);
    static void destroyInstance(int deviceID);

    FastMemoryManager(int deviceID) : _deviceID(deviceID), _data(NULL), _size(0), _buckets(0) {
    }

    ~FastMemoryManager() {
        freeInitialSegment();
        for (int i = 0; i < _freeSegments.size(); ++i) {
            for (int j = 0; j < _freeSegments[i].size(); ++j) {
                delete _freeSegments[i][j];
            }
        }
    }

    virtual MemoryManager* init() {
        allocateInitialSegment();

        for (int i = 0; i < NUM_BUCKETS; ++i) {
            _freeSegments.push_back(std::vector<MemorySegment*>());
        }
        int bucket = GET_DEALLOC_BUCKET(_size);
        SET_BIT(_buckets, bucket);
        _freeSegments[bucket].push_back(new MemorySegment(_data, _size, _deviceID));
        return this;
    }

    MemorySegment* malloc(size_t size) {
        assert(size > 0);
        int requestedBucket = GET_ALLOC_BUCKET(size);
        _lock.acquire();

        int bucket = GET_AVAILABLE_BUCKET(requestedBucket, _buckets);
//        if (bucket - requestedBucket > BUCKET_DIVISION_THRESHOLD) {
//            printf("MemoryManager[%d] requested size: %lu, requested bucket: %d, available bucket: %d\n", _deviceID, size, requestedBucket, bucket);
//        }

        assert(bucket >= requestedBucket); // Out of memory

        MemorySegment* sourceSegment = _freeSegments[bucket].back();
        MemorySegment* ret = sourceSegment;
        if (bucket - requestedBucket > BUCKET_DIVISION_THRESHOLD) { // We got a much bigger chunk than we wanted
            ret = sourceSegment->subdivide(GET_BUCKET_SIZE(requestedBucket));
            int newSrcBucket = GET_DEALLOC_BUCKET(sourceSegment->getSize());
            if (newSrcBucket != bucket) {
                _freeSegments[bucket].pop_back();
                _freeSegments[newSrcBucket].push_back(sourceSegment);
                SET_BIT(_buckets, newSrcBucket);
            }
        } else {
            _freeSegments[bucket].pop_back();
        }
        if (_freeSegments[bucket].size() == 0) {
            CLEAR_BIT(_buckets, bucket);
        }
        _lock.release();
        return ret;
    }

    void free(MemorySegment* mem) {
        assert(mem != NULL);
        assert(mem->getSize() >= FIRST_BUCKET_SIZE);
        int bucket = GET_DEALLOC_BUCKET(mem->getSize());
        // Synchronize for safety, so that we don't free memory that's being used. Not synchronizing
        // could potentially cause a problem if we re-allocate the just-freed chunk and attempt to
        // use it in a different stream.
        if (SYNC_ON_FREE) {
            int d;
            checkCudaErrors(cudaGetDevice(&d));
            checkCudaErrors(cudaSetDevice(mem->getDeviceID()));
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaSetDevice(d));
        }
        _lock.acquire();
        _freeSegments[bucket].push_back(mem);
        SET_BIT(_buckets, bucket);
//        printf("MemoryManager[%d] Freed segment of size %lu into bucket %lu\n", _deviceID, mem->getSize(), bucket);
        _lock.release();
    }
};

class FastHostMemoryManager : public FastMemoryManager {
protected:
    static MemoryManager* _memoryManager;
    void allocateInitialSegment() {
        _size = HOST_ALLOC_CHUNK;
        checkCudaErrors(cudaHostAlloc(&_data, _size, cudaHostAllocPortable));
    }
    void freeInitialSegment () {
        checkCudaErrors(cudaFreeHost(_data));
    }
public:
    FastHostMemoryManager() : FastMemoryManager(DEVICE_HOST) {
    }

    static MemoryManager& getInstance();
    static void destroyInstance();
};

class CUDAMemoryManager : public MemoryManager {
protected:
    static MemoryManager* _memoryManager;

    virtual void _malloc(DataType** data, size_t size) {
        checkCudaErrors(cudaMalloc(data, size));
    }
    virtual void _free(MemorySegment* mem) {
        checkCudaErrors(cudaFree(mem->getData<DataType>()));
    }
public:
    static MemoryManager& getInstance(int deviceID);
    static void destroyInstance(int deviceID);
    CUDAMemoryManager() {
    }

    MemoryManager* init() {
        return this;
    }

    MemorySegment* malloc(size_t size) {
        MemorySegment* seg = new MemorySegment(reinterpret_cast<DataType*>(NULL));
        DataType** data = seg->getDataPtr<DataType>();
        _malloc(data, size);
        return seg;
    }

    void free(MemorySegment* mem) {
        assert(mem != NULL);
        _free(mem);
        delete mem;
    }
};

class CUDAHostMemoryManager : public CUDAMemoryManager {
protected:
    static MemoryManager* _memoryManager;
    void _free(MemorySegment* mem) {
        checkCudaErrors(cudaFreeHost(mem->getData<DataType>()));
    }
    void _malloc(DataType** data, size_t size) {
        checkCudaErrors(cudaHostAlloc(data, size, cudaHostAllocPortable));
    }
public:
    static MemoryManager& getInstance();
    static void destroyInstance();
    CUDAHostMemoryManager() : CUDAMemoryManager() {

    }
};
#endif /* MEMORY_CUH_H_ */
