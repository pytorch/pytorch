///////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "cnmem.h"
#include <cstddef>
#include <vector>
#include <cuda_runtime_api.h>

#if !defined(WIN32) && defined(_MSC_VER)
#define WIN32
#endif

#ifdef WIN32
#include <Windows.h>
#else
#include <pthread.h>
#endif

#define CNMEM_GRANULARITY 512

///////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" const char* cnmemGetErrorString(cnmemStatus_t status) {
    switch(status) {
        case CNMEM_STATUS_SUCCESS: return "CNMEM_STATUS_SUCCESS";
        case CNMEM_STATUS_CUDA_ERROR: return "CNMEM_STATUS_CUDA_ERROR";
        case CNMEM_STATUS_INVALID_ARGUMENT: return "CNMEM_STATUS_INVALID_ARGUMENT";
        case CNMEM_STATUS_NOT_INITIALIZED: return "CNMEM_STATUS_NOT_INITIALIZED";
        case CNMEM_STATUS_OUT_OF_MEMORY: return "CNMEM_STATUS_OUT_OF_MEMORY";
        default: return "CNMEM_STATUS_UNKNOWN_ERROR";
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#if 0
#ifdef WIN32
#define CNMEM_DEBUG_ERROR(...) do { \
    fprintf(stderr, "Error at line: %d\n", __LINE__); \
    fprintf(stderr, __VA_ARGS__); \
} while(0)
#else
#include <execinfo.h>
static inline void printBacktrace() {
    void *stackBuffer[64]; 
    int numAddresses = backtrace((void**) &stackBuffer, 64); 
    char **addresses = backtrace_symbols(stackBuffer, numAddresses); 
    for( int i = 0 ; i < numAddresses ; ++i ) { 
        fprintf(stderr, "[%2d]: %s\n", i, addresses[i]); 
    } 
    free(addresses); 
}
#define CNMEM_DEBUG_ERROR(...) do { \
    fprintf(stderr, "Error at line: %d\n", __LINE__); \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "Backtrace:\n"); \
    printBacktrace(); \
} while(0)
#endif
#else
#define CNMEM_DEBUG_ERROR(...)
#endif

#if 0
#define CNMEM_DEBUG_INFO printf
#else
#define CNMEM_DEBUG_INFO(...)
#endif

#if 0 // Enable/disable assertions
#include <cassert>
#define CNMEM_ASSERT assert
#else
#define CNMEM_ASSERT(...)
#endif

#define CNMEM_CHECK_TRUE(cond, error) do { \
    if( !(cond) ) { \
        CNMEM_DEBUG_ERROR("CNMEM_CHECK_TRUE evaluates to false\n"); \
        return error; \
    } \
} while(0) 

#define CNMEM_CHECK(call) do { \
    cnmemStatus_t status = (call); \
    if( status != CNMEM_STATUS_SUCCESS ) { \
        CNMEM_DEBUG_ERROR("CNMEM_CHECK failed with status \"%s\"\n", \
                cnmemGetErrorString(status)); \
        return status; \
    } \
} while(0)

#define CNMEM_CHECK_OR_UNLOCK(call, mutex) do { \
    cnmemStatus_t status = (call); \
    if( status != CNMEM_STATUS_SUCCESS ) { \
        CNMEM_DEBUG_ERROR("CNMEM_CHECK_OR_UNLOCK failed with status \"%s\"\n", \
                cnmemGetErrorString(status)); \
        (mutex).unlock(); \
        return status; \
    } \
} while(0)

#define CNMEM_CHECK_CUDA(call) do { \
    cudaError_t cudaError = (call); \
    if( cudaError == cudaErrorMemoryAllocation ) { \
        CNMEM_DEBUG_ERROR("CNMEM_CHECK_CUDA failed with CUDA error \"%s\"\n", \
                cudaGetErrorString(cudaError)); \
        return CNMEM_STATUS_OUT_OF_MEMORY; \
    } \
    else if( cudaError != cudaSuccess ) { \
        CNMEM_DEBUG_ERROR("CNMEM_CHECK_CUDA failed with CUDA error \"%s\"\n", \
                cudaGetErrorString(cudaError)); \
        return CNMEM_STATUS_CUDA_ERROR; \
    } \
} while(0)

#define CNMEM_CHECK_CUDA_OR_UNLOCK(call, mutex) do { \
    cudaError_t cudaError = (call); \
    if( cudaError == cudaErrorMemoryAllocation ) { \
        CNMEM_DEBUG_ERROR("CNMEM_CHECK_CUDA_OR_UNLOCK failed with CUDA error \"%s\"\n", \
                cudaGetErrorString(cudaError)); \
        (mutex).unlock(); \
        return CNMEM_STATUS_OUT_OF_MEMORY; \
    } \
    else if( cudaError != cudaSuccess ) { \
        CNMEM_DEBUG_ERROR("CNMEM_CHECK_CUDA_OR_UNLOCK failed with CUDA error \"%s\"\n", \
                cudaGetErrorString(cudaError)); \
        (mutex).unlock(); \
        return CNMEM_STATUS_CUDA_ERROR; \
    } \
} while(0)

#ifdef WIN32
#define CNMEM_CHECK_WIN32(call, error_code) do { \
    SetLastError(0); /* Clean the flag. */ \
    call; \
    DWORD status = GetLastError(); \
    if( status ) \
        return error_code; \
} while(0)
#else
#define CNMEM_CHECK_PTHREAD(call, error_code) do { \
    int status = call; \
    if( status ) { \
        CNMEM_DEBUG_ERROR("CNMEM_CHECK_PTHREAD failed with status %d\n", status); \
        return error_code; \
    } \
} while(0)
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cnmem {

static inline std::size_t ceilInt(std::size_t m, std::size_t n) {
    CNMEM_ASSERT(n > 0);
    return (m + n-1) / n * n;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

class Mutex {
#ifdef WIN32
    mutable CRITICAL_SECTION mCriticalSection;
#else
    pthread_mutex_t  mMutex;
#endif

public:
    /// Initialize the mutex.
    cnmemStatus_t initialize();
    /// Finalize the mutex.
    cnmemStatus_t finalize();
    /// Lock the mutex.
    cnmemStatus_t lock() const;
    /// Unlock the mutex.
    cnmemStatus_t unlock() const;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Mutex::initialize() {
#ifdef WIN32
    CNMEM_CHECK_WIN32(InitializeCriticalSection((CRITICAL_SECTION*) &mCriticalSection), CNMEM_STATUS_UNKNOWN_ERROR);
#else
#if 0
    pthread_mutexattr_t attr;
    CNMEM_CHECK_PTHREAD(pthread_mutexattr_init(&attr), CNMEM_STATUS_UNKNOWN_ERROR);
    CNMEM_CHECK_PTHREAD(pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE), CNMEM_STATUS_UNKNOWN_ERROR);
    CNMEM_CHECK_PTHREAD(pthread_mutex_init(&mMutex, &attr), CNMEM_STATUS_UNKNOWN_ERROR);
#else
    CNMEM_CHECK_PTHREAD(pthread_mutex_init(&mMutex, NULL), CNMEM_STATUS_UNKNOWN_ERROR);
#endif
#endif
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Mutex::finalize() {
#ifdef WIN32
    CNMEM_CHECK_WIN32(DeleteCriticalSection((CRITICAL_SECTION*) &mCriticalSection), CNMEM_STATUS_UNKNOWN_ERROR);
#else
    CNMEM_CHECK_PTHREAD(pthread_mutex_destroy(&mMutex), CNMEM_STATUS_UNKNOWN_ERROR);
#endif
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Mutex::lock() const {
#ifdef WIN32
    CNMEM_CHECK_WIN32(EnterCriticalSection(&mCriticalSection), CNMEM_STATUS_UNKNOWN_ERROR);
#else
    CNMEM_CHECK_PTHREAD(pthread_mutex_lock((pthread_mutex_t*) &mMutex), CNMEM_STATUS_UNKNOWN_ERROR);
#endif
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Mutex::unlock() const {
#ifdef WIN32
    CNMEM_CHECK_WIN32(LeaveCriticalSection(&mCriticalSection), CNMEM_STATUS_UNKNOWN_ERROR);
#else
    CNMEM_CHECK_PTHREAD(pthread_mutex_unlock((pthread_mutex_t*) &mMutex), CNMEM_STATUS_UNKNOWN_ERROR);
#endif
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

class Block {
    /// The pointer to the memory region on the device. 
    char *mData;
    /// The size of the memory buffer.
    std::size_t mSize;
    /// The prev/next blocks in the linked list of blocks.
    Block *mNext;
    /// Is it a head node (i.e. a node obtained from parent->allocate or cudaMalloc).
    bool mIsHead;

public:
    /// Create a block.
    Block(char *data, std::size_t size, Block *next, bool isHead)
        : mData(data)
        , mSize(size)
        , mNext(next)
        , mIsHead(isHead) {
    }
    
    /// The data.
    inline const char* getData() const { return mData; }
    /// The data (mutable).
    inline char* getData() { return mData; }
    
    /// The size of the block.
    inline std::size_t getSize() const { return mSize; }

    /// The next block in the linked list.
    inline const Block* getNext() const { return mNext; }
    /// The next block in the linked list (mutable).
    inline Block* getNext() { return mNext; }
    
    /// Is it a head block.
    inline bool isHead() const { return mIsHead; }

    /// Change the next block.
    inline void setNext(Block *next) { mNext = next; }
    /// Change the size of the block.
    inline void setSize(std::size_t size) { mSize = size; }
    /// Set the head flag.
    inline void setHeadFlag(bool isHead) { mIsHead = isHead; }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

class Manager {

    /// The parent manager.
    Manager *mParent;
    /// The children managers.
    std::vector<Manager*> mChildren;
    /// The GPU device where the memory is allocated.
    int mDevice;
    /// The stream this manager is associated with. It could be NULL.
    cudaStream_t mStream;
    /// Is the stream blocking?
    bool mIsStreamBlocking;
    /// The list of used blocks.
    Block *mUsedBlocks;
    /// The list of free blocks.
    Block *mFreeBlocks;
    /// The managed memory size.
    std::size_t mSize;
    /// The flags.
    unsigned mFlags;
    /// To support multi-threading. Each manager has its own mutex.
    Mutex mMutex;

public:
    /// Create an unitialized manager.
    Manager();
    /// Dtor.
    ~Manager();

    /// Allocate a block of memory.
    cnmemStatus_t allocate(void *&ptr, std::size_t size, bool isBlocking = true);
    /// Release a block of memory.
    cnmemStatus_t release(void *ptr);
    /// Release memory. It returns true if we have no memory leak.
    cnmemStatus_t releaseAllUnsafe();
    /// Reserve memory for a manager.
    cnmemStatus_t reserve(std::size_t size);
    /// Steal memory from another manager.
    cnmemStatus_t stealUnsafe(void *&ptr, std::size_t size);

    /// Print the full memory state.
    cnmemStatus_t printMemoryState(FILE *file) const;

    /// The amount of used memory.
    inline cnmemStatus_t getUsedMemoryUnsafe(std::size_t &usedMemory) const { 
        return getMemoryUnsafe(usedMemory, mUsedBlocks); 
    }
    /// The amount of used memory.
    inline cnmemStatus_t getFreeMemoryUnsafe(std::size_t &freeMemory) const { 
        return getMemoryUnsafe(freeMemory, mFreeBlocks); 
    }
    
    /// Get a specific child based on the stream id. 
    cnmemStatus_t getChildFromStream(Manager *&manager, cudaStream_t stream) const;
    /// Get a specific child based on the stream id. 
    cnmemStatus_t getChild(Manager *&manager, std::size_t i) const;
    /// Add a new child.
    cnmemStatus_t addChild(Manager *manager);
    /// The number of children.
    cnmemStatus_t getNumChildren(std::size_t &numChildren) const;

    /// The associated device.
    inline int getDevice() const { return mDevice; }
    /// The flags.
    inline unsigned getFlags() const { return mFlags; }
    /// Get the mutex.
    inline const Mutex* getMutex() const { return &mMutex; }
    /// The size allocated to that manager.
    inline std::size_t getSize() const { return mSize; }
    /// The CUDA stream.
    inline cudaStream_t getStream() const { return mStream; }
    
    /// Define the parent.
    inline void setParent(Manager *parent) { mParent = parent; }
    /// Define the device.
    inline void setDevice(int device) { mDevice = device; }
    /// Define the stream.
    inline cnmemStatus_t setStream(cudaStream_t stream) { 
        mStream = stream; 
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
        mIsStreamBlocking = false;
#elif CUDART_VERSION < 5050
        mIsStreamBlocking = true;
#else
        unsigned flags = 0;
        CNMEM_CHECK_CUDA(cudaStreamGetFlags(mStream, &flags));
        mIsStreamBlocking = !mStream || !(flags & cudaStreamNonBlocking);
#endif
        return CNMEM_STATUS_SUCCESS;
    }
    /// Define the flags.
    inline void setFlags(unsigned flags) { mFlags = flags; }
    
private:
    /// The member functions below which are marked "Unsafe" are not thread-safe when called on a
    /// same Manager object. Make sure they are called by a single thread in that case.

    /// Allocate a new block and add it to the free list.
    cnmemStatus_t allocateBlockUnsafe(Block *&curr, Block *&prev, std::size_t size);
    /// Release a block from the active list.
    cnmemStatus_t releaseBlockUnsafe(Block *curr, Block *prev);
    /// Find the best free node based on the size.
    cnmemStatus_t findBestBlockUnsafe(Block *&curr, Block *&prev, std::size_t size);
    /// Extract a node from the list of free blocks.
    cnmemStatus_t extractBlockUnsafe(Block *curr, Block *prev, std::size_t size, bool stolen);
    
    /// Give a free block from that manager.
    cnmemStatus_t giveBlockUnsafe(void *&data, std::size_t &dataSize, std::size_t size);
    /// Steal a block from another manager.
    cnmemStatus_t stealBlockUnsafe(void *&data, std::size_t &dataSize, std::size_t size);
    
    /// The memory consumption of a list.
    cnmemStatus_t getMemoryUnsafe(std::size_t &memSize, const Block *head) const;
    /// Print an internal linked list.
    cnmemStatus_t printListUnsafe(FILE *file, const char *name, const Block *head) const;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

Manager::Manager()
    : mParent(NULL)
    , mChildren()
    , mDevice(-1)
    , mStream(NULL)
    , mIsStreamBlocking(false)
    , mUsedBlocks(NULL)
    , mFreeBlocks(NULL)
    , mSize(0)
    , mFlags(CNMEM_FLAGS_DEFAULT)
    , mMutex() {

    mMutex.initialize();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

Manager::~Manager() {
    if( mDevice == -1 || cudaSetDevice(mDevice) != cudaSuccess ) { // Invalid device, skip it.
        return;
    }
    releaseAllUnsafe();
    mMutex.finalize();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::addChild(Manager *manager) {
    CNMEM_CHECK(mMutex.lock());
    mChildren.push_back(manager);
    CNMEM_CHECK(mMutex.unlock());
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::allocate(void *&ptr, std::size_t size, bool isBlocking) {
    CNMEM_CHECK(mMutex.lock());

    // If the client is not blocking, we have to explicitly synchronize before giving one buffer.
    if( !isBlocking ) {
        CNMEM_CHECK_CUDA_OR_UNLOCK(cudaStreamSynchronize(mStream), mMutex);
    }

    // Find the best fit.
    Block *best = NULL, *prev = NULL;
    CNMEM_CHECK_OR_UNLOCK(findBestBlockUnsafe(best, prev, size), mMutex);

    // If there's no block left in the list of free blocks (with a sufficient size). Request a new block. 
    if( best == NULL && !(mFlags & CNMEM_FLAGS_CANNOT_GROW) ) {
        CNMEM_CHECK_OR_UNLOCK(allocateBlockUnsafe(best, prev, size), mMutex);
    }
    
    // Make sure we do have a block or quit.
    if( !best ) {
        ptr = NULL;
        CNMEM_CHECK(mMutex.unlock());
        return CNMEM_STATUS_OUT_OF_MEMORY;
    }

    // Split the free block if needed.
    CNMEM_CHECK_OR_UNLOCK(extractBlockUnsafe(best, prev, size, false), mMutex);

    // Push the node to the list of used nodes.
    best->setNext(mUsedBlocks);
    mUsedBlocks = best;

    // Return the new pointer into memory.
    ptr = mUsedBlocks->getData();
    CNMEM_CHECK(mMutex.unlock());
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::allocateBlockUnsafe(Block *&curr, Block *&prev, std::size_t size) {
    // Reset the outputs.
    curr = prev = NULL;

    // Try to allocate data from the parent or the device.
    void *data = NULL;
    if( mParent ) {
        CNMEM_CHECK(mParent->allocate(data, size, mIsStreamBlocking));
    }
    else {
        CNMEM_DEBUG_INFO("cudaMalloc(%lu)\n", size);
        CNMEM_CHECK_CUDA(cudaMalloc(&data, size));
        CNMEM_DEBUG_INFO(">> returned address=0x%016lx\n", (size_t) data);
    }
    
    // If it failed, there's an unexpected issue.
    CNMEM_ASSERT(data);

    // We have data, we now need to add it to the list of free nodes. We keep the list sorted.
    Block *next = mFreeBlocks;
    for( ; next && next->getData() < data ; next = next->getNext() ) {
        prev = next;
    }
    curr = new Block((char*) data, size, next, true);
    if( !curr ) {
        return CNMEM_STATUS_OUT_OF_MEMORY;
    }
    if( prev ) {
        prev->setNext(curr);
    }
    else {
        mFreeBlocks = curr;
    }

    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::extractBlockUnsafe(Block *curr, Block *prev, std::size_t size, bool stolen) {
    // We have two cases: 1/ It is the right size so we keep it or 2/ it is too large and we split the node.
    Block *next;
    if( curr->getSize() == size ) {
        next = curr->getNext();
    }
    else {
        std::size_t remaining = curr->getSize()-size;
        Block *newBlock = new Block(curr->getData() + size, remaining, curr->getNext(), stolen);
        if( !newBlock ) {
            return CNMEM_STATUS_OUT_OF_MEMORY;
        }
        next = newBlock;
        curr->setSize(size);
    }
    
    // Redo the "branching" in the nodes.
    if( prev ) {
        prev->setNext(next);
    }
    else {
        mFreeBlocks = next;
    }
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::findBestBlockUnsafe(Block *&best, Block *&prev, std::size_t size) {
    best = NULL, prev = NULL;
    for( Block *temp = mFreeBlocks, *tempPrev = NULL ; temp ; temp = temp->getNext() ) {
        if( temp->getSize() >= size && (!best || temp->getSize() < best->getSize()) ) {
            best = temp;
            prev = tempPrev;
        }
        tempPrev = temp;
    }
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::getChildFromStream(Manager *&manager, cudaStream_t stream) const {
    CNMEM_CHECK(mMutex.lock());
    std::size_t i = 0, numChildren = mChildren.size();
    for( ; i < numChildren ; ++i ) {
        if( mChildren[i]->mStream == stream ) {
            manager = mChildren[i];
            break;
        }
    }
    CNMEM_CHECK(mMutex.unlock());
    return i < numChildren ? CNMEM_STATUS_SUCCESS : CNMEM_STATUS_INVALID_ARGUMENT;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::getChild(Manager *&manager, std::size_t i) const {
    CNMEM_CHECK(mMutex.lock());
    if( i >= mChildren.size() ) {
        CNMEM_CHECK(mMutex.unlock());
        return CNMEM_STATUS_INVALID_ARGUMENT;
    }
    manager = mChildren[i];

    CNMEM_CHECK(mMutex.unlock());
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::getMemoryUnsafe(std::size_t &size, const Block *head) const {
    size = 0;
    for( Block *curr = (Block*) head ; curr ; curr = curr->getNext() ) {
        size += curr->getSize();
    }
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#if 0
cnmemStatus_t Manager::getMemory(std::size_t &size, const Block *head) const {
    CNMEM_CHECK(mMutex.lock());
    CNMEM_CHECK_OR_UNLOCK(getMemoryUnsafe(size, head));
    CNMEM_CHECK(mMutex.unlock());
    return status;
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::getNumChildren(std::size_t &numChildren) const {
    CNMEM_CHECK(mMutex.lock());
    numChildren = mChildren.size();
    CNMEM_CHECK(mMutex.unlock());
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::giveBlockUnsafe(void *&blockData, std::size_t &blockSize, std::size_t size) {
    // Make sure the block is not in use any more. It could be too coarse grain and we may change 
    // it in the future.
    CNMEM_CHECK_CUDA(cudaStreamSynchronize(mStream));
    
    // Init the returned values to 0.
    blockData = NULL;
    blockSize = 0;
    
    // Find the best node to steal and reserve it.
    Block *best = NULL, *prev = NULL;
    CNMEM_CHECK(findBestBlockUnsafe(best, prev, size));
    if( !best ) {
        return CNMEM_STATUS_OUT_OF_MEMORY;
    }
    CNMEM_CHECK(extractBlockUnsafe(best, prev, size, true));
    blockData = best->getData();
    blockSize = best->getSize();
    
    // Release the memory used by that block.
    delete best;
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::printListUnsafe(FILE *file, const char *name, const Block *head) const {
    std::size_t size = 0;
    for( Block *curr = (Block*) head; curr; curr = curr->getNext() ) {
        size += curr->getSize();
    }
    fprintf(file, "| list=\"%s\", size=%lu\n", name, size);
    for( Block *curr = (Block*) head ; curr ; curr = curr->getNext() ) {
        fprintf(file, "| | node=0x%016lx, data=0x%016lx, size=%lu, next=0x%016lx, head=%2lu\n", 
            (std::size_t) curr, 
            (std::size_t) curr->getData(),
            (std::size_t) curr->getSize(),
            (std::size_t) curr->getNext(),
            (std::size_t) curr->isHead ());
    }
    fprintf(file, "|\n");
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::printMemoryState(FILE *file) const {
    CNMEM_CHECK(mMutex.lock());
    std::size_t streamCode = (std::size_t) mStream;
    std::size_t usedMemory, freeMemory;
    CNMEM_CHECK_OR_UNLOCK(getUsedMemoryUnsafe(usedMemory), mMutex);
    CNMEM_CHECK_OR_UNLOCK(getFreeMemoryUnsafe(freeMemory), mMutex);

    fprintf(file, ">> [%s] device=%d, stream=0x%016lx, used=%luB, free=%luB\n", 
            mParent ? "child" : "root",
            mDevice, 
            streamCode,
            usedMemory,
            freeMemory);
    CNMEM_CHECK_OR_UNLOCK(printListUnsafe(file, "used", mUsedBlocks), mMutex);
    CNMEM_CHECK_OR_UNLOCK(printListUnsafe(file, "free", mFreeBlocks), mMutex);
    fprintf(file, "\n");
    CNMEM_CHECK(mMutex.unlock());

    if( mParent ) {
        CNMEM_CHECK(mParent->printMemoryState(file));
    }
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::release(void *ptr) {
    // Skip if ptr is NULL.
    if( ptr == NULL ) {
        return CNMEM_STATUS_SUCCESS;
    }
        
    // Lock to make sure only one thread execute that fragment of code.
    CNMEM_CHECK(mMutex.lock());
    
    // Find the node in the list of used blocks.
    Block *curr = mUsedBlocks, *prev = NULL;
    for( ; curr && curr->getData() != ptr ; curr = curr->getNext() ) {
        prev = curr;
    }
    
    // Make sure we have found a node.
    if( curr == NULL ) {
        CNMEM_CHECK(mMutex.unlock());
        return CNMEM_STATUS_INVALID_ARGUMENT;
    }

    // We have the node so release it.
    cnmemStatus_t result = releaseBlockUnsafe(curr, prev);
    CNMEM_CHECK(mMutex.unlock());
    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::releaseAllUnsafe() {
    // Destroy the children if any.
    for( std::size_t i = 0; i < mChildren.size(); ++i ) {
        Manager *child = mChildren[i];
        CNMEM_CHECK(child->releaseAllUnsafe());
        delete child;
    }
    mChildren.clear();

    // Destroy used blocks. It's a kind of panic mode to avoid leaks. NOTE: Do that only with roots!!!
    if( !mParent ) {
        while( mUsedBlocks ) {
            CNMEM_CHECK(releaseBlockUnsafe(mUsedBlocks, NULL));
        }
    }

    // We should be having only free blocks that are head blocks. Release those blocks.
    while( mFreeBlocks ) {
        if( mParent ) {
            CNMEM_CHECK(mParent->release(mFreeBlocks->getData()));
        }
        else if( mFreeBlocks->isHead() ) {
            void *data = mFreeBlocks->getData();
            CNMEM_DEBUG_INFO("cudaFree(%lu, 0x%016lx)\n", mFreeBlocks->getSize(), (size_t) data);
            CNMEM_CHECK_CUDA(cudaFree(data));
            CNMEM_DEBUG_INFO(">> success\n");
        }
        Block *block = mFreeBlocks;
        mFreeBlocks = mFreeBlocks->getNext();
        delete block;
    }

    // We shouldn't have any used block left. Or, it means the user is causing memory leaks!
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::releaseBlockUnsafe(Block *curr, Block *prev) {
    // The current node cannot be NULL!
    CNMEM_ASSERT(curr != NULL);
    
    // Change the connection of the node.
    if( prev ) {
        prev->setNext(curr->getNext());
    }
    else {
        mUsedBlocks = curr->getNext();
    }
        
    // Find the location where this block should be added to the free list.
    prev = NULL;
    Block *iter = mFreeBlocks;
    for( ; iter && iter->getData() < curr->getData() ; iter = iter->getNext() ) {
        prev = iter;
    }
    
    // Keep track of the successor of pred. We may lose track of it in the following "else".
    Block *next = prev ? prev->getNext() : mFreeBlocks;
    
    // We first check if we can merge the block with its predecessor in the list and curr can be merged.
    if( prev && prev->getData() + prev->getSize() == curr->getData() && !curr->isHead() ) {
        prev->setSize(prev->getSize() + curr->getSize());
        delete curr;
        curr = prev;
    }
    else if( prev ) {
        prev->setNext(curr);
    }
    else {
        mFreeBlocks = curr;
    }
    
    // Check if we can merge curr and next. We can't merge over "cudaMalloc" boundaries.
    if( next && curr->getData() + curr->getSize() == next->getData() && !next->isHead() ) {
        curr->setSize(curr->getSize() + next->getSize());
        curr->setNext(next->getNext());
        delete next;
    }
    else {
        curr->setNext(next);
    }
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::reserve(std::size_t size) {
    CNMEM_CHECK(mMutex.lock());
    Block *curr, *prev;
    CNMEM_CHECK_OR_UNLOCK(allocateBlockUnsafe(curr, prev, size), mMutex);
    mSize = size;
    CNMEM_CHECK(mMutex.unlock());
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::stealUnsafe(void *&stolen, std::size_t size) {
    // If we cannot steal, don't even try.
    if( mFlags & CNMEM_FLAGS_CANNOT_STEAL ) {
        stolen = NULL;
        return CNMEM_STATUS_INVALID_ARGUMENT;
    }

    // The stolen block.
    void *data = NULL; std::size_t dataSize = 0;
    if( !mChildren.empty() ) {
        CNMEM_CHECK(stealBlockUnsafe(data, dataSize, size));
    }
    else if( mParent ) {
        CNMEM_CHECK(mParent->stealBlockUnsafe(data, dataSize, size));
    }
    
    // Make sure we do have a block of memory or quit.
    if( !data ) {
        stolen = NULL;
        return CNMEM_STATUS_OUT_OF_MEMORY;
    }

    // Push the block in the used list.
    mUsedBlocks = new Block((char*) data, dataSize, mUsedBlocks, true);
    if( !mUsedBlocks ) {
        return CNMEM_STATUS_OUT_OF_MEMORY;
    }

    // Return the new pointer into memory.
    stolen = data;
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Manager::stealBlockUnsafe(void *&data, std::size_t &dataSize, ::size_t size) {
    // No block found and no room to grow. Try to steal from a children (if we have any).
    data = NULL;
    for( std::size_t i = 0 ; !data && i < mChildren.size() ; ++i ) {
        Manager *child = mChildren[i];
        if( child->giveBlockUnsafe(data, dataSize, size) == CNMEM_STATUS_SUCCESS ) {
            break;
        }
    }
        
    // If no memory space found, simply return NULL. We have failed to allocate. Quit miserably.
    if( !data ) {
        return CNMEM_STATUS_OUT_OF_MEMORY;
    }

    // We have got a node from a children. We need to update our "used" list before we can do 
    // anything with it.
    Block *curr = mUsedBlocks, *prev = NULL;
    for( ; curr ; curr = curr->getNext() ) { 
        if( curr->getData() <= data && data < curr->getData()+curr->getSize() ) {
            break;
        }
        prev = curr;
    }
    
    // Curr points to the node which contains that memory region.
    CNMEM_ASSERT(curr);

    // If it is exactly the same memory region, we are done!!!
    if( curr->getData() == data && curr->getSize() == dataSize ) {
        return CNMEM_STATUS_SUCCESS;
    }
    
    // Track the blocks before and after curr.
    Block *next = curr->getNext();
    
    // We may have up to 3 blocks.
    std::size_t sizeBefore = (std::size_t) ((char*) data - curr->getData());
    std::size_t sizeAfter = (curr->getSize() - sizeBefore - dataSize);

    // The resulting block.
    Block *result = curr;
    
    // If we have no space between curr->getData and block->getData.
    if( sizeBefore == 0 ) {
        curr->setSize(dataSize);
    }
    else {
        curr->setSize(sizeBefore);
        Block *block = new Block((char*) data, dataSize, next, false);
        if( !block ) {
            return CNMEM_STATUS_OUT_OF_MEMORY;
        }
        curr->setNext(block);
        curr = block;
        data = (char*) data + dataSize;
        dataSize = sizeAfter; 
        result = block;
    }
    
    // We have space at the end so we may need to add a new node.
    if( sizeAfter > 0 ) {
        Block *block = new Block(curr->getData() + curr->getSize(), sizeAfter, next, false);
        if( !block ) {
            return CNMEM_STATUS_OUT_OF_MEMORY;
        }
        curr->setNext(block);
        curr = block;
    }
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

class Context {
    /// Use a magic number to specify that the context is valid.
    enum { CTX_VALID = 0x1f5632a3 };

    /// The reference counting mechanism.
    int mRefCount;
    /// The mutex to increase/decrease the reference counter. TODO: Use atomics.
    Mutex mMutex;
    /// The memory managers.
    std::vector<Manager> mManagers;
    /// The global context.
    static Context *sCtx;
    /// Use a magic number to specify that the context was created.
    static int sCtxCheck;

public:
    /// Ctor.
    Context() : mRefCount(1) { mMutex.initialize(); }
    /// Dtor.
    ~Context();
    /// Get the managers.
    inline std::vector<Manager>& getManagers() { return mManagers; }
    /// Get a single manager associated with a device.
    inline Manager& getManager(int i) { return mManagers[i]; }

    /// Create the global context.
    static cnmemStatus_t create();
    /// Check that the context was created.
    static inline bool check() { return sCtxCheck == CTX_VALID && sCtx; }
    /// Get the global context.
    static Context* get();
    /// Retain.
    static cnmemStatus_t retain();
    /// Release.
    static cnmemStatus_t release();
};

Context *Context::sCtx;
int Context::sCtxCheck;

///////////////////////////////////////////////////////////////////////////////////////////////////

Context::~Context() { 
    int oldDevice;
    cudaGetDevice(&oldDevice);
    for( std::size_t i = 0 ; i < mManagers.size() ; ++i ) {
        if( mManagers[i].getDevice() != -1 ) { // Skip invalid managers.
            cudaSetDevice(mManagers[i].getDevice());
            mManagers[i].releaseAllUnsafe();
        }
    }
    mManagers.clear();
    mMutex.finalize();
    cudaSetDevice(oldDevice);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Context::create() {
    sCtx = new Context;
    sCtxCheck = CTX_VALID;
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

Context* Context::get() {
    CNMEM_ASSERT(Context::check());
    return Context::sCtx;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Context::retain() { 
    CNMEM_CHECK(sCtx->mMutex.lock());
    sCtx->mRefCount++; 
    CNMEM_CHECK(sCtx->mMutex.unlock());
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t Context::release() {
    CNMEM_CHECK(sCtx->mMutex.lock());
    int refCount = --sCtx->mRefCount;
    CNMEM_CHECK(sCtx->mMutex.unlock());

    if( refCount == 0 ) { // Kill the context.
        delete sCtx;
        Context::sCtx = NULL;
        Context::sCtxCheck = 0;
    }
    return CNMEM_STATUS_SUCCESS;
}

} // namespace cnmem

///////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" {

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t cnmemInit(int numDevices, const cnmemDevice_t *devices, unsigned flags) {
    // Make sure we have at least one device declared.
    CNMEM_CHECK_TRUE(numDevices > 0, CNMEM_STATUS_INVALID_ARGUMENT);
    
    // Find the largest ID of the device.
    int maxDevice = 0;
    for( int i = 0 ; i < numDevices ; ++i ) {
        if( devices[i].device > maxDevice ) {
            maxDevice = devices[i].device;
        }
    }

    // Create the global context.
    cnmem::Context::create();
    cnmem::Context *ctx = cnmem::Context::get();
        
    // Allocate enough managers.
    CNMEM_CHECK_TRUE(maxDevice >= 0, CNMEM_STATUS_INVALID_ARGUMENT);
    std::vector<cnmem::Manager> &managers = ctx->getManagers();
    managers.resize(maxDevice+1);

    // Create a root manager for each device and create the children.
    int oldDevice;
    CNMEM_CHECK_CUDA(cudaGetDevice(&oldDevice));
    for( int i = 0 ; i < numDevices ; ++i ) {
        CNMEM_CHECK_CUDA(cudaSetDevice(devices[i].device));
        std::size_t size = devices[i].size;
        if( size == 0 ) {
            cudaDeviceProp props;
            CNMEM_CHECK_CUDA(cudaGetDeviceProperties(&props, devices[i].device));
            size = props.totalGlobalMem / 2;
        }
        CNMEM_CHECK_TRUE(size > 0, CNMEM_STATUS_INVALID_ARGUMENT);
        
        cnmem::Manager &manager = ctx->getManager(devices[i].device);
        manager.setDevice(devices[i].device);
        manager.setFlags(flags);
        
        size = cnmem::ceilInt(size, CNMEM_GRANULARITY);
        CNMEM_CHECK(manager.reserve(size));
        
        for( int j = 0 ; j < devices[i].numStreams ; ++j ) {
            cnmem::Manager *child = new cnmem::Manager;
            child->setParent(&manager);
            child->setDevice(devices[i].device);
            child->setStream(devices[i].streams[j]);
            child->setFlags(flags & ~CNMEM_FLAGS_CANNOT_GROW);
            if( devices[i].streamSizes && devices[i].streamSizes[j] > 0 ) {
                CNMEM_CHECK(child->reserve(devices[i].streamSizes[j]));
            }
            CNMEM_CHECK(manager.addChild(child));
        }
    }
    CNMEM_CHECK_CUDA(cudaSetDevice(oldDevice));
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t cnmemFinalize() {
    CNMEM_CHECK_TRUE(cnmem::Context::check(), CNMEM_STATUS_NOT_INITIALIZED);
    return cnmem::Context::release();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t cnmemRetain() {
    CNMEM_CHECK_TRUE(cnmem::Context::check(), CNMEM_STATUS_NOT_INITIALIZED);
    return cnmem::Context::retain();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t cnmemRelease() {
    CNMEM_CHECK_TRUE(cnmem::Context::check(), CNMEM_STATUS_NOT_INITIALIZED);
    return cnmem::Context::release();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t cnmemRegisterStream(cudaStream_t stream) {
    CNMEM_CHECK_TRUE(cnmem::Context::check(), CNMEM_STATUS_NOT_INITIALIZED);
    CNMEM_CHECK_TRUE(stream, CNMEM_STATUS_INVALID_ARGUMENT);
    
    int device;
    CNMEM_CHECK_CUDA(cudaGetDevice(&device));

    cnmem::Manager &root = cnmem::Context::get()->getManager(device);
    cnmem::Manager *child = new cnmem::Manager;
    child->setParent(&root);
    child->setDevice(device);
    child->setStream(stream);
    child->setFlags(root.getFlags() & ~CNMEM_FLAGS_CANNOT_GROW);
    root.addChild(child);

    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t cnmemMalloc(void **ptr, std::size_t size, cudaStream_t stream) {
    CNMEM_CHECK_TRUE(cnmem::Context::check(), CNMEM_STATUS_NOT_INITIALIZED);
    if( !ptr && !size ) {
        return CNMEM_STATUS_SUCCESS;
    }
    else if( !size ) {
        ptr[0] = NULL;
        return CNMEM_STATUS_SUCCESS;
    }
    CNMEM_CHECK_TRUE(ptr,  CNMEM_STATUS_INVALID_ARGUMENT);
    
    int device;
    CNMEM_CHECK_CUDA(cudaGetDevice(&device));

    cnmem::Manager &root = cnmem::Context::get()->getManager(device);
    cnmem::Manager *manager = &root;
    if( stream ) {
        CNMEM_CHECK(root.getChildFromStream(manager, stream));
    }
    CNMEM_ASSERT(manager);
    
    size = cnmem::ceilInt(size, CNMEM_GRANULARITY);
    cnmemStatus_t result = manager->allocate(ptr[0], size);

    // We failed to allocate but there might still be a buffer available in another manager. Try to 
    // steal it.
    if( result == CNMEM_STATUS_OUT_OF_MEMORY ) {

        // Try to acquire locks on all the children.
        std::size_t numChildren;
        CNMEM_CHECK(root.getNumChildren(numChildren));
        std::vector<const cnmem::Mutex*> mutexes(numChildren);

        std::size_t numLocked = 0;
        for( size_t i = 0 ; i < numChildren ; ++i, ++numLocked ) {
            cnmem::Manager *child;
            CNMEM_CHECK(root.getChild(child, i));
            mutexes[numLocked] = child->getMutex();
            if( mutexes[numLocked]->lock() != CNMEM_STATUS_SUCCESS ) {
                break;
            }
        }

        // One lock failed, quit. Reduce the damage as much as possible, though.
        if( numLocked != numChildren ) {
            for( std::size_t i = 0 ; i < numLocked ; ++i ) {
                cnmemStatus_t lockStatus = mutexes[i]->unlock();
            }
            return CNMEM_STATUS_UNKNOWN_ERROR;
        }

        // Grab the lock on the root, first.
        const cnmem::Mutex *rootMutex = root.getMutex();
        CNMEM_CHECK(rootMutex->lock());

        // We acquired all the lock so we try to steal a node from another child.
        if( numLocked == mutexes.size() ) {
            result = manager->stealUnsafe(ptr[0], size);
        }
        for( std::size_t i = 0 ; i < numLocked ; ++i ) {
            cnmemStatus_t lockStatus = mutexes[i]->unlock();
            if( lockStatus != CNMEM_STATUS_SUCCESS ) { 
                // Starting from now we are panicking!!! One lock failed to be released, we try
                // we others. We could also give up because we are already screwed. I don't know
                // what's best! Comment are welcome.
                result = lockStatus;
            }
        }
        CNMEM_CHECK(rootMutex->unlock());
    }
    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t cnmemFree(void *ptr, cudaStream_t stream) {
    CNMEM_CHECK_TRUE(cnmem::Context::check(), CNMEM_STATUS_NOT_INITIALIZED);
    if( ptr == NULL ) {
        return CNMEM_STATUS_SUCCESS;
    }

    int device;
    CNMEM_CHECK_CUDA(cudaGetDevice(&device));

    cnmem::Manager &root = cnmem::Context::get()->getManager(device);
    cnmem::Manager *manager = &root;
    if( stream ) {
        CNMEM_CHECK(root.getChildFromStream(manager, stream));
    }
    CNMEM_ASSERT(manager);
    return manager->release(ptr);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t cnmemMemGetInfo(size_t *freeMem, size_t *totalMem, cudaStream_t stream) {
    CNMEM_CHECK_TRUE(cnmem::Context::check(), CNMEM_STATUS_NOT_INITIALIZED);
    CNMEM_CHECK_TRUE(totalMem && freeMem, CNMEM_STATUS_INVALID_ARGUMENT);

    int device;
    CNMEM_CHECK_CUDA(cudaGetDevice(&device));
    cnmem::Manager &root = cnmem::Context::get()->getManager(device);
    cnmem::Manager *manager = &root;
    if( stream ) {
        CNMEM_CHECK(root.getChildFromStream(manager, stream));
    }
    CNMEM_ASSERT(manager);

    const cnmem::Mutex *mutex = manager->getMutex();
    CNMEM_CHECK(mutex->lock());
    CNMEM_CHECK_OR_UNLOCK(manager->getFreeMemoryUnsafe(*freeMem), *mutex);
    size_t usedMem;
    CNMEM_CHECK_OR_UNLOCK(manager->getUsedMemoryUnsafe(usedMem), *mutex);
    CNMEM_CHECK(mutex->unlock());
    totalMem[0] = usedMem + freeMem[0];
    return CNMEM_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cnmemStatus_t cnmemPrintMemoryState(FILE *file, cudaStream_t stream) {
    CNMEM_CHECK_TRUE(cnmem::Context::check(), CNMEM_STATUS_NOT_INITIALIZED);

    int device;
    CNMEM_CHECK_CUDA(cudaGetDevice(&device));
    cnmem::Manager &root = cnmem::Context::get()->getManager(device);
    cnmem::Manager *manager = &root;
    if( stream ) {
        CNMEM_CHECK(root.getChildFromStream(manager, stream));
    }
    CNMEM_ASSERT(manager);
    return manager->printMemoryState(file); 
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // extern "C"

