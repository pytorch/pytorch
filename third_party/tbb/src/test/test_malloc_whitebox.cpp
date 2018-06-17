/*
    Copyright (c) 2005-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

/* to prevent loading dynamic TBBmalloc at startup, that is not needed
   for the whitebox test */
#define __TBB_SOURCE_DIRECTLY_INCLUDED 1

// According to C99 standard INTPTR_MIN defined for C++
// iff __STDC_LIMIT_MACROS pre-defined
#define __STDC_LIMIT_MACROS 1

#define HARNESS_TBBMALLOC_THREAD_SHUTDOWN 1

#include "harness.h"
#include "harness_barrier.h"

// To not depends on ITT support stuff
#ifdef DO_ITT_NOTIFY
#undef DO_ITT_NOTIFY
#endif

#define __TBB_MALLOC_WHITEBOX_TEST 1 // to get access to allocator internals
// help trigger rare race condition
#define WhiteboxTestingYield() (__TBB_Yield(), __TBB_Yield(), __TBB_Yield(), __TBB_Yield())

#if __INTEL_COMPILER && __TBB_MIC_OFFLOAD
// 2571 is variable has not been declared with compatible "target" attribute
// 3218 is class/struct may fail when offloaded because this field is misaligned
//         or contains data that is misaligned
    #pragma warning(push)
    #pragma warning(disable:2571 3218)
#endif
#define protected public
#define private public
#include "../tbbmalloc/frontend.cpp"
#undef protected
#undef private
#if __INTEL_COMPILER && __TBB_MIC_OFFLOAD
    #pragma warning(pop)
#endif
#include "../tbbmalloc/backend.cpp"
#include "../tbbmalloc/backref.cpp"

namespace tbbmalloc_whitebox {
    size_t locGetProcessed = 0;
    size_t locPutProcessed = 0;
}
#include "../tbbmalloc/large_objects.cpp"
#include "../tbbmalloc/tbbmalloc.cpp"

const int LARGE_MEM_SIZES_NUM = 10;

class AllocInfo {
    int *p;
    int val;
    int size;
public:
    AllocInfo() : p(NULL), val(0), size(0) {}
    explicit AllocInfo(int sz) : p((int*)scalable_malloc(sz*sizeof(int))),
                                   val(rand()), size(sz) {
        ASSERT(p, NULL);
        for (int k=0; k<size; k++)
            p[k] = val;
    }
    void check() const {
        for (int k=0; k<size; k++)
            ASSERT(p[k] == val, NULL);
    }
    void clear() {
        scalable_free(p);
    }
};

class SimpleBarrier: NoAssign {
protected:
    static Harness::SpinBarrier barrier;
public:
    static void initBarrier(unsigned thrds) { barrier.initialize(thrds); }
};

Harness::SpinBarrier SimpleBarrier::barrier;

class TestLargeObjCache: public SimpleBarrier {
public:
    static int largeMemSizes[LARGE_MEM_SIZES_NUM];

    TestLargeObjCache( ) {}

    void operator()( int /*mynum*/ ) const {
        AllocInfo allocs[LARGE_MEM_SIZES_NUM];

        // push to maximal cache limit
        for (int i=0; i<2; i++) {
            const int sizes[] = { MByte/sizeof(int),
                                  (MByte-2*LargeObjectCache::largeBlockCacheStep)/sizeof(int) };
            for (int q=0; q<2; q++) {
                size_t curr = 0;
                for (int j=0; j<LARGE_MEM_SIZES_NUM; j++, curr++)
                    new (allocs+curr) AllocInfo(sizes[q]);

                for (size_t j=0; j<curr; j++) {
                    allocs[j].check();
                    allocs[j].clear();
                }
            }
        }

        barrier.wait();

        // check caching correctness
        for (int i=0; i<1000; i++) {
            size_t curr = 0;
            for (int j=0; j<LARGE_MEM_SIZES_NUM-1; j++, curr++)
                new (allocs+curr) AllocInfo(largeMemSizes[j]);

            new (allocs+curr)
                AllocInfo((int)(4*minLargeObjectSize +
                                2*minLargeObjectSize*(1.*rand()/RAND_MAX)));
            curr++;

            for (size_t j=0; j<curr; j++) {
                allocs[j].check();
                allocs[j].clear();
            }
        }
    }
};

int TestLargeObjCache::largeMemSizes[LARGE_MEM_SIZES_NUM];

void TestLargeObjectCache()
{
    for (int i=0; i<LARGE_MEM_SIZES_NUM; i++)
        TestLargeObjCache::largeMemSizes[i] =
            (int)(minLargeObjectSize + 2*minLargeObjectSize*(1.*rand()/RAND_MAX));

    for( int p=MaxThread; p>=MinThread; --p ) {
        TestLargeObjCache::initBarrier( p );
        NativeParallelFor( p, TestLargeObjCache() );
    }
}

#if MALLOC_CHECK_RECURSION

class TestStartupAlloc: public SimpleBarrier {
    struct TestBlock {
        void *ptr;
        size_t sz;
    };
    static const int ITERS = 100;
public:
    TestStartupAlloc() {}
    void operator()(int) const {
        TestBlock blocks1[ITERS], blocks2[ITERS];

        barrier.wait();

        for (int i=0; i<ITERS; i++) {
            blocks1[i].sz = rand() % minLargeObjectSize;
            blocks1[i].ptr = StartupBlock::allocate(blocks1[i].sz);
            ASSERT(blocks1[i].ptr && StartupBlock::msize(blocks1[i].ptr)>=blocks1[i].sz
                   && 0==(uintptr_t)blocks1[i].ptr % sizeof(void*), NULL);
            memset(blocks1[i].ptr, i, blocks1[i].sz);
        }
        for (int i=0; i<ITERS; i++) {
            blocks2[i].sz = rand() % minLargeObjectSize;
            blocks2[i].ptr = StartupBlock::allocate(blocks2[i].sz);
            ASSERT(blocks2[i].ptr && StartupBlock::msize(blocks2[i].ptr)>=blocks2[i].sz
                   && 0==(uintptr_t)blocks2[i].ptr % sizeof(void*), NULL);
            memset(blocks2[i].ptr, i, blocks2[i].sz);

            for (size_t j=0; j<blocks1[i].sz; j++)
                ASSERT(*((char*)blocks1[i].ptr+j) == i, NULL);
            Block *block = (Block *)alignDown(blocks1[i].ptr, slabSize);
            ((StartupBlock *)block)->free(blocks1[i].ptr);
        }
        for (int i=ITERS-1; i>=0; i--) {
            for (size_t j=0; j<blocks2[i].sz; j++)
                ASSERT(*((char*)blocks2[i].ptr+j) == i, NULL);
            Block *block = (Block *)alignDown(blocks2[i].ptr, slabSize);
            ((StartupBlock *)block)->free(blocks2[i].ptr);
        }
    }
};

#endif /* MALLOC_CHECK_RECURSION */

#include <deque>

template<int ITERS>
class BackRefWork: NoAssign {
    struct TestBlock {
        BackRefIdx idx;
        char       data;
        TestBlock(BackRefIdx idx_) : idx(idx_) {}
    };
public:
    BackRefWork() {}
    void operator()(int) const {
        size_t cnt;
        // it's important to not invalidate pointers to the contents of the container
        std::deque<TestBlock> blocks;

        // for ITERS==0 consume all available backrefs
        for (cnt=0; !ITERS || cnt<ITERS; cnt++) {
            BackRefIdx idx = BackRefIdx::newBackRef(/*largeObj=*/false);
            if (idx.isInvalid())
                break;
            blocks.push_back(TestBlock(idx));
            setBackRef(blocks.back().idx, &blocks.back().data);
        }
        for (int i=0; i<cnt; i++)
            ASSERT((Block*)&blocks[i].data == getBackRef(blocks[i].idx), NULL);
        for (int i=cnt-1; i>=0; i--)
            removeBackRef(blocks[i].idx);
    }
};

class LocalCachesHit: NoAssign {
    // set ITERS to trigger possible leak of backreferences
    // during cleanup on cache overflow and on thread termination
    static const int ITERS = 2*(FreeBlockPool::POOL_HIGH_MARK +
                                LocalLOC::LOC_HIGH_MARK);
public:
    LocalCachesHit() {}
    void operator()(int) const {
        void *objsSmall[ITERS], *objsLarge[ITERS];

        for (int i=0; i<ITERS; i++) {
            objsSmall[i] = scalable_malloc(minLargeObjectSize-1);
            objsLarge[i] = scalable_malloc(minLargeObjectSize);
        }
        for (int i=0; i<ITERS; i++) {
            scalable_free(objsSmall[i]);
            scalable_free(objsLarge[i]);
        }
    }
};

static size_t allocatedBackRefCount()
{
    size_t cnt = 0;
    for (int i=0; i<=backRefMaster->lastUsed; i++)
        cnt += backRefMaster->backRefBl[i]->allocatedCount;
    return cnt;
}

class TestInvalidBackrefs: public SimpleBarrier {
#if __ANDROID__
    // Android requires lower iters due to lack of virtual memory.
    static const int BACKREF_GROWTH_ITERS = 50*1024;
#else
    static const int BACKREF_GROWTH_ITERS = 200*1024;
#endif

    static tbb::atomic<bool> backrefGrowthDone;
    static void *ptrs[BACKREF_GROWTH_ITERS];
public:
    TestInvalidBackrefs() {}
    void operator()(int id) const {

        if (!id) {
            backrefGrowthDone = false;
            barrier.wait();

            for (int i=0; i<BACKREF_GROWTH_ITERS; i++)
                ptrs[i] = scalable_malloc(minLargeObjectSize);
            backrefGrowthDone = true;
            for (int i=0; i<BACKREF_GROWTH_ITERS; i++)
                scalable_free(ptrs[i]);
        } else {
            void *p2 = scalable_malloc(minLargeObjectSize-1);
            char *p1 = (char*)scalable_malloc(minLargeObjectSize-1);
            LargeObjectHdr *hdr =
                (LargeObjectHdr*)(p1+minLargeObjectSize-1 - sizeof(LargeObjectHdr));
            hdr->backRefIdx.master = 7;
            hdr->backRefIdx.largeObj = 1;
            hdr->backRefIdx.offset = 2000;

            barrier.wait();

            while (!backrefGrowthDone) {
                scalable_free(p2);
                p2 = scalable_malloc(minLargeObjectSize-1);
            }
            scalable_free(p1);
            scalable_free(p2);
        }
    }
};

tbb::atomic<bool> TestInvalidBackrefs::backrefGrowthDone;
void *TestInvalidBackrefs::ptrs[BACKREF_GROWTH_ITERS];

void TestBackRef() {
    size_t beforeNumBackRef, afterNumBackRef;

    beforeNumBackRef = allocatedBackRefCount();
    for( int p=MaxThread; p>=MinThread; --p )
        NativeParallelFor( p, BackRefWork<2*BR_MAX_CNT+2>() );
    afterNumBackRef = allocatedBackRefCount();
    ASSERT(beforeNumBackRef==afterNumBackRef, "backreference leak detected");

    // lastUsed marks peak resource consumption. As we allocate below the mark,
    // it must not move up, otherwise there is a resource leak.
    int sustLastUsed = backRefMaster->lastUsed;
    NativeParallelFor( 1, BackRefWork<2*BR_MAX_CNT+2>() );
    ASSERT(sustLastUsed == backRefMaster->lastUsed, "backreference leak detected");

    // check leak of back references while per-thread caches are in use
    // warm up needed to cover bootStrapMalloc call
    NativeParallelFor( 1, LocalCachesHit() );
    beforeNumBackRef = allocatedBackRefCount();
    NativeParallelFor( 2, LocalCachesHit() );
    int res = scalable_allocation_command(TBBMALLOC_CLEAN_ALL_BUFFERS, NULL);
    ASSERT(res == TBBMALLOC_OK, NULL);
    afterNumBackRef = allocatedBackRefCount();
    ASSERT(beforeNumBackRef>=afterNumBackRef, "backreference leak detected");

    // This is a regression test against race condition between backreference
    // extension and checking invalid BackRefIdx.
    // While detecting is object large or small, scalable_free 1st check for
    // large objects, so there is a chance to prepend small object with
    // seems valid BackRefIdx for large objects, and thus trigger the bug.
    TestInvalidBackrefs::initBarrier(MaxThread);
    NativeParallelFor( MaxThread, TestInvalidBackrefs() );
    // Consume all available backrefs and check they work correctly.
    // For now test 32-bit machines only, because for 64-bit memory consumption is too high.
    if (sizeof(uintptr_t) == 4)
        NativeParallelFor( MaxThread, BackRefWork<0>() );
}

void *getMem(intptr_t /*pool_id*/, size_t &bytes)
{
    const size_t BUF_SIZE = 8*1024*1024;
    static char space[BUF_SIZE];
    static size_t pos;

    if (pos + bytes > BUF_SIZE)
        return NULL;

    void *ret = space + pos;
    pos += bytes;

    return ret;
}

int putMem(intptr_t /*pool_id*/, void* /*raw_ptr*/, size_t /*raw_bytes*/)
{
    return 0;
}

struct MallocPoolHeader {
    void  *rawPtr;
    size_t userSize;
};

void *getMallocMem(intptr_t /*pool_id*/, size_t &bytes)
{
    void *rawPtr = malloc(bytes+sizeof(MallocPoolHeader));
    void *ret = (void *)((uintptr_t)rawPtr+sizeof(MallocPoolHeader));

    MallocPoolHeader *hdr = (MallocPoolHeader*)ret-1;
    hdr->rawPtr = rawPtr;
    hdr->userSize = bytes;

    return ret;
}

int putMallocMem(intptr_t /*pool_id*/, void *ptr, size_t bytes)
{
    MallocPoolHeader *hdr = (MallocPoolHeader*)ptr-1;
    ASSERT(bytes == hdr->userSize, "Invalid size in pool callback.");
    free(hdr->rawPtr);

    return 0;
}

class StressLOCacheWork: NoAssign {
    rml::MemoryPool *mallocPool;
public:
    StressLOCacheWork(rml::MemoryPool *mallocPool) : mallocPool(mallocPool) {}
    void operator()(int) const {
        for (size_t sz=minLargeObjectSize; sz<1*1024*1024;
             sz+=LargeObjectCache::largeBlockCacheStep) {
            void *ptr = pool_malloc(mallocPool, sz);
            ASSERT(ptr, "Memory was not allocated");
            memset(ptr, sz, sz);
            pool_free(mallocPool, ptr);
        }
    }
};

void TestPools() {
    rml::MemPoolPolicy pol(getMem, putMem);
    size_t beforeNumBackRef, afterNumBackRef;

    rml::MemoryPool *pool1;
    rml::MemoryPool *pool2;
    pool_create_v1(0, &pol, &pool1);
    pool_create_v1(0, &pol, &pool2);
    pool_destroy(pool1);
    pool_destroy(pool2);

    scalable_allocation_command(TBBMALLOC_CLEAN_ALL_BUFFERS, NULL);
    beforeNumBackRef = allocatedBackRefCount();
    rml::MemoryPool *fixedPool;

    pool_create_v1(0, &pol, &fixedPool);
    pol.pAlloc = getMallocMem;
    pol.pFree = putMallocMem;
    pol.granularity = 8;
    rml::MemoryPool *mallocPool;

    pool_create_v1(0, &pol, &mallocPool);
/* check that large object cache (LOC) returns correct size for cached objects
   passBackendSz Byte objects are cached in LOC, but bypassed the backend, so
   memory requested directly from allocation callback.
   nextPassBackendSz Byte objects must fit to another LOC bin,
   so that their allocation/realeasing leads to cache cleanup.
   All this is expecting to lead to releasing of passBackendSz Byte object
   from LOC during LOC cleanup, and putMallocMem checks that returned size
   is correct.
*/
    const size_t passBackendSz = Backend::maxBinned_HugePage+1,
        anotherLOCBinSz = minLargeObjectSize+1;
    for (int i=0; i<10; i++) { // run long enough to be cached
        void *p = pool_malloc(mallocPool, passBackendSz);
        ASSERT(p, "Memory was not allocated");
        pool_free(mallocPool, p);
    }
    // run long enough to passBackendSz allocation was cleaned from cache
    // and returned back to putMallocMem for size checking
    for (int i=0; i<1000; i++) {
        void *p = pool_malloc(mallocPool, anotherLOCBinSz);
        ASSERT(p, "Memory was not allocated");
        pool_free(mallocPool, p);
    }

    void *smallObj =  pool_malloc(fixedPool, 10);
    ASSERT(smallObj, "Memory was not allocated");
    memset(smallObj, 1, 10);
    void *ptr = pool_malloc(fixedPool, 1024);
    ASSERT(ptr, "Memory was not allocated");
    memset(ptr, 1, 1024);
    void *largeObj = pool_malloc(fixedPool, minLargeObjectSize);
    ASSERT(largeObj, "Memory was not allocated");
    memset(largeObj, 1, minLargeObjectSize);
    ptr = pool_malloc(fixedPool, minLargeObjectSize);
    ASSERT(ptr, "Memory was not allocated");
    memset(ptr, minLargeObjectSize, minLargeObjectSize);
    pool_malloc(fixedPool, 10*minLargeObjectSize); // no leak for unsuccessful allocations
    pool_free(fixedPool, smallObj);
    pool_free(fixedPool, largeObj);

    // provoke large object cache cleanup and hope no leaks occurs
    for( int p=MaxThread; p>=MinThread; --p )
        NativeParallelFor( p, StressLOCacheWork(mallocPool) );
    pool_destroy(mallocPool);
    pool_destroy(fixedPool);

    scalable_allocation_command(TBBMALLOC_CLEAN_ALL_BUFFERS, NULL);
    afterNumBackRef = allocatedBackRefCount();
    ASSERT(beforeNumBackRef==afterNumBackRef, "backreference leak detected");

    {
        // test usedSize/cachedSize and LOC bitmask correctness
        void *p[5];
        pool_create_v1(0, &pol, &mallocPool);
        const LargeObjectCache *loc = &((rml::internal::MemoryPool*)mallocPool)->extMemPool.loc;
        p[3] = pool_malloc(mallocPool, minLargeObjectSize+2*LargeObjectCache::largeBlockCacheStep);
        for (int i=0; i<10; i++) {
            p[0] = pool_malloc(mallocPool, minLargeObjectSize);
            p[1] = pool_malloc(mallocPool, minLargeObjectSize+LargeObjectCache::largeBlockCacheStep);
            pool_free(mallocPool, p[0]);
            pool_free(mallocPool, p[1]);
        }
        ASSERT(loc->getUsedSize(), NULL);
        pool_free(mallocPool, p[3]);
        ASSERT(loc->getLOCSize() < 3*(minLargeObjectSize+LargeObjectCache::largeBlockCacheStep), NULL);
        const size_t maxLocalLOCSize = LocalLOCImpl<3,30>::getMaxSize();
        ASSERT(loc->getUsedSize() <= maxLocalLOCSize, NULL);
        for (int i=0; i<3; i++)
            p[i] = pool_malloc(mallocPool, minLargeObjectSize+i*LargeObjectCache::largeBlockCacheStep);
        size_t currUser = loc->getUsedSize();
        ASSERT(!loc->getLOCSize() && currUser >= 3*(minLargeObjectSize+LargeObjectCache::largeBlockCacheStep), NULL);
        p[4] = pool_malloc(mallocPool, minLargeObjectSize+3*LargeObjectCache::largeBlockCacheStep);
        ASSERT(loc->getUsedSize() - currUser >= minLargeObjectSize+3*LargeObjectCache::largeBlockCacheStep, NULL);
        pool_free(mallocPool, p[4]);
        ASSERT(loc->getUsedSize() <= currUser+maxLocalLOCSize, NULL);
        pool_reset(mallocPool);
        ASSERT(!loc->getLOCSize() && !loc->getUsedSize(), NULL);
        pool_destroy(mallocPool);
    }
    // To test LOC we need bigger lists than released by current LocalLOC
    //   in production code. Create special LocalLOC.
    {
        LocalLOCImpl<2, 20> lLOC;
        pool_create_v1(0, &pol, &mallocPool);
        rml::internal::ExtMemoryPool *mPool = &((rml::internal::MemoryPool*)mallocPool)->extMemPool;
        const LargeObjectCache *loc = &((rml::internal::MemoryPool*)mallocPool)->extMemPool.loc;
        for (int i=0; i<22; i++) {
            void *o = pool_malloc(mallocPool, minLargeObjectSize+i*LargeObjectCache::largeBlockCacheStep);
            bool ret = lLOC.put(((LargeObjectHdr*)o - 1)->memoryBlock, mPool);
            ASSERT(ret, NULL);

            o = pool_malloc(mallocPool, minLargeObjectSize+i*LargeObjectCache::largeBlockCacheStep);
            ret = lLOC.put(((LargeObjectHdr*)o - 1)->memoryBlock, mPool);
            ASSERT(ret, NULL);
        }
        lLOC.externalCleanup(mPool);
        ASSERT(!loc->getUsedSize(), NULL);

        pool_destroy(mallocPool);
    }
}

void TestObjectRecognition() {
    size_t headersSize = sizeof(LargeMemoryBlock)+sizeof(LargeObjectHdr);
    unsigned falseObjectSize = 113; // unsigned is the type expected by getObjectSize
    size_t obtainedSize;

    ASSERT(sizeof(BackRefIdx)==sizeof(uintptr_t), "Unexpected size of BackRefIdx");
    ASSERT(getObjectSize(falseObjectSize)!=falseObjectSize, "Error in test: bad choice for false object size");

    void* mem = scalable_malloc(2*slabSize);
    ASSERT(mem, "Memory was not allocated");
    Block* falseBlock = (Block*)alignUp((uintptr_t)mem, slabSize);
    falseBlock->objectSize = falseObjectSize;
    char* falseSO = (char*)falseBlock + falseObjectSize*7;
    ASSERT(alignDown(falseSO, slabSize)==(void*)falseBlock, "Error in test: false object offset is too big");

    void* bufferLOH = scalable_malloc(2*slabSize + headersSize);
    ASSERT(bufferLOH, "Memory was not allocated");
    LargeObjectHdr* falseLO =
        (LargeObjectHdr*)alignUp((uintptr_t)bufferLOH + headersSize, slabSize);
    LargeObjectHdr* headerLO = (LargeObjectHdr*)falseLO-1;
    headerLO->memoryBlock = (LargeMemoryBlock*)bufferLOH;
    headerLO->memoryBlock->unalignedSize = 2*slabSize + headersSize;
    headerLO->memoryBlock->objectSize = slabSize + headersSize;
    headerLO->backRefIdx = BackRefIdx::newBackRef(/*largeObj=*/true);
    setBackRef(headerLO->backRefIdx, headerLO);
    ASSERT(scalable_msize(falseLO) == slabSize + headersSize,
           "Error in test: LOH falsification failed");
    removeBackRef(headerLO->backRefIdx);

    const int NUM_OF_IDX = BR_MAX_CNT+2;
    BackRefIdx idxs[NUM_OF_IDX];
    for (int cnt=0; cnt<2; cnt++) {
        for (int master = -10; master<10; master++) {
            falseBlock->backRefIdx.master = (uint16_t)master;
            headerLO->backRefIdx.master = (uint16_t)master;

            for (int bl = -10; bl<BR_MAX_CNT+10; bl++) {
                falseBlock->backRefIdx.offset = (uint16_t)bl;
                headerLO->backRefIdx.offset = (uint16_t)bl;

                for (int largeObj = 0; largeObj<2; largeObj++) {
                    falseBlock->backRefIdx.largeObj = largeObj;
                    headerLO->backRefIdx.largeObj = largeObj;

                    obtainedSize = __TBB_malloc_safer_msize(falseSO, NULL);
                    ASSERT(obtainedSize==0, "Incorrect pointer accepted");
                    obtainedSize = __TBB_malloc_safer_msize(falseLO, NULL);
                    ASSERT(obtainedSize==0, "Incorrect pointer accepted");
                }
            }
        }
        if (cnt == 1) {
            for (int i=0; i<NUM_OF_IDX; i++)
                removeBackRef(idxs[i]);
            break;
        }
        for (int i=0; i<NUM_OF_IDX; i++) {
            idxs[i] = BackRefIdx::newBackRef(/*largeObj=*/false);
            setBackRef(idxs[i], NULL);
        }
    }
    char *smallPtr = (char*)scalable_malloc(falseObjectSize);
    obtainedSize = __TBB_malloc_safer_msize(smallPtr, NULL);
    ASSERT(obtainedSize==getObjectSize(falseObjectSize), "Correct pointer not accepted?");
    scalable_free(smallPtr);

    obtainedSize = __TBB_malloc_safer_msize(mem, NULL);
    ASSERT(obtainedSize>=2*slabSize, "Correct pointer not accepted?");
    scalable_free(mem);
    scalable_free(bufferLOH);
}

class TestBackendWork: public SimpleBarrier {
    struct TestBlock {
        intptr_t   data;
        BackRefIdx idx;
    };
    static const int ITERS = 20;

    rml::internal::Backend *backend;
public:
    TestBackendWork(rml::internal::Backend *bknd) : backend(bknd) {}
    void operator()(int) const {
        barrier.wait();

        for (int i=0; i<ITERS; i++) {
            BlockI *slabBlock = backend->getSlabBlock(1);
            ASSERT(slabBlock, "Memory was not allocated");
            LargeMemoryBlock *lmb = backend->getLargeBlock(8*1024);
            backend->putSlabBlock(slabBlock);
            backend->putLargeBlock(lmb);
        }
    }
};

void TestBackend()
{
    rml::MemPoolPolicy pol(getMallocMem, putMallocMem);
    rml::MemoryPool *mPool;
    pool_create_v1(0, &pol, &mPool);
    rml::internal::ExtMemoryPool *ePool =
        &((rml::internal::MemoryPool*)mPool)->extMemPool;
    rml::internal::Backend *backend = &ePool->backend;

    for( int p=MaxThread; p>=MinThread; --p ) {
        // regression test against an race condition in backend synchronization,
        // triggered only when WhiteboxTestingYield() call yields
        for (int i=0; i<100; i++) {
            TestBackendWork::initBarrier(p);
            NativeParallelFor( p, TestBackendWork(backend) );
        }
    }

    BlockI *block = backend->getSlabBlock(1);
    ASSERT(block, "Memory was not allocated");
    backend->putSlabBlock(block);

    // Checks if the backend increases and decreases the amount of allocated memory when memory is allocated.
    const size_t memSize0 = backend->getTotalMemSize();
    LargeMemoryBlock *lmb = backend->getLargeBlock(4*MByte);
    ASSERT( lmb, ASSERT_TEXT );

    const size_t memSize1 = backend->getTotalMemSize();
    ASSERT( (intptr_t)(memSize1-memSize0) >= 4*MByte, "The backend has not increased the amount of using memory." );

    backend->putLargeBlock(lmb);
    const size_t memSize2 = backend->getTotalMemSize();
    ASSERT( memSize2 == memSize0, "The backend has not decreased the amount of using memory." );

    pool_destroy(mPool);
}

void TestBitMask()
{
    BitMaskMin<256> mask;

    mask.reset();
    mask.set(10, 1);
    mask.set(5, 1);
    mask.set(1, 1);
    ASSERT(mask.getMinTrue(2) == 5, NULL);

    mask.reset();
    mask.set(0, 1);
    mask.set(64, 1);
    mask.set(63, 1);
    mask.set(200, 1);
    mask.set(255, 1);
    ASSERT(mask.getMinTrue(0) == 0, NULL);
    ASSERT(mask.getMinTrue(1) == 63, NULL);
    ASSERT(mask.getMinTrue(63) == 63, NULL);
    ASSERT(mask.getMinTrue(64) == 64, NULL);
    ASSERT(mask.getMinTrue(101) == 200, NULL);
    ASSERT(mask.getMinTrue(201) == 255, NULL);
    mask.set(255, 0);
    ASSERT(mask.getMinTrue(201) == -1, NULL);
}

size_t getMemSize()
{
    return defaultMemPool->extMemPool.backend.getTotalMemSize();
}

class CheckNotCached {
    static size_t memSize;
public:
    void operator() () const {
        int res = scalable_allocation_mode(TBBMALLOC_SET_SOFT_HEAP_LIMIT, 1);
        ASSERT(res == TBBMALLOC_OK, NULL);
        if (memSize==(size_t)-1) {
            memSize = getMemSize();
        } else {
            ASSERT(getMemSize() == memSize, NULL);
            memSize=(size_t)-1;
        }
    }
};

size_t CheckNotCached::memSize = (size_t)-1;

class RunTestHeapLimit: public SimpleBarrier {
public:
    void operator()( int /*mynum*/ ) const {
        // Provoke bootstrap heap initialization before recording memory size.
        // NOTE: The initialization should be processed only with a "large"
        // object. Since the "small" object allocation lead to blocking of a
        // slab as an active block and it is impossible to release it with
        // foreign thread.
        scalable_free(scalable_malloc(minLargeObjectSize));
        barrier.wait(CheckNotCached());
        for (size_t n = minLargeObjectSize; n < 5*1024*1024; n += 128*1024)
            scalable_free(scalable_malloc(n));
        barrier.wait(CheckNotCached());
    }
};

void TestHeapLimit()
{
    if(!isMallocInitialized()) doInitialization();
    // tiny limit to stop caching
    int res = scalable_allocation_mode(TBBMALLOC_SET_SOFT_HEAP_LIMIT, 1);
    ASSERT(res == TBBMALLOC_OK, NULL);
     // Provoke bootstrap heap initialization before recording memory size.
    scalable_free(scalable_malloc(8));
    size_t n, sizeBefore = getMemSize();

    // Try to provoke call to OS for memory to check that
    // requests are not fulfilled from caches.
    // Single call is not enough here because of backend fragmentation.
    for (n = minLargeObjectSize; n < 10*1024*1024; n += 16*1024) {
        void *p = scalable_malloc(n);
        bool leave = (sizeBefore != getMemSize());
        scalable_free(p);
        if (leave)
            break;
        ASSERT(sizeBefore == getMemSize(), "No caching expected");
    }
    ASSERT(n < 10*1024*1024, "scalable_malloc doesn't provoke OS request for memory, "
           "is some internal cache still used?");

    for( int p=MaxThread; p>=MinThread; --p ) {
        RunTestHeapLimit::initBarrier( p );
        NativeParallelFor( p, RunTestHeapLimit() );
    }
    // it's try to match limit as well as set limit, so call here
    res = scalable_allocation_mode(TBBMALLOC_SET_SOFT_HEAP_LIMIT, 1);
    ASSERT(res == TBBMALLOC_OK, NULL);
    size_t m = getMemSize();
    ASSERT(sizeBefore == m, NULL);
    // restore default
    res = scalable_allocation_mode(TBBMALLOC_SET_SOFT_HEAP_LIMIT, 0);
    ASSERT(res == TBBMALLOC_OK, NULL);
}

void checkNoHugePages()
{
    ASSERT(!hugePages.enabled, "scalable_allocation_mode "
           "must have priority over environment variable");
}

/*---------------------------------------------------------------------------*/
// The regression test against bugs in TBBMALLOC_CLEAN_ALL_BUFFERS allocation command.
// The idea is to allocate and deallocate a set of objects randomly in parallel.
// For large sizes (16K), it forces conflicts in backend during coalescing.
// For small sizes (4K), it forces cross-thread deallocations and then orphaned slabs.
// Global cleanup should process orphaned slabs and the queue of postponed coalescing
// requests, otherwise it will not be able to unmap all unused memory.

const int num_allocs = 10*1024;
void *ptrs[num_allocs];
tbb::atomic<int> alloc_counter;

template<int AllocSize>
struct TestCleanAllBuffersBody : public SimpleBarrier {
    void operator() ( int ) const {
        barrier.wait();
        for( int i = alloc_counter++; i < num_allocs; i = alloc_counter++ ) {
           ptrs[i] = scalable_malloc( AllocSize );
           ASSERT( ptrs[i] != NULL, "scalable_malloc returned zero." );
        }
        barrier.wait();
        for( int i = --alloc_counter; i >= 0; i = --alloc_counter )
           if (i<num_allocs) scalable_free( ptrs[i] );
    }
};

template<int AllocSize>
void TestCleanAllBuffers() {
    const int num_threads = 8;
    // Clean up if something was allocated before the test
    scalable_allocation_command(TBBMALLOC_CLEAN_ALL_BUFFERS,0);

    size_t memory_in_use_before = getMemSize();
    alloc_counter = 0;
    TestCleanAllBuffersBody<AllocSize>::initBarrier(num_threads);
    NativeParallelFor(num_threads, TestCleanAllBuffersBody<AllocSize>());
    // TODO: reproduce the bug conditions more reliably
    if ( defaultMemPool->extMemPool.backend.coalescQ.blocksToFree == NULL )
        REMARK( "Warning: The queue of postponed coalescing requests is empty. Unable to create the condition for bug reproduction.\n" );
    int result = scalable_allocation_command(TBBMALLOC_CLEAN_ALL_BUFFERS,0);
    ASSERT( result == TBBMALLOC_OK, "The cleanup request has not cleaned anything." );
    size_t memory_in_use_after = getMemSize();

    size_t memory_leak = memory_in_use_after - memory_in_use_before;
    REMARK( "memory_in_use_before = %ld\nmemory_in_use_after = %ld\n", memory_in_use_before, memory_in_use_after );
    ASSERT( memory_leak == 0, "Cleanup was unable to release all allocated memory." );
}
/*---------------------------------------------------------------------------*/
/*------------------------- Large Object Cache tests ------------------------*/
#if _MSC_VER==1600 || _MSC_VER==1500
    // ignore C4275: non dll-interface class 'stdext::exception' used as
    // base for dll-interface class 'std::bad_cast'
    #pragma warning (disable: 4275)
#endif
#include <vector>
#include <list>
#include __TBB_STD_SWAP_HEADER

// default constructor of CacheBin
template<typename Props>
rml::internal::LargeObjectCacheImpl<Props>::CacheBin::CacheBin() {}

template<typename Props>
class CacheBinModel {

    typedef typename rml::internal::LargeObjectCacheImpl<Props>::CacheBin CacheBinType;

    // The emulated cache bin.
    CacheBinType cacheBinModel;
    // The reference to real cahce bin inside the large object cache.
    CacheBinType &cacheBin;

    const size_t size;

    // save only current time
    std::list<uintptr_t> objects;

    void doCleanup() {
        if ( cacheBinModel.cachedSize > Props::TooLargeFactor*cacheBinModel.usedSize ) tooLargeLOC++;
        else tooLargeLOC = 0;

        if (tooLargeLOC>3 && cacheBinModel.ageThreshold)
            cacheBinModel.ageThreshold = (cacheBinModel.ageThreshold + cacheBinModel.meanHitRange)/2;

        uintptr_t currTime = cacheCurrTime;
        while (!objects.empty() && (intptr_t)(currTime - objects.front()) > cacheBinModel.ageThreshold) {
            cacheBinModel.cachedSize -= size;
            cacheBinModel.lastCleanedAge = objects.front();
            objects.pop_front();
        }

        cacheBinModel.oldest = objects.empty() ? 0 : objects.front();
    }

public:
    CacheBinModel(CacheBinType &_cacheBin, size_t allocSize) : cacheBin(_cacheBin), size(allocSize) {
        cacheBinModel.oldest = cacheBin.oldest;
        cacheBinModel.lastCleanedAge = cacheBin.lastCleanedAge;
        cacheBinModel.ageThreshold = cacheBin.ageThreshold;
        cacheBinModel.usedSize = cacheBin.usedSize;
        cacheBinModel.cachedSize = cacheBin.cachedSize;
        cacheBinModel.meanHitRange = cacheBin.meanHitRange;
        cacheBinModel.lastGet = cacheBin.lastGet;
    }
    void get() {
        uintptr_t currTime = ++cacheCurrTime;

        if ( objects.empty() ) {
            const uintptr_t sinceLastGet = currTime - cacheBinModel.lastGet;
            if ( ( cacheBinModel.ageThreshold && sinceLastGet > Props::LongWaitFactor*cacheBinModel.ageThreshold ) ||
                 ( cacheBinModel.lastCleanedAge && sinceLastGet > Props::LongWaitFactor*(cacheBinModel.lastCleanedAge - cacheBinModel.lastGet) ) )
                cacheBinModel.lastCleanedAge = cacheBinModel.ageThreshold = 0;

            if (cacheBinModel.lastCleanedAge)
                cacheBinModel.ageThreshold = Props::OnMissFactor*(currTime - cacheBinModel.lastCleanedAge);
        } else {
            uintptr_t obj_age = objects.back();
            objects.pop_back();
            if ( objects.empty() ) cacheBinModel.oldest = 0;

            intptr_t hitRange = currTime - obj_age;
            cacheBinModel.meanHitRange = cacheBinModel.meanHitRange? (cacheBinModel.meanHitRange + hitRange)/2 : hitRange;

            cacheBinModel.cachedSize -= size;
        }

        cacheBinModel.usedSize += size;
        cacheBinModel.lastGet = currTime;

        if ( currTime % rml::internal::cacheCleanupFreq == 0 ) doCleanup();
    }

    void putList( int num ) {
        uintptr_t currTime = cacheCurrTime;
        cacheCurrTime += num;

        cacheBinModel.usedSize -= num*size;

        bool cleanUpNeeded = false;
        if ( !cacheBinModel.lastCleanedAge ) {
            cacheBinModel.lastCleanedAge = ++currTime;
            cleanUpNeeded |= currTime % rml::internal::cacheCleanupFreq == 0;
            num--;
        }

        for ( int i=1; i<=num; ++i ) {
            currTime+=1;
            cleanUpNeeded |= currTime % rml::internal::cacheCleanupFreq == 0;
            if ( objects.empty() )
                cacheBinModel.oldest = currTime;
            objects.push_back(currTime);
        }

        cacheBinModel.cachedSize += num*size;

        if ( cleanUpNeeded ) doCleanup();
    }

    void check() {
        ASSERT(cacheBinModel.oldest == cacheBin.oldest, ASSERT_TEXT);
        ASSERT(cacheBinModel.lastCleanedAge == cacheBin.lastCleanedAge, ASSERT_TEXT);
        ASSERT(cacheBinModel.ageThreshold == cacheBin.ageThreshold, ASSERT_TEXT);
        ASSERT(cacheBinModel.usedSize == cacheBin.usedSize, ASSERT_TEXT);
        ASSERT(cacheBinModel.cachedSize == cacheBin.cachedSize, ASSERT_TEXT);
        ASSERT(cacheBinModel.meanHitRange == cacheBin.meanHitRange, ASSERT_TEXT);
        ASSERT(cacheBinModel.lastGet == cacheBin.lastGet, ASSERT_TEXT);
    }

    static uintptr_t cacheCurrTime;
    static intptr_t tooLargeLOC;
};

template<typename Props> uintptr_t CacheBinModel<Props>::cacheCurrTime;
template<typename Props> intptr_t CacheBinModel<Props>::tooLargeLOC;

template <typename Scenarion>
void LOCModelTester() {
    defaultMemPool->extMemPool.loc.cleanAll();
    defaultMemPool->extMemPool.loc.reset();

    const size_t size = 16 * 1024;
    const size_t headersSize = sizeof(rml::internal::LargeMemoryBlock)+sizeof(rml::internal::LargeObjectHdr);
    const size_t allocationSize = LargeObjectCache::alignToBin(size+headersSize+rml::internal::largeObjectAlignment);
    const int binIdx = defaultMemPool->extMemPool.loc.largeCache.sizeToIdx( allocationSize );

    CacheBinModel<rml::internal::LargeObjectCache::LargeCacheTypeProps>::cacheCurrTime = defaultMemPool->extMemPool.loc.cacheCurrTime;
    CacheBinModel<rml::internal::LargeObjectCache::LargeCacheTypeProps>::tooLargeLOC = defaultMemPool->extMemPool.loc.largeCache.tooLargeLOC;
    CacheBinModel<rml::internal::LargeObjectCache::LargeCacheTypeProps> cacheBinModel(defaultMemPool->extMemPool.loc.largeCache.bin[binIdx], allocationSize);

    Scenarion scen;
    for (rml::internal::LargeMemoryBlock *lmb = scen.next(); (intptr_t)lmb != (intptr_t)-1; lmb = scen.next()) {
        if ( lmb ) {
            int num=1;
            for (rml::internal::LargeMemoryBlock *curr = lmb; curr->next; curr=curr->next) num+=1;
            defaultMemPool->extMemPool.freeLargeObject(lmb);
            cacheBinModel.putList(num);
        } else {
            scen.saveLmb(defaultMemPool->extMemPool.mallocLargeObject(defaultMemPool, allocationSize));
            cacheBinModel.get();
        }

        cacheBinModel.check();
    }
}

class TestBootstrap {
    bool allocating;
    std::vector<rml::internal::LargeMemoryBlock*> lmbArray;
public:
    TestBootstrap() : allocating(true) {}

    rml::internal::LargeMemoryBlock* next() {
        if ( allocating )
            return NULL;
        if ( !lmbArray.empty() ) {
            rml::internal::LargeMemoryBlock *ret = lmbArray.back();
            lmbArray.pop_back();
            return ret;
        }
        return (rml::internal::LargeMemoryBlock*)-1;
    }

    void saveLmb( rml::internal::LargeMemoryBlock *lmb ) {
        lmb->next = NULL;
        lmbArray.push_back(lmb);
        if ( lmbArray.size() == 1000 ) allocating = false;
    }
};

class TestRandom {
    std::vector<rml::internal::LargeMemoryBlock*> lmbArray;
    int numOps;
public:
    TestRandom() : numOps(100000) {
        srand(1234);
    }

    rml::internal::LargeMemoryBlock* next() {
        if ( numOps-- ) {
            if ( lmbArray.empty() || rand() / (RAND_MAX>>1) == 0 )
                return NULL;
            size_t ind = rand()%lmbArray.size();
            if ( ind != lmbArray.size()-1 ) std::swap(lmbArray[ind],lmbArray[lmbArray.size()-1]);
            rml::internal::LargeMemoryBlock *lmb = lmbArray.back();
            lmbArray.pop_back();
            return lmb;
        }
        return (rml::internal::LargeMemoryBlock*)-1;
    }

    void saveLmb( rml::internal::LargeMemoryBlock *lmb ) {
        lmb->next = NULL;
        lmbArray.push_back(lmb);
    }
};

class TestCollapsingMallocFree : public SimpleBarrier {
public:
    static const int NUM_ALLOCS = 100000;
    const int num_threads;

    TestCollapsingMallocFree( int _num_threads ) : num_threads(_num_threads) {
        initBarrier( num_threads );
    }

    void operator() ( int ) const {
        const size_t size = 16 * 1024;
        const size_t headersSize = sizeof(rml::internal::LargeMemoryBlock)+sizeof(rml::internal::LargeObjectHdr);
        const size_t allocationSize = LargeObjectCache::alignToBin(size+headersSize+rml::internal::largeObjectAlignment);

        barrier.wait();
        for ( int i=0; i<NUM_ALLOCS; ++i ) {
            defaultMemPool->extMemPool.freeLargeObject(
                defaultMemPool->extMemPool.mallocLargeObject(defaultMemPool, allocationSize) );
        }
    }

    void check() {
        ASSERT( tbbmalloc_whitebox::locGetProcessed == tbbmalloc_whitebox::locPutProcessed, ASSERT_TEXT );
        ASSERT( tbbmalloc_whitebox::locGetProcessed < num_threads*NUM_ALLOCS, "No one Malloc/Free pair was collapsed." );
    }
};

class TestCollapsingBootstrap : public SimpleBarrier {
    class CheckNumAllocs {
        const int num_threads;
    public:
        CheckNumAllocs( int _num_threads ) : num_threads(_num_threads) {}
        void operator()() const {
            ASSERT( tbbmalloc_whitebox::locGetProcessed == num_threads*NUM_ALLOCS, ASSERT_TEXT );
            ASSERT( tbbmalloc_whitebox::locPutProcessed == 0, ASSERT_TEXT );
        }
    };
public:
    static const int NUM_ALLOCS = 1000;
    const int num_threads;

    TestCollapsingBootstrap( int _num_threads ) : num_threads(_num_threads) {
        initBarrier( num_threads );
    }

    void operator() ( int ) const {
        const size_t size = 16 * 1024;
        size_t headersSize = sizeof(rml::internal::LargeMemoryBlock)+sizeof(rml::internal::LargeObjectHdr);
        size_t allocationSize = LargeObjectCache::alignToBin(size+headersSize+rml::internal::largeObjectAlignment);

        barrier.wait();
        rml::internal::LargeMemoryBlock *lmbArray[NUM_ALLOCS];
        for ( int i=0; i<NUM_ALLOCS; ++i )
            lmbArray[i] = defaultMemPool->extMemPool.mallocLargeObject(defaultMemPool, allocationSize);

        barrier.wait(CheckNumAllocs(num_threads));
        for ( int i=0; i<NUM_ALLOCS; ++i )
            defaultMemPool->extMemPool.freeLargeObject( lmbArray[i] );
    }

    void check() {
        ASSERT( tbbmalloc_whitebox::locGetProcessed == tbbmalloc_whitebox::locPutProcessed, ASSERT_TEXT );
        ASSERT( tbbmalloc_whitebox::locGetProcessed == num_threads*NUM_ALLOCS, ASSERT_TEXT );
    }
};

template <typename Scenario>
void LOCCollapsingTester( int num_threads ) {
    tbbmalloc_whitebox::locGetProcessed = 0;
    tbbmalloc_whitebox::locPutProcessed = 0;
    defaultMemPool->extMemPool.loc.cleanAll();
    defaultMemPool->extMemPool.loc.reset();

    Scenario scen(num_threads);
    NativeParallelFor(num_threads, scen);

    scen.check();
}

void TestLOC() {
    LOCModelTester<TestBootstrap>();
    LOCModelTester<TestRandom>();

    const int num_threads = 16;
    LOCCollapsingTester<TestCollapsingBootstrap>( num_threads );
    if ( num_threads > 1 ) {
        REMARK( "num_threads = %d\n", num_threads );
        LOCCollapsingTester<TestCollapsingMallocFree>( num_threads );
    } else {
        REPORT( "Warning: concurrency is too low for TestMallocFreeCollapsing ( num_threads = %d )\n", num_threads );
    }
}
/*---------------------------------------------------------------------------*/

void *findCacheLine(void *p) {
    return (void*)alignDown((uintptr_t)p, estimatedCacheLineSize);
}

// test that internals of Block are at expected cache lines
void TestSlabAlignment() {
    const size_t min_sz = 8;
    const int space = 2*16*1024; // fill at least 2 slabs
    void *ptrs[space / min_sz];  // the worst case is min_sz byte object

    for (size_t sz = min_sz; sz <= 64; sz *= 2) {
        for (int i = 0; i < space/sz; i++) {
            ptrs[i] = scalable_malloc(sz);
            Block *block = (Block *)alignDown(ptrs[i], slabSize);
            MALLOC_ASSERT(findCacheLine(&block->isFull) != findCacheLine(ptrs[i]),
                          "A user object must not share a cache line with slab control structures.");
            MALLOC_ASSERT(findCacheLine(&block->next) != findCacheLine(&block->nextPrivatizable),
                          "GlobalBlockFields and LocalBlockFields must be on different cache lines.");
        }
        for (int i = 0; i < space/sz; i++)
            scalable_free(ptrs[i]);
    }
}

int TestMain () {
    scalable_allocation_mode(USE_HUGE_PAGES, 0);
#if !__TBB_WIN8UI_SUPPORT
    Harness::SetEnv("TBB_MALLOC_USE_HUGE_PAGES","yes");
#endif
    checkNoHugePages();
    // backreference requires that initialization was done
    if(!isMallocInitialized()) doInitialization();
    checkNoHugePages();
    // to succeed, leak detection must be the 1st memory-intensive test
    TestBackRef();
    TestCleanAllBuffers<4*1024>();
    TestCleanAllBuffers<16*1024>();
    TestPools();
    TestBackend();

#if MALLOC_CHECK_RECURSION
    for( int p=MaxThread; p>=MinThread; --p ) {
        TestStartupAlloc::initBarrier( p );
        NativeParallelFor( p, TestStartupAlloc() );
        ASSERT(!firstStartupBlock, "Startup heap memory leak detected");
    }
#endif

    TestLargeObjectCache();
    TestObjectRecognition();
    TestBitMask();
    TestHeapLimit();
    TestLOC();
    TestSlabAlignment();
    return Harness::Done;
}
