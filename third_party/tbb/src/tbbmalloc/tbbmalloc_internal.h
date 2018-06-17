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

#ifndef __TBB_tbbmalloc_internal_H
#define __TBB_tbbmalloc_internal_H 1


#include "TypeDefinitions.h" /* Also includes customization layer Customize.h */

#if USE_PTHREAD
    // Some pthreads documentation says that <pthreads.h> must be first header.
    #include <pthread.h>
    typedef pthread_key_t tls_key_t;
#elif USE_WINTHREAD
    #include "tbb/machine/windows_api.h"
    typedef DWORD tls_key_t;
#else
    #error Must define USE_PTHREAD or USE_WINTHREAD
#endif

// TODO: *BSD also has it
#define BACKEND_HAS_MREMAP __linux__
#define CHECK_ALLOCATION_RANGE MALLOC_DEBUG || MALLOC_ZONE_OVERLOAD_ENABLED || MALLOC_UNIXLIKE_OVERLOAD_ENABLED

#include "tbb/tbb_config.h" // for __TBB_LIBSTDCPP_EXCEPTION_HEADERS_BROKEN
#if __TBB_LIBSTDCPP_EXCEPTION_HEADERS_BROKEN
  #define _EXCEPTION_PTR_H /* prevents exception_ptr.h inclusion */
  #define _GLIBCXX_NESTED_EXCEPTION_H /* prevents nested_exception.h inclusion */
#endif

#include <stdio.h>
#include <stdlib.h>
#include <limits.h> // for CHAR_BIT
#include <string.h> // for memset
#if MALLOC_CHECK_RECURSION
#include <new>        /* for placement new */
#endif
#include "tbb/scalable_allocator.h"
#include "tbbmalloc_internal_api.h"

/********* Various compile-time options        **************/

#if !__TBB_DEFINE_MIC && __TBB_MIC_NATIVE
 #error Intel(R) Many Integrated Core Compiler does not define __MIC__ anymore.
#endif

#define MALLOC_TRACE 0

#if MALLOC_TRACE
#define TRACEF(x) printf x
#else
#define TRACEF(x) ((void)0)
#endif /* MALLOC_TRACE */

#define ASSERT_TEXT NULL

#define COLLECT_STATISTICS ( MALLOC_DEBUG && MALLOCENV_COLLECT_STATISTICS )
#ifndef USE_INTERNAL_TID
#define USE_INTERNAL_TID COLLECT_STATISTICS || MALLOC_TRACE
#endif

#include "Statistics.h"

// call yield for whitebox testing, skip in real library
#ifndef WhiteboxTestingYield
#define WhiteboxTestingYield() ((void)0)
#endif


/********* End compile-time options        **************/

namespace rml {

namespace internal {

#if __TBB_MALLOC_LOCACHE_STAT
extern intptr_t mallocCalls, cacheHits;
extern intptr_t memAllocKB, memHitKB;
#endif

//! Utility template function to prevent "unused" warnings by various compilers.
template<typename T>
void suppress_unused_warning( const T& ) {}

/********** Various numeric parameters controlling allocations ********/

/*
 * slabSize - the size of a block for allocation of small objects,
 * it must be larger than maxSegregatedObjectSize.
 */
const uintptr_t slabSize = 16*1024;

/*
 * Large blocks cache cleanup frequency.
 * It should be power of 2 for the fast checking.
 */
const unsigned cacheCleanupFreq = 256;

/*
 * Alignment of large (>= minLargeObjectSize) objects.
 */
const size_t largeObjectAlignment = estimatedCacheLineSize;

/*
 * This number of bins in the TLS that leads to blocks that we can allocate in.
 */
const uint32_t numBlockBinLimit = 31;

/********** End of numeric parameters controlling allocations *********/

class BlockI;
class Block;
struct LargeMemoryBlock;
struct ExtMemoryPool;
struct MemRegion;
class FreeBlock;
class TLSData;
class Backend;
class MemoryPool;
struct CacheBinOperation;
extern const uint32_t minLargeObjectSize;

enum DecreaseOrIncrease {
    decrease, increase
};

class TLSKey {
    tls_key_t TLS_pointer_key;
public:
    bool init();
    bool destroy();
    TLSData* getThreadMallocTLS() const;
    void setThreadMallocTLS( TLSData * newvalue );
    TLSData* createTLS(MemoryPool *memPool, Backend *backend);
};

template<typename Arg, typename Compare>
inline void AtomicUpdate(Arg &location, Arg newVal, const Compare &cmp)
{
    MALLOC_STATIC_ASSERT(sizeof(Arg) == sizeof(intptr_t),
                         "Type of argument must match AtomicCompareExchange type.");
    for (Arg old = location; cmp(old, newVal); ) {
        Arg val = AtomicCompareExchange((intptr_t&)location, (intptr_t)newVal, old);
        if (val == old)
            break;
        // TODO: do we need backoff after unsuccessful CAS?
        old = val;
    }
}

// TODO: make BitMaskBasic more general
// (currently, it fits BitMaskMin well, but not as suitable for BitMaskMax)
template<unsigned NUM>
class BitMaskBasic {
    static const unsigned SZ = (NUM-1)/(CHAR_BIT*sizeof(uintptr_t))+1;
    static const unsigned WORD_LEN = CHAR_BIT*sizeof(uintptr_t);
    uintptr_t mask[SZ];
protected:
    void set(size_t idx, bool val) {
        MALLOC_ASSERT(idx<NUM, ASSERT_TEXT);

        size_t i = idx / WORD_LEN;
        int pos = WORD_LEN - idx % WORD_LEN - 1;
        if (val)
            AtomicOr(&mask[i], 1ULL << pos);
        else
            AtomicAnd(&mask[i], ~(1ULL << pos));
    }
    int getMinTrue(unsigned startIdx) const {
        unsigned idx = startIdx / WORD_LEN;
        int pos;

        if (startIdx % WORD_LEN) {
            // only interested in part of a word, clear bits before startIdx
            pos = WORD_LEN - startIdx % WORD_LEN;
            uintptr_t actualMask = mask[idx] & (((uintptr_t)1<<pos) - 1);
            idx++;
            if (-1 != (pos = BitScanRev(actualMask)))
                return idx*WORD_LEN - pos - 1;
        }

        while (idx<SZ)
            if (-1 != (pos = BitScanRev(mask[idx++])))
                return idx*WORD_LEN - pos - 1;
        return -1;
    }
public:
    void reset() { for (unsigned i=0; i<SZ; i++) mask[i] = 0; }
};

template<unsigned NUM>
class BitMaskMin : public BitMaskBasic<NUM> {
public:
    void set(size_t idx, bool val) { BitMaskBasic<NUM>::set(idx, val); }
    int getMinTrue(unsigned startIdx) const {
        return BitMaskBasic<NUM>::getMinTrue(startIdx);
    }
};

template<unsigned NUM>
class BitMaskMax : public BitMaskBasic<NUM> {
public:
    void set(size_t idx, bool val) {
        BitMaskBasic<NUM>::set(NUM - 1 - idx, val);
    }
    int getMaxTrue(unsigned startIdx) const {
        int p = BitMaskBasic<NUM>::getMinTrue(NUM-startIdx-1);
        return -1==p? -1 : (int)NUM - 1 - p;
    }
};


// The part of thread-specific data that can be modified by other threads.
// Such modifications must be protected by AllLocalCaches::listLock.
struct TLSRemote {
    TLSRemote *next,
              *prev;
};

// The list of all thread-local data; supporting cleanup of thread caches
class AllLocalCaches {
    TLSRemote  *head;
    MallocMutex listLock; // protects operations in the list
public:
    void registerThread(TLSRemote *tls);
    void unregisterThread(TLSRemote *tls);
    bool cleanup(ExtMemoryPool *extPool, bool cleanOnlyUnused);
    void markUnused();
    void reset() { head = NULL; }
};

class LifoList {
public:
    inline LifoList();
    inline void push(Block *block);
    inline Block *pop();
    inline Block *grab();

private:
    Block *top;
    MallocMutex lock;
};

/*
 * When a block that is not completely free is returned for reuse by other threads
 * this is where the block goes.
 *
 * LifoList assumes zero initialization; so below its constructors are omitted,
 * to avoid linking with C++ libraries on Linux.
 */

class OrphanedBlocks {
    LifoList bins[numBlockBinLimit];
public:
    Block *get(TLSData *tls, unsigned int size);
    void put(intptr_t binTag, Block *block);
    void reset();
    bool cleanup(Backend* backend);
};

/* cache blocks in range [MinSize; MaxSize) in bins with CacheStep
 TooLargeFactor -- when cache size treated "too large" in comparison to user data size
 OnMissFactor -- If cache miss occurred and cache was cleaned,
                 set ageThreshold to OnMissFactor * the difference
                 between current time and last time cache was cleaned.
 LongWaitFactor -- to detect rarely-used bins and forget about their usage history
*/
template<size_t MIN_SIZE, size_t MAX_SIZE, uint32_t CACHE_STEP, int TOO_LARGE,
         int ON_MISS, int LONG_WAIT>
struct LargeObjectCacheProps {
    static const size_t MinSize = MIN_SIZE, MaxSize = MAX_SIZE;
    static const uint32_t CacheStep = CACHE_STEP;
    static const int TooLargeFactor = TOO_LARGE, OnMissFactor = ON_MISS,
        LongWaitFactor = LONG_WAIT;
};

template<typename Props>
class LargeObjectCacheImpl {
private:
    // The number of bins to cache large objects.
    static const uint32_t numBins = (Props::MaxSize-Props::MinSize)/Props::CacheStep;
    // Current sizes of used and cached objects. It's calculated while we are
    // traversing bins, and used for isLOCTooLarge() check at the same time.
    class BinsSummary {
        size_t usedSz;
        size_t cachedSz;
    public:
        BinsSummary() : usedSz(0), cachedSz(0) {}
        // "too large" criteria
        bool isLOCTooLarge() const { return cachedSz > Props::TooLargeFactor*usedSz; }
        void update(size_t usedSize, size_t cachedSize) {
            usedSz += usedSize;
            cachedSz += cachedSize;
        }
        void reset() { usedSz = cachedSz = 0; }
    };
public:
    typedef BitMaskMax<numBins> BinBitMask;

    // 2-linked list of same-size cached blocks ordered by age (oldest on top)
    // TODO: are we really want the list to be 2-linked? This allows us
    // reduce memory consumption and do less operations under lock.
    // TODO: try to switch to 32-bit logical time to save space in CacheBin
    // and move bins to different cache lines.
    class CacheBin {
    private:
        LargeMemoryBlock *first,
                         *last;
  /* age of an oldest block in the list; equal to last->age, if last defined,
     used for quick cheching it without acquiring the lock. */
        uintptr_t         oldest;
  /* currAge when something was excluded out of list because of the age,
     not because of cache hit */
        uintptr_t         lastCleanedAge;
  /* Current threshold value for the blocks of a particular size.
     Set on cache miss. */
        intptr_t          ageThreshold;

  /* total size of all objects corresponding to the bin and allocated by user */
        size_t            usedSize,
  /* total size of all objects cached in the bin */
                          cachedSize;
  /* mean time of presence of block in the bin before successful reuse */
        intptr_t          meanHitRange;
  /* time of last get called for the bin */
        uintptr_t         lastGet;

        typename MallocAggregator<CacheBinOperation>::type aggregator;

        void ExecuteOperation(CacheBinOperation *op, ExtMemoryPool *extMemPool, BinBitMask *bitMask, int idx, bool longLifeTime = true);
  /* should be placed in zero-initialized memory, ctor not needed. */
        CacheBin();
    public:
        void init() { memset(this, 0, sizeof(CacheBin)); }
        void putList(ExtMemoryPool *extMemPool, LargeMemoryBlock *head, BinBitMask *bitMask, int idx);
        LargeMemoryBlock *get(ExtMemoryPool *extMemPool, size_t size, BinBitMask *bitMask, int idx);
        bool cleanToThreshold(ExtMemoryPool *extMemPool, BinBitMask *bitMask, uintptr_t currTime, int idx);
        bool releaseAllToBackend(ExtMemoryPool *extMemPool, BinBitMask *bitMask, int idx);
        void updateUsedSize(ExtMemoryPool *extMemPool, size_t size, BinBitMask *bitMask, int idx);

        void decreaseThreshold() {
            if (ageThreshold)
                ageThreshold = (ageThreshold + meanHitRange)/2;
        }
        void updateBinsSummary(BinsSummary *binsSummary) const {
            binsSummary->update(usedSize, cachedSize);
        }
        size_t getSize() const { return cachedSize; }
        size_t getUsedSize() const { return usedSize; }
        size_t reportStat(int num, FILE *f);
  /* ---------- unsafe methods used with the aggregator ---------- */
        void forgetOutdatedState(uintptr_t currTime);
        LargeMemoryBlock *putList(LargeMemoryBlock *head, LargeMemoryBlock *tail, BinBitMask *bitMask, int idx, int num);
        LargeMemoryBlock *get();
        LargeMemoryBlock *cleanToThreshold(uintptr_t currTime, BinBitMask *bitMask, int idx);
        LargeMemoryBlock *cleanAll(BinBitMask *bitMask, int idx);
        void updateUsedSize(size_t size, BinBitMask *bitMask, int idx) {
            if (!usedSize) bitMask->set(idx, true);
            usedSize += size;
            if (!usedSize && !first) bitMask->set(idx, false);
        }
        void updateMeanHitRange( intptr_t hitRange ) {
            hitRange = hitRange >= 0 ? hitRange : 0;
            meanHitRange = meanHitRange ? (meanHitRange + hitRange)/2 : hitRange;
        }
        void updateAgeThreshold( uintptr_t currTime ) {
            if (lastCleanedAge)
                ageThreshold = Props::OnMissFactor*(currTime - lastCleanedAge);
        }
        void updateCachedSize(size_t size) { cachedSize += size; }
        void setLastGet( uintptr_t newLastGet ) { lastGet = newLastGet; }
  /* -------------------------------------------------------- */
    };
private:
    intptr_t     tooLargeLOC; // how many times LOC was "too large"
    // for fast finding of used bins and bins with non-zero usedSize;
    // indexed from the end, as we need largest 1st
    BinBitMask   bitMask;
    // bins with lists of recently freed large blocks cached for re-use
    CacheBin bin[numBins];

public:
    static int sizeToIdx(size_t size) {
        MALLOC_ASSERT(Props::MinSize <= size && size < Props::MaxSize, ASSERT_TEXT);
        return (size-Props::MinSize)/Props::CacheStep;
    }
    static int getNumBins() { return numBins; }

    void putList(ExtMemoryPool *extMemPool, LargeMemoryBlock *largeBlock);
    LargeMemoryBlock *get(ExtMemoryPool *extMemPool, size_t size);

    void updateCacheState(ExtMemoryPool *extMemPool, DecreaseOrIncrease op, size_t size);
    bool regularCleanup(ExtMemoryPool *extMemPool, uintptr_t currAge, bool doThreshDecr);
    bool cleanAll(ExtMemoryPool *extMemPool);
    void reset() {
        tooLargeLOC = 0;
        for (int i = numBins-1; i >= 0; i--)
            bin[i].init();
        bitMask.reset();
    }
    void reportStat(FILE *f);
#if __TBB_MALLOC_WHITEBOX_TEST
    size_t getLOCSize() const;
    size_t getUsedSize() const;
#endif
};

class LargeObjectCache {
    static const size_t minLargeSize =  8*1024,
                        maxLargeSize =  8*1024*1024,
    // There are benchmarks of interest that should work well with objects of this size
                        maxHugeSize = 129*1024*1024;
public:
    // Difference between object sizes in large block bins
    static const uint32_t largeBlockCacheStep =  8*1024,
                          hugeBlockCacheStep = 512*1024;
private:
    typedef LargeObjectCacheProps<minLargeSize, maxLargeSize, largeBlockCacheStep, 2, 2, 16> LargeCacheTypeProps;
    typedef LargeObjectCacheProps<maxLargeSize, maxHugeSize, hugeBlockCacheStep, 1, 1, 4> HugeCacheTypeProps;
    typedef LargeObjectCacheImpl< LargeCacheTypeProps > LargeCacheType;
    typedef LargeObjectCacheImpl< HugeCacheTypeProps > HugeCacheType;

    // beginning of largeCache is more actively used and smaller than hugeCache,
    // so put hugeCache first to prevent false sharing
    // with LargeObjectCache's predecessor
    HugeCacheType hugeCache;
    LargeCacheType largeCache;

    /* logical time, incremented on each put/get operation
       To prevent starvation between pools, keep separately for each pool.
       Overflow is OK, as we only want difference between
       its current value and some recent.

       Both malloc and free should increment logical time, as in
       a different case multiple cached blocks would have same age,
       and accuracy of predictors suffers.
    */
    uintptr_t cacheCurrTime;

                     // memory pool that owns this LargeObjectCache,
    ExtMemoryPool *extMemPool; // strict 1:1 relation, never changed

    static int sizeToIdx(size_t size);
public:
    void init(ExtMemoryPool *memPool) { extMemPool = memPool; }
    void put(LargeMemoryBlock *largeBlock);
    void putList(LargeMemoryBlock *head);
    LargeMemoryBlock *get(size_t size);

    void updateCacheState(DecreaseOrIncrease op, size_t size);
    bool isCleanupNeededOnRange(uintptr_t range, uintptr_t currTime);
    bool doCleanup(uintptr_t currTime, bool doThreshDecr);

    bool decreasingCleanup();
    bool regularCleanup();
    bool cleanAll();
    void reset() {
        largeCache.reset();
        hugeCache.reset();
    }
    void reportStat(FILE *f);
#if __TBB_MALLOC_WHITEBOX_TEST
    size_t getLOCSize() const;
    size_t getUsedSize() const;
#endif
    static size_t alignToBin(size_t size) {
        return size<maxLargeSize? alignUp(size, largeBlockCacheStep)
            : alignUp(size, hugeBlockCacheStep);
    }

    uintptr_t getCurrTime() { return (uintptr_t)AtomicIncrement((intptr_t&)cacheCurrTime); }
    uintptr_t getCurrTimeRange(uintptr_t range) { return (uintptr_t)AtomicAdd((intptr_t&)cacheCurrTime, range)+1; }
    void registerRealloc(size_t oldSize, size_t newSize);
};

// select index size for BackRefMaster based on word size: default is uint32_t,
// uint16_t for 32-bit platforms
template<bool>
struct MasterIndexSelect {
    typedef uint32_t master_type;
};

template<>
struct MasterIndexSelect<false> {
    typedef uint16_t master_type;
};

class BackRefIdx { // composite index to backreference array
public:
    typedef MasterIndexSelect<4 < sizeof(uintptr_t)>::master_type master_t;
private:
    static const master_t invalid = ~master_t(0);
    master_t master;      // index in BackRefMaster
    uint16_t largeObj:1;  // is this object "large"?
    uint16_t offset  :15; // offset from beginning of BackRefBlock
public:
    BackRefIdx() : master(invalid) {}
    bool isInvalid() const { return master == invalid; }
    bool isLargeObject() const { return largeObj; }
    master_t getMaster() const { return master; }
    uint16_t getOffset() const { return offset; }

    // only newBackRef can modify BackRefIdx
    static BackRefIdx newBackRef(bool largeObj);
};

// Block header is used during block coalescing
// and must be preserved in used blocks.
class BlockI {
    intptr_t     blockState[2];
};

struct LargeMemoryBlock : public BlockI {
    MemoryPool       *pool;          // owner pool
    LargeMemoryBlock *next,          // ptrs in list of cached blocks
                     *prev,
    // 2-linked list of pool's large objects
    // Used to destroy backrefs on pool destroy (backrefs are global)
    // and for object releasing during pool reset.
                     *gPrev,
                     *gNext;
    uintptr_t         age;           // age of block while in cache
    size_t            objectSize;    // the size requested by a client
    size_t            unalignedSize; // the size requested from backend
    BackRefIdx        backRefIdx;    // cached here, used copy is in LargeObjectHdr
};

// global state of blocks currently in processing
class BackendSync {
    // Class instances should reside in zero-initialized memory!
    // The number of blocks currently removed from a bin and not returned back
    intptr_t inFlyBlocks;         // to another
    intptr_t binsModifications;   // incremented on every bin modification
    Backend *backend;
public:
    void init(Backend *b) { backend = b; }
    void blockConsumed() { AtomicIncrement(inFlyBlocks); }
    void binsModified() { AtomicIncrement(binsModifications); }
    void blockReleased() {
#if __TBB_MALLOC_BACKEND_STAT
        MALLOC_ITT_SYNC_RELEASING(&inFlyBlocks);
#endif
        AtomicIncrement(binsModifications);
        intptr_t prev = AtomicAdd(inFlyBlocks, -1);
        MALLOC_ASSERT(prev > 0, ASSERT_TEXT);
        suppress_unused_warning(prev);
    }
    intptr_t getNumOfMods() const { return FencedLoad(binsModifications); }
    // return true if need re-do the blocks search
    inline bool waitTillBlockReleased(intptr_t startModifiedCnt);
};

class CoalRequestQ { // queue of free blocks that coalescing was delayed
private:
    FreeBlock   *blocksToFree;
    BackendSync *bkndSync;
    // counted blocks in blocksToFree and that are leaved blocksToFree
    // and still in active coalescing
    intptr_t     inFlyBlocks;
public:
    void init(BackendSync *bSync) { bkndSync = bSync; }
    FreeBlock *getAll(); // return current list of blocks and make queue empty
    void putBlock(FreeBlock *fBlock);
    inline void blockWasProcessed();
    intptr_t blocksInFly() const { return FencedLoad(inFlyBlocks); }
};

class MemExtendingSema {
    intptr_t     active;
public:
    bool wait() {
        bool rescanBins = false;
        // up to 3 threads can add more memory from OS simultaneously,
        // rest of threads have to wait
        for (;;) {
            intptr_t prevCnt = FencedLoad(active);
            if (prevCnt < 3) {
                intptr_t n = AtomicCompareExchange(active, prevCnt+1, prevCnt);
                if (n == prevCnt)
                    break;
            } else {
                SpinWaitWhileEq(active, prevCnt);
                rescanBins = true;
                break;
            }
        }
        return rescanBins;
    }
    void signal() { AtomicAdd(active, -1); }
};

enum MemRegionType {
    // The region does not guarantee the block size.
    MEMREG_FLEXIBLE_SIZE = 0,
    // The region can hold exact number of blocks with the size of the
    // first reqested block.
    MEMREG_SEVERAL_BLOCKS,
    // The region holds only one block with a reqested size.
    MEMREG_ONE_BLOCK
};

class MemRegionList {
    MallocMutex regionListLock;
public:
    MemRegion  *head;
    void add(MemRegion *r);
    void remove(MemRegion *r);
    int reportStat(FILE *f);
};

class Backend {
private:
/* Blocks in range [minBinnedSize; getMaxBinnedSize()] are kept in bins,
   one region can contains several blocks. Larger blocks are allocated directly
   and one region always contains one block.
*/
    enum {
        minBinnedSize = 8*1024UL,
        /*   If huge pages are available, maxBinned_HugePage used.
             If not, maxBinned_SmallPage is the threshold.
             TODO: use pool's granularity for upper bound setting.*/
        maxBinned_SmallPage = 1024*1024UL,
        // TODO: support other page sizes
        maxBinned_HugePage = 4*1024*1024UL
    };
    enum {
        VALID_BLOCK_IN_BIN = 1 // valid block added to bin, not returned as result
    };
public:
    static const int freeBinsNum =
        (maxBinned_HugePage-minBinnedSize)/LargeObjectCache::largeBlockCacheStep + 1;

    // if previous access missed per-thread slabs pool,
    // allocate numOfSlabAllocOnMiss blocks in advance
    static const int numOfSlabAllocOnMiss = 2;

    enum {
        NO_BIN = -1,
        // special bin for blocks >= maxBinned_HugePage, blocks go to this bin
        // when pool is created with keepAllMemory policy
        // TODO: currently this bin is scanned using "1st fit", as it accumulates
        // blocks of different sizes, "best fit" is preferred in terms of fragmentation
        HUGE_BIN = freeBinsNum-1
    };

    // Bin keeps 2-linked list of free blocks. It must be 2-linked
    // because during coalescing a block it's removed from a middle of the list.
    struct Bin {
        FreeBlock   *head,
                    *tail;
        MallocMutex  tLock;

        void removeBlock(FreeBlock *fBlock);
        void reset() { head = tail = 0; }
        bool empty() const { return !head; }

        size_t countFreeBlocks();
        size_t reportFreeBlocks(FILE *f);
        void reportStat(FILE *f);
    };

    typedef BitMaskMin<Backend::freeBinsNum> BitMaskBins;

    // array of bins supplemented with bitmask for fast finding of non-empty bins
    class IndexedBins {
        BitMaskBins bitMask;
        Bin         freeBins[Backend::freeBinsNum];
        FreeBlock *getFromBin(int binIdx, BackendSync *sync, size_t size,
                              bool resSlabAligned, bool alignedBin, bool wait,
                              int *resLocked);
    public:
        FreeBlock *findBlock(int nativeBin, BackendSync *sync, size_t size,
                             bool resSlabAligned, bool alignedBin, int *numOfLockedBins);
        bool tryReleaseRegions(int binIdx, Backend *backend);
        void lockRemoveBlock(int binIdx, FreeBlock *fBlock);
        void addBlock(int binIdx, FreeBlock *fBlock, size_t blockSz, bool addToTail);
        bool tryAddBlock(int binIdx, FreeBlock *fBlock, bool addToTail);
        int getMinNonemptyBin(unsigned startBin) const {
            int p = bitMask.getMinTrue(startBin);
            return p == -1 ? Backend::freeBinsNum : p;
        }
        void verify();
        void reset();
        void reportStat(FILE *f);
    };

private:
    class AdvRegionsBins {
        BitMaskBins bins;
    public:
        void registerBin(int regBin) { bins.set(regBin, 1); }
        int getMinUsedBin(int start) const { return bins.getMinTrue(start); }
        void reset() { bins.reset(); }
    };
    // auxiliary class to atomic maximum request finding
    class MaxRequestComparator {
        const Backend *backend;
    public:
        MaxRequestComparator(const Backend *be) : backend(be) {}
        inline bool operator()(size_t oldMaxReq, size_t requestSize) const;
    };

#if CHECK_ALLOCATION_RANGE
    // Keep min and max of all addresses requested from OS,
    // use it for checking memory possibly allocated by replaced allocators
    // and for debugging purposes. Valid only for default memory pool.
    class UsedAddressRange {
        static const uintptr_t ADDRESS_UPPER_BOUND = UINTPTR_MAX;

        uintptr_t   leftBound,
                    rightBound;
        MallocMutex mutex;
    public:
        // rightBound is zero-initialized
        void init() { leftBound = ADDRESS_UPPER_BOUND; }
        void registerAlloc(uintptr_t left, uintptr_t right);
        void registerFree(uintptr_t left, uintptr_t right);
        // as only left and right bounds are kept, we can return true
        // for pointer not allocated by us, if more than single region
        // was requested from OS
        bool inRange(void *ptr) const {
            const uintptr_t p = (uintptr_t)ptr;
            return leftBound<=p && p<=rightBound;
        }
    };
#else
    class UsedAddressRange {
    public:
        void init() { }
        void registerAlloc(uintptr_t, uintptr_t) {}
        void registerFree(uintptr_t, uintptr_t) {}
        bool inRange(void *) const { return true; }
    };
#endif

    ExtMemoryPool   *extMemPool;
    // used for release every region on pool destroying
    MemRegionList    regionList;

    CoalRequestQ     coalescQ; // queue of coalescing requests
    BackendSync      bkndSync;
    // semaphore protecting adding more more memory from OS
    MemExtendingSema memExtendingSema;
    size_t           totalMemSize,
                     memSoftLimit;
    UsedAddressRange usedAddrRange;
    // to keep 1st allocation large than requested, keep bootstrapping status
    enum {
        bootsrapMemNotDone = 0,
        bootsrapMemInitializing,
        bootsrapMemDone
    };
    intptr_t         bootsrapMemStatus;
    MallocMutex      bootsrapMemStatusMutex;

    // Using of maximal observed requested size allows decrease
    // memory consumption for small requests and decrease fragmentation
    // for workloads when small and large allocation requests are mixed.
    // TODO: decrease, not only increase it
    size_t           maxRequestedSize;

    FreeBlock *addNewRegion(size_t size, MemRegionType type, bool addToBin);
    FreeBlock *findBlockInRegion(MemRegion *region, size_t exactBlockSize);
    void startUseBlock(MemRegion *region, FreeBlock *fBlock, bool addToBin);
    void releaseRegion(MemRegion *region);

    FreeBlock *releaseMemInCaches(intptr_t startModifiedCnt,
                                  int *lockedBinsThreshold, int numOfLockedBins);
    void requestBootstrapMem();
    FreeBlock *askMemFromOS(size_t totalReqSize, intptr_t startModifiedCnt,
                            int *lockedBinsThreshold, int numOfLockedBins,
                            bool *splittable);
    FreeBlock *genericGetBlock(int num, size_t size, bool resSlabAligned);
    void genericPutBlock(FreeBlock *fBlock, size_t blockSz);
    FreeBlock *splitUnalignedBlock(FreeBlock *fBlock, int num, size_t size,
                              bool needAlignedRes);
    FreeBlock *splitAlignedBlock(FreeBlock *fBlock, int num, size_t size,
                            bool needAlignedRes);

    FreeBlock *doCoalesc(FreeBlock *fBlock, MemRegion **memRegion);
    bool coalescAndPutList(FreeBlock *head, bool forceCoalescQDrop, bool reportBlocksProcessed);
    void coalescAndPut(FreeBlock *fBlock, size_t blockSz);

    void removeBlockFromBin(FreeBlock *fBlock);

    void *allocRawMem(size_t &size);
    bool freeRawMem(void *object, size_t size);

    void putLargeBlock(LargeMemoryBlock *lmb);
    void releaseCachesToLimit();
public:
    bool scanCoalescQ(bool forceCoalescQDrop);
    intptr_t blocksInCoalescing() const { return coalescQ.blocksInFly(); }
    void verify();
    void init(ExtMemoryPool *extMemoryPool);
    void reset();
    bool destroy();
    bool clean(); // clean on caches cleanup
    void reportStat(FILE *f);

    BlockI *getSlabBlock(int num) {
        BlockI *b = (BlockI*)
            genericGetBlock(num, slabSize, /*resSlabAligned=*/true);
        MALLOC_ASSERT(isAligned(b, slabSize), ASSERT_TEXT);
        return b;
    }
    void putSlabBlock(BlockI *block) {
        genericPutBlock((FreeBlock *)block, slabSize);
    }
    void *getBackRefSpace(size_t size, bool *rawMemUsed);
    void putBackRefSpace(void *b, size_t size, bool rawMemUsed);

    bool inUserPool() const;

    LargeMemoryBlock *getLargeBlock(size_t size);
    void returnLargeObject(LargeMemoryBlock *lmb);

    void *remap(void *ptr, size_t oldSize, size_t newSize, size_t alignment);

    void setRecommendedMaxSize(size_t softLimit) {
        memSoftLimit = softLimit;
        releaseCachesToLimit();
    }
    inline size_t getMaxBinnedSize() const;

    bool ptrCanBeValid(void *ptr) const { return usedAddrRange.inRange(ptr); }

#if __TBB_MALLOC_WHITEBOX_TEST
    size_t getTotalMemSize() const { return totalMemSize; }
#endif
private:
    static int sizeToBin(size_t size) {
        if (size >= maxBinned_HugePage)
            return HUGE_BIN;
        else if (size < minBinnedSize)
            return NO_BIN;

        int bin = (size - minBinnedSize)/LargeObjectCache::largeBlockCacheStep;

        MALLOC_ASSERT(bin < HUGE_BIN, "Invalid size.");
        return bin;
    }
#if __TBB_MALLOC_BACKEND_STAT
    static size_t binToSize(int bin) {
        MALLOC_ASSERT(bin <= HUGE_BIN, "Invalid bin.");

        return bin*LargeObjectCache::largeBlockCacheStep + minBinnedSize;
    }
#endif
    static bool toAlignedBin(FreeBlock *block, size_t size) {
        return isAligned((char*)block+size, slabSize)
            && size >= slabSize;
    }

    // register bins related to advance regions
    AdvRegionsBins advRegBins;
    IndexedBins freeLargeBins,
                freeAlignedBins;
};

class AllLargeBlocksList {
    MallocMutex       largeObjLock;
    LargeMemoryBlock *loHead;
public:
    void add(LargeMemoryBlock *lmb);
    void remove(LargeMemoryBlock *lmb);
    template<bool poolDestroy> void releaseAll(Backend *backend);
};

struct ExtMemoryPool {
    Backend           backend;
    LargeObjectCache  loc;
    AllLocalCaches    allLocalCaches;
    OrphanedBlocks    orphanedBlocks;

    intptr_t          poolId;
    // To find all large objects. Used during user pool destruction,
    // to release all backreferences in large blocks (slab blocks do not have them).
    AllLargeBlocksList lmbList;
    // Callbacks to be used instead of MapMemory/UnmapMemory.
    rawAllocType      rawAlloc;
    rawFreeType       rawFree;
    size_t            granularity;
    bool              keepAllMemory,
                      delayRegsReleasing,
    // TODO: implements fixedPool with calling rawFree on destruction
                      fixedPool;
    TLSKey            tlsPointerKey;  // per-pool TLS key

    bool init(intptr_t poolId, rawAllocType rawAlloc, rawFreeType rawFree,
              size_t granularity, bool keepAllMemory, bool fixedPool);
    bool initTLS();

    // i.e., not system default pool for scalable_malloc/scalable_free
    bool userPool() const { return rawAlloc; }

     // true if something has been released
    bool softCachesCleanup();
    bool releaseAllLocalCaches();
    bool hardCachesCleanup();
    void *remap(void *ptr, size_t oldSize, size_t newSize, size_t alignment);
    bool reset() {
        loc.reset();
        allLocalCaches.reset();
        orphanedBlocks.reset();
        bool ret = tlsPointerKey.destroy();
        backend.reset();
        return ret;
    }
    bool destroy() {
        MALLOC_ASSERT(isPoolValid(),
                      "Possible double pool_destroy or heap corruption");
        if (!userPool()) {
            loc.reset();
            allLocalCaches.reset();
        }
        // pthread_key_dtors must be disabled before memory unmapping
        // TODO: race-free solution
        bool ret = tlsPointerKey.destroy();
        if (rawFree || !userPool())
            ret &= backend.destroy();
        // pool is not valid after this point
        granularity = 0;
        return ret;
    }
    void delayRegionsReleasing(bool mode) { delayRegsReleasing = mode; }
    inline bool regionsAreReleaseable() const;

    LargeMemoryBlock *mallocLargeObject(MemoryPool *pool, size_t allocationSize);
    void freeLargeObject(LargeMemoryBlock *lmb);
    void freeLargeObjectList(LargeMemoryBlock *head);
    // use granulatity as marker for pool validity
    bool isPoolValid() const { return granularity; }
};

inline bool Backend::inUserPool() const { return extMemPool->userPool(); }

struct LargeObjectHdr {
    LargeMemoryBlock *memoryBlock;
    /* Backreference points to LargeObjectHdr.
       Duplicated in LargeMemoryBlock to reuse in subsequent allocations. */
    BackRefIdx       backRefIdx;
};

struct FreeObject {
    FreeObject  *next;
};

// An TBB allocator mode that can be controlled by user
// via API/environment variable. Must be placed in zero-initialized memory.
// External synchronization assumed.
// TODO: TBB_VERSION support
class AllocControlledMode {
    intptr_t val;
    bool     setDone;
public:
    bool ready() const { return setDone; }
    intptr_t get() const {
        MALLOC_ASSERT(setDone, ASSERT_TEXT);
        return val;
    }
    void set(intptr_t newVal) { // note set() can be called before init()
        val = newVal;
        setDone = true;
    }
    // envName - environment variable to get controlled mode
    void initReadEnv(const char *envName, intptr_t defaultVal);
};

// init() and printStatus() is called only under global initialization lock.
// Race is possible between registerAllocation() and registerReleasing(),
// harm is that up to single huge page releasing is missed (because failure
// to get huge page is registered only 1st time), that is negligible.
// setMode is also can be called concurrently.
// Object must reside in zero-initialized memory
// TODO: can we check for huge page presence during every 10th mmap() call
// in case huge page is released by another process?
class HugePagesStatus {
private:
    AllocControlledMode requestedMode; // changed only by user
               // to keep enabled and requestedMode consistent
    MallocMutex setModeLock;
    size_t      pageSize;
    intptr_t    needActualStatusPrint;

    static void doPrintStatus(bool state, const char *stateName);
public:
    // both variables are changed only inside HugePagesStatus
    intptr_t    enabled;
    // Have we got huge pages at all? It's used when large hugepage-aligned
    // region is releasing, to find can it release some huge pages or not.
    intptr_t    wasObserved;

    // If memory mapping size is a multiple of huge page size, some OS kernels
    // can use huge pages transparently (i.e. even if not explicitly enabled).
    // Use this when huge pages are requested.
    size_t recommendedGranularity() const {
        if (requestedMode.ready())
            return requestedMode.get()? pageSize : 0;
        else
            return 2048*1024; // the mode is not yet known; assume typical 2MB huge pages
    }
    void printStatus();
    void registerAllocation(bool available);
    void registerReleasing(void* addr, size_t size);

    void init(size_t hugePageSize) {
        MALLOC_ASSERT(!hugePageSize || isPowerOfTwo(hugePageSize),
                      "Only memory pages of a power-of-two size are supported.");
        MALLOC_ASSERT(!pageSize, "Huge page size can't be set twice.");
        pageSize = hugePageSize;

        MallocMutex::scoped_lock lock(setModeLock);
        requestedMode.initReadEnv("TBB_MALLOC_USE_HUGE_PAGES", 0);
        enabled = pageSize && requestedMode.get();
    }
    void setMode(intptr_t newVal) {
        MallocMutex::scoped_lock lock(setModeLock);
        requestedMode.set(newVal);
        enabled = pageSize && newVal;
    }
    void reset() {
        pageSize = 0;
        needActualStatusPrint = enabled = wasObserved = 0;
    }
};

extern HugePagesStatus hugePages;

/******* A helper class to support overriding malloc with scalable_malloc *******/
#if MALLOC_CHECK_RECURSION

class RecursiveMallocCallProtector {
    // pointer to an automatic data of holding thread
    static void       *autoObjPtr;
    static MallocMutex rmc_mutex;
    static pthread_t   owner_thread;
/* Under FreeBSD 8.0 1st call to any pthread function including pthread_self
   leads to pthread initialization, that causes malloc calls. As 1st usage of
   RecursiveMallocCallProtector can be before pthread initialized, pthread calls
   can't be used in 1st instance of RecursiveMallocCallProtector.
   RecursiveMallocCallProtector is used 1st time in checkInitialization(),
   so there is a guarantee that on 2nd usage pthread is initialized.
   No such situation observed with other supported OSes.
 */
#if __FreeBSD__
    static bool        canUsePthread;
#else
    static const bool  canUsePthread = true;
#endif
/*
  The variable modified in checkInitialization,
  so can be read without memory barriers.
 */
    static bool mallocRecursionDetected;

    MallocMutex::scoped_lock* lock_acquired;
    char scoped_lock_space[sizeof(MallocMutex::scoped_lock)+1];

    static uintptr_t absDiffPtr(void *x, void *y) {
        uintptr_t xi = (uintptr_t)x, yi = (uintptr_t)y;
        return xi > yi ? xi - yi : yi - xi;
    }
public:

    RecursiveMallocCallProtector() : lock_acquired(NULL) {
        lock_acquired = new (scoped_lock_space) MallocMutex::scoped_lock( rmc_mutex );
        if (canUsePthread)
            owner_thread = pthread_self();
        autoObjPtr = &scoped_lock_space;
    }
    ~RecursiveMallocCallProtector() {
        if (lock_acquired) {
            autoObjPtr = NULL;
            lock_acquired->~scoped_lock();
        }
    }
    static bool sameThreadActive() {
        if (!autoObjPtr) // fast path
            return false;
        // Some thread has an active recursive call protector; check if the current one.
        // Exact pthread_self based test
        if (canUsePthread) {
            if (pthread_equal( owner_thread, pthread_self() )) {
                mallocRecursionDetected = true;
                return true;
            } else
                return false;
        }
        // inexact stack size based test
        const uintptr_t threadStackSz = 2*1024*1024;
        int dummy;
        return absDiffPtr(autoObjPtr, &dummy)<threadStackSz;
    }
    static bool noRecursion();
/* The function is called on 1st scalable_malloc call to check if malloc calls
   scalable_malloc (nested call must set mallocRecursionDetected). */
    static void detectNaiveOverload() {
        if (!malloc_proxy) {
#if __FreeBSD__
/* If !canUsePthread, we can't call pthread_self() before, but now pthread
   is already on, so can do it. */
            if (!canUsePthread) {
                canUsePthread = true;
                owner_thread = pthread_self();
            }
#endif
            free(malloc(1));
        }
    }
};

#else

class RecursiveMallocCallProtector {
public:
    RecursiveMallocCallProtector() {}
    ~RecursiveMallocCallProtector() {}
};

#endif  /* MALLOC_CHECK_RECURSION */

bool isMallocInitializedExt();

unsigned int getThreadId();

bool initBackRefMaster(Backend *backend);
void destroyBackRefMaster(Backend *backend);
void removeBackRef(BackRefIdx backRefIdx);
void setBackRef(BackRefIdx backRefIdx, void *newPtr);
void *getBackRef(BackRefIdx backRefIdx);

} // namespace internal
} // namespace rml

#endif // __TBB_tbbmalloc_internal_H
