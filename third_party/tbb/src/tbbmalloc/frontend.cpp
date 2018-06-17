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


#include "tbbmalloc_internal.h"
#include <errno.h>
#include <new>        /* for placement new */
#include <string.h>   /* for memset */

#include "../tbb/tbb_version.h"
#include "../tbb/itt_notify.h" // for __TBB_load_ittnotify()

#if USE_PTHREAD
    #define TlsSetValue_func pthread_setspecific
    #define TlsGetValue_func pthread_getspecific
    #define GetMyTID() pthread_self()
    #include <sched.h>
    inline void do_yield() {sched_yield();}
    extern "C" { static void mallocThreadShutdownNotification(void*); }
    #if __sun || __SUNPRO_CC
    #define __asm__ asm
    #endif
    #include <unistd.h> // sysconf(_SC_PAGESIZE)
#elif USE_WINTHREAD
    #define GetMyTID() GetCurrentThreadId()
#if __TBB_WIN8UI_SUPPORT
    #include<thread>
    #define TlsSetValue_func FlsSetValue
    #define TlsGetValue_func FlsGetValue
    #define TlsAlloc() FlsAlloc(NULL)
    #define TLS_ALLOC_FAILURE FLS_OUT_OF_INDEXES
    #define TlsFree FlsFree
    inline void do_yield() {std::this_thread::yield();}
#else
    #define TlsSetValue_func TlsSetValue
    #define TlsGetValue_func TlsGetValue
    #define TLS_ALLOC_FAILURE TLS_OUT_OF_INDEXES
    inline void do_yield() {SwitchToThread();}
#endif
#else
    #error Must define USE_PTHREAD or USE_WINTHREAD

#endif


#define FREELIST_NONBLOCKING 1

namespace rml {
class MemoryPool;
namespace internal {

class Block;
class MemoryPool;

#if MALLOC_CHECK_RECURSION

inline bool isMallocInitialized();

bool RecursiveMallocCallProtector::noRecursion() {
    MALLOC_ASSERT(isMallocInitialized(),
                  "Recursion status can be checked only when initialization was done.");
    return !mallocRecursionDetected;
}

#endif // MALLOC_CHECK_RECURSION

/** Support for handling the special UNUSABLE pointer state **/
const intptr_t UNUSABLE = 0x1;
inline bool isSolidPtr( void* ptr ) {
    return (UNUSABLE|(intptr_t)ptr)!=UNUSABLE;
}
inline bool isNotForUse( void* ptr ) {
    return (intptr_t)ptr==UNUSABLE;
}

/*
 * Block::objectSize value used to mark blocks allocated by startupAlloc
 */
const uint16_t startupAllocObjSizeMark = ~(uint16_t)0;

/*
 * The following constant is used to define the size of struct Block, the block header.
 * The intent is to have the size of a Block multiple of the cache line size, this allows us to
 * get good alignment at the cost of some overhead equal to the amount of padding included in the Block.
 */
const int blockHeaderAlignment = estimatedCacheLineSize;

/********* The data structures and global objects        **************/

/*
 * The malloc routines themselves need to be able to occasionally malloc some space,
 * in order to set up the structures used by the thread local structures. This
 * routine performs that functions.
 */
class BootStrapBlocks {
    MallocMutex bootStrapLock;
    Block      *bootStrapBlock;
    Block      *bootStrapBlockUsed;
    FreeObject *bootStrapObjectList;
public:
    void *allocate(MemoryPool *memPool, size_t size);
    void free(void* ptr);
    void reset();
};

#if USE_INTERNAL_TID
class ThreadId {
    static tls_key_t Tid_key;
    static intptr_t ThreadCount;

    unsigned int id;

    static unsigned int tlsNumber() {
        unsigned int result = reinterpret_cast<intptr_t>(TlsGetValue_func(Tid_key));
        if( !result ) {
            RecursiveMallocCallProtector scoped;
            // Thread-local value is zero -> first call from this thread,
            // need to initialize with next ID value (IDs start from 1)
            result = AtomicIncrement(ThreadCount); // returned new value!
            TlsSetValue_func( Tid_key, reinterpret_cast<void*>(result) );
        }
        return result;
    }
public:
    static void init() {
#if USE_WINTHREAD
        Tid_key = TlsAlloc();
#else
        int status = pthread_key_create( &Tid_key, NULL );
        if ( status ) {
            fprintf (stderr, "The memory manager cannot create tls key during initialization; exiting \n");
            exit(1);
        }
#endif /* USE_WINTHREAD */
    }
    static void destroy() {
        if( Tid_key ) {
#if USE_WINTHREAD
            BOOL status = !(TlsFree( Tid_key ));  // fail is zero
#else
            int status = pthread_key_delete( Tid_key );
#endif /* USE_WINTHREAD */
            if ( status ) {
                fprintf (stderr, "The memory manager cannot delete tls key; exiting \n");
                exit(1);
            }
            Tid_key = 0;
        }
    }

    ThreadId() : id(ThreadId::tlsNumber()) {}
    bool isCurrentThreadId() const { return id == ThreadId::tlsNumber(); }

#if COLLECT_STATISTICS || MALLOC_TRACE
    friend unsigned int getThreadId() { return ThreadId::tlsNumber(); }
#endif
#if COLLECT_STATISTICS
    static unsigned getMaxThreadId() { return ThreadCount; }

    friend int STAT_increment(ThreadId tid, int bin, int ctr);
#endif
};

tls_key_t ThreadId::Tid_key;
intptr_t ThreadId::ThreadCount;

#if COLLECT_STATISTICS
int STAT_increment(ThreadId tid, int bin, int ctr)
{
    return ::STAT_increment(tid.id, bin, ctr);
}
#endif

#else // USE_INTERNAL_TID

class ThreadId {
#if USE_PTHREAD
    pthread_t tid;
#else
    DWORD     tid;
#endif
public:
    ThreadId() : tid(GetMyTID()) {}
#if USE_PTHREAD
    bool isCurrentThreadId() const { return pthread_equal(pthread_self(), tid); }
#else
    bool isCurrentThreadId() const { return GetCurrentThreadId() == tid; }
#endif
    static void init() {}
    static void destroy() {}
};

#endif // USE_INTERNAL_TID

/*********** Code to provide thread ID and a thread-local void pointer **********/

bool TLSKey::init()
{
#if USE_WINTHREAD
    TLS_pointer_key = TlsAlloc();
    if (TLS_pointer_key == TLS_ALLOC_FAILURE)
        return false;
#else
    int status = pthread_key_create( &TLS_pointer_key, mallocThreadShutdownNotification );
    if ( status )
        return false;
#endif /* USE_WINTHREAD */
    return true;
}

bool TLSKey::destroy()
{
#if USE_WINTHREAD
    BOOL status1 = !(TlsFree(TLS_pointer_key)); // fail is zero
#else
    int status1 = pthread_key_delete(TLS_pointer_key);
#endif /* USE_WINTHREAD */
    MALLOC_ASSERT(!status1, "The memory manager cannot delete tls key.");
    return status1==0;
}

inline TLSData* TLSKey::getThreadMallocTLS() const
{
    return (TLSData *)TlsGetValue_func( TLS_pointer_key );
}

inline void TLSKey::setThreadMallocTLS( TLSData * newvalue ) {
    RecursiveMallocCallProtector scoped;
    TlsSetValue_func( TLS_pointer_key, newvalue );
}

/* The 'next' field in the block header has to maintain some invariants:
 *   it needs to be on a 16K boundary and the first field in the block.
 *   Any value stored there needs to have the lower 14 bits set to 0
 *   so that various assert work. This means that if you want to smash this memory
 *   for debugging purposes you will need to obey this invariant.
 * The total size of the header needs to be a power of 2 to simplify
 * the alignment requirements. For now it is a 128 byte structure.
 * To avoid false sharing, the fields changed only locally are separated
 * from the fields changed by foreign threads.
 * Changing the size of the block header would require to change
 * some bin allocation sizes, in particular "fitting" sizes (see above).
 */
class Bin;
class StartupBlock;

class MemoryPool {
    // if no explicit grainsize, expect to see malloc in user's pAlloc
    // and set reasonable low granularity
    static const size_t defaultGranularity = estimatedCacheLineSize;

    MemoryPool();                  // deny
public:
    static MallocMutex  memPoolListLock;

    // list of all active pools is used to release
    // all TLS data on thread termination or library unload
    MemoryPool    *next,
                  *prev;
    ExtMemoryPool  extMemPool;
    BootStrapBlocks bootStrapBlocks;

    bool init(intptr_t poolId, const MemPoolPolicy* memPoolPolicy);
    static void initDefaultPool();
    bool reset();
    bool destroy();
    void onThreadShutdown(TLSData *tlsData);

    inline TLSData *getTLS(bool create);
    void clearTLS() { extMemPool.tlsPointerKey.setThreadMallocTLS(NULL); }

    Block *getEmptyBlock(size_t size);
    void returnEmptyBlock(Block *block, bool poolTheBlock);

    // get/put large object to/from local large object cache
    void *getFromLLOCache(TLSData *tls, size_t size, size_t alignment);
    void putToLLOCache(TLSData *tls, void *object);
};

static intptr_t defaultMemPool_space[sizeof(MemoryPool)/sizeof(intptr_t) +
                                     (sizeof(MemoryPool)%sizeof(intptr_t)? 1 : 0)];
static MemoryPool *defaultMemPool = (MemoryPool*)defaultMemPool_space;
const size_t MemoryPool::defaultGranularity;
// zero-initialized
MallocMutex  MemoryPool::memPoolListLock;
// TODO: move huge page status to default pool, because that's its states
HugePagesStatus hugePages;
static bool usedBySrcIncluded = false;

// Padding helpers
template<size_t padd>
struct PaddingImpl {
    size_t       __padding[padd];
};

template<>
struct PaddingImpl<0> {};

template<int N>
struct Padding : PaddingImpl<N/sizeof(size_t)> {};

// Slab block is 16KB-aligned. To prevent false sharing, separate locally-accessed
// fields and fields commonly accessed by not owner threads.
class GlobalBlockFields : public BlockI {
protected:
    FreeObject  *publicFreeList;
    Block       *nextPrivatizable;
    MemoryPool  *poolPtr;
};

class LocalBlockFields : public GlobalBlockFields, Padding<blockHeaderAlignment - sizeof(GlobalBlockFields)>  {
protected:
    Block       *next;
    Block       *previous;        /* Use double linked list to speed up removal */
    FreeObject  *bumpPtr;         /* Bump pointer moves from the end to the beginning of a block */
    FreeObject  *freeList;
    /* Pointer to local data for the owner thread. Used for fast finding tls
       when releasing object from a block that current thread owned.
       NULL for orphaned blocks. */
    TLSData     *tlsPtr;
    ThreadId     ownerTid;        /* the ID of the thread that owns or last owned the block */
    BackRefIdx   backRefIdx;
    uint16_t     allocatedCount;  /* Number of objects allocated (obviously by the owning thread) */
    uint16_t     objectSize;
    bool         isFull;

    friend class FreeBlockPool;
    friend class StartupBlock;
    friend class LifoList;
    friend void *BootStrapBlocks::allocate(MemoryPool *, size_t);
    friend bool OrphanedBlocks::cleanup(Backend*);
    friend Block *MemoryPool::getEmptyBlock(size_t);
};

// Use inheritance to guarantee that a user data start on next cache line.
// Can't use member for it, because when LocalBlockFields already on cache line,
// we must have no additional memory consumption for all compilers.
class Block : public LocalBlockFields,
              Padding<2*blockHeaderAlignment - sizeof(LocalBlockFields)> {
public:
    bool empty() const { return allocatedCount==0 && !isSolidPtr(publicFreeList); }
    inline FreeObject* allocate();
    inline FreeObject *allocateFromFreeList();
    inline bool emptyEnoughToUse();
    bool freeListNonNull() { return freeList; }
    void freePublicObject(FreeObject *objectToFree);
    inline void freeOwnObject(void *object);
    void reset();
    void privatizePublicFreeList( bool cleanup = false );
    void restoreBumpPtr();
    void privatizeOrphaned(TLSData *tls, unsigned index);
    void shareOrphaned(intptr_t binTag, unsigned index);
    unsigned int getSize() const {
        MALLOC_ASSERT(isStartupAllocObject() || objectSize<minLargeObjectSize,
                      "Invalid object size");
        return isStartupAllocObject()? 0 : objectSize;
    }
    const BackRefIdx *getBackRefIdx() const { return &backRefIdx; }
    inline bool isOwnedByCurrentThread() const;
    bool isStartupAllocObject() const { return objectSize == startupAllocObjSizeMark; }
    inline FreeObject *findObjectToFree(const void *object) const;
    void checkFreePrecond(const void *object) const {
#if MALLOC_DEBUG
        const char *msg = "Possible double free or heap corruption.";
        // small objects are always at least sizeof(size_t) Byte aligned,
        // try to check this before this dereference as for invalid objects
        // this may be unreadable
        MALLOC_ASSERT(isAligned(object, sizeof(size_t)), "Try to free invalid small object");
        // releasing to free slab
        MALLOC_ASSERT(allocatedCount>0, msg);
        // must not point to slab's header
        MALLOC_ASSERT((uintptr_t)object - (uintptr_t)this >= sizeof(Block), msg);
        if (startupAllocObjSizeMark == objectSize) // startup block
            MALLOC_ASSERT(object<=bumpPtr, msg);
        else {
            // non-startup objects are 8 Byte aligned
            MALLOC_ASSERT(isAligned(object, 8), "Try to free invalid small object");
            MALLOC_ASSERT(allocatedCount <= (slabSize-sizeof(Block))/objectSize
                          && (!bumpPtr || object>bumpPtr), msg);
            FreeObject *toFree = findObjectToFree(object);
            // check against head of freeList, as this is mostly
            // expected after double free
            MALLOC_ASSERT(toFree != freeList, msg);
            // check against head of publicFreeList, to detect double free
            // involving foreign thread
            MALLOC_ASSERT(toFree != publicFreeList, msg);
        }
#else
        suppress_unused_warning(object);
#endif
    }
    void initEmptyBlock(TLSData *tls, size_t size);
    size_t findObjectSize(void *object) const;
    MemoryPool *getMemPool() const { return poolPtr; } // do not use on the hot path!

protected:
    void cleanBlockHeader();

private:
    static const float emptyEnoughRatio; /* "Reactivate" a block if this share of its objects is free. */

    inline FreeObject *allocateFromBumpPtr();
    inline FreeObject *findAllocatedObject(const void *address) const;
    inline bool isProperlyPlaced(const void *object) const;
    inline void markOwned(TLSData *tls) {
        MALLOC_ASSERT(!tlsPtr, ASSERT_TEXT);
        ownerTid = ThreadId(); /* save the ID of the current thread */
        tlsPtr = tls;
    }
    inline void markOrphaned() {
        MALLOC_ASSERT(tlsPtr, ASSERT_TEXT);
        tlsPtr = NULL;
    }

    friend class Bin;
    friend class TLSData;
    friend bool MemoryPool::destroy();
};

const float Block::emptyEnoughRatio = 1.0 / 4.0;

MALLOC_STATIC_ASSERT(sizeof(Block) <= 2*estimatedCacheLineSize,
    "The class Block does not fit into 2 cache lines on this platform. "
    "Defining USE_INTERNAL_TID may help to fix it.");

class Bin {
    Block      *activeBlk;
    Block      *mailbox;
    MallocMutex mailLock;

public:
    inline Block* getActiveBlock() const { return activeBlk; }
    void resetActiveBlock() { activeBlk = 0; }
    bool activeBlockUnused() const { return activeBlk && !activeBlk->allocatedCount; }
    inline void setActiveBlock(Block *block);
    inline Block* setPreviousBlockActive();
    Block* getPublicFreeListBlock();
    void moveBlockToFront(Block *block);
    void processLessUsedBlock(MemoryPool *memPool, Block *block);

    void outofTLSBin(Block* block);
    void verifyTLSBin(size_t size) const;
    void pushTLSBin(Block* block);

    void verifyInitState() const {
        MALLOC_ASSERT( activeBlk == 0, ASSERT_TEXT );
        MALLOC_ASSERT( mailbox == 0, ASSERT_TEXT );
    }

    friend void Block::freePublicObject (FreeObject *objectToFree);
};

/********* End of the data structures                    **************/

/*
 * There are bins for all 8 byte aligned objects less than this segregated size; 8 bins in total
 */
const uint32_t minSmallObjectIndex = 0;
const uint32_t numSmallObjectBins = 8;
const uint32_t maxSmallObjectSize = 64;

/*
 * There are 4 bins between each couple of powers of 2 [64-128-256-...]
 * from maxSmallObjectSize till this size; 16 bins in total
 */
const uint32_t minSegregatedObjectIndex = minSmallObjectIndex+numSmallObjectBins;
const uint32_t numSegregatedObjectBins = 16;
const uint32_t maxSegregatedObjectSize = 1024;

/*
 * And there are 5 bins with allocation sizes that are multiples of estimatedCacheLineSize
 * and selected to fit 9, 6, 4, 3, and 2 allocations in a block.
 */
const uint32_t minFittingIndex = minSegregatedObjectIndex+numSegregatedObjectBins;
const uint32_t numFittingBins = 5;

const uint32_t fittingAlignment = estimatedCacheLineSize;

#define SET_FITTING_SIZE(N) ( (slabSize-sizeof(Block))/N ) & ~(fittingAlignment-1)
// For blockSize=16*1024, sizeof(Block)=2*estimatedCacheLineSize and fittingAlignment=estimatedCacheLineSize,
// the comments show the fitting sizes and the amounts left unused for estimatedCacheLineSize=64/128:
const uint32_t fittingSize1 = SET_FITTING_SIZE(9); // 1792/1792 128/000
const uint32_t fittingSize2 = SET_FITTING_SIZE(6); // 2688/2688 128/000
const uint32_t fittingSize3 = SET_FITTING_SIZE(4); // 4032/3968 128/256
const uint32_t fittingSize4 = SET_FITTING_SIZE(3); // 5376/5376 128/000
const uint32_t fittingSize5 = SET_FITTING_SIZE(2); // 8128/8064 000/000
#undef SET_FITTING_SIZE

/*
 * The total number of thread-specific Block-based bins
 */
const uint32_t numBlockBins = minFittingIndex+numFittingBins;

/*
 * Objects of this size and larger are considered large objects.
 */
const uint32_t minLargeObjectSize = fittingSize5 + 1;

/*
 * Per-thread pool of slab blocks. Idea behind it is to not share with other
 * threads memory that are likely in local cache(s) of our CPU.
 */
class FreeBlockPool {
    Block      *head;
    int         size;
    Backend    *backend;
    bool        lastAccessMiss;
public:
    static const int POOL_HIGH_MARK = 32;
    static const int POOL_LOW_MARK  = 8;

    class ResOfGet {
        ResOfGet();
    public:
        Block* block;
        bool   lastAccMiss;
        ResOfGet(Block *b, bool lastMiss) : block(b), lastAccMiss(lastMiss) {}
    };

    // allocated in zero-initialized memory
    FreeBlockPool(Backend *bknd) : backend(bknd) {}
    ResOfGet getBlock();
    void returnBlock(Block *block);
    bool externalCleanup(); // can be called by another thread
};

template<int LOW_MARK, int HIGH_MARK>
class LocalLOCImpl {
    static const size_t MAX_TOTAL_SIZE = 4*1024*1024;
    // TODO: can single-linked list be faster here?
    LargeMemoryBlock *head,
                     *tail; // need it when do releasing on overflow
    size_t            totalSize;
    int               numOfBlocks;
public:
    bool put(LargeMemoryBlock *object, ExtMemoryPool *extMemPool);
    LargeMemoryBlock *get(size_t size);
    bool externalCleanup(ExtMemoryPool *extMemPool);
#if __TBB_MALLOC_WHITEBOX_TEST
    LocalLOCImpl() : head(NULL), tail(NULL), totalSize(0), numOfBlocks(0) {}
    static size_t getMaxSize() { return MAX_TOTAL_SIZE; }
    static const int LOC_HIGH_MARK = HIGH_MARK;
#else
    // no ctor, object must be created in zero-initialized memory
#endif
};

typedef LocalLOCImpl<8,32> LocalLOC; // set production code parameters

class TLSData : public TLSRemote {
    MemoryPool   *memPool;
public:
    Bin           bin[numBlockBinLimit];
    FreeBlockPool freeSlabBlocks;
    LocalLOC      lloc;
    unsigned      currCacheIdx;
private:
    bool unused;
public:
    TLSData(MemoryPool *mPool, Backend *bknd) : memPool(mPool), freeSlabBlocks(bknd) {}
    MemoryPool *getMemPool() const { return memPool; }
    Bin* getAllocationBin(size_t size);
    void release(MemoryPool *mPool);
    bool externalCleanup(ExtMemoryPool *mPool, bool cleanOnlyUnused) {
        if (!unused && cleanOnlyUnused) return false;
        // both cleanups to be called, and the order is not important
        return lloc.externalCleanup(mPool) | freeSlabBlocks.externalCleanup();
    }
    bool cleanUnusedActiveBlocks(Backend *backend, bool userPool);
    void markUsed() { unused = false; } // called by owner when TLS touched
    void markUnused() { unused =  true; } // can be called by not owner thread
};

TLSData *TLSKey::createTLS(MemoryPool *memPool, Backend *backend)
{
    MALLOC_ASSERT( sizeof(TLSData) >= sizeof(Bin) * numBlockBins + sizeof(FreeBlockPool), ASSERT_TEXT );
    TLSData* tls = (TLSData*) memPool->bootStrapBlocks.allocate(memPool, sizeof(TLSData));
    if ( !tls )
        return NULL;
    new(tls) TLSData(memPool, backend);
    /* the block contains zeroes after bootStrapMalloc, so bins are initialized */
#if MALLOC_DEBUG
    for (uint32_t i = 0; i < numBlockBinLimit; i++)
        tls->bin[i].verifyInitState();
#endif
    setThreadMallocTLS(tls);
    memPool->extMemPool.allLocalCaches.registerThread(tls);
    return tls;
}

bool TLSData::cleanUnusedActiveBlocks(Backend *backend, bool userPool)
{
    bool released = false;
    // active blocks can be not used, so return them to backend
    for (uint32_t i=0; i<numBlockBinLimit; i++)
        if (bin[i].activeBlockUnused()) {
            Block *block = bin[i].getActiveBlock();
            bin[i].outofTLSBin(block);
            // slab blocks in user's pools do not have valid backRefIdx
            if (!userPool)
                removeBackRef(*(block->getBackRefIdx()));
            backend->putSlabBlock(block);

            released = true;
        }
    return released;
}

bool ExtMemoryPool::releaseAllLocalCaches()
{
    bool released = allLocalCaches.cleanup(this, /*cleanOnlyUnused=*/false);

    if (TLSData *tlsData = tlsPointerKey.getThreadMallocTLS())
        // released only for current thread for now
        released |= tlsData->cleanUnusedActiveBlocks(&backend, userPool());

    return released;
}

void AllLocalCaches::registerThread(TLSRemote *tls)
{
    tls->prev = NULL;
    MallocMutex::scoped_lock lock(listLock);
    MALLOC_ASSERT(head!=tls, ASSERT_TEXT);
    tls->next = head;
    if (head)
        head->prev = tls;
    head = tls;
    MALLOC_ASSERT(head->next!=head, ASSERT_TEXT);
}

void AllLocalCaches::unregisterThread(TLSRemote *tls)
{
    MallocMutex::scoped_lock lock(listLock);
    MALLOC_ASSERT(head, "Can't unregister thread: no threads are registered.");
    if (head == tls)
        head = tls->next;
    if (tls->next)
        tls->next->prev = tls->prev;
    if (tls->prev)
        tls->prev->next = tls->next;
    MALLOC_ASSERT(!tls->next || tls->next->next!=tls->next, ASSERT_TEXT);
}

bool AllLocalCaches::cleanup(ExtMemoryPool *extPool, bool cleanOnlyUnused)
{
    bool total = false;
    {
        MallocMutex::scoped_lock lock(listLock);

        for (TLSRemote *curr=head; curr; curr=curr->next)
            total |= static_cast<TLSData*>(curr)->
                         externalCleanup(extPool, cleanOnlyUnused);
    }
    return total;
}

void AllLocalCaches::markUnused()
{
    bool locked;
    MallocMutex::scoped_lock lock(listLock, /*block=*/false, &locked);
    if (!locked) // not wait for marking if someone doing something with it
        return;

    for (TLSRemote *curr=head; curr; curr=curr->next)
        static_cast<TLSData*>(curr)->markUnused();
}

#if MALLOC_CHECK_RECURSION
MallocMutex RecursiveMallocCallProtector::rmc_mutex;
pthread_t   RecursiveMallocCallProtector::owner_thread;
void       *RecursiveMallocCallProtector::autoObjPtr;
bool        RecursiveMallocCallProtector::mallocRecursionDetected;
#if __FreeBSD__
bool        RecursiveMallocCallProtector::canUsePthread;
#endif

#endif

/*********** End code to provide thread ID and a TLS pointer **********/

// Parameter for isLargeObject, keeps our expectations on memory origin.
// Assertions must use unknownMem to reliably report object invalidity.
enum MemoryOrigin {
    ourMem,    // allocated by TBB allocator
    unknownMem // can be allocated by system allocator or TBB allocator
};

template<MemoryOrigin> bool isLargeObject(void *object);
static void *internalMalloc(size_t size);
static void internalFree(void *object);
static void *internalPoolMalloc(MemoryPool* mPool, size_t size);
static bool internalPoolFree(MemoryPool *mPool, void *object, size_t size);

#if !MALLOC_DEBUG
#if __INTEL_COMPILER || _MSC_VER
#define NOINLINE(decl) __declspec(noinline) decl
#define ALWAYSINLINE(decl) __forceinline decl
#elif __GNUC__
#define NOINLINE(decl) decl __attribute__ ((noinline))
#define ALWAYSINLINE(decl) decl __attribute__ ((always_inline))
#else
#define NOINLINE(decl) decl
#define ALWAYSINLINE(decl) decl
#endif

static NOINLINE( bool doInitialization() );
ALWAYSINLINE( bool isMallocInitialized() );

#undef ALWAYSINLINE
#undef NOINLINE
#endif /* !MALLOC_DEBUG */


/********* Now some rough utility code to deal with indexing the size bins. **************/

/*
 * Given a number return the highest non-zero bit in it. It is intended to work with 32-bit values only.
 * Moreover, on IPF, for sake of simplicity and performance, it is narrowed to only serve for 64 to 1023.
 * This is enough for current algorithm of distribution of sizes among bins.
 * __TBB_Log2 is not used here to minimize dependencies on TBB specific sources.
 */
#if _WIN64 && _MSC_VER>=1400 && !__INTEL_COMPILER
extern "C" unsigned char _BitScanReverse( unsigned long* i, unsigned long w );
#pragma intrinsic(_BitScanReverse)
#endif
static inline unsigned int highestBitPos(unsigned int n)
{
    MALLOC_ASSERT( n>=64 && n<1024, ASSERT_TEXT ); // only needed for bsr array lookup, but always true
    unsigned int pos;
#if __ARCH_x86_32||__ARCH_x86_64

# if __linux__||__APPLE__||__FreeBSD__||__NetBSD__||__sun||__MINGW32__
    __asm__ ("bsr %1,%0" : "=r"(pos) : "r"(n));
# elif (_WIN32 && (!_WIN64 || __INTEL_COMPILER))
    __asm
    {
        bsr eax, n
        mov pos, eax
    }
# elif _WIN64 && _MSC_VER>=1400
    _BitScanReverse((unsigned long*)&pos, (unsigned long)n);
# else
#   error highestBitPos() not implemented for this platform
# endif
#elif __arm__
    __asm__ __volatile__
    (
       "clz %0, %1\n"
       "rsb %0, %0, %2\n"
       :"=r" (pos) :"r" (n), "I" (31)
    );
#else
    static unsigned int bsr[16] = {0/*N/A*/,6,7,7,8,8,8,8,9,9,9,9,9,9,9,9};
    pos = bsr[ n>>6 ];
#endif /* __ARCH_* */
    return pos;
}

template<bool Is32Bit>
unsigned int getSmallObjectIndex(unsigned int size)
{
    return (size-1)>>3;
}
template<>
unsigned int getSmallObjectIndex</*Is32Bit=*/false>(unsigned int size)
{
    // For 64-bit malloc, 16 byte alignment is needed except for bin 0.
    unsigned int result = (size-1)>>3;
    if (result) result |= 1; // 0,1,3,5,7; bins 2,4,6 are not aligned to 16 bytes
    return result;
}
/*
 * Depending on indexRequest, for a given size return either the index into the bin
 * for objects of this size, or the actual size of objects in this bin.
 */
template<bool indexRequest>
static unsigned int getIndexOrObjectSize (unsigned int size)
{
    if (size <= maxSmallObjectSize) { // selection from 8/16/24/32/40/48/56/64
        unsigned int index = getSmallObjectIndex</*Is32Bit=*/(sizeof(size_t)<=4)>( size );
         /* Bin 0 is for 8 bytes, bin 1 is for 16, and so forth */
        return indexRequest ? index : (index+1)<<3;
    }
    else if (size <= maxSegregatedObjectSize ) { // 80/96/112/128 / 160/192/224/256 / 320/384/448/512 / 640/768/896/1024
        unsigned int order = highestBitPos(size-1); // which group of bin sizes?
        MALLOC_ASSERT( 6<=order && order<=9, ASSERT_TEXT );
        if (indexRequest)
            return minSegregatedObjectIndex - (4*6) - 4 + (4*order) + ((size-1)>>(order-2));
        else {
            unsigned int alignment = 128 >> (9-order); // alignment in the group
            MALLOC_ASSERT( alignment==16 || alignment==32 || alignment==64 || alignment==128, ASSERT_TEXT );
            return alignUp(size,alignment);
        }
    }
    else {
        if( size <= fittingSize3 ) {
            if( size <= fittingSize2 ) {
                if( size <= fittingSize1 )
                    return indexRequest ? minFittingIndex : fittingSize1;
                else
                    return indexRequest ? minFittingIndex+1 : fittingSize2;
            } else
                return indexRequest ? minFittingIndex+2 : fittingSize3;
        } else {
            if( size <= fittingSize5 ) {
                if( size <= fittingSize4 )
                    return indexRequest ? minFittingIndex+3 : fittingSize4;
                else
                    return indexRequest ? minFittingIndex+4 : fittingSize5;
            } else {
                MALLOC_ASSERT( 0,ASSERT_TEXT ); // this should not happen
                return ~0U;
            }
        }
    }
}

static unsigned int getIndex (unsigned int size)
{
    return getIndexOrObjectSize</*indexRequest=*/true>(size);
}

static unsigned int getObjectSize (unsigned int size)
{
    return getIndexOrObjectSize</*indexRequest=*/false>(size);
}


void *BootStrapBlocks::allocate(MemoryPool *memPool, size_t size)
{
    FreeObject *result;

    MALLOC_ASSERT( size == sizeof(TLSData), ASSERT_TEXT );

    { // Lock with acquire
        MallocMutex::scoped_lock scoped_cs(bootStrapLock);

        if( bootStrapObjectList) {
            result = bootStrapObjectList;
            bootStrapObjectList = bootStrapObjectList->next;
        } else {
            if (!bootStrapBlock) {
                bootStrapBlock = memPool->getEmptyBlock(size);
                if (!bootStrapBlock) return NULL;
            }
            result = bootStrapBlock->bumpPtr;
            bootStrapBlock->bumpPtr = (FreeObject *)((uintptr_t)bootStrapBlock->bumpPtr - bootStrapBlock->objectSize);
            if ((uintptr_t)bootStrapBlock->bumpPtr < (uintptr_t)bootStrapBlock+sizeof(Block)) {
                bootStrapBlock->bumpPtr = NULL;
                bootStrapBlock->next = bootStrapBlockUsed;
                bootStrapBlockUsed = bootStrapBlock;
                bootStrapBlock = NULL;
            }
        }
    } // Unlock with release

    memset (result, 0, size);
    return (void*)result;
}

void BootStrapBlocks::free(void* ptr)
{
    MALLOC_ASSERT( ptr, ASSERT_TEXT );
    { // Lock with acquire
        MallocMutex::scoped_lock scoped_cs(bootStrapLock);
        ((FreeObject*)ptr)->next = bootStrapObjectList;
        bootStrapObjectList = (FreeObject*)ptr;
    } // Unlock with release
}

void BootStrapBlocks::reset()
{
    bootStrapBlock = bootStrapBlockUsed = NULL;
    bootStrapObjectList = NULL;
}

#if !(FREELIST_NONBLOCKING)
static MallocMutex publicFreeListLock; // lock for changes of publicFreeList
#endif

/********* End rough utility code  **************/

/* LifoList assumes zero initialization so a vector of it can be created
 * by just allocating some space with no call to constructor.
 * On Linux, it seems to be necessary to avoid linking with C++ libraries.
 *
 * By usage convention there is no race on the initialization. */
LifoList::LifoList( ) : top(NULL)
{
    // MallocMutex assumes zero initialization
    memset(&lock, 0, sizeof(MallocMutex));
}

void LifoList::push(Block *block)
{
    MallocMutex::scoped_lock scoped_cs(lock);
    block->next = top;
    top = block;
}

Block *LifoList::pop()
{
    Block *block=NULL;
    if (top) {
        MallocMutex::scoped_lock scoped_cs(lock);
        if (top) {
            block = top;
            top = block->next;
        }
    }
    return block;
}

Block *LifoList::grab()
{
    Block *block = NULL;
    if (top) {
        MallocMutex::scoped_lock scoped_cs(lock);
        block = top;
        top = NULL;
    }
    return block;
}

/********* Thread and block related code      *************/

template<bool poolDestroy> void AllLargeBlocksList::releaseAll(Backend *backend) {
     LargeMemoryBlock *next, *lmb = loHead;
     loHead = NULL;

     for (; lmb; lmb = next) {
         next = lmb->gNext;
         if (poolDestroy) {
             // as it's pool destruction, no need to return object to backend,
             // only remove backrefs, as they are global
             removeBackRef(lmb->backRefIdx);
         } else {
             // clean g(Next|Prev) to prevent removing lmb
             // from AllLargeBlocksList inside returnLargeObject
             lmb->gNext = lmb->gPrev = NULL;
             backend->returnLargeObject(lmb);
         }
     }
}

TLSData* MemoryPool::getTLS(bool create)
{
    TLSData* tls = extMemPool.tlsPointerKey.getThreadMallocTLS();
    if (create && !tls)
        tls = extMemPool.tlsPointerKey.createTLS(this, &extMemPool.backend);
    return tls;
}

/*
 * Return the bin for the given size.
 */
inline Bin* TLSData::getAllocationBin(size_t size)
{
    return bin + getIndex(size);
}

/* Return an empty uninitialized block in a non-blocking fashion. */
Block *MemoryPool::getEmptyBlock(size_t size)
{
    TLSData* tls = extMemPool.tlsPointerKey.getThreadMallocTLS();
    // try to use per-thread cache, if TLS available
    FreeBlockPool::ResOfGet resOfGet = tls?
        tls->freeSlabBlocks.getBlock() : FreeBlockPool::ResOfGet(NULL, false);
    Block *result = resOfGet.block;

    if (!result) { // not found in local cache, asks backend for slabs
        int num = resOfGet.lastAccMiss? Backend::numOfSlabAllocOnMiss : 1;
        BackRefIdx backRefIdx[Backend::numOfSlabAllocOnMiss];

        result = static_cast<Block*>(extMemPool.backend.getSlabBlock(num));
        if (!result) return NULL;

        if (!extMemPool.userPool())
            for (int i=0; i<num; i++) {
                backRefIdx[i] = BackRefIdx::newBackRef(/*largeObj=*/false);
                if (backRefIdx[i].isInvalid()) {
                    // roll back resource allocation
                    for (int j=0; j<i; j++)
                        removeBackRef(backRefIdx[j]);
                    Block *b = result;
                    for (int j=0; j<num; b=(Block*)((uintptr_t)b+slabSize), j++)
                        extMemPool.backend.putSlabBlock(b);
                    return NULL;
                }
            }
        // resources were allocated, register blocks
        Block *b = result;
        for (int i=0; i<num; b=(Block*)((uintptr_t)b+slabSize), i++) {
            // slab block in user's pool must have invalid backRefIdx
            if (extMemPool.userPool()) {
                new (&b->backRefIdx) BackRefIdx();
            } else {
                setBackRef(backRefIdx[i], b);
                b->backRefIdx = backRefIdx[i];
            }
            b->tlsPtr = tls;
            b->poolPtr = this;
            // all but first one go to per-thread pool
            if (i > 0) {
                MALLOC_ASSERT(tls, ASSERT_TEXT);
                tls->freeSlabBlocks.returnBlock(b);
            }
        }
    }
    MALLOC_ASSERT(result, ASSERT_TEXT);
    result->initEmptyBlock(tls, size);
    STAT_increment(getThreadId(), getIndex(result->objectSize), allocBlockNew);
    return result;
}

void MemoryPool::returnEmptyBlock(Block *block, bool poolTheBlock)
{
    block->reset();
    if (poolTheBlock) {
        extMemPool.tlsPointerKey.getThreadMallocTLS()->freeSlabBlocks.returnBlock(block);
    }
    else {
        // slab blocks in user's pools do not have valid backRefIdx
        if (!extMemPool.userPool())
            removeBackRef(*(block->getBackRefIdx()));
        extMemPool.backend.putSlabBlock(block);
    }
}

bool ExtMemoryPool::init(intptr_t poolId, rawAllocType rawAlloc,
                         rawFreeType rawFree, size_t granularity,
                         bool keepAllMemory, bool fixedPool)
{
    this->poolId = poolId;
    this->rawAlloc = rawAlloc;
    this->rawFree = rawFree;
    this->granularity = granularity;
    this->keepAllMemory = keepAllMemory;
    this->fixedPool = fixedPool;
    this->delayRegsReleasing = false;
    if (!initTLS())
        return false;
    loc.init(this);
    backend.init(this);
    MALLOC_ASSERT(isPoolValid(), NULL);
    return true;
}

bool ExtMemoryPool::initTLS() { return tlsPointerKey.init(); }

bool MemoryPool::init(intptr_t poolId, const MemPoolPolicy *policy)
{
    if (!extMemPool.init(poolId, policy->pAlloc, policy->pFree,
               policy->granularity? policy->granularity : defaultGranularity,
               policy->keepAllMemory, policy->fixedPool))
        return false;
    {
        MallocMutex::scoped_lock lock(memPoolListLock);
        next = defaultMemPool->next;
        defaultMemPool->next = this;
        prev = defaultMemPool;
        if (next)
            next->prev = this;
    }
    return true;
}

bool MemoryPool::reset()
{
    MALLOC_ASSERT(extMemPool.userPool(), "No reset for the system pool.");
    // memory is not releasing during pool reset
    // TODO: mark regions to release unused on next reset()
    extMemPool.delayRegionsReleasing(true);

    bootStrapBlocks.reset();
    extMemPool.lmbList.releaseAll</*poolDestroy=*/false>(&extMemPool.backend);
    if (!extMemPool.reset())
        return false;

    if (!extMemPool.initTLS())
        return false;
    extMemPool.delayRegionsReleasing(false);
    return true;
}

bool MemoryPool::destroy()
{
#if __TBB_MALLOC_LOCACHE_STAT
    extMemPool.loc.reportStat(stdout);
#endif
#if __TBB_MALLOC_BACKEND_STAT
    extMemPool.backend.reportStat(stdout);
#endif
    {
        MallocMutex::scoped_lock lock(memPoolListLock);
        // remove itself from global pool list
        if (prev)
            prev->next = next;
        if (next)
            next->prev = prev;
    }
    // slab blocks in non-default pool do not have backreferences,
    // only large objects do
    if (extMemPool.userPool())
        extMemPool.lmbList.releaseAll</*poolDestroy=*/true>(&extMemPool.backend);
    else {
        // only one non-userPool() is supported now
        MALLOC_ASSERT(this==defaultMemPool, NULL);
        // There and below in extMemPool.destroy(), do not restore initial state
        // for user pool, because it's just about to be released. But for system
        // pool restoring, we do not want to do zeroing of it on subsequent reload.
        bootStrapBlocks.reset();
        extMemPool.orphanedBlocks.reset();
    }
    return extMemPool.destroy();
}

void MemoryPool::onThreadShutdown(TLSData *tlsData)
{
    if (tlsData) { // might be called for "empty" TLS
        tlsData->release(this);
        bootStrapBlocks.free(tlsData);
        clearTLS();
    }
}

#if MALLOC_DEBUG
void Bin::verifyTLSBin (size_t size) const
{
/* The debug version verifies the TLSBin as needed */
    uint32_t objSize = getObjectSize(size);

    if (activeBlk) {
        MALLOC_ASSERT( activeBlk->isOwnedByCurrentThread(), ASSERT_TEXT );
        MALLOC_ASSERT( activeBlk->objectSize == objSize, ASSERT_TEXT );
#if MALLOC_DEBUG>1
        for (Block* temp = activeBlk->next; temp; temp=temp->next) {
            MALLOC_ASSERT( temp!=activeBlk, ASSERT_TEXT );
            MALLOC_ASSERT( temp->isOwnedByCurrentThread(), ASSERT_TEXT );
            MALLOC_ASSERT( temp->objectSize == objSize, ASSERT_TEXT );
            MALLOC_ASSERT( temp->previous->next == temp, ASSERT_TEXT );
            if (temp->next) {
                MALLOC_ASSERT( temp->next->previous == temp, ASSERT_TEXT );
            }
        }
        for (Block* temp = activeBlk->previous; temp; temp=temp->previous) {
            MALLOC_ASSERT( temp!=activeBlk, ASSERT_TEXT );
            MALLOC_ASSERT( temp->isOwnedByCurrentThread(), ASSERT_TEXT );
            MALLOC_ASSERT( temp->objectSize == objSize, ASSERT_TEXT );
            MALLOC_ASSERT( temp->next->previous == temp, ASSERT_TEXT );
            if (temp->previous) {
                MALLOC_ASSERT( temp->previous->next == temp, ASSERT_TEXT );
            }
        }
#endif /* MALLOC_DEBUG>1 */
    }
}
#else /* MALLOC_DEBUG */
inline void Bin::verifyTLSBin (size_t) const { }
#endif /* MALLOC_DEBUG */

/*
 * Add a block to the start of this tls bin list.
 */
void Bin::pushTLSBin(Block* block)
{
    /* The objectSize should be defined and not a parameter
       because the function is applied to partially filled blocks as well */
    unsigned int size = block->objectSize;

    MALLOC_ASSERT( block->isOwnedByCurrentThread(), ASSERT_TEXT );
    MALLOC_ASSERT( block->objectSize != 0, ASSERT_TEXT );
    MALLOC_ASSERT( block->next == NULL, ASSERT_TEXT );
    MALLOC_ASSERT( block->previous == NULL, ASSERT_TEXT );

    MALLOC_ASSERT( this, ASSERT_TEXT );
    verifyTLSBin(size);

    block->next = activeBlk;
    if( activeBlk ) {
        block->previous = activeBlk->previous;
        activeBlk->previous = block;
        if( block->previous )
            block->previous->next = block;
    } else {
        activeBlk = block;
    }

    verifyTLSBin(size);
}

/*
 * Take a block out of its tls bin (e.g. before removal).
 */
void Bin::outofTLSBin(Block* block)
{
    unsigned int size = block->objectSize;

    MALLOC_ASSERT( block->isOwnedByCurrentThread(), ASSERT_TEXT );
    MALLOC_ASSERT( block->objectSize != 0, ASSERT_TEXT );

    MALLOC_ASSERT( this, ASSERT_TEXT );
    verifyTLSBin(size);

    if (block == activeBlk) {
        activeBlk = block->previous? block->previous : block->next;
    }
    /* Delink the block */
    if (block->previous) {
        MALLOC_ASSERT( block->previous->next == block, ASSERT_TEXT );
        block->previous->next = block->next;
    }
    if (block->next) {
        MALLOC_ASSERT( block->next->previous == block, ASSERT_TEXT );
        block->next->previous = block->previous;
    }
    block->next = NULL;
    block->previous = NULL;

    verifyTLSBin(size);
}

Block* Bin::getPublicFreeListBlock()
{
    Block* block;
    MALLOC_ASSERT( this, ASSERT_TEXT );
    // if this method is called, active block usage must be unsuccessful
    MALLOC_ASSERT( !activeBlk && !mailbox || activeBlk && activeBlk->isFull, ASSERT_TEXT );

// the counter should be changed    STAT_increment(getThreadId(), ThreadCommonCounters, lockPublicFreeList);
    if (!FencedLoad((intptr_t&)mailbox)) // hotpath is empty mailbox
        return NULL;
    else { // mailbox is not empty, take lock and inspect it
        MallocMutex::scoped_lock scoped_cs(mailLock);
        block = mailbox;
        if( block ) {
            MALLOC_ASSERT( block->isOwnedByCurrentThread(), ASSERT_TEXT );
            MALLOC_ASSERT( !isNotForUse(block->nextPrivatizable), ASSERT_TEXT );
            mailbox = block->nextPrivatizable;
            block->nextPrivatizable = (Block*) this;
        }
    }
    if( block ) {
        MALLOC_ASSERT( isSolidPtr(block->publicFreeList), ASSERT_TEXT );
        block->privatizePublicFreeList();
    }
    return block;
}

bool Block::emptyEnoughToUse()
{
    const float threshold = (slabSize - sizeof(Block)) * (1-emptyEnoughRatio);

    if (bumpPtr) {
        /* If we are still using a bump ptr for this block it is empty enough to use. */
        STAT_increment(getThreadId(), getIndex(objectSize), examineEmptyEnough);
        isFull = false;
        return 1;
    }

    /* allocatedCount shows how many objects in the block are in use; however it still counts
       blocks freed by other threads; so prior call to privatizePublicFreeList() is recommended */
    isFull = (allocatedCount*objectSize > threshold)? true: false;
#if COLLECT_STATISTICS
    if (isFull)
        STAT_increment(getThreadId(), getIndex(objectSize), examineNotEmpty);
    else
        STAT_increment(getThreadId(), getIndex(objectSize), examineEmptyEnough);
#endif
    return !isFull;
}

/* Restore the bump pointer for an empty block that is planned to use */
void Block::restoreBumpPtr()
{
    MALLOC_ASSERT( allocatedCount == 0, ASSERT_TEXT );
    MALLOC_ASSERT( publicFreeList == NULL, ASSERT_TEXT );
    STAT_increment(getThreadId(), getIndex(objectSize), freeRestoreBumpPtr);
    bumpPtr = (FreeObject *)((uintptr_t)this + slabSize - objectSize);
    freeList = NULL;
    isFull = 0;
}

void Block::freeOwnObject(void *object)
{
    tlsPtr->markUsed();
    allocatedCount--;
    MALLOC_ASSERT( allocatedCount < (slabSize-sizeof(Block))/objectSize, ASSERT_TEXT );
#if COLLECT_STATISTICS
    // Note that getAllocationBin is not called on the hottest path with statistics off.
    if (tlsPtr->getAllocationBin(objectSize)->getActiveBlock() != this)
        STAT_increment(getThreadId(), getIndex(objectSize), freeToInactiveBlock);
    else
        STAT_increment(getThreadId(), getIndex(objectSize), freeToActiveBlock);
#endif
    if (empty()) {
        // The bump pointer is about to be restored for the block,
        // no need to find objectToFree here (this is costly).

        // if the last object of a slab is freed, the slab cannot be marked full
        MALLOC_ASSERT(!isFull, ASSERT_TEXT);
        tlsPtr->getAllocationBin(objectSize)->processLessUsedBlock(poolPtr, this);
    } else {
        FreeObject *objectToFree = findObjectToFree(object);
        objectToFree->next = freeList;
        freeList = objectToFree;

        if (isFull && emptyEnoughToUse())
            tlsPtr->getAllocationBin(objectSize)->moveBlockToFront(this);
    }
}

void Block::freePublicObject (FreeObject *objectToFree)
{
    FreeObject *localPublicFreeList;

    MALLOC_ITT_SYNC_RELEASING(&publicFreeList);
#if FREELIST_NONBLOCKING
    FreeObject *temp = publicFreeList;
    do {
        localPublicFreeList = objectToFree->next = temp;
        temp = (FreeObject*)AtomicCompareExchange(
                                (intptr_t&)publicFreeList,
                                (intptr_t)objectToFree, (intptr_t)localPublicFreeList );
        // no backoff necessary because trying to make change, not waiting for a change
    } while( temp != localPublicFreeList );
#else
    STAT_increment(getThreadId(), ThreadCommonCounters, lockPublicFreeList);
    {
        MallocMutex::scoped_lock scoped_cs(publicFreeListLock);
        localPublicFreeList = objectToFree->next = publicFreeList;
        publicFreeList = objectToFree;
    }
#endif

    if( localPublicFreeList==NULL ) {
        // if the block is abandoned, its nextPrivatizable pointer should be UNUSABLE
        // otherwise, it should point to the bin the block belongs to.
        // reading nextPrivatizable is thread-safe below, because:
        // 1) the executing thread atomically got publicFreeList==NULL and changed it to non-NULL;
        // 2) only owning thread can change it back to NULL,
        // 3) but it can not be done until the block is put to the mailbox
        // So the executing thread is now the only one that can change nextPrivatizable
        if( !isNotForUse(nextPrivatizable) ) {
            MALLOC_ASSERT( nextPrivatizable!=NULL, ASSERT_TEXT );
            Bin* theBin = (Bin*) nextPrivatizable;
            MallocMutex::scoped_lock scoped_cs(theBin->mailLock);
            nextPrivatizable = theBin->mailbox;
            theBin->mailbox = this;
        }
    }
    STAT_increment(getThreadId(), ThreadCommonCounters, freeToOtherThread);
    STAT_increment(ownerTid, getIndex(objectSize), freeByOtherThread);
}

void Block::privatizePublicFreeList( bool cleanup )
{
    FreeObject *temp, *localPublicFreeList;
    const intptr_t endMarker = cleanup? UNUSABLE : 0;

    // During cleanup of orphaned blocks, the calling thread is not registered as the owner 
    MALLOC_ASSERT( cleanup || isOwnedByCurrentThread(), ASSERT_TEXT );
#if FREELIST_NONBLOCKING
    temp = publicFreeList;
    do {
        localPublicFreeList = temp;
        temp = (FreeObject*)AtomicCompareExchange( (intptr_t&)publicFreeList,
                                        endMarker, (intptr_t)localPublicFreeList);
        // no backoff necessary because trying to make change, not waiting for a change
    } while( temp != localPublicFreeList );
#else
    STAT_increment(getThreadId(), ThreadCommonCounters, lockPublicFreeList);
    {
        MallocMutex::scoped_lock scoped_cs(publicFreeListLock);
        localPublicFreeList = publicFreeList;
        publicFreeList = endMarker;
    }
    temp = localPublicFreeList;
#endif
    MALLOC_ITT_SYNC_ACQUIRED(&publicFreeList);

     // publicFreeList must have been UNUSABLE (possible for orphaned blocks) or valid, but not NULL
    MALLOC_ASSERT( localPublicFreeList!=NULL, ASSERT_TEXT );
    MALLOC_ASSERT( localPublicFreeList==temp, ASSERT_TEXT );
    if( isSolidPtr(temp) ) {
        MALLOC_ASSERT( allocatedCount <= (slabSize-sizeof(Block))/objectSize, ASSERT_TEXT );
        /* other threads did not change the counter freeing our blocks */
        allocatedCount--;
        while( isSolidPtr(temp->next) ){ // the list will end with either NULL or UNUSABLE
            temp = temp->next;
            allocatedCount--;
            MALLOC_ASSERT( allocatedCount < (slabSize-sizeof(Block))/objectSize, ASSERT_TEXT );
        }
        /* merge with local freeList */
        temp->next = freeList;
        freeList = localPublicFreeList;
        STAT_increment(getThreadId(), getIndex(objectSize), allocPrivatized);
    }
}

void Block::privatizeOrphaned(TLSData *tls, unsigned index)
{
    Bin* bin = tls->bin + index;
    STAT_increment(getThreadId(), index, allocBlockPublic);
    next = NULL;
    previous = NULL;
    MALLOC_ASSERT( publicFreeList!=NULL, ASSERT_TEXT );
    /* There is not a race here since no other thread owns this block */
    markOwned(tls);
    // It is safe to change nextPrivatizable, as publicFreeList is not null
    MALLOC_ASSERT( isNotForUse(nextPrivatizable), ASSERT_TEXT );
    nextPrivatizable = (Block*)bin;
    // the next call is required to change publicFreeList to 0
    privatizePublicFreeList();
    if( allocatedCount ) {
        emptyEnoughToUse(); // check its fullness and set result->isFull
    } else {
        restoreBumpPtr();
    }
    MALLOC_ASSERT( !isNotForUse(publicFreeList), ASSERT_TEXT );
}

void Block::shareOrphaned(intptr_t binTag, unsigned index)
{
    MALLOC_ASSERT( binTag, ASSERT_TEXT );
    STAT_increment(getThreadId(), index, freeBlockPublic);
    markOrphaned();
    // need to set publicFreeList to non-zero, so other threads
    // will not change nextPrivatizable and it can be zeroed.
    if ((intptr_t)nextPrivatizable==binTag) {
        void* oldval;
#if FREELIST_NONBLOCKING
        oldval = (void*)AtomicCompareExchange((intptr_t&)publicFreeList, UNUSABLE, 0);
#else
        STAT_increment(getThreadId(), ThreadCommonCounters, lockPublicFreeList);
        {
            MallocMutex::scoped_lock scoped_cs(publicFreeListLock);
            if ( (oldval=publicFreeList)==NULL )
                (intptr_t&)(publicFreeList) = UNUSABLE;
        }
#endif
        if ( oldval!=NULL ) {
            // another thread freed an object; we need to wait until it finishes.
            // There is no need for exponential backoff, as the wait here is not for a lock;
            // but need to yield, so the thread we wait has a chance to run.
            // TODO: add a pause to also be friendly to hyperthreads
            int count = 256;
            while( (intptr_t)const_cast<Block* volatile &>(nextPrivatizable)==binTag ) {
                if (--count==0) {
                    do_yield();
                    count = 256;
                }
            }
        }
    } else {
        MALLOC_ASSERT( isSolidPtr(publicFreeList), ASSERT_TEXT );
    }
    MALLOC_ASSERT( publicFreeList!=NULL, ASSERT_TEXT );
    // now it is safe to change our data
    previous = NULL;
    // it is caller responsibility to ensure that the list of blocks
    // formed by nextPrivatizable pointers is kept consistent if required.
    // if only called from thread shutdown code, it does not matter.
    (intptr_t&)(nextPrivatizable) = UNUSABLE;
}

void Block::cleanBlockHeader()
{
    next = NULL;
    previous = NULL;
    freeList = NULL;
    allocatedCount = 0;
    isFull = 0;
    tlsPtr = NULL;

    publicFreeList = NULL;
}

void Block::initEmptyBlock(TLSData *tls, size_t size)
{
    // Having getIndex and getObjectSize called next to each other
    // allows better compiler optimization as they basically share the code.
    unsigned int index = getIndex(size);
    unsigned int objSz = getObjectSize(size);

    cleanBlockHeader();
    objectSize = objSz;
    markOwned(tls);
    // bump pointer should be prepared for first allocation - thus mode it down to objectSize
    bumpPtr = (FreeObject *)((uintptr_t)this + slabSize - objectSize);

    // each block should have the address where the head of the list of "privatizable" blocks is kept
    // the only exception is a block for boot strap which is initialized when TLS is yet NULL
    nextPrivatizable = tls? (Block*)(tls->bin + index) : NULL;
    TRACEF(( "[ScalableMalloc trace] Empty block %p is initialized, owner is %ld, objectSize is %d, bumpPtr is %p\n",
             this, tlsPtr ? getThreadId() : -1, objectSize, bumpPtr ));
}

Block *OrphanedBlocks::get(TLSData *tls, unsigned int size)
{
    // TODO: try to use index from getAllocationBin
    unsigned int index = getIndex(size);
    Block *block = bins[index].pop();
    if (block) {
        MALLOC_ITT_SYNC_ACQUIRED(bins+index);
        block->privatizeOrphaned(tls, index);
    }
    return block;
}

void OrphanedBlocks::put(intptr_t binTag, Block *block)
{
    unsigned int index = getIndex(block->getSize());
    block->shareOrphaned(binTag, index);
    MALLOC_ITT_SYNC_RELEASING(bins+index);
    bins[index].push(block);
}

void OrphanedBlocks::reset()
{
    for (uint32_t i=0; i<numBlockBinLimit; i++)
        new (bins+i) LifoList();
}

bool OrphanedBlocks::cleanup(Backend* backend)
{
    bool result = false;
    for (uint32_t i=0; i<numBlockBinLimit; i++) {
        Block* block = bins[i].grab();
        MALLOC_ITT_SYNC_ACQUIRED(bins+i);
        while (block) {
            Block* next = block->next;
            block->privatizePublicFreeList( /*cleanup=*/true );
            if (block->empty()) {
                block->reset();
                // slab blocks in user's pools do not have valid backRefIdx
                if (!backend->inUserPool())
                    removeBackRef(*(block->getBackRefIdx()));
                backend->putSlabBlock(block);
                result = true;
            } else {
                MALLOC_ITT_SYNC_RELEASING(bins+i);
                bins[i].push(block);
            }
            block = next;
        }
    }
    return result;
}

FreeBlockPool::ResOfGet FreeBlockPool::getBlock()
{
    Block *b = (Block*)AtomicFetchStore(&head, 0);

    if (b) {
        size--;
        Block *newHead = b->next;
        lastAccessMiss = false;
        FencedStore((intptr_t&)head, (intptr_t)newHead);
    } else
        lastAccessMiss = true;

    return ResOfGet(b, lastAccessMiss);
}

void FreeBlockPool::returnBlock(Block *block)
{
    MALLOC_ASSERT( size <= POOL_HIGH_MARK, ASSERT_TEXT );
    Block *localHead = (Block*)AtomicFetchStore(&head, 0);

    if (!localHead)
        size = 0; // head was stolen by externalClean, correct size accordingly
    else if (size == POOL_HIGH_MARK) {
        // release cold blocks and add hot one,
        // so keep POOL_LOW_MARK-1 blocks and add new block to head
        Block *headToFree = localHead, *helper;
        for (int i=0; i<POOL_LOW_MARK-2; i++)
            headToFree = headToFree->next;
        Block *last = headToFree;
        headToFree = headToFree->next;
        last->next = NULL;
        size = POOL_LOW_MARK-1;
        for (Block *currBl = headToFree; currBl; currBl = helper) {
            helper = currBl->next;
            // slab blocks in user's pools do not have valid backRefIdx
            if (!backend->inUserPool())
                removeBackRef(currBl->backRefIdx);
            backend->putSlabBlock(currBl);
        }
    }
    size++;
    block->next = localHead;
    FencedStore((intptr_t&)head, (intptr_t)block);
}

bool FreeBlockPool::externalCleanup()
{
    Block *helper;
    bool nonEmpty = false;

    for (Block *currBl=(Block*)AtomicFetchStore(&head, 0); currBl; currBl=helper) {
        helper = currBl->next;
        // slab blocks in user's pools do not have valid backRefIdx
        if (!backend->inUserPool())
            removeBackRef(currBl->backRefIdx);
        backend->putSlabBlock(currBl);
        nonEmpty = true;
    }
    return nonEmpty;
}

/* Prepare the block for returning to FreeBlockPool */
void Block::reset()
{
    // it is caller's responsibility to ensure no data is lost before calling this
    MALLOC_ASSERT( allocatedCount==0, ASSERT_TEXT );
    MALLOC_ASSERT( !isSolidPtr(publicFreeList), ASSERT_TEXT );
    if (!isStartupAllocObject())
        STAT_increment(getThreadId(), getIndex(objectSize), freeBlockBack);

    cleanBlockHeader();

    nextPrivatizable = NULL;

    objectSize = 0;
    // for an empty block, bump pointer should point right after the end of the block
    bumpPtr = (FreeObject *)((uintptr_t)this + slabSize);
}

inline void Bin::setActiveBlock (Block *block)
{
//    MALLOC_ASSERT( bin, ASSERT_TEXT );
    MALLOC_ASSERT( block->isOwnedByCurrentThread(), ASSERT_TEXT );
    // it is the caller responsibility to keep bin consistence (i.e. ensure this block is in the bin list)
    activeBlk = block;
}

inline Block* Bin::setPreviousBlockActive()
{
    MALLOC_ASSERT( activeBlk, ASSERT_TEXT );
    Block* temp = activeBlk->previous;
    if( temp ) {
        MALLOC_ASSERT( temp->isFull == 0, ASSERT_TEXT );
        activeBlk = temp;
    }
    return temp;
}

inline bool Block::isOwnedByCurrentThread() const {
    return tlsPtr && ownerTid.isCurrentThreadId();
}

FreeObject *Block::findObjectToFree(const void *object) const
{
    FreeObject *objectToFree;
    // Due to aligned allocations, a pointer passed to scalable_free
    // might differ from the address of internally allocated object.
    // Small objects however should always be fine.
    if (objectSize <= maxSegregatedObjectSize)
        objectToFree = (FreeObject*)object;
    // "Fitting size" allocations are suspicious if aligned higher than naturally
    else {
        if ( ! isAligned(object,2*fittingAlignment) )
            // TODO: the above check is questionable - it gives false negatives in ~50% cases,
            //       so might even be slower in average than unconditional use of findAllocatedObject.
            // here it should be a "real" object
            objectToFree = (FreeObject*)object;
        else
            // here object can be an aligned address, so applying additional checks
            objectToFree = findAllocatedObject(object);
        MALLOC_ASSERT( isAligned(objectToFree,fittingAlignment), ASSERT_TEXT );
    }
    MALLOC_ASSERT( isProperlyPlaced(objectToFree), ASSERT_TEXT );

    return objectToFree;
}

void TLSData::release(MemoryPool *mPool)
{
    mPool->extMemPool.allLocalCaches.unregisterThread(this);
    externalCleanup(&mPool->extMemPool, /*cleanOnlyUnused=*/false);

    for (unsigned index = 0; index < numBlockBins; index++) {
        Block *activeBlk = bin[index].getActiveBlock();
        if (!activeBlk)
            continue;
        Block *threadlessBlock = activeBlk->previous;
        while (threadlessBlock) {
            Block *threadBlock = threadlessBlock->previous;
            if (threadlessBlock->empty()) {
                /* we destroy the thread, so not use its block pool */
                mPool->returnEmptyBlock(threadlessBlock, /*poolTheBlock=*/false);
            } else {
                mPool->extMemPool.orphanedBlocks.put(intptr_t(bin+index), threadlessBlock);
            }
            threadlessBlock = threadBlock;
        }
        threadlessBlock = activeBlk;
        while (threadlessBlock) {
            Block *threadBlock = threadlessBlock->next;
            if (threadlessBlock->empty()) {
                /* we destroy the thread, so not use its block pool */
                mPool->returnEmptyBlock(threadlessBlock, /*poolTheBlock=*/false);
            } else {
                mPool->extMemPool.orphanedBlocks.put(intptr_t(bin+index), threadlessBlock);
            }
            threadlessBlock = threadBlock;
        }
        bin[index].resetActiveBlock();
    }
}


#if MALLOC_CHECK_RECURSION
// TODO: Use dedicated heap for this

/*
 * It's a special kind of allocation that can be used when malloc is
 * not available (either during startup or when malloc was already called and
 * we are, say, inside pthread_setspecific's call).
 * Block can contain objects of different sizes,
 * allocations are performed by moving bump pointer and increasing of object counter,
 * releasing is done via counter of objects allocated in the block
 * or moving bump pointer if releasing object is on a bound.
 * TODO: make bump pointer to grow to the same backward direction as all the others.
 */

class StartupBlock : public Block {
    size_t availableSize() const {
        return slabSize - ((uintptr_t)bumpPtr - (uintptr_t)this);
    }
    static StartupBlock *getBlock();
public:
    static FreeObject *allocate(size_t size);
    static size_t msize(void *ptr) { return *((size_t*)ptr - 1); }
    void free(void *ptr);
};

static MallocMutex startupMallocLock;
static StartupBlock *firstStartupBlock;

StartupBlock *StartupBlock::getBlock()
{
    BackRefIdx backRefIdx = BackRefIdx::newBackRef(/*largeObj=*/false);
    if (backRefIdx.isInvalid()) return NULL;

    StartupBlock *block = static_cast<StartupBlock*>(
        defaultMemPool->extMemPool.backend.getSlabBlock(1));
    if (!block) return NULL;

    block->cleanBlockHeader();
    setBackRef(backRefIdx, block);
    block->backRefIdx = backRefIdx;
    // use startupAllocObjSizeMark to mark objects from startup block marker
    block->objectSize = startupAllocObjSizeMark;
    block->bumpPtr = (FreeObject *)((uintptr_t)block + sizeof(StartupBlock));
    return block;
}

FreeObject *StartupBlock::allocate(size_t size)
{
    FreeObject *result;
    StartupBlock *newBlock = NULL;
    bool newBlockUnused = false;

    /* Objects must be aligned on their natural bounds,
       and objects bigger than word on word's bound. */
    size = alignUp(size, sizeof(size_t));
    // We need size of an object to implement msize.
    size_t reqSize = size + sizeof(size_t);
    // speculatively allocates newBlock to try avoid allocation while holding lock
    /* TODO: The function is called when malloc nested call is detected,
             so simultaneous usage from different threads seems unlikely.
             If pre-allocation is found useless, the code might be simplified. */
    if (!firstStartupBlock || firstStartupBlock->availableSize() < reqSize) {
        newBlock = StartupBlock::getBlock();
        if (!newBlock) return NULL;
    }
    {
        MallocMutex::scoped_lock scoped_cs(startupMallocLock);
        // Re-check whether we need a new block (conditions might have changed)
        if (!firstStartupBlock || firstStartupBlock->availableSize() < reqSize) {
            if (!newBlock) {
                newBlock = StartupBlock::getBlock();
                if (!newBlock) return NULL;
            }
            newBlock->next = (Block*)firstStartupBlock;
            if (firstStartupBlock)
                firstStartupBlock->previous = (Block*)newBlock;
            firstStartupBlock = newBlock;
        } else
            newBlockUnused = true;
        result = firstStartupBlock->bumpPtr;
        firstStartupBlock->allocatedCount++;
        firstStartupBlock->bumpPtr =
            (FreeObject *)((uintptr_t)firstStartupBlock->bumpPtr + reqSize);
    }
    if (newBlock && newBlockUnused)
        defaultMemPool->returnEmptyBlock(newBlock, /*poolTheBlock=*/false);

    // keep object size at the negative offset
    *((size_t*)result) = size;
    return (FreeObject*)((size_t*)result+1);
}

void StartupBlock::free(void *ptr)
{
    Block* blockToRelease = NULL;
    {
        MallocMutex::scoped_lock scoped_cs(startupMallocLock);

        MALLOC_ASSERT(firstStartupBlock, ASSERT_TEXT);
        MALLOC_ASSERT(startupAllocObjSizeMark==objectSize
                      && allocatedCount>0, ASSERT_TEXT);
        MALLOC_ASSERT((uintptr_t)ptr>=(uintptr_t)this+sizeof(StartupBlock)
                      && (uintptr_t)ptr+StartupBlock::msize(ptr)<=(uintptr_t)this+slabSize,
                      ASSERT_TEXT);
        if (0 == --allocatedCount) {
            if (this == firstStartupBlock)
                firstStartupBlock = (StartupBlock*)firstStartupBlock->next;
            if (previous)
                previous->next = next;
            if (next)
                next->previous = previous;
            blockToRelease = this;
        } else if ((uintptr_t)ptr + StartupBlock::msize(ptr) == (uintptr_t)bumpPtr) {
            // last object in the block released
            FreeObject *newBump = (FreeObject*)((size_t*)ptr - 1);
            MALLOC_ASSERT((uintptr_t)newBump>(uintptr_t)this+sizeof(StartupBlock),
                          ASSERT_TEXT);
            bumpPtr = newBump;
        }
    }
    if (blockToRelease) {
        blockToRelease->previous = blockToRelease->next = NULL;
        defaultMemPool->returnEmptyBlock(blockToRelease, /*poolTheBlock=*/false);
    }
}

#endif /* MALLOC_CHECK_RECURSION */

/********* End thread related code  *************/

/********* Library initialization *************/

//! Value indicating the state of initialization.
/* 0 = initialization not started.
 * 1 = initialization started but not finished.
 * 2 = initialization finished.
 * In theory, we only need values 0 and 2. But value 1 is nonetheless
 * useful for detecting errors in the double-check pattern.
 */
static intptr_t mallocInitialized;   // implicitly initialized to 0
static MallocMutex initMutex;

/** The leading "\0" is here so that applying "strings" to the binary
    delivers a clean result. */
static char VersionString[] = "\0" TBBMALLOC_VERSION_STRINGS;

#if __TBB_WIN8UI_SUPPORT
bool GetBoolEnvironmentVariable(const char *) { return false; }
#else
bool GetBoolEnvironmentVariable(const char *name)
{
    if (const char* s = getenv(name))
        return strcmp(s,"0") != 0;
    return false;
}
#endif

void AllocControlledMode::initReadEnv(const char *envName, intptr_t defaultVal)
{
    if (!setDone) {
#if !__TBB_WIN8UI_SUPPORT
    // TODO: use strtol to get the actual value of the envirable
        const char *envVal = getenv(envName);
        if (envVal && !strcmp(envVal, "1"))
            val = 1;
        else
#endif
            val = defaultVal;
        setDone = true;
    }
}

void MemoryPool::initDefaultPool()
{
    long long unsigned hugePageSize = 0;
#if __linux__
    if (FILE *f = fopen("/proc/meminfo", "r")) {
        const int READ_BUF_SIZE = 100;
        char buf[READ_BUF_SIZE];
        MALLOC_STATIC_ASSERT(sizeof(hugePageSize) >= 8,
              "At least 64 bits required for keeping page size/numbers.");

        while (fgets(buf, READ_BUF_SIZE, f)) {
            if (1 == sscanf(buf, "Hugepagesize: %llu kB", &hugePageSize)) {
                hugePageSize *= 1024;
                break;
            }
        }
        fclose(f);
    }
#endif
    hugePages.init(hugePageSize);
}

#if USE_PTHREAD && (__TBB_SOURCE_DIRECTLY_INCLUDED || __TBB_USE_DLOPEN_REENTRANCY_WORKAROUND)

/* Decrease race interval between dynamic library unloading and pthread key
   destructor. Protect only Pthreads with supported unloading. */
class ShutdownSync {
/* flag is the number of threads in pthread key dtor body
   (i.e., between threadDtorStart() and threadDtorDone())
   or the signal to skip dtor, if flag < 0 */
    intptr_t flag;
    static const intptr_t skipDtor = INTPTR_MIN/2;
public:
    void init() { flag = 0; }
/* Suppose that 2*abs(skipDtor) or more threads never call threadDtorStart()
   simultaneously, so flag never becomes negative because of that. */
    bool threadDtorStart() {
        if (flag < 0)
            return false;
        if (AtomicIncrement(flag) <= 0) { // note that new value returned
            AtomicAdd(flag, -1);  // flag is spoiled by us, restore it
            return false;
        }
        return true;
    }
    void threadDtorDone() {
        AtomicAdd(flag, -1);
    }
    void processExit() {
        if (AtomicAdd(flag, skipDtor) != 0)
            SpinWaitUntilEq(flag, skipDtor);
    }
};

#else

class ShutdownSync {
public:
    void init() { }
    bool threadDtorStart() { return true; }
    void threadDtorDone() { }
    void processExit() { }
};

#endif // USE_PTHREAD && (__TBB_SOURCE_DIRECTLY_INCLUDED || __TBB_USE_DLOPEN_REENTRANCY_WORKAROUND)

static ShutdownSync shutdownSync;

inline bool isMallocInitialized() {
    // Load must have acquire fence; otherwise thread taking "initialized" path
    // might perform textually later loads *before* mallocInitialized becomes 2.
    return 2 == FencedLoad(mallocInitialized);
}

bool isMallocInitializedExt() {
    return isMallocInitialized();
}

/** Caller is responsible for ensuring this routine is called exactly once. */
extern "C" void MallocInitializeITT() {
#if DO_ITT_NOTIFY
    if (!usedBySrcIncluded)
        tbb::internal::__TBB_load_ittnotify();
#endif
}

/*
 * Allocator initialization routine;
 * it is called lazily on the very first scalable_malloc call.
 */
static bool initMemoryManager()
{
    TRACEF(( "[ScalableMalloc trace] sizeof(Block) is %d (expected 128); sizeof(uintptr_t) is %d\n",
             sizeof(Block), sizeof(uintptr_t) ));
    MALLOC_ASSERT( 2*blockHeaderAlignment == sizeof(Block), ASSERT_TEXT );
    MALLOC_ASSERT( sizeof(FreeObject) == sizeof(void*), ASSERT_TEXT );
    MALLOC_ASSERT( isAligned(defaultMemPool, sizeof(intptr_t)),
                   "Memory pool must be void*-aligned for atomic to work over aligned arguments.");

#if USE_WINTHREAD
    const size_t granularity = 64*1024; // granulatity of VirtualAlloc
#else
    // POSIX.1-2001-compliant way to get page size
    const size_t granularity = sysconf(_SC_PAGESIZE);
#endif
    bool initOk = defaultMemPool->
        extMemPool.init(0, NULL, NULL, granularity,
                        /*keepAllMemory=*/false, /*fixedPool=*/false);
// TODO: extMemPool.init() to not allocate memory
    if (!initOk || !initBackRefMaster(&defaultMemPool->extMemPool.backend))
        return false;
    ThreadId::init();      // Create keys for thread id
    MemoryPool::initDefaultPool();
    // init() is required iff initMemoryManager() is called
    // after mallocProcessShutdownNotification()
    shutdownSync.init();
#if COLLECT_STATISTICS
    initStatisticsCollection();
#endif
    return true;
}

//! Ensures that initMemoryManager() is called once and only once.
/** Does not return until initMemoryManager() has been completed by a thread.
    There is no need to call this routine if mallocInitialized==2 . */
static bool doInitialization()
{
    MallocMutex::scoped_lock lock( initMutex );
    if (mallocInitialized!=2) {
        MALLOC_ASSERT( mallocInitialized==0, ASSERT_TEXT );
        mallocInitialized = 1;
        RecursiveMallocCallProtector scoped;
        if (!initMemoryManager()) {
            mallocInitialized = 0; // restore and out
            return false;
        }
#ifdef  MALLOC_EXTRA_INITIALIZATION
        MALLOC_EXTRA_INITIALIZATION;
#endif
#if MALLOC_CHECK_RECURSION
        RecursiveMallocCallProtector::detectNaiveOverload();
#endif
        MALLOC_ASSERT( mallocInitialized==1, ASSERT_TEXT );
        // Store must have release fence, otherwise mallocInitialized==2
        // might become remotely visible before side effects of
        // initMemoryManager() become remotely visible.
        FencedStore( mallocInitialized, 2 );
        if( GetBoolEnvironmentVariable("TBB_VERSION") ) {
            fputs(VersionString+1,stderr);
            hugePages.printStatus();
        }
    }
    /* It can't be 0 or I would have initialized it */
    MALLOC_ASSERT( mallocInitialized==2, ASSERT_TEXT );
    return true;
}

/********* End library initialization *************/

/********* The malloc show begins     *************/


FreeObject *Block::allocateFromFreeList()
{
    FreeObject *result;

    if (!freeList) return NULL;

    result = freeList;
    MALLOC_ASSERT( result, ASSERT_TEXT );

    freeList = result->next;
    MALLOC_ASSERT( allocatedCount < (slabSize-sizeof(Block))/objectSize, ASSERT_TEXT );
    allocatedCount++;
    STAT_increment(getThreadId(), getIndex(objectSize), allocFreeListUsed);

    return result;
}

FreeObject *Block::allocateFromBumpPtr()
{
    FreeObject *result = bumpPtr;
    if (result) {
        bumpPtr = (FreeObject *) ((uintptr_t) bumpPtr - objectSize);
        if ( (uintptr_t)bumpPtr < (uintptr_t)this+sizeof(Block) ) {
            bumpPtr = NULL;
        }
        MALLOC_ASSERT( allocatedCount < (slabSize-sizeof(Block))/objectSize, ASSERT_TEXT );
        allocatedCount++;
        STAT_increment(getThreadId(), getIndex(objectSize), allocBumpPtrUsed);
    }
    return result;
}

inline FreeObject* Block::allocate()
{
    MALLOC_ASSERT( isOwnedByCurrentThread(), ASSERT_TEXT );

    /* for better cache locality, first looking in the free list. */
    if ( FreeObject *result = allocateFromFreeList() ) {
        return result;
    }
    MALLOC_ASSERT( !freeList, ASSERT_TEXT );

    /* if free list is empty, try thread local bump pointer allocation. */
    if ( FreeObject *result = allocateFromBumpPtr() ) {
        return result;
    }
    MALLOC_ASSERT( !bumpPtr, ASSERT_TEXT );

    /* the block is considered full. */
    isFull = 1;
    return NULL;
}

size_t Block::findObjectSize(void *object) const
{
    size_t blSize = getSize();
#if MALLOC_CHECK_RECURSION
    // Currently, there is no aligned allocations from startup blocks,
    // so we can return just StartupBlock::msize().
    // TODO: This must be extended if we add aligned allocation from startup blocks.
    if (!blSize)
        return StartupBlock::msize(object);
#endif
    // object can be aligned, so real size can be less than block's
    size_t size =
        blSize - ((uintptr_t)object - (uintptr_t)findObjectToFree(object));
    MALLOC_ASSERT(size>0 && size<minLargeObjectSize, ASSERT_TEXT);
    return size;
}

void Bin::moveBlockToFront(Block *block)
{
    /* move the block to the front of the bin */
    if (block == activeBlk) return;
    outofTLSBin(block);
    pushTLSBin(block);
}

void Bin::processLessUsedBlock(MemoryPool *memPool, Block *block)
{
    if (block != activeBlk) {
        /* We are not actively using this block; return it to the general block pool */
        outofTLSBin(block);
        memPool->returnEmptyBlock(block, /*poolTheBlock=*/true);
    } else {
        /* all objects are free - let's restore the bump pointer */
        block->restoreBumpPtr();
    }
}

template<int LOW_MARK, int HIGH_MARK>
bool LocalLOCImpl<LOW_MARK, HIGH_MARK>::put(LargeMemoryBlock *object, ExtMemoryPool *extMemPool)
{
    const size_t size = object->unalignedSize;
    // not spoil cache with too large object, that can cause its total cleanup
    if (size > MAX_TOTAL_SIZE)
        return false;
    LargeMemoryBlock *localHead = (LargeMemoryBlock*)AtomicFetchStore(&head, 0);

    object->prev = NULL;
    object->next = localHead;
    if (localHead)
        localHead->prev = object;
    else {
        // those might not be cleaned during local cache stealing, correct them
        totalSize = 0;
        numOfBlocks = 0;
        tail = object;
    }
    localHead = object;
    totalSize += size;
    numOfBlocks++;
    // must meet both size and number of cached objects constrains
    if (totalSize > MAX_TOTAL_SIZE || numOfBlocks >= HIGH_MARK) {
        // scanning from tail until meet conditions
        while (totalSize > MAX_TOTAL_SIZE || numOfBlocks > LOW_MARK) {
            totalSize -= tail->unalignedSize;
            numOfBlocks--;
            tail = tail->prev;
        }
        LargeMemoryBlock *headToRelease = tail->next;
        tail->next = NULL;

        extMemPool->freeLargeObjectList(headToRelease);
    }

    FencedStore((intptr_t&)head, (intptr_t)localHead);
    return true;
}

template<int LOW_MARK, int HIGH_MARK>
LargeMemoryBlock *LocalLOCImpl<LOW_MARK, HIGH_MARK>::get(size_t size)
{
    LargeMemoryBlock *localHead, *res=NULL;

    if (size > MAX_TOTAL_SIZE)
        return NULL;

    if (!head || !(localHead = (LargeMemoryBlock*)AtomicFetchStore(&head, 0))) {
        // do not restore totalSize, numOfBlocks and tail at this point,
        // as they are used only in put(), where they must be restored
        return NULL;
    }

    for (LargeMemoryBlock *curr = localHead; curr; curr=curr->next) {
        if (curr->unalignedSize == size) {
            res = curr;
            if (curr->next)
                curr->next->prev = curr->prev;
            else
                tail = curr->prev;
            if (curr != localHead)
                curr->prev->next = curr->next;
            else
                localHead = curr->next;
            totalSize -= size;
            numOfBlocks--;
            break;
        }
    }
    FencedStore((intptr_t&)head, (intptr_t)localHead);
    return res;
}

template<int LOW_MARK, int HIGH_MARK>
bool LocalLOCImpl<LOW_MARK, HIGH_MARK>::externalCleanup(ExtMemoryPool *extMemPool)
{
    if (LargeMemoryBlock *localHead = (LargeMemoryBlock*)AtomicFetchStore(&head, 0)) {
        extMemPool->freeLargeObjectList(localHead);
        return true;
    }
    return false;
}

void *MemoryPool::getFromLLOCache(TLSData* tls, size_t size, size_t alignment)
{
    LargeMemoryBlock *lmb = NULL;

    size_t headersSize = sizeof(LargeMemoryBlock)+sizeof(LargeObjectHdr);
    size_t allocationSize = LargeObjectCache::alignToBin(size+headersSize+alignment);
    if (allocationSize < size) // allocationSize is wrapped around after alignToBin
        return NULL;
    MALLOC_ASSERT(allocationSize >= alignment, "Overflow must be checked before.");

    if (tls) {
        tls->markUsed();
        lmb = tls->lloc.get(allocationSize);
    }
    if (!lmb)
        lmb = extMemPool.mallocLargeObject(this, allocationSize);

    if (lmb) {
        // doing shuffle we suppose that alignment offset guarantees
        // that different cache lines are in use
        MALLOC_ASSERT(alignment >= estimatedCacheLineSize, ASSERT_TEXT);

        void *alignedArea = (void*)alignUp((uintptr_t)lmb+headersSize, alignment);
        uintptr_t alignedRight =
            alignDown((uintptr_t)lmb+lmb->unalignedSize - size, alignment);
        // Has some room to shuffle object between cache lines?
        // Note that alignedRight and alignedArea are aligned at alignment.
        unsigned ptrDelta = alignedRight - (uintptr_t)alignedArea;
        if (ptrDelta && tls) { // !tls is cold path
            // for the hot path of alignment==estimatedCacheLineSize,
            // allow compilers to use shift for division
            // (since estimatedCacheLineSize is a power-of-2 constant)
            unsigned numOfPossibleOffsets = alignment == estimatedCacheLineSize?
                  ptrDelta / estimatedCacheLineSize :
                  ptrDelta / alignment;
            unsigned myCacheIdx = ++tls->currCacheIdx;
            unsigned offset = myCacheIdx % numOfPossibleOffsets;

            // Move object to a cache line with an offset that is different from
            // previous allocation. This supposedly allows us to use cache
            // associativity more efficiently.
            alignedArea = (void*)((uintptr_t)alignedArea + offset*alignment);
        }
        MALLOC_ASSERT((uintptr_t)lmb+lmb->unalignedSize >=
                      (uintptr_t)alignedArea+size, "Object doesn't fit the block.");
        LargeObjectHdr *header = (LargeObjectHdr*)alignedArea-1;
        header->memoryBlock = lmb;
        header->backRefIdx = lmb->backRefIdx;
        setBackRef(header->backRefIdx, header);

        lmb->objectSize = size;

        MALLOC_ASSERT( isLargeObject<unknownMem>(alignedArea), ASSERT_TEXT );
        MALLOC_ASSERT( isAligned(alignedArea, alignment), ASSERT_TEXT );

        return alignedArea;
    }
    return NULL;
}

void MemoryPool::putToLLOCache(TLSData *tls, void *object)
{
    LargeObjectHdr *header = (LargeObjectHdr*)object - 1;
    // overwrite backRefIdx to simplify double free detection
    header->backRefIdx = BackRefIdx();

    if (tls) {
        tls->markUsed();
        if (tls->lloc.put(header->memoryBlock, &extMemPool))
            return;
    }
    extMemPool.freeLargeObject(header->memoryBlock);
}

/*
 * All aligned allocations fall into one of the following categories:
 *  1. if both request size and alignment are <= maxSegregatedObjectSize,
 *       we just align the size up, and request this amount, because for every size
 *       aligned to some power of 2, the allocated object is at least that aligned.
 * 2. for size<minLargeObjectSize, check if already guaranteed fittingAlignment is enough.
 * 3. if size+alignment<minLargeObjectSize, we take an object of fittingSizeN and align
 *       its address up; given such pointer, scalable_free could find the real object.
 *       Wrapping of size+alignment is impossible because maximal allowed
 *       alignment plus minLargeObjectSize can't lead to wrapping.
 * 4. otherwise, aligned large object is allocated.
 */
static void *allocateAligned(MemoryPool *memPool, size_t size, size_t alignment)
{
    MALLOC_ASSERT( isPowerOfTwo(alignment), ASSERT_TEXT );

    if (!isMallocInitialized())
        if (!doInitialization())
            return NULL;

    void *result;
    if (size<=maxSegregatedObjectSize && alignment<=maxSegregatedObjectSize)
        result = internalPoolMalloc(memPool, alignUp(size? size: sizeof(size_t), alignment));
    else if (size<minLargeObjectSize) {
        if (alignment<=fittingAlignment)
            result = internalPoolMalloc(memPool, size);
        else if (size+alignment < minLargeObjectSize) {
            void *unaligned = internalPoolMalloc(memPool, size+alignment);
            if (!unaligned) return NULL;
            result = alignUp(unaligned, alignment);
        } else
            goto LargeObjAlloc;
    } else {
    LargeObjAlloc:
        TLSData *tls = memPool->getTLS(/*create=*/true);
        // take into account only alignment that are higher then natural
        result =
            memPool->getFromLLOCache(tls, size, largeObjectAlignment>alignment?
                                               largeObjectAlignment: alignment);
    }

    MALLOC_ASSERT( isAligned(result, alignment), ASSERT_TEXT );
    return result;
}

static void *reallocAligned(MemoryPool *memPool, void *ptr,
                            size_t size, size_t alignment = 0)
{
    void *result;
    size_t copySize;

    if (isLargeObject<ourMem>(ptr)) {
        LargeMemoryBlock* lmb = ((LargeObjectHdr *)ptr - 1)->memoryBlock;
        copySize = lmb->unalignedSize-((uintptr_t)ptr-(uintptr_t)lmb);
        if (size <= copySize && (0==alignment || isAligned(ptr, alignment))) {
            lmb->objectSize = size;
            return ptr;
        } else {
            copySize = lmb->objectSize;
#if BACKEND_HAS_MREMAP
            if (void *r = memPool->extMemPool.remap(ptr, copySize, size,
                              alignment<largeObjectAlignment?
                              largeObjectAlignment : alignment))
                return r;
#endif
            result = alignment ? allocateAligned(memPool, size, alignment) :
                internalPoolMalloc(memPool, size);
        }
    } else {
        Block* block = (Block *)alignDown(ptr, slabSize);
        copySize = block->findObjectSize(ptr);
        if (size <= copySize && (0==alignment || isAligned(ptr, alignment))) {
            return ptr;
        } else {
            result = alignment ? allocateAligned(memPool, size, alignment) :
                internalPoolMalloc(memPool, size);
        }
    }
    if (result) {
        memcpy(result, ptr, copySize<size? copySize: size);
        internalPoolFree(memPool, ptr, 0);
    }
    return result;
}

/* A predicate checks if an object is properly placed inside its block */
inline bool Block::isProperlyPlaced(const void *object) const
{
    return 0 == ((uintptr_t)this + slabSize - (uintptr_t)object) % objectSize;
}

/* Finds the real object inside the block */
FreeObject *Block::findAllocatedObject(const void *address) const
{
    // calculate offset from the end of the block space
    uint16_t offset = (uintptr_t)this + slabSize - (uintptr_t)address;
    MALLOC_ASSERT( offset<=slabSize-sizeof(Block), ASSERT_TEXT );
    // find offset difference from a multiple of allocation size
    offset %= objectSize;
    // and move the address down to where the real object starts.
    return (FreeObject*)((uintptr_t)address - (offset? objectSize-offset: 0));
}

/*
 * Bad dereference caused by a foreign pointer is possible only here, not earlier in call chain.
 * Separate function isolates SEH code, as it has bad influence on compiler optimization.
 */
static inline BackRefIdx safer_dereference (const BackRefIdx *ptr)
{
    BackRefIdx id;
#if _MSC_VER
    __try {
#endif
        id = *ptr;
#if _MSC_VER
    } __except( GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION?
                EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH ) {
        id = BackRefIdx();
    }
#endif
    return id;
}

template<MemoryOrigin memOrigin>
bool isLargeObject(void *object)
{
    if (!isAligned(object, largeObjectAlignment))
        return false;
    LargeObjectHdr *header = (LargeObjectHdr*)object - 1;
    BackRefIdx idx = memOrigin==unknownMem? safer_dereference(&header->backRefIdx) :
        header->backRefIdx;

    return idx.isLargeObject()
        // in valid LargeObjectHdr memoryBlock is not NULL
        && header->memoryBlock
        // in valid LargeObjectHdr memoryBlock points somewhere before header
        // TODO: more strict check
        && (uintptr_t)header->memoryBlock < (uintptr_t)header
        && getBackRef(idx) == header;
}

static inline bool isSmallObject (void *ptr)
{
    Block* expectedBlock = (Block*)alignDown(ptr, slabSize);
    const BackRefIdx* idx = expectedBlock->getBackRefIdx();

    bool isSmall = expectedBlock == getBackRef(safer_dereference(idx));
    if (isSmall)
        expectedBlock->checkFreePrecond(ptr);
    return isSmall;
}

/**** Check if an object was allocated by scalable_malloc ****/
static inline bool isRecognized (void* ptr)
{
    return defaultMemPool->extMemPool.backend.ptrCanBeValid(ptr) &&
        (isLargeObject<unknownMem>(ptr) || isSmallObject(ptr));
}

static inline void freeSmallObject(void *object)
{
    /* mask low bits to get the block */
    Block *block = (Block *)alignDown(object, slabSize);
    block->checkFreePrecond(object);

#if MALLOC_CHECK_RECURSION
    if (block->isStartupAllocObject()) {
        ((StartupBlock *)block)->free(object);
        return;
    }
#endif
    if (block->isOwnedByCurrentThread()) {
        block->freeOwnObject(object);
    } else { /* Slower path to add to the shared list, the allocatedCount is updated by the owner thread in malloc. */
        FreeObject *objectToFree = block->findObjectToFree(object);
        block->freePublicObject(objectToFree);
    }
}

static void *internalPoolMalloc(MemoryPool* memPool, size_t size)
{
    Bin* bin;
    Block * mallocBlock;

    if (!memPool) return NULL;

    if (!size) size = sizeof(size_t);

    TLSData *tls = memPool->getTLS(/*create=*/true);

    /* Allocate a large object */
    if (size >= minLargeObjectSize)
        return memPool->getFromLLOCache(tls, size, largeObjectAlignment);

    if (!tls) return NULL;

    tls->markUsed();
    /*
     * Get an element in thread-local array corresponding to the given size;
     * It keeps ptr to the active block for allocations of this size
     */
    bin = tls->getAllocationBin(size);
    if ( !bin ) return NULL;

    /* Get a block to try to allocate in. */
    for( mallocBlock = bin->getActiveBlock(); mallocBlock;
         mallocBlock = bin->setPreviousBlockActive() ) // the previous block should be empty enough
    {
        if( FreeObject *result = mallocBlock->allocate() )
            return result;
    }

    /*
     * else privatize publicly freed objects in some block and allocate from it
     */
    mallocBlock = bin->getPublicFreeListBlock();
    if (mallocBlock) {
        if (mallocBlock->emptyEnoughToUse()) {
            bin->moveBlockToFront(mallocBlock);
        }
        MALLOC_ASSERT( mallocBlock->freeListNonNull(), ASSERT_TEXT );
        if ( FreeObject *result = mallocBlock->allocateFromFreeList() )
            return result;
        /* Else something strange happened, need to retry from the beginning; */
        TRACEF(( "[ScalableMalloc trace] Something is wrong: no objects in public free list; reentering.\n" ));
        return internalPoolMalloc(memPool, size);
    }

    /*
     * no suitable own blocks, try to get a partial block that some other thread has discarded.
     */
    mallocBlock = memPool->extMemPool.orphanedBlocks.get(tls, size);
    while (mallocBlock) {
        bin->pushTLSBin(mallocBlock);
        bin->setActiveBlock(mallocBlock); // TODO: move under the below condition?
        if( FreeObject *result = mallocBlock->allocate() )
            return result;
        mallocBlock = memPool->extMemPool.orphanedBlocks.get(tls, size);
    }

    /*
     * else try to get a new empty block
     */
    mallocBlock = memPool->getEmptyBlock(size);
    if (mallocBlock) {
        bin->pushTLSBin(mallocBlock);
        bin->setActiveBlock(mallocBlock);
        if( FreeObject *result = mallocBlock->allocate() )
            return result;
        /* Else something strange happened, need to retry from the beginning; */
        TRACEF(( "[ScalableMalloc trace] Something is wrong: no objects in empty block; reentering.\n" ));
        return internalPoolMalloc(memPool, size);
    }
    /*
     * else nothing works so return NULL
     */
    TRACEF(( "[ScalableMalloc trace] No memory found, returning NULL.\n" ));
    return NULL;
}

// When size==0 (i.e. unknown), detect here whether the object is large.
// For size is known and < minLargeObjectSize, we still need to check
// if the actual object is large, because large objects might be used
// for aligned small allocations.
static bool internalPoolFree(MemoryPool *memPool, void *object, size_t size)
{
    if (!memPool || !object) return false;

    // The library is initialized at allocation call, so releasing while
    // not initialized means foreign object is releasing.
    MALLOC_ASSERT(isMallocInitialized(), ASSERT_TEXT);
    MALLOC_ASSERT(memPool->extMemPool.userPool() || isRecognized(object),
                  "Invalid pointer during object releasing is detected.");

    if (size >= minLargeObjectSize || isLargeObject<ourMem>(object))
        memPool->putToLLOCache(memPool->getTLS(/*create=*/false), object);
    else
        freeSmallObject(object);
    return true;
}

static void *internalMalloc(size_t size)
{
    if (!size) size = sizeof(size_t);

#if MALLOC_CHECK_RECURSION
    if (RecursiveMallocCallProtector::sameThreadActive())
        return size<minLargeObjectSize? StartupBlock::allocate(size) :
            // nested allocation, so skip tls
            (FreeObject*)defaultMemPool->getFromLLOCache(NULL, size, slabSize);
#endif

    if (!isMallocInitialized())
        if (!doInitialization())
            return NULL;
    return internalPoolMalloc(defaultMemPool, size);
}

static void internalFree(void *object)
{
    internalPoolFree(defaultMemPool, object, 0);
}

static size_t internalMsize(void* ptr)
{
    if (ptr) {
        MALLOC_ASSERT(isRecognized(ptr), "Invalid pointer in scalable_msize detected.");
        if (isLargeObject<ourMem>(ptr)) {
            LargeMemoryBlock* lmb = ((LargeObjectHdr*)ptr - 1)->memoryBlock;
            return lmb->objectSize;
        } else
            return ((Block*)alignDown(ptr, slabSize))->findObjectSize(ptr);
    }
    errno = EINVAL;
    // Unlike _msize, return 0 in case of parameter error.
    // Returning size_t(-1) looks more like the way to troubles.
    return 0;
}

} // namespace internal

using namespace rml::internal;

// legacy entry point saved for compatibility with binaries complied
// with pre-6003 versions of TBB
rml::MemoryPool *pool_create(intptr_t pool_id, const MemPoolPolicy *policy)
{
    rml::MemoryPool *pool;
    MemPoolPolicy pol(policy->pAlloc, policy->pFree, policy->granularity);

    pool_create_v1(pool_id, &pol, &pool);
    return pool;
}

rml::MemPoolError pool_create_v1(intptr_t pool_id, const MemPoolPolicy *policy,
                                 rml::MemoryPool **pool)
{
    if ( !policy->pAlloc || policy->version<MemPoolPolicy::TBBMALLOC_POOL_VERSION
         // empty pFree allowed only for fixed pools
         || !(policy->fixedPool || policy->pFree)) {
        *pool = NULL;
        return INVALID_POLICY;
    }
    if ( policy->version>MemPoolPolicy::TBBMALLOC_POOL_VERSION // future versions are not supported
         // new flags can be added in place of reserved, but default
         // behaviour must be supported by this version
         || policy->reserved ) {
        *pool = NULL;
        return UNSUPPORTED_POLICY;
    }
    if (!isMallocInitialized())
        if (!doInitialization())
            return NO_MEMORY;
    rml::internal::MemoryPool *memPool =
        (rml::internal::MemoryPool*)internalMalloc((sizeof(rml::internal::MemoryPool)));
    if (!memPool) {
        *pool = NULL;
        return NO_MEMORY;
    }
    memset(memPool, 0, sizeof(rml::internal::MemoryPool));
    if (!memPool->init(pool_id, policy)) {
        internalFree(memPool);
        *pool = NULL;
        return NO_MEMORY;
    }

    *pool = (rml::MemoryPool*)memPool;
    return POOL_OK;
}

bool pool_destroy(rml::MemoryPool* memPool)
{
    if (!memPool) return false;
    bool ret = ((rml::internal::MemoryPool*)memPool)->destroy();
    internalFree(memPool);

    return ret;
}

bool pool_reset(rml::MemoryPool* memPool)
{
    if (!memPool) return false;

    return ((rml::internal::MemoryPool*)memPool)->reset();
}

void *pool_malloc(rml::MemoryPool* mPool, size_t size)
{
    return internalPoolMalloc((rml::internal::MemoryPool*)mPool, size);
}

void *pool_realloc(rml::MemoryPool* mPool, void *object, size_t size)
{
    if (!object)
        return internalPoolMalloc((rml::internal::MemoryPool*)mPool, size);
    if (!size) {
        internalPoolFree((rml::internal::MemoryPool*)mPool, object, 0);
        return NULL;
    }
    return reallocAligned((rml::internal::MemoryPool*)mPool, object, size, 0);
}

void *pool_aligned_malloc(rml::MemoryPool* mPool, size_t size, size_t alignment)
{
    if (!isPowerOfTwo(alignment) || 0==size)
        return NULL;

    return allocateAligned((rml::internal::MemoryPool*)mPool, size, alignment);
}

void *pool_aligned_realloc(rml::MemoryPool* memPool, void *ptr, size_t size, size_t alignment)
{
    if (!isPowerOfTwo(alignment))
        return NULL;
    rml::internal::MemoryPool *mPool = (rml::internal::MemoryPool*)memPool;
    void *tmp;

    if (!ptr)
        tmp = allocateAligned(mPool, size, alignment);
    else if (!size) {
        internalPoolFree(mPool, ptr, 0);
        return NULL;
    } else
        tmp = reallocAligned(mPool, ptr, size, alignment);

    return tmp;
}

bool pool_free(rml::MemoryPool *mPool, void *object)
{
    return internalPoolFree((rml::internal::MemoryPool*)mPool, object, 0);
}

rml::MemoryPool *pool_identify(void *object)
{
    rml::internal::MemoryPool *pool;
    if (isLargeObject<ourMem>(object)) {
        LargeObjectHdr *header = (LargeObjectHdr*)object - 1;
        pool = header->memoryBlock->pool;
    } else {
        Block *block = (Block*)alignDown(object, slabSize);
        pool = block->getMemPool();
    }
    // do not return defaultMemPool, as it can't be used in pool_free() etc
    __TBB_ASSERT_RELEASE(pool!=defaultMemPool,
        "rml::pool_identify() can't be used for scalable_malloc() etc results.");
    return (rml::MemoryPool*)pool;
}

} // namespace rml

using namespace rml::internal;

#if MALLOC_TRACE
static unsigned int threadGoingDownCount = 0;
#endif

/*
 * When a thread is shutting down this routine should be called to remove all the thread ids
 * from the malloc blocks and replace them with a NULL thread id.
 *
 * For pthreads, the function is set as a callback in pthread_key_create for TLS bin.
 * It will be automatically called at thread exit with the key value as the argument,
 * unless that value is NULL.
 * For Windows, it is called from DllMain( DLL_THREAD_DETACH ).
 *
 * However neither of the above is called for the main process thread, so the routine
 * also needs to be called during the process shutdown.
 *
*/
// TODO: Consider making this function part of class MemoryPool.
void doThreadShutdownNotification(TLSData* tls, bool main_thread)
{
    TRACEF(( "[ScalableMalloc trace] Thread id %d blocks return start %d\n",
             getThreadId(),  threadGoingDownCount++ ));

#if USE_PTHREAD
    if (tls) {
        if (!shutdownSync.threadDtorStart()) return;
        tls->getMemPool()->onThreadShutdown(tls);
        shutdownSync.threadDtorDone();
    } else
#endif
    {
        suppress_unused_warning(tls); // not used on Windows
        // The default pool is safe to use at this point:
        //   on Linux, only the main thread can go here before destroying defaultMemPool;
        //   on Windows, shutdown is synchronized via loader lock and isMallocInitialized().
        // See also __TBB_mallocProcessShutdownNotification()
        defaultMemPool->onThreadShutdown(defaultMemPool->getTLS(/*create=*/false));
        // Take lock to walk through other pools; but waiting might be dangerous at this point
        // (e.g. on Windows the main thread might deadlock)
        bool locked;
        MallocMutex::scoped_lock lock(MemoryPool::memPoolListLock, /*wait=*/!main_thread, &locked);
        if (locked) { // the list is safe to process
            for (MemoryPool *memPool = defaultMemPool->next; memPool; memPool = memPool->next)
                memPool->onThreadShutdown(memPool->getTLS(/*create=*/false));
        }
    }

    TRACEF(( "[ScalableMalloc trace] Thread id %d blocks return end\n", getThreadId() ));
}

#if USE_PTHREAD
void mallocThreadShutdownNotification(void* arg)
{
    // The routine is called for each pool (as TLS dtor) on each thread, except for the main thread
    if (!isMallocInitialized()) return;
    doThreadShutdownNotification((TLSData*)arg, false);
}
#else
extern "C" void __TBB_mallocThreadShutdownNotification()
{
    // The routine is called once per thread on Windows
    if (!isMallocInitialized()) return;
    doThreadShutdownNotification(NULL, false);
}
#endif

extern "C" void __TBB_mallocProcessShutdownNotification()
{
    if (!isMallocInitialized()) return;

    doThreadShutdownNotification(NULL, /*main_thread=*/true);
#if  __TBB_MALLOC_LOCACHE_STAT
    printf("cache hit ratio %f, size hit %f\n",
           1.*cacheHits/mallocCalls, 1.*memHitKB/memAllocKB);
    defaultMemPool->extMemPool.loc.reportStat(stdout);
#endif

    shutdownSync.processExit();
#if __TBB_SOURCE_DIRECTLY_INCLUDED
/* Pthread keys must be deleted as soon as possible to not call key dtor
   on thread termination when then the tbbmalloc code can be already unloaded.
*/
    defaultMemPool->destroy();
    destroyBackRefMaster(&defaultMemPool->extMemPool.backend);
    ThreadId::destroy();      // Delete key for thread id
    hugePages.reset();
    // new total malloc initialization is possible after this point
    FencedStore(mallocInitialized, 0);
#elif __TBB_USE_DLOPEN_REENTRANCY_WORKAROUND
/* In most cases we prevent unloading tbbmalloc, and don't clean up memory
   on process shutdown. When impossible to prevent, library unload results
   in shutdown notification, and it makes sense to release unused memory
   at that point (we can't release all memory because it's possible that
   it will be accessed after this point).
   TODO: better support systems where we can't prevent unloading by removing
   pthread destructors and releasing caches.
 */
    defaultMemPool->extMemPool.hardCachesCleanup();
#endif // __TBB_SOURCE_DIRECTLY_INCLUDED

#if COLLECT_STATISTICS
    unsigned nThreads = ThreadId::getMaxThreadId();
    for( int i=1; i<=nThreads && i<MAX_THREADS; ++i )
        STAT_print(i);
#endif
    if (!usedBySrcIncluded)
        MALLOC_ITT_FINI_ITTLIB();
}

extern "C" void * scalable_malloc(size_t size)
{
    void *ptr = internalMalloc(size);
    if (!ptr) errno = ENOMEM;
    return ptr;
}

extern "C" void scalable_free (void *object) {
    internalFree(object);
}

#if MALLOC_ZONE_OVERLOAD_ENABLED
extern "C" void __TBB_malloc_free_definite_size(void *object, size_t size) {
    internalPoolFree(defaultMemPool, object, size);
}
#endif

/*
 * A variant that provides additional memory safety, by checking whether the given address
 * was obtained with this allocator, and if not redirecting to the provided alternative call.
 */
extern "C" void __TBB_malloc_safer_free(void *object, void (*original_free)(void*))
{
    if (!object)
        return;

    // tbbmalloc can allocate object only when tbbmalloc has been initialized
    if (FencedLoad(mallocInitialized) && defaultMemPool->extMemPool.backend.ptrCanBeValid(object)) {
        if (isLargeObject<unknownMem>(object)) {
            // must check 1st for large object, because small object check touches 4 pages on left,
            // and it can be inaccessible
            TLSData *tls = defaultMemPool->getTLS(/*create=*/false);

            defaultMemPool->putToLLOCache(tls, object);
            return;
        } else if (isSmallObject(object)) {
            freeSmallObject(object);
            return;
        }
    }
    if (original_free)
        original_free(object);
}

/********* End the free code        *************/

/********* Code for scalable_realloc       ***********/

/*
 * From K&R
 * "realloc changes the size of the object pointed to by p to size. The contents will
 * be unchanged up to the minimum of the old and the new sizes. If the new size is larger,
 * the new space is uninitialized. realloc returns a pointer to the new space, or
 * NULL if the request cannot be satisfied, in which case *p is unchanged."
 *
 */
extern "C" void* scalable_realloc(void* ptr, size_t size)
{
    void *tmp;

    if (!ptr)
        tmp = internalMalloc(size);
    else if (!size) {
        internalFree(ptr);
        return NULL;
    } else
        tmp = reallocAligned(defaultMemPool, ptr, size, 0);

    if (!tmp) errno = ENOMEM;
    return tmp;
}

/*
 * A variant that provides additional memory safety, by checking whether the given address
 * was obtained with this allocator, and if not redirecting to the provided alternative call.
 */
extern "C" void* __TBB_malloc_safer_realloc(void* ptr, size_t sz, void* original_realloc)
{
    void *tmp; // TODO: fix warnings about uninitialized use of tmp

    if (!ptr) {
        tmp = internalMalloc(sz);
    } else if (FencedLoad(mallocInitialized) && isRecognized(ptr)) {
        if (!sz) {
            internalFree(ptr);
            return NULL;
        } else {
            tmp = reallocAligned(defaultMemPool, ptr, sz, 0);
        }
    }
#if USE_WINTHREAD
    else if (original_realloc && sz) {
        orig_ptrs *original_ptrs = static_cast<orig_ptrs*>(original_realloc);
        if ( original_ptrs->msize ){
            size_t oldSize = original_ptrs->msize(ptr);
            tmp = internalMalloc(sz);
            if (tmp) {
                memcpy(tmp, ptr, sz<oldSize? sz : oldSize);
                if ( original_ptrs->free ){
                    original_ptrs->free( ptr );
                }
            }
        } else
            tmp = NULL;
    }
#else
    else if (original_realloc) {
        typedef void* (*realloc_ptr_t)(void*,size_t);
        realloc_ptr_t original_realloc_ptr;
        (void *&)original_realloc_ptr = original_realloc;
        tmp = original_realloc_ptr(ptr,sz);
    }
#endif
    else tmp = NULL;

    if (!tmp) errno = ENOMEM;
    return tmp;
}

/********* End code for scalable_realloc   ***********/

/********* Code for scalable_calloc   ***********/

/*
 * From K&R
 * calloc returns a pointer to space for an array of nobj objects,
 * each of size size, or NULL if the request cannot be satisfied.
 * The space is initialized to zero bytes.
 *
 */

extern "C" void * scalable_calloc(size_t nobj, size_t size)
{
    // it's square root of maximal size_t value
    const size_t mult_not_overflow = size_t(1) << (sizeof(size_t)*CHAR_BIT/2);
    const size_t arraySize = nobj * size;

    // check for overflow during multiplication:
    if (nobj>=mult_not_overflow || size>=mult_not_overflow) // 1) heuristic check
        if (nobj && arraySize / nobj != size) {             // 2) exact check
            errno = ENOMEM;
            return NULL;
        }
    void* result = internalMalloc(arraySize);
    if (result)
        memset(result, 0, arraySize);
    else
        errno = ENOMEM;
    return result;
}

/********* End code for scalable_calloc   ***********/

/********* Code for aligned allocation API **********/

extern "C" int scalable_posix_memalign(void **memptr, size_t alignment, size_t size)
{
    if ( !isPowerOfTwoAtLeast(alignment, sizeof(void*)) )
        return EINVAL;
    void *result = allocateAligned(defaultMemPool, size, alignment);
    if (!result)
        return ENOMEM;
    *memptr = result;
    return 0;
}

extern "C" void * scalable_aligned_malloc(size_t size, size_t alignment)
{
    if (!isPowerOfTwo(alignment) || 0==size) {
        errno = EINVAL;
        return NULL;
    }
    void *tmp = allocateAligned(defaultMemPool, size, alignment);
    if (!tmp) errno = ENOMEM;
    return tmp;
}

extern "C" void * scalable_aligned_realloc(void *ptr, size_t size, size_t alignment)
{
    if (!isPowerOfTwo(alignment)) {
        errno = EINVAL;
        return NULL;
    }
    void *tmp;

    if (!ptr)
        tmp = allocateAligned(defaultMemPool, size, alignment);
    else if (!size) {
        scalable_free(ptr);
        return NULL;
    } else
        tmp = reallocAligned(defaultMemPool, ptr, size, alignment);

    if (!tmp) errno = ENOMEM;
    return tmp;
}

extern "C" void * __TBB_malloc_safer_aligned_realloc(void *ptr, size_t size, size_t alignment, void* orig_function)
{
    /* corner cases left out of reallocAligned to not deal with errno there */
    if (!isPowerOfTwo(alignment)) {
        errno = EINVAL;
        return NULL;
    }
    void *tmp = NULL;

    if (!ptr) {
        tmp = allocateAligned(defaultMemPool, size, alignment);
    } else if (FencedLoad(mallocInitialized) && isRecognized(ptr)) {
        if (!size) {
            internalFree(ptr);
            return NULL;
        } else {
            tmp = reallocAligned(defaultMemPool, ptr, size, alignment);
        }
    }
#if USE_WINTHREAD
    else {
        orig_aligned_ptrs *original_ptrs = static_cast<orig_aligned_ptrs*>(orig_function);
        if (size) {
            // Without orig_msize, we can't do anything with this.
            // Just keeping old pointer.
            if ( original_ptrs->aligned_msize ){
                // set alignment and offset to have possibly correct oldSize
                size_t oldSize = original_ptrs->aligned_msize(ptr, sizeof(void*), 0);
                tmp = allocateAligned(defaultMemPool, size, alignment);
                if (tmp) {
                    memcpy(tmp, ptr, size<oldSize? size : oldSize);
                    if ( original_ptrs->aligned_free ){
                        original_ptrs->aligned_free( ptr );
                    }
                }
            }
        } else {
            if ( original_ptrs->aligned_free ){
                original_ptrs->aligned_free( ptr );
            }
            return NULL;
        }
    }
#else
    // As original_realloc can't align result, and there is no way to find
    // size of reallocating object, we are giving up.
    suppress_unused_warning(orig_function);
#endif
    if (!tmp) errno = ENOMEM;
    return tmp;
}

extern "C" void scalable_aligned_free(void *ptr)
{
    internalFree(ptr);
}

/********* end code for aligned allocation API **********/

/********* Code for scalable_msize       ***********/

/*
 * Returns the size of a memory block allocated in the heap.
 */
extern "C" size_t scalable_msize(void* ptr)
{
    return internalMsize(ptr);
}

/*
 * A variant that provides additional memory safety, by checking whether the given address
 * was obtained with this allocator, and if not redirecting to the provided alternative call.
 */
extern "C" size_t __TBB_malloc_safer_msize(void *object, size_t (*original_msize)(void*))
{
    if (object) {
        // Check if the memory was allocated by scalable_malloc
        if (FencedLoad(mallocInitialized) && isRecognized(object))
            return internalMsize(object);
        else if (original_msize)
            return original_msize(object);
    }
    // object is NULL or unknown, or foreign and no original_msize
#if USE_WINTHREAD
    errno = EINVAL; // errno expected to be set only on this platform
#endif
    return 0;
}

/*
 * The same as above but for _aligned_msize case
 */
extern "C" size_t __TBB_malloc_safer_aligned_msize(void *object, size_t alignment, size_t offset, size_t (*orig_aligned_msize)(void*,size_t,size_t))
{
    if (object) {
        // Check if the memory was allocated by scalable_malloc
        if (FencedLoad(mallocInitialized) && isRecognized(object))
            return internalMsize(object);
        else if (orig_aligned_msize)
            return orig_aligned_msize(object,alignment,offset);
    }
    // object is NULL or unknown
    errno = EINVAL;
    return 0;
}

/********* End code for scalable_msize   ***********/

extern "C" int scalable_allocation_mode(int param, intptr_t value)
{
    if (param == TBBMALLOC_SET_SOFT_HEAP_LIMIT) {
        defaultMemPool->extMemPool.backend.setRecommendedMaxSize((size_t)value);
        return TBBMALLOC_OK;
    } else if (param == USE_HUGE_PAGES) {
#if __linux__
        switch (value) {
        case 0:
        case 1:
            hugePages.setMode(value);
            return TBBMALLOC_OK;
        default:
            return TBBMALLOC_INVALID_PARAM;
        }
#else
        return TBBMALLOC_NO_EFFECT;
#endif
#if __TBB_SOURCE_DIRECTLY_INCLUDED
    } else if (param == TBBMALLOC_INTERNAL_SOURCE_INCLUDED) {
        switch (value) {
        case 0: // used by dynamic library
        case 1: // used by static library or directly included sources
            usedBySrcIncluded = value;
            return TBBMALLOC_OK;
        default:
            return TBBMALLOC_INVALID_PARAM;
        }
#endif
    }
    return TBBMALLOC_INVALID_PARAM;
}

extern "C" int scalable_allocation_command(int cmd, void *param)
{
    if (param)
        return TBBMALLOC_INVALID_PARAM;
    switch(cmd) {
    case TBBMALLOC_CLEAN_THREAD_BUFFERS:
        if (TLSData *tls = defaultMemPool->getTLS(/*create=*/false))
            return tls->externalCleanup(&defaultMemPool->extMemPool,
                                        /*cleanOnlyUnused=*/false)?
                TBBMALLOC_OK : TBBMALLOC_NO_EFFECT;
        return TBBMALLOC_NO_EFFECT;
    case TBBMALLOC_CLEAN_ALL_BUFFERS:
        return defaultMemPool->extMemPool.hardCachesCleanup()?
            TBBMALLOC_OK : TBBMALLOC_NO_EFFECT;
    }
    return TBBMALLOC_INVALID_PARAM;
}
