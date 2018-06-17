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

/********* Allocation of large objects ************/


namespace rml {
namespace internal {


/* The functor called by the aggregator for the operation list */
template<typename Props>
class CacheBinFunctor {
    typename LargeObjectCacheImpl<Props>::CacheBin *const bin;
    ExtMemoryPool *const extMemPool;
    typename LargeObjectCacheImpl<Props>::BinBitMask *const bitMask;
    const int idx;

    LargeMemoryBlock *toRelease;
    bool needCleanup;
    uintptr_t currTime;

    /* Do preprocessing under the operation list. */
    /* All the OP_PUT_LIST operations are merged in the one operation.
       All OP_GET operations are merged with the OP_PUT_LIST operations but
       it demands the update of the moving average value in the bin.
       Only the last OP_CLEAN_TO_THRESHOLD operation has sense.
       The OP_CLEAN_ALL operation also should be performed only once.
       Moreover it cancels the OP_CLEAN_TO_THRESHOLD operation. */
    class OperationPreprocessor {
        // TODO: remove the dependency on CacheBin.
        typename LargeObjectCacheImpl<Props>::CacheBin *const  bin;

        /* Contains the relative time in the operation list.
           It counts in the reverse order since the aggregator also
           provides operations in the reverse order. */
        uintptr_t lclTime;

        /* opGet contains only OP_GET operations which cannot be merge with OP_PUT operations
           opClean contains all OP_CLEAN_TO_THRESHOLD and OP_CLEAN_ALL operations. */
        CacheBinOperation *opGet, *opClean;
        /* The time of the last OP_CLEAN_TO_THRESHOLD operations */
        uintptr_t cleanTime;

        /* lastGetOpTime - the time of the last OP_GET operation.
           lastGet - the same meaning as CacheBin::lastGet */
        uintptr_t lastGetOpTime, lastGet;

        /* The total sum of all usedSize changes requested with CBOP_UPDATE_USED_SIZE operations. */
        size_t updateUsedSize;

        /* The list of blocks for the OP_PUT_LIST operation. */
        LargeMemoryBlock *head, *tail;
        int putListNum;

        /* if the OP_CLEAN_ALL is requested. */
        bool isCleanAll;

        inline void commitOperation(CacheBinOperation *op) const;
        inline void addOpToOpList(CacheBinOperation *op, CacheBinOperation **opList) const;
        bool getFromPutList(CacheBinOperation* opGet, uintptr_t currTime);
        void addToPutList( LargeMemoryBlock *head, LargeMemoryBlock *tail, int num );

    public:
        OperationPreprocessor(typename LargeObjectCacheImpl<Props>::CacheBin *bin) :
            bin(bin), lclTime(0), opGet(NULL), opClean(NULL), cleanTime(0),
            lastGetOpTime(0), updateUsedSize(0), head(NULL), isCleanAll(false)  {}
        void operator()(CacheBinOperation* opList);
        uintptr_t getTimeRange() const { return -lclTime; }

        friend class CacheBinFunctor;
    };

public:
    CacheBinFunctor(typename LargeObjectCacheImpl<Props>::CacheBin *bin, ExtMemoryPool *extMemPool,
                    typename LargeObjectCacheImpl<Props>::BinBitMask *bitMask, int idx) :
        bin(bin), extMemPool(extMemPool), bitMask(bitMask), idx(idx), toRelease(NULL), needCleanup(false) {}
    void operator()(CacheBinOperation* opList);

    bool isCleanupNeeded() const { return needCleanup; }
    LargeMemoryBlock *getToRelease() const { return toRelease; }
    uintptr_t getCurrTime() const { return currTime; }
};

// ---------------- Cache Bin Aggregator Operation Helpers ---------------- //
// The list of possible operations.
enum CacheBinOperationType {
    CBOP_INVALID = 0,
    CBOP_GET,
    CBOP_PUT_LIST,
    CBOP_CLEAN_TO_THRESHOLD,
    CBOP_CLEAN_ALL,
    CBOP_UPDATE_USED_SIZE
};

// The operation status list. CBST_NOWAIT can be specified for non-blocking operations.
enum CacheBinOperationStatus {
    CBST_WAIT = 0,
    CBST_NOWAIT,
    CBST_DONE
};

// The list of structures which describe the operation data
struct OpGet {
    static const CacheBinOperationType type = CBOP_GET;
    LargeMemoryBlock **res;
    size_t size;
    uintptr_t currTime;
};

struct OpPutList {
    static const CacheBinOperationType type = CBOP_PUT_LIST;
    LargeMemoryBlock *head;
};

struct OpCleanToThreshold {
    static const CacheBinOperationType type = CBOP_CLEAN_TO_THRESHOLD;
    LargeMemoryBlock **res;
    uintptr_t currTime;
};

struct OpCleanAll {
    static const CacheBinOperationType type = CBOP_CLEAN_ALL;
    LargeMemoryBlock **res;
};

struct OpUpdateUsedSize {
    static const CacheBinOperationType type = CBOP_UPDATE_USED_SIZE;
    size_t size;
};

union CacheBinOperationData {
private:
    OpGet opGet;
    OpPutList opPutList;
    OpCleanToThreshold opCleanToThreshold;
    OpCleanAll opCleanAll;
    OpUpdateUsedSize opUpdateUsedSize;
};

// Forward declarations
template <typename OpTypeData> OpTypeData& opCast(CacheBinOperation &op);

// Describes the aggregator operation
struct CacheBinOperation : public MallocAggregatedOperation<CacheBinOperation>::type {
    CacheBinOperationType type;

    template <typename OpTypeData>
    CacheBinOperation(OpTypeData &d, CacheBinOperationStatus st = CBST_WAIT) {
        opCast<OpTypeData>(*this) = d;
        type = OpTypeData::type;
        MallocAggregatedOperation<CacheBinOperation>::type::status = st;
    }
private:
    CacheBinOperationData data;

    template <typename OpTypeData>
    friend OpTypeData& opCast(CacheBinOperation &op);
};

// The opCast function can be the member of CacheBinOperation but it will have
// small stylistic ambiguity: it will look like a getter (with a cast) for the
// CacheBinOperation::data data member but it should return a reference to
// simplify the code from a lot of getter/setter calls. So the global cast in
// the style of static_cast (or reinterpret_cast) seems to be more readable and
// have more explicit semantic.
template <typename OpTypeData>
OpTypeData& opCast(CacheBinOperation &op) {
    return *reinterpret_cast<OpTypeData*>(&op.data);
}
// ------------------------------------------------------------------------ //

#if __TBB_MALLOC_LOCACHE_STAT
intptr_t mallocCalls, cacheHits;
intptr_t memAllocKB, memHitKB;
#endif

inline bool lessThanWithOverflow(intptr_t a, intptr_t b)
{
    return (a < b && (b - a < UINTPTR_MAX/2)) ||
           (a > b && (a - b > UINTPTR_MAX/2));
}

/* ----------------------------------- Operation processing methods ------------------------------------ */

template<typename Props> void CacheBinFunctor<Props>::
    OperationPreprocessor::commitOperation(CacheBinOperation *op) const
{
    FencedStore( (intptr_t&)(op->status), CBST_DONE );
}

template<typename Props> void CacheBinFunctor<Props>::
    OperationPreprocessor::addOpToOpList(CacheBinOperation *op, CacheBinOperation **opList) const
{
    op->next = *opList;
    *opList = op;
}

template<typename Props> bool CacheBinFunctor<Props>::
    OperationPreprocessor::getFromPutList(CacheBinOperation *opGet, uintptr_t currTime)
{
    if ( head ) {
        uintptr_t age = head->age;
        LargeMemoryBlock *next = head->next;
        *opCast<OpGet>(*opGet).res = head;
        commitOperation( opGet );
        head = next;
        putListNum--;
        MALLOC_ASSERT( putListNum>=0, ASSERT_TEXT );

        // use moving average with current hit interval
        bin->updateMeanHitRange( currTime - age );
        return true;
    }
    return false;
}

template<typename Props> void CacheBinFunctor<Props>::
    OperationPreprocessor::addToPutList(LargeMemoryBlock *h, LargeMemoryBlock *t, int num)
{
    if ( head ) {
        MALLOC_ASSERT( tail, ASSERT_TEXT );
        tail->next = h;
        h->prev = tail;
        tail = t;
        putListNum += num;
    } else {
        head = h;
        tail = t;
        putListNum = num;
    }
}

template<typename Props> void CacheBinFunctor<Props>::
    OperationPreprocessor::operator()(CacheBinOperation* opList)
{
    for ( CacheBinOperation *op = opList, *opNext; op; op = opNext ) {
        opNext = op->next;
        switch ( op->type ) {
        case CBOP_GET:
            {
                lclTime--;
                if ( !lastGetOpTime ) {
                    lastGetOpTime = lclTime;
                    lastGet = 0;
                } else if ( !lastGet ) lastGet = lclTime;

                if ( !getFromPutList(op,lclTime) ) {
                    opCast<OpGet>(*op).currTime = lclTime;
                    addOpToOpList( op, &opGet );
                }
            }
            break;

        case CBOP_PUT_LIST:
            {
                LargeMemoryBlock *head = opCast<OpPutList>(*op).head;
                LargeMemoryBlock *curr = head, *prev = NULL;

                int num = 0;
                do {
                    // we do not kept prev pointers during assigning blocks to bins, set them now
                    curr->prev = prev;

                    // Save the local times to the memory blocks. Local times are necessary
                    // for the getFromPutList function which updates the hit range value in
                    // CacheBin when OP_GET and OP_PUT_LIST operations are merged successfully.
                    // The age will be updated to the correct global time after preprocessing
                    // when global cache time is updated.
                    curr->age = --lclTime;

                    prev = curr;
                    num += 1;

                    STAT_increment(getThreadId(), ThreadCommonCounters, cacheLargeObj);
                } while (( curr = curr->next ));

                LargeMemoryBlock *tail = prev;
                addToPutList(head, tail, num);

                while ( opGet ) {
                    CacheBinOperation *next = opGet->next;
                    if ( !getFromPutList(opGet, opCast<OpGet>(*opGet).currTime) )
                        break;
                    opGet = next;
                }
            }
            break;

        case CBOP_UPDATE_USED_SIZE:
            updateUsedSize += opCast<OpUpdateUsedSize>(*op).size;
            commitOperation( op );
            break;

        case CBOP_CLEAN_ALL:
            isCleanAll = true;
            addOpToOpList( op, &opClean );
            break;

        case CBOP_CLEAN_TO_THRESHOLD:
            {
                uintptr_t currTime = opCast<OpCleanToThreshold>(*op).currTime;
                // We don't worry about currTime overflow since it is a rare
                // occurrence and doesn't affect correctness
                cleanTime = cleanTime < currTime ? currTime : cleanTime;
                addOpToOpList( op, &opClean );
            }
            break;

        default:
            MALLOC_ASSERT( false, "Unknown operation." );
        }
    }
    MALLOC_ASSERT( !( opGet && head ), "Not all put/get pairs are processed!" );
}

template<typename Props> void CacheBinFunctor<Props>::operator()(CacheBinOperation* opList)
{
    MALLOC_ASSERT( opList, "Empty operation list is passed into operation handler." );

    OperationPreprocessor prep(bin);
    prep(opList);

    if ( uintptr_t timeRange = prep.getTimeRange() ) {
        uintptr_t startTime = extMemPool->loc.getCurrTimeRange(timeRange);
        // endTime is used as the current (base) time since the local time is negative.
        uintptr_t endTime = startTime + timeRange;

        if ( prep.lastGetOpTime && prep.lastGet ) bin->setLastGet(prep.lastGet+endTime);

        if ( CacheBinOperation *opGet = prep.opGet ) {
            bool isEmpty = false;
            do {
#if __TBB_MALLOC_WHITEBOX_TEST
                tbbmalloc_whitebox::locGetProcessed++;
#endif
                const OpGet &opGetData = opCast<OpGet>(*opGet);
                if ( !isEmpty ) {
                    if ( LargeMemoryBlock *res = bin->get() ) {
                        uintptr_t getTime = opGetData.currTime + endTime;
                        // use moving average with current hit interval
                        bin->updateMeanHitRange( getTime - res->age);
                        bin->updateCachedSize( -opGetData.size );
                        *opGetData.res = res;
                    } else {
                        isEmpty = true;
                        uintptr_t lastGetOpTime = prep.lastGetOpTime+endTime;
                        bin->forgetOutdatedState(lastGetOpTime);
                        bin->updateAgeThreshold(lastGetOpTime);
                    }
                }

                CacheBinOperation *opNext = opGet->next;
                bin->updateUsedSize( opGetData.size, bitMask, idx );
                prep.commitOperation( opGet );
                opGet = opNext;
            } while ( opGet );
            if ( prep.lastGetOpTime )
                bin->setLastGet( prep.lastGetOpTime + endTime );
        } else if ( LargeMemoryBlock *curr = prep.head ) {
            curr->prev = NULL;
            while ( curr ) {
                // Update local times to global times
                curr->age += endTime;
                curr=curr->next;
            }
#if __TBB_MALLOC_WHITEBOX_TEST
            tbbmalloc_whitebox::locPutProcessed+=prep.putListNum;
#endif
            toRelease = bin->putList(prep.head, prep.tail, bitMask, idx, prep.putListNum);
        }
        needCleanup = extMemPool->loc.isCleanupNeededOnRange(timeRange, startTime);
        currTime = endTime - 1;
    }

    if ( CacheBinOperation *opClean = prep.opClean ) {
        if ( prep.isCleanAll )
            *opCast<OpCleanAll>(*opClean).res = bin->cleanAll(bitMask, idx);
        else
            *opCast<OpCleanToThreshold>(*opClean).res = bin->cleanToThreshold(prep.cleanTime, bitMask, idx);

        CacheBinOperation *opNext = opClean->next;
        prep.commitOperation( opClean );

        while (( opClean = opNext )) {
            opNext = opClean->next;
            prep.commitOperation(opClean);
        }
    }

    if ( size_t size = prep.updateUsedSize )
        bin->updateUsedSize(size, bitMask, idx);
}
/* ----------------------------------------------------------------------------------------------------- */
/* --------------------------- Methods for creating and executing operations --------------------------- */
template<typename Props> void LargeObjectCacheImpl<Props>::
    CacheBin::ExecuteOperation(CacheBinOperation *op, ExtMemoryPool *extMemPool, BinBitMask *bitMask, int idx, bool longLifeTime)
{
    CacheBinFunctor<Props> func( this, extMemPool, bitMask, idx );
    aggregator.execute( op, func, longLifeTime );

    if (  LargeMemoryBlock *toRelease = func.getToRelease() )
        extMemPool->backend.returnLargeObject(toRelease);

    if ( func.isCleanupNeeded() )
        extMemPool->loc.doCleanup( func.getCurrTime(), /*doThreshDecr=*/false);
}

template<typename Props> LargeMemoryBlock *LargeObjectCacheImpl<Props>::
    CacheBin::get(ExtMemoryPool *extMemPool, size_t size, BinBitMask *bitMask, int idx)
{
    LargeMemoryBlock *lmb=NULL;
    OpGet data = {&lmb, size};
    CacheBinOperation op(data);
    ExecuteOperation( &op, extMemPool, bitMask, idx );
    return lmb;
}

template<typename Props> void LargeObjectCacheImpl<Props>::
    CacheBin::putList(ExtMemoryPool *extMemPool, LargeMemoryBlock *head, BinBitMask *bitMask, int idx)
{
    MALLOC_ASSERT(sizeof(LargeMemoryBlock)+sizeof(CacheBinOperation)<=head->unalignedSize, "CacheBinOperation is too large to be placed in LargeMemoryBlock!");

    OpPutList data = {head};
    CacheBinOperation *op = new (head+1) CacheBinOperation(data, CBST_NOWAIT);
    ExecuteOperation( op, extMemPool, bitMask, idx, false );
}

template<typename Props> bool LargeObjectCacheImpl<Props>::
    CacheBin::cleanToThreshold(ExtMemoryPool *extMemPool, BinBitMask *bitMask, uintptr_t currTime, int idx)
{
    LargeMemoryBlock *toRelease = NULL;

    /* oldest may be more recent then age, that's why cast to signed type
       was used. age overflow is also processed correctly. */
    if (last && (intptr_t)(currTime - oldest) > ageThreshold) {
        OpCleanToThreshold data = {&toRelease, currTime};
        CacheBinOperation op(data);
        ExecuteOperation( &op, extMemPool, bitMask, idx );
    }
    bool released = toRelease;

    Backend *backend = &extMemPool->backend;
    while ( toRelease ) {
        LargeMemoryBlock *helper = toRelease->next;
        backend->returnLargeObject(toRelease);
        toRelease = helper;
    }
    return released;
}

template<typename Props> bool LargeObjectCacheImpl<Props>::
    CacheBin::releaseAllToBackend(ExtMemoryPool *extMemPool, BinBitMask *bitMask, int idx)
{
    LargeMemoryBlock *toRelease = NULL;

    if (last) {
        OpCleanAll data = {&toRelease};
        CacheBinOperation op(data);
        ExecuteOperation(&op, extMemPool, bitMask, idx);
    }
    bool released = toRelease;

    Backend *backend = &extMemPool->backend;
    while ( toRelease ) {
        LargeMemoryBlock *helper = toRelease->next;
        MALLOC_ASSERT(!helper || lessThanWithOverflow(helper->age, toRelease->age),
                      ASSERT_TEXT);
        backend->returnLargeObject(toRelease);
        toRelease = helper;
    }
    return released;
}

template<typename Props> void LargeObjectCacheImpl<Props>::
    CacheBin::updateUsedSize(ExtMemoryPool *extMemPool, size_t size, BinBitMask *bitMask, int idx) {
    OpUpdateUsedSize data = {size};
    CacheBinOperation op(data);
    ExecuteOperation( &op, extMemPool, bitMask, idx );
}
/* ----------------------------------------------------------------------------------------------------- */
/* ------------------------------ Unsafe methods used with the aggregator ------------------------------ */
template<typename Props> LargeMemoryBlock *LargeObjectCacheImpl<Props>::
    CacheBin::putList(LargeMemoryBlock *head, LargeMemoryBlock *tail, BinBitMask *bitMask, int idx, int num)
{
    size_t size = head->unalignedSize;
    usedSize -= num*size;
    MALLOC_ASSERT( !last || (last->age != 0 && last->age != -1U), ASSERT_TEXT );
    MALLOC_ASSERT( (tail==head && num==1) || (tail!=head && num>1), ASSERT_TEXT );
    LargeMemoryBlock *toRelease = NULL;
    if (!lastCleanedAge) {
        // 1st object of such size was released.
        // Not cache it, and remember when this occurs
        // to take into account during cache miss.
        lastCleanedAge = tail->age;
        toRelease = tail;
        tail = tail->prev;
        if (tail)
            tail->next = NULL;
        else
            head = NULL;
        num--;
    }
    if (num) {
        // add [head;tail] list to cache
        MALLOC_ASSERT( tail, ASSERT_TEXT );
        tail->next = first;
        if (first)
            first->prev = tail;
        first = head;
        if (!last) {
            MALLOC_ASSERT(0 == oldest, ASSERT_TEXT);
            oldest = tail->age;
            last = tail;
        }

        cachedSize += num*size;
    }

    // No used object, and nothing in the bin, mark the bin as empty
    if (!usedSize && !first)
        bitMask->set(idx, false);

    return toRelease;
}

template<typename Props> LargeMemoryBlock *LargeObjectCacheImpl<Props>::
    CacheBin::get()
{
    LargeMemoryBlock *result=first;
    if (result) {
        first = result->next;
        if (first)
            first->prev = NULL;
        else {
            last = NULL;
            oldest = 0;
        }
    }

    return result;
}

// forget the history for the bin if it was unused for long time
template<typename Props> void LargeObjectCacheImpl<Props>::
    CacheBin::forgetOutdatedState(uintptr_t currTime)
{
    // If the time since the last get is LongWaitFactor times more than ageThreshold
    // for the bin, treat the bin as rarely-used and forget everything we know
    // about it.
    // If LongWaitFactor is too small, we forget too early and
    // so prevents good caching, while if too high, caching blocks
    // with unrelated usage pattern occurs.
    const uintptr_t sinceLastGet = currTime - lastGet;
    bool doCleanup = false;

    if (ageThreshold)
        doCleanup = sinceLastGet > Props::LongWaitFactor*ageThreshold;
    else if (lastCleanedAge)
        doCleanup = sinceLastGet > Props::LongWaitFactor*(lastCleanedAge - lastGet);

    if (doCleanup) {
        lastCleanedAge = 0;
        ageThreshold = 0;
    }

}

template<typename Props> LargeMemoryBlock *LargeObjectCacheImpl<Props>::
    CacheBin::cleanToThreshold(uintptr_t currTime, BinBitMask *bitMask, int idx)
{
    /* oldest may be more recent then age, that's why cast to signed type
    was used. age overflow is also processed correctly. */
    if ( !last || (intptr_t)(currTime - last->age) < ageThreshold ) return NULL;

#if MALLOC_DEBUG
    uintptr_t nextAge = 0;
#endif
    do {
#if MALLOC_DEBUG
        // check that list ordered
        MALLOC_ASSERT(!nextAge || lessThanWithOverflow(nextAge, last->age),
            ASSERT_TEXT);
        nextAge = last->age;
#endif
        cachedSize -= last->unalignedSize;
        last = last->prev;
    } while (last && (intptr_t)(currTime - last->age) > ageThreshold);

    LargeMemoryBlock *toRelease = NULL;
    if (last) {
        toRelease = last->next;
        oldest = last->age;
        last->next = NULL;
    } else {
        toRelease = first;
        first = NULL;
        oldest = 0;
        if (!usedSize)
            bitMask->set(idx, false);
    }
    MALLOC_ASSERT( toRelease, ASSERT_TEXT );
    lastCleanedAge = toRelease->age;

    return toRelease;
}

template<typename Props> LargeMemoryBlock *LargeObjectCacheImpl<Props>::
    CacheBin::cleanAll(BinBitMask *bitMask, int idx)
{
    if (!last) return NULL;

    LargeMemoryBlock *toRelease = first;
    last = NULL;
    first = NULL;
    oldest = 0;
    cachedSize = 0;
    if (!usedSize)
        bitMask->set(idx, false);

    return toRelease;
}
/* ----------------------------------------------------------------------------------------------------- */

template<typename Props> size_t LargeObjectCacheImpl<Props>::
    CacheBin::reportStat(int num, FILE *f)
{
#if __TBB_MALLOC_LOCACHE_STAT
    if (first)
        printf("%d(%lu): total %lu KB thr %ld lastCln %lu oldest %lu\n",
               num, num*Props::CacheStep+Props::MinSize,
               cachedSize/1024, ageThreshold, lastCleanedAge, oldest);
#else
    suppress_unused_warning(num);
    suppress_unused_warning(f);
#endif
    return cachedSize;
}

// release from cache blocks that are older than ageThreshold
template<typename Props>
bool LargeObjectCacheImpl<Props>::regularCleanup(ExtMemoryPool *extMemPool, uintptr_t currTime, bool doThreshDecr)
{
    bool released = false;
    BinsSummary binsSummary;

    for (int i = bitMask.getMaxTrue(numBins-1); i >= 0;
         i = bitMask.getMaxTrue(i-1)) {
        bin[i].updateBinsSummary(&binsSummary);
        if (!doThreshDecr && tooLargeLOC>2 && binsSummary.isLOCTooLarge()) {
            // if LOC is too large for quite long time, decrease the threshold
            // based on bin hit statistics.
            // For this, redo cleanup from the beginning.
            // Note: on this iteration total usedSz can be not too large
            // in comparison to total cachedSz, as we calculated it only
            // partially. We are ok with it.
            i = bitMask.getMaxTrue(numBins-1)+1;
            doThreshDecr = true;
            binsSummary.reset();
            continue;
        }
        if (doThreshDecr)
            bin[i].decreaseThreshold();
        if (bin[i].cleanToThreshold(extMemPool, &bitMask, currTime, i))
            released = true;
    }

    // We want to find if LOC was too large for some time continuously,
    // so OK with races between incrementing and zeroing, but incrementing
    // must be atomic.
    if (binsSummary.isLOCTooLarge())
        AtomicIncrement(tooLargeLOC);
    else
        tooLargeLOC = 0;
    return released;
}

template<typename Props>
bool LargeObjectCacheImpl<Props>::cleanAll(ExtMemoryPool *extMemPool)
{
    bool released = false;
    for (int i = numBins-1; i >= 0; i--)
        released |= bin[i].releaseAllToBackend(extMemPool, &bitMask, i);
    return released;
}

#if __TBB_MALLOC_WHITEBOX_TEST
template<typename Props>
size_t LargeObjectCacheImpl<Props>::getLOCSize() const
{
    size_t size = 0;
    for (int i = numBins-1; i >= 0; i--)
        size += bin[i].getSize();
    return size;
}

size_t LargeObjectCache::getLOCSize() const
{
    return largeCache.getLOCSize() + hugeCache.getLOCSize();
}

template<typename Props>
size_t LargeObjectCacheImpl<Props>::getUsedSize() const
{
    size_t size = 0;
    for (int i = numBins-1; i >= 0; i--)
        size += bin[i].getUsedSize();
    return size;
}

size_t LargeObjectCache::getUsedSize() const
{
    return largeCache.getUsedSize() + hugeCache.getUsedSize();
}
#endif // __TBB_MALLOC_WHITEBOX_TEST

inline bool LargeObjectCache::isCleanupNeededOnRange(uintptr_t range, uintptr_t currTime)
{
    return range >= cacheCleanupFreq
        || currTime+range < currTime-1 // overflow, 0 is power of 2, do cleanup
        // (prev;prev+range] contains n*cacheCleanupFreq
        || alignUp(currTime, cacheCleanupFreq)<currTime+range;
}

bool LargeObjectCache::doCleanup(uintptr_t currTime, bool doThreshDecr)
{
    if (!doThreshDecr)
        extMemPool->allLocalCaches.markUnused();
    return largeCache.regularCleanup(extMemPool, currTime, doThreshDecr)
        | hugeCache.regularCleanup(extMemPool, currTime, doThreshDecr);
}

bool LargeObjectCache::decreasingCleanup()
{
    return doCleanup(FencedLoad((intptr_t&)cacheCurrTime), /*doThreshDecr=*/true);
}

bool LargeObjectCache::regularCleanup()
{
    return doCleanup(FencedLoad((intptr_t&)cacheCurrTime), /*doThreshDecr=*/false);
}

bool LargeObjectCache::cleanAll()
{
    return largeCache.cleanAll(extMemPool) | hugeCache.cleanAll(extMemPool);
}

template<typename Props>
LargeMemoryBlock *LargeObjectCacheImpl<Props>::get(ExtMemoryPool *extMemoryPool, size_t size)
{
    MALLOC_ASSERT( size%Props::CacheStep==0, ASSERT_TEXT );
    int idx = sizeToIdx(size);

    LargeMemoryBlock *lmb = bin[idx].get(extMemoryPool, size, &bitMask, idx);

    if (lmb) {
        MALLOC_ITT_SYNC_ACQUIRED(bin+idx);
        STAT_increment(getThreadId(), ThreadCommonCounters, allocCachedLargeObj);
    }
    return lmb;
}

template<typename Props>
void LargeObjectCacheImpl<Props>::updateCacheState(ExtMemoryPool *extMemPool, DecreaseOrIncrease op, size_t size)
{
    int idx = sizeToIdx(size);
    MALLOC_ASSERT(idx<numBins, ASSERT_TEXT);
    bin[idx].updateUsedSize(extMemPool, op==decrease? -size : size, &bitMask, idx);
}

#if __TBB_MALLOC_LOCACHE_STAT
template<typename Props>
void LargeObjectCacheImpl<Props>::reportStat(FILE *f)
{
    size_t cachedSize = 0;
    for (int i=0; i<numBins; i++)
        cachedSize += bin[i].reportStat(i, f);
    fprintf(f, "total LOC size %lu MB\n", cachedSize/1024/1024);
}

void LargeObjectCache::reportStat(FILE *f)
{
    largeCache.reportStat(f);
    hugeCache.reportStat(f);
    fprintf(f, "cache time %lu\n", cacheCurrTime);
}
#endif

template<typename Props>
void LargeObjectCacheImpl<Props>::putList(ExtMemoryPool *extMemPool, LargeMemoryBlock *toCache)
{
    int toBinIdx = sizeToIdx(toCache->unalignedSize);

    MALLOC_ITT_SYNC_RELEASING(bin+toBinIdx);
    bin[toBinIdx].putList(extMemPool, toCache, &bitMask, toBinIdx);
}

void LargeObjectCache::updateCacheState(DecreaseOrIncrease op, size_t size)
{
    if (size < maxLargeSize)
        largeCache.updateCacheState(extMemPool, op, size);
    else if (size < maxHugeSize)
        hugeCache.updateCacheState(extMemPool, op, size);
}

void LargeObjectCache::registerRealloc(size_t oldSize, size_t newSize)
{
    updateCacheState(decrease, oldSize);
    updateCacheState(increase, newSize);
}

// return artificial bin index, it's used only during sorting and never saved
int LargeObjectCache::sizeToIdx(size_t size)
{
    MALLOC_ASSERT(size < maxHugeSize, ASSERT_TEXT);
    return size < maxLargeSize?
        LargeCacheType::sizeToIdx(size) :
        LargeCacheType::getNumBins()+HugeCacheType::sizeToIdx(size);
}

void LargeObjectCache::putList(LargeMemoryBlock *list)
{
    LargeMemoryBlock *toProcess, *n;

    for (LargeMemoryBlock *curr = list; curr; curr = toProcess) {
        LargeMemoryBlock *tail = curr;
        toProcess = curr->next;
        if (curr->unalignedSize >= maxHugeSize) {
            extMemPool->backend.returnLargeObject(curr);
            continue;
        }
        int currIdx = sizeToIdx(curr->unalignedSize);

        // Find all blocks fitting to same bin. Not use more efficient sorting
        // algorithm because list is short (commonly,
        // LocalLOC's HIGH_MARK-LOW_MARK, i.e. 24 items).
        for (LargeMemoryBlock *b = toProcess; b; b = n) {
            n = b->next;
            if (sizeToIdx(b->unalignedSize) == currIdx) {
                tail->next = b;
                tail = b;
                if (toProcess == b)
                    toProcess = toProcess->next;
                else {
                    b->prev->next = b->next;
                    if (b->next)
                        b->next->prev = b->prev;
                }
            }
        }
        tail->next = NULL;
        if (curr->unalignedSize < maxLargeSize)
            largeCache.putList(extMemPool, curr);
        else
            hugeCache.putList(extMemPool, curr);
    }
}

void LargeObjectCache::put(LargeMemoryBlock *largeBlock)
{
    if (largeBlock->unalignedSize < maxHugeSize) {
        largeBlock->next = NULL;
        if (largeBlock->unalignedSize<maxLargeSize)
            largeCache.putList(extMemPool, largeBlock);
        else
            hugeCache.putList(extMemPool, largeBlock);
    } else
        extMemPool->backend.returnLargeObject(largeBlock);
}

LargeMemoryBlock *LargeObjectCache::get(size_t size)
{
    MALLOC_ASSERT( size%largeBlockCacheStep==0, ASSERT_TEXT );
    MALLOC_ASSERT( size>=minLargeSize, ASSERT_TEXT );

    if ( size < maxHugeSize) {
        return size < maxLargeSize?
            largeCache.get(extMemPool, size) : hugeCache.get(extMemPool, size);
    }
    return NULL;
}

LargeMemoryBlock *ExtMemoryPool::mallocLargeObject(MemoryPool *pool, size_t allocationSize)
{
#if __TBB_MALLOC_LOCACHE_STAT
    AtomicIncrement(mallocCalls);
    AtomicAdd(memAllocKB, allocationSize/1024);
#endif
    LargeMemoryBlock* lmb = loc.get(allocationSize);
    if (!lmb) {
        BackRefIdx backRefIdx = BackRefIdx::newBackRef(/*largeObj=*/true);
        if (backRefIdx.isInvalid())
            return NULL;

        // unalignedSize is set in getLargeBlock
        lmb = backend.getLargeBlock(allocationSize);
        if (!lmb) {
            removeBackRef(backRefIdx);
            loc.updateCacheState(decrease, allocationSize);
            return NULL;
        }
        lmb->backRefIdx = backRefIdx;
        lmb->pool = pool;
        STAT_increment(getThreadId(), ThreadCommonCounters, allocNewLargeObj);
    } else {
#if __TBB_MALLOC_LOCACHE_STAT
        AtomicIncrement(cacheHits);
        AtomicAdd(memHitKB, allocationSize/1024);
#endif
    }
    return lmb;
}

void ExtMemoryPool::freeLargeObject(LargeMemoryBlock *mBlock)
{
    loc.put(mBlock);
}

void ExtMemoryPool::freeLargeObjectList(LargeMemoryBlock *head)
{
    loc.putList(head);
}

bool ExtMemoryPool::softCachesCleanup()
{
    return loc.regularCleanup();
}

bool ExtMemoryPool::hardCachesCleanup()
{
    // thread-local caches must be cleaned before LOC,
    // because object from thread-local cache can be released to LOC
    bool ret = releaseAllLocalCaches();
    ret |= orphanedBlocks.cleanup(&backend);
    ret |= loc.cleanAll();
    ret |= backend.clean();
    return ret;
}

#if BACKEND_HAS_MREMAP
void *ExtMemoryPool::remap(void *ptr, size_t oldSize, size_t newSize, size_t alignment)
{
    const size_t oldUnalignedSize = ((LargeObjectHdr*)ptr - 1)->memoryBlock->unalignedSize;
    void *o = backend.remap(ptr, oldSize, newSize, alignment);
    if (o) {
        LargeMemoryBlock *lmb = ((LargeObjectHdr*)o - 1)->memoryBlock;
        loc.registerRealloc(lmb->unalignedSize, oldUnalignedSize);
    }
    return o;
}
#endif /* BACKEND_HAS_MREMAP */

/*********** End allocation of large objects **********/

} // namespace internal
} // namespace rml

