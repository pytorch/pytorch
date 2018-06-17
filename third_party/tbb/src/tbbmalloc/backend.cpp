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

#include <string.h>   /* for memset */
#include <errno.h>
#include "tbbmalloc_internal.h"

namespace rml {
namespace internal {

/*********** Code to acquire memory from the OS or other executive ****************/

/*
  syscall/malloc can set non-zero errno in case of failure,
  but later allocator might be able to find memory to fulfill the request.
  And we do not want changing of errno by successful scalable_malloc call.
  To support this, restore old errno in (get|free)RawMemory, and set errno
  in frontend just before returning to user code.
  Please note: every syscall/libc call used inside scalable_malloc that
  sets errno must be protected this way, not just memory allocation per se.
*/

#if USE_DEFAULT_MEMORY_MAPPING
#include "MapMemory.h"
#else
/* assume MapMemory and UnmapMemory are customized */
#endif

void* getRawMemory (size_t size, bool hugePages) {
    return MapMemory(size, hugePages);
}

int freeRawMemory (void *object, size_t size) {
    return UnmapMemory(object, size);
}

void HugePagesStatus::registerAllocation(bool gotPage)
{
    if (gotPage) {
        if (!wasObserved)
            FencedStore(wasObserved, 1);
    } else
        FencedStore(enabled, 0);
    // reports huge page status only once
    if (needActualStatusPrint
        && AtomicCompareExchange(needActualStatusPrint, 0, 1))
        doPrintStatus(gotPage, "available");
}

void HugePagesStatus::registerReleasing(void* addr, size_t size)
{
    // We: 1) got huge page at least once,
    // 2) something that looks like a huge page is been released,
    // and 3) user requested huge pages,
    // so a huge page might be available at next allocation.
    // TODO: keep page status in regions and use exact check here
    if (FencedLoad(wasObserved) && size>=pageSize && isAligned(addr, pageSize))
        FencedStore(enabled, requestedMode.get());
}

void HugePagesStatus::printStatus() {
    doPrintStatus(requestedMode.get(), "requested");
    if (requestedMode.get()) { // report actual status iff requested
        if (pageSize)
            FencedStore(needActualStatusPrint, 1);
        else
            doPrintStatus(/*state=*/false, "available");
    }
}

void HugePagesStatus::doPrintStatus(bool state, const char *stateName)
{
    // Under macOS* fprintf/snprintf acquires an internal lock, so when
    // 1st allocation is done under the lock, we got a deadlock.
    // Do not use fprintf etc during initialization.
    fputs("TBBmalloc: huge pages\t", stderr);
    if (!state)
        fputs("not ", stderr);
    fputs(stateName, stderr);
    fputs("\n", stderr);
}

#if CHECK_ALLOCATION_RANGE

void Backend::UsedAddressRange::registerAlloc(uintptr_t left, uintptr_t right)
{
    MallocMutex::scoped_lock lock(mutex);
    if (left < leftBound)
        leftBound = left;
    if (right > rightBound)
        rightBound = right;
    MALLOC_ASSERT(leftBound, ASSERT_TEXT);
    MALLOC_ASSERT(leftBound < rightBound, ASSERT_TEXT);
    MALLOC_ASSERT(leftBound <= left && right <= rightBound, ASSERT_TEXT);
}

void Backend::UsedAddressRange::registerFree(uintptr_t left, uintptr_t right)
{
    MallocMutex::scoped_lock lock(mutex);
    if (leftBound == left) {
        if (rightBound == right) {
            leftBound = ADDRESS_UPPER_BOUND;
            rightBound = 0;
        } else
            leftBound = right;
    } else if (rightBound == right)
        rightBound = left;
    MALLOC_ASSERT((!rightBound && leftBound == ADDRESS_UPPER_BOUND)
                  || leftBound < rightBound, ASSERT_TEXT);
}
#endif // CHECK_ALLOCATION_RANGE

void *Backend::allocRawMem(size_t &size)
{
    void *res = NULL;
    size_t allocSize;

    if (extMemPool->userPool()) {
        if (extMemPool->fixedPool && bootsrapMemDone==FencedLoad(bootsrapMemStatus))
            return NULL;
        MALLOC_ASSERT(bootsrapMemStatus!=bootsrapMemNotDone,
                      "Backend::allocRawMem() called prematurely?");
        // TODO: support for raw mem not aligned at sizeof(uintptr_t)
        // memory from fixed pool is asked once and only once
        allocSize = alignUpGeneric(size, extMemPool->granularity);
        res = (*extMemPool->rawAlloc)(extMemPool->poolId, allocSize);
    } else {
        // check if alignment to huge page size is recommended
        size_t hugePageSize = hugePages.recommendedGranularity();
        allocSize = alignUpGeneric(size, hugePageSize? hugePageSize : extMemPool->granularity);
        // try to get them at 1st allocation and still use, if successful
        // if 1st try is unsuccessful, no more trying
        if (FencedLoad(hugePages.enabled)) {
            MALLOC_ASSERT(hugePageSize, "Inconsistent state of HugePagesStatus");
            res = getRawMemory(allocSize, /*hugePages=*/true);
            hugePages.registerAllocation(res);
        }

        if (!res)
            res = getRawMemory(allocSize, /*hugePages=*/false);
    }

    if (res) {
        MALLOC_ASSERT(allocSize > 0, "Invalid size of an allocated region.");
        size = allocSize;
        if (!extMemPool->userPool())
            usedAddrRange.registerAlloc((uintptr_t)res, (uintptr_t)res+size);
#if MALLOC_DEBUG
        volatile size_t curTotalSize = totalMemSize; // to read global value once
        MALLOC_ASSERT(curTotalSize+size > curTotalSize, "Overflow allocation size.");
#endif
        AtomicAdd((intptr_t&)totalMemSize, size);
    }

    return res;
}

bool Backend::freeRawMem(void *object, size_t size)
{
    bool fail;
#if MALLOC_DEBUG
    volatile size_t curTotalSize = totalMemSize; // to read global value once
    MALLOC_ASSERT(curTotalSize-size < curTotalSize, "Negative allocation size.");
#endif
    AtomicAdd((intptr_t&)totalMemSize, -size);
    if (extMemPool->userPool()) {
        MALLOC_ASSERT(!extMemPool->fixedPool, "No free for fixed-size pools.");
        fail = (*extMemPool->rawFree)(extMemPool->poolId, object, size);
    } else {
        usedAddrRange.registerFree((uintptr_t)object, (uintptr_t)object + size);
        hugePages.registerReleasing(object, size);
        fail = freeRawMemory(object, size);
    }
    // TODO: use result in all freeRawMem() callers
    return !fail;
}

/********* End memory acquisition code ********************************/

// Protected object size. After successful locking returns size of locked block,
// and releasing requires setting block size.
class GuardedSize : tbb::internal::no_copy {
    uintptr_t value;
public:
    enum State {
        LOCKED,
        COAL_BLOCK,        // block is coalescing now
        MAX_LOCKED_VAL = COAL_BLOCK,
        LAST_REGION_BLOCK, // used to mark last block in region
        // values after this are "normal" block sizes
        MAX_SPEC_VAL = LAST_REGION_BLOCK
    };

    void initLocked() { value = LOCKED; }
    void makeCoalscing() {
        MALLOC_ASSERT(value == LOCKED, ASSERT_TEXT);
        value = COAL_BLOCK;
    }
    size_t tryLock(State state) {
        size_t szVal, sz;
        MALLOC_ASSERT(state <= MAX_LOCKED_VAL, ASSERT_TEXT);
        for (;;) {
            sz = FencedLoad((intptr_t&)value);
            if (sz <= MAX_LOCKED_VAL)
                break;
            szVal = AtomicCompareExchange((intptr_t&)value, state, sz);

            if (szVal==sz)
                break;
        }
        return sz;
    }
    void unlock(size_t size) {
        MALLOC_ASSERT(value <= MAX_LOCKED_VAL, "The lock is not locked");
        MALLOC_ASSERT(size > MAX_LOCKED_VAL, ASSERT_TEXT);
        FencedStore((intptr_t&)value, size);
    }
    bool isLastRegionBlock() const { return value==LAST_REGION_BLOCK; }
    friend void Backend::IndexedBins::verify();
};

struct MemRegion {
    MemRegion *next,      // keep all regions in any pool to release all them on
              *prev;      // pool destroying, 2-linked list to release individual
                          // regions.
    size_t     allocSz,   // got from pool callback
               blockSz;   // initial and maximal inner block size
    MemRegionType type;
};

// this data must be unmodified while block is in use, so separate it
class BlockMutexes {
protected:
    GuardedSize myL,   // lock for me
                leftL; // lock for left neighbor
};

class FreeBlock : BlockMutexes {
public:
    static const size_t minBlockSize;
    friend void Backend::IndexedBins::verify();

    FreeBlock    *prev,       // in 2-linked list related to bin
                 *next,
                 *nextToFree; // used to form a queue during coalescing
    // valid only when block is in processing, i.e. one is not free and not
    size_t        sizeTmp;    // used outside of backend
    int           myBin;      // bin that is owner of the block
    bool          aligned;
    bool          blockInBin; // this block in myBin already

    FreeBlock *rightNeig(size_t sz) const {
        MALLOC_ASSERT(sz, ASSERT_TEXT);
        return (FreeBlock*)((uintptr_t)this+sz);
    }
    FreeBlock *leftNeig(size_t sz) const {
        MALLOC_ASSERT(sz, ASSERT_TEXT);
        return (FreeBlock*)((uintptr_t)this - sz);
    }

    void initHeader() { myL.initLocked(); leftL.initLocked(); }
    void setMeFree(size_t size) { myL.unlock(size); }
    size_t trySetMeUsed(GuardedSize::State s) { return myL.tryLock(s); }
    bool isLastRegionBlock() const { return myL.isLastRegionBlock(); }

    void setLeftFree(size_t sz) { leftL.unlock(sz); }
    size_t trySetLeftUsed(GuardedSize::State s) { return leftL.tryLock(s); }

    size_t tryLockBlock() {
        size_t rSz, sz = trySetMeUsed(GuardedSize::LOCKED);

        if (sz <= GuardedSize::MAX_LOCKED_VAL)
            return false;
        rSz = rightNeig(sz)->trySetLeftUsed(GuardedSize::LOCKED);
        if (rSz <= GuardedSize::MAX_LOCKED_VAL) {
            setMeFree(sz);
            return false;
        }
        MALLOC_ASSERT(rSz == sz, ASSERT_TEXT);
        return sz;
    }
    void markCoalescing(size_t blockSz) {
        myL.makeCoalscing();
        rightNeig(blockSz)->leftL.makeCoalscing();
        sizeTmp = blockSz;
        nextToFree = NULL;
    }
    void markUsed() {
        myL.initLocked();
        rightNeig(sizeTmp)->leftL.initLocked();
        nextToFree = NULL;
    }
    static void markBlocks(FreeBlock *fBlock, int num, size_t size) {
        for (int i=1; i<num; i++) {
            fBlock = (FreeBlock*)((uintptr_t)fBlock + size);
            fBlock->initHeader();
        }
    }
};

// Last block in any region. Its "size" field is GuardedSize::LAST_REGION_BLOCK,
// This kind of blocks used to find region header
// and have a possibility to return region back to OS
struct LastFreeBlock : public FreeBlock {
    MemRegion *memRegion;
};

const size_t FreeBlock::minBlockSize = sizeof(FreeBlock);

inline bool BackendSync::waitTillBlockReleased(intptr_t startModifiedCnt)
{
    AtomicBackoff backoff;
#if __TBB_MALLOC_BACKEND_STAT
    class ITT_Guard {
        void *ptr;
    public:
        ITT_Guard(void *p) : ptr(p) {
            MALLOC_ITT_SYNC_PREPARE(ptr);
        }
        ~ITT_Guard() {
            MALLOC_ITT_SYNC_ACQUIRED(ptr);
        }
    };
    ITT_Guard ittGuard(&inFlyBlocks);
#endif
    for (intptr_t myBinsInFlyBlocks = FencedLoad(inFlyBlocks),
             myCoalescQInFlyBlocks = backend->blocksInCoalescing(); ;
         backoff.pause()) {
        MALLOC_ASSERT(myBinsInFlyBlocks>=0 && myCoalescQInFlyBlocks>=0, NULL);
        intptr_t currBinsInFlyBlocks = FencedLoad(inFlyBlocks),
            currCoalescQInFlyBlocks = backend->blocksInCoalescing();
        WhiteboxTestingYield();
        // Stop waiting iff:

        // 1) blocks were removed from processing, not added
        if (myBinsInFlyBlocks > currBinsInFlyBlocks
        // 2) released during delayed coalescing queue
            || myCoalescQInFlyBlocks > currCoalescQInFlyBlocks)
            break;
        // 3) if there are blocks in coalescing, and no progress in its processing,
        // try to scan coalescing queue and stop waiting, if changes were made
        // (if there are no changes and in-fly blocks exist, we continue
        //  waiting to not increase load on coalescQ)
        if (currCoalescQInFlyBlocks > 0 && backend->scanCoalescQ(/*forceCoalescQDrop=*/false))
            break;
        // 4) when there are no blocks
        if (!currBinsInFlyBlocks && !currCoalescQInFlyBlocks)
            // re-scan make sense only if bins were modified since scanned
            return startModifiedCnt != getNumOfMods();
        myBinsInFlyBlocks = currBinsInFlyBlocks;
        myCoalescQInFlyBlocks = currCoalescQInFlyBlocks;
    }
    return true;
}

void CoalRequestQ::putBlock(FreeBlock *fBlock)
{
    MALLOC_ASSERT(fBlock->sizeTmp >= FreeBlock::minBlockSize, ASSERT_TEXT);
    fBlock->markUsed();
    // the block is in the queue, do not forget that it's here
    AtomicIncrement(inFlyBlocks);

    for (;;) {
        FreeBlock *myBlToFree = (FreeBlock*)FencedLoad((intptr_t&)blocksToFree);

        fBlock->nextToFree = myBlToFree;
        if (myBlToFree ==
            (FreeBlock*)AtomicCompareExchange((intptr_t&)blocksToFree,
                                              (intptr_t)fBlock,
                                              (intptr_t)myBlToFree))
            return;
    }
}

FreeBlock *CoalRequestQ::getAll()
{
    for (;;) {
        FreeBlock *myBlToFree = (FreeBlock*)FencedLoad((intptr_t&)blocksToFree);

        if (!myBlToFree)
            return NULL;
        else {
            if (myBlToFree ==
                (FreeBlock*)AtomicCompareExchange((intptr_t&)blocksToFree,
                                                  0, (intptr_t)myBlToFree))
                return myBlToFree;
            else
                continue;
        }
    }
}

inline void CoalRequestQ::blockWasProcessed()
{
    bkndSync->binsModified();
    int prev = AtomicAdd(inFlyBlocks, -1);
    MALLOC_ASSERT(prev > 0, ASSERT_TEXT);
}

// Try to get a block from a bin.
// If the remaining free space would stay in the same bin,
//     split the block without removing it.
// If the free space should go to other bin(s), remove the block.
// alignedBin is true, if all blocks in the bin have slab-aligned right side.
FreeBlock *Backend::IndexedBins::getFromBin(int binIdx, BackendSync *sync,
                size_t size, bool needAlignedRes, bool alignedBin, bool wait,
                int *binLocked)
{
    Bin *b = &freeBins[binIdx];
try_next:
    FreeBlock *fBlock = NULL;
    if (b->head) {
        bool locked;
        MallocMutex::scoped_lock scopedLock(b->tLock, wait, &locked);

        if (!locked) {
            if (binLocked) (*binLocked)++;
            return NULL;
        }

        for (FreeBlock *curr = b->head; curr; curr = curr->next) {
            size_t szBlock = curr->tryLockBlock();
            if (!szBlock) {
                // block is locked, re-do bin lock, as there is no place to spin
                // while block coalescing
                goto try_next;
            }

            if (alignedBin || !needAlignedRes) {
                size_t splitSz = szBlock - size;
                // If we got a block as split result,
                // it must have a room for control structures.
                if (szBlock >= size && (splitSz >= FreeBlock::minBlockSize ||
                                        !splitSz))
                    fBlock = curr;
            } else {
                void *newB = alignUp(curr, slabSize);
                uintptr_t rightNew = (uintptr_t)newB + size;
                uintptr_t rightCurr = (uintptr_t)curr + szBlock;
                // appropriate size, and left and right split results
                // are either big enough or non-existent
                if (rightNew <= rightCurr
                    && (newB==curr ||
                        (uintptr_t)newB-(uintptr_t)curr >= FreeBlock::minBlockSize)
                    && (rightNew==rightCurr ||
                        rightCurr - rightNew >= FreeBlock::minBlockSize))
                    fBlock = curr;
            }
            if (fBlock) {
                // consume must be called before result of removing from a bin
                // is visible externally.
                sync->blockConsumed();
                if (alignedBin && needAlignedRes &&
                    Backend::sizeToBin(szBlock-size) == Backend::sizeToBin(szBlock)) {
                    // free remainder of fBlock stay in same bin,
                    // so no need to remove it from the bin
                    // TODO: add more "still here" cases
                    FreeBlock *newFBlock = fBlock;
                    // return block from right side of fBlock
                    fBlock = (FreeBlock*)((uintptr_t)newFBlock + szBlock - size);
                    MALLOC_ASSERT(isAligned(fBlock, slabSize), "Invalid free block");
                    fBlock->initHeader();
                    fBlock->setLeftFree(szBlock - size);
                    newFBlock->setMeFree(szBlock - size);

                    fBlock->sizeTmp = size;
                } else {
                    b->removeBlock(fBlock);
                    if (freeBins[binIdx].empty())
                        bitMask.set(binIdx, false);
                    fBlock->sizeTmp = szBlock;
                }
                break;
            } else { // block size is not valid, search for next block in the bin
                curr->setMeFree(szBlock);
                curr->rightNeig(szBlock)->setLeftFree(szBlock);
            }
        }
    }
    return fBlock;
}

bool Backend::IndexedBins::tryReleaseRegions(int binIdx, Backend *backend)
{
    Bin *b = &freeBins[binIdx];
    FreeBlock *fBlockList = NULL;

    // got all blocks from the bin and re-do coalesce on them
    // to release single-block regions
try_next:
    if (b->head) {
        MallocMutex::scoped_lock binLock(b->tLock);
        for (FreeBlock *curr = b->head; curr; ) {
            size_t szBlock = curr->tryLockBlock();
            if (!szBlock)
                goto try_next;

            FreeBlock *next = curr->next;

            b->removeBlock(curr);
            curr->sizeTmp = szBlock;
            curr->nextToFree = fBlockList;
            fBlockList = curr;
            curr = next;
        }
    }
    return backend->coalescAndPutList(fBlockList, /*forceCoalescQDrop=*/true,
                                      /*reportBlocksProcessed=*/false);
}

void Backend::Bin::removeBlock(FreeBlock *fBlock)
{
    MALLOC_ASSERT(fBlock->next||fBlock->prev||fBlock==head,
                  "Detected that a block is not in the bin.");
    if (head == fBlock)
        head = fBlock->next;
    if (tail == fBlock)
        tail = fBlock->prev;
    if (fBlock->prev)
        fBlock->prev->next = fBlock->next;
    if (fBlock->next)
        fBlock->next->prev = fBlock->prev;
}

void Backend::IndexedBins::addBlock(int binIdx, FreeBlock *fBlock, size_t blockSz, bool addToTail)
{
    Bin *b = &freeBins[binIdx];

    fBlock->myBin = binIdx;
    fBlock->aligned = toAlignedBin(fBlock, blockSz);
    fBlock->next = fBlock->prev = NULL;
    {
        MallocMutex::scoped_lock scopedLock(b->tLock);
        if (addToTail) {
            fBlock->prev = b->tail;
            b->tail = fBlock;
            if (fBlock->prev)
                fBlock->prev->next = fBlock;
            if (!b->head)
                b->head = fBlock;
        } else {
            fBlock->next = b->head;
            b->head = fBlock;
            if (fBlock->next)
                fBlock->next->prev = fBlock;
            if (!b->tail)
                b->tail = fBlock;
        }
    }
    bitMask.set(binIdx, true);
}

bool Backend::IndexedBins::tryAddBlock(int binIdx, FreeBlock *fBlock, bool addToTail)
{
    bool locked;
    Bin *b = &freeBins[binIdx];

    fBlock->myBin = binIdx;
    fBlock->aligned = toAlignedBin(fBlock, fBlock->sizeTmp);
    if (addToTail) {
        fBlock->next = NULL;
        {
            MallocMutex::scoped_lock scopedLock(b->tLock, /*wait=*/false, &locked);
            if (!locked)
                return false;
            fBlock->prev = b->tail;
            b->tail = fBlock;
            if (fBlock->prev)
                fBlock->prev->next = fBlock;
            if (!b->head)
                b->head = fBlock;
        }
    } else {
        fBlock->prev = NULL;
        {
            MallocMutex::scoped_lock scopedLock(b->tLock, /*wait=*/false, &locked);
            if (!locked)
                return false;
            fBlock->next = b->head;
            b->head = fBlock;
            if (fBlock->next)
                fBlock->next->prev = fBlock;
            if (!b->tail)
                b->tail = fBlock;
        }
    }
    bitMask.set(binIdx, true);
    return true;
}

void Backend::IndexedBins::reset()
{
    for (int i=0; i<Backend::freeBinsNum; i++)
        freeBins[i].reset();
    bitMask.reset();
}

void Backend::IndexedBins::lockRemoveBlock(int binIdx, FreeBlock *fBlock)
{
    MallocMutex::scoped_lock scopedLock(freeBins[binIdx].tLock);
    freeBins[binIdx].removeBlock(fBlock);
    if (freeBins[binIdx].empty())
        bitMask.set(binIdx, false);
}

bool ExtMemoryPool::regionsAreReleaseable() const
{
    return !keepAllMemory && !delayRegsReleasing;
}

FreeBlock *Backend::splitUnalignedBlock(FreeBlock *fBlock, int num, size_t size,
                                        bool needAlignedBlock)
{
    const size_t totalSize = num*size;
    if (needAlignedBlock) {
        size_t fBlockSz = fBlock->sizeTmp;
        uintptr_t fBlockEnd = (uintptr_t)fBlock + fBlockSz;
        FreeBlock *newB = alignUp(fBlock, slabSize);
        FreeBlock *rightPart = (FreeBlock*)((uintptr_t)newB + totalSize);

        // Space to use is in the middle,
        // ... return free right part
        if ((uintptr_t)rightPart != fBlockEnd) {
            rightPart->initHeader();  // to prevent coalescing rightPart with fBlock
            coalescAndPut(rightPart, fBlockEnd - (uintptr_t)rightPart);
        }
        // ... and free left part
        if (newB != fBlock) {
            newB->initHeader(); // to prevent coalescing fBlock with newB
            coalescAndPut(fBlock, (uintptr_t)newB - (uintptr_t)fBlock);
        }

        fBlock = newB;
        MALLOC_ASSERT(isAligned(fBlock, slabSize), ASSERT_TEXT);
    } else {
        if (size_t splitSz = fBlock->sizeTmp - totalSize) {
            // split block and return free right part
            FreeBlock *splitB = (FreeBlock*)((uintptr_t)fBlock + totalSize);
            splitB->initHeader();
            coalescAndPut(splitB, splitSz);
        }
    }
    FreeBlock::markBlocks(fBlock, num, size);
    return fBlock;
}

FreeBlock *Backend::splitAlignedBlock(FreeBlock *fBlock, int num, size_t size,
                                      bool needAlignedBlock)
{
    if (fBlock->sizeTmp != num*size) { // i.e., need to split the block
        FreeBlock *newAlgnd;
        size_t newSz;

        if (needAlignedBlock) {
            newAlgnd = fBlock;
            fBlock = (FreeBlock*)((uintptr_t)newAlgnd + newAlgnd->sizeTmp
                                  - num*size);
            MALLOC_ASSERT(isAligned(fBlock, slabSize), "Invalid free block");
            fBlock->initHeader();
            newSz = newAlgnd->sizeTmp - num*size;
        } else {
            newAlgnd = (FreeBlock*)((uintptr_t)fBlock + num*size);
            newSz = fBlock->sizeTmp - num*size;
            newAlgnd->initHeader();
        }
        coalescAndPut(newAlgnd, newSz);
    }
    MALLOC_ASSERT(!needAlignedBlock || isAligned(fBlock, slabSize),
                  "Expect to get aligned block, if one was requested.");
    FreeBlock::markBlocks(fBlock, num, size);
    return fBlock;
}

inline size_t Backend::getMaxBinnedSize() const
{
    return hugePages.wasObserved && !inUserPool()?
        maxBinned_HugePage : maxBinned_SmallPage;
}

inline bool Backend::MaxRequestComparator::operator()(size_t oldMaxReq,
                                                      size_t requestSize) const
{
    return requestSize > oldMaxReq && requestSize < backend->getMaxBinnedSize();
}

// last chance to get memory
FreeBlock *Backend::releaseMemInCaches(intptr_t startModifiedCnt,
                                    int *lockedBinsThreshold, int numOfLockedBins)
{
    // something released from caches
    if (extMemPool->hardCachesCleanup()
        // ..or can use blocks that are in processing now
        || bkndSync.waitTillBlockReleased(startModifiedCnt))
        return (FreeBlock*)VALID_BLOCK_IN_BIN;
    // OS can't give us more memory, but we have some in locked bins
    if (*lockedBinsThreshold && numOfLockedBins) {
        *lockedBinsThreshold = 0;
        return (FreeBlock*)VALID_BLOCK_IN_BIN;
    }
    return NULL; // nothing found, give up
}

FreeBlock *Backend::askMemFromOS(size_t blockSize, intptr_t startModifiedCnt,
                                 int *lockedBinsThreshold, int numOfLockedBins,
                                 bool *splittableRet)
{
    FreeBlock *block;
    // The block sizes can be divided into 3 groups:
    //   1. "quite small": popular object size, we are in bootstarp or something
    //      like; request several regions.
    //   2. "quite large": we want to have several such blocks in the region
    //      but not want several pre-allocated regions.
    //   3. "huge": exact fit, we allocate only one block and do not allow
    //       any other allocations to placed in a region.
    // Dividing the block sizes in these groups we are trying to balance between
    // too small regions (that leads to fragmentation) and too large ones (that
    // leads to excessive address space consumption). If a region is "too
    // large", allocate only one, to prevent fragmentation. It supposedly
    // doesn't hurt performance, because the object requested by user is large.
    // Bounds for the groups are:
    const size_t maxBinned = getMaxBinnedSize();
    const size_t quiteSmall = maxBinned / 8;
    const size_t quiteLarge = maxBinned;

    if (blockSize >= quiteLarge) {
        // Do not interact with other threads via semaphores, as for exact fit
        // we can't share regions with them, memory requesting is individual.
        block = addNewRegion(blockSize, MEMREG_ONE_BLOCK, /*addToBin=*/false);
        if (!block)
            return releaseMemInCaches(startModifiedCnt, lockedBinsThreshold, numOfLockedBins);
        *splittableRet = false;
    } else {
        const size_t regSz_sizeBased = alignUp(4*maxRequestedSize, 1024*1024);
        // Another thread is modifying backend while we can't get the block.
        // Wait while it leaves and re-do the scan
        // before trying other ways to extend the backend.
        if (bkndSync.waitTillBlockReleased(startModifiedCnt)
            // semaphore is protecting adding more more memory from OS
            || memExtendingSema.wait())
            return (FreeBlock*)VALID_BLOCK_IN_BIN;

        if (startModifiedCnt != bkndSync.getNumOfMods()) {
            memExtendingSema.signal();
            return (FreeBlock*)VALID_BLOCK_IN_BIN;
        }

        if (blockSize < quiteSmall) {
            // For this size of blocks, add NUM_OF_REG "advance" regions in bin,
            // and return one as a result.
            // TODO: add to bin first, because other threads can use them right away.
            // This must be done carefully, because blocks in bins can be released
            // in releaseCachesToLimit().
            const unsigned NUM_OF_REG = 3;
            block = addNewRegion(regSz_sizeBased, MEMREG_FLEXIBLE_SIZE, /*addToBin=*/false);
            if (block)
                for (unsigned idx=0; idx<NUM_OF_REG; idx++)
                    if (! addNewRegion(regSz_sizeBased, MEMREG_FLEXIBLE_SIZE, /*addToBin=*/true))
                        break;
        } else {
            block = addNewRegion(regSz_sizeBased, MEMREG_SEVERAL_BLOCKS, /*addToBin=*/false);
        }
        memExtendingSema.signal();

        // no regions found, try to clean cache
        if (!block || block == (FreeBlock*)VALID_BLOCK_IN_BIN)
            return releaseMemInCaches(startModifiedCnt, lockedBinsThreshold, numOfLockedBins);
        // Since a region can hold more than one block it can be splitted.
        *splittableRet = true;
    }
    // after asking memory from OS, release caches if we above the memory limits
    releaseCachesToLimit();

    return block;
}

void Backend::releaseCachesToLimit()
{
    if (!memSoftLimit || totalMemSize <= memSoftLimit)
        return;
    size_t locTotalMemSize, locMemSoftLimit;

    scanCoalescQ(/*forceCoalescQDrop=*/false);
    if (extMemPool->softCachesCleanup() &&
        (locTotalMemSize = FencedLoad((intptr_t&)totalMemSize)) <=
        (locMemSoftLimit = FencedLoad((intptr_t&)memSoftLimit)))
        return;
    // clean global large-object cache, if this is not enough, clean local caches
    // do this in several tries, because backend fragmentation can prevent
    // region from releasing
    for (int cleanLocal = 0; cleanLocal<2; cleanLocal++)
        while (cleanLocal?
               extMemPool->allLocalCaches.cleanup(extMemPool, /*cleanOnlyUnused=*/true)
               : extMemPool->loc.decreasingCleanup())
            if ((locTotalMemSize = FencedLoad((intptr_t&)totalMemSize)) <=
                (locMemSoftLimit = FencedLoad((intptr_t&)memSoftLimit)))
                return;
    // last chance to match memSoftLimit
    extMemPool->hardCachesCleanup();
}

FreeBlock *Backend::IndexedBins::
    findBlock(int nativeBin, BackendSync *sync, size_t size,
              bool resSlabAligned, bool alignedBin, int *numOfLockedBins)
{
    for (int i=getMinNonemptyBin(nativeBin); i<freeBinsNum; i=getMinNonemptyBin(i+1))
        if (FreeBlock *block = getFromBin(i, sync, size, resSlabAligned, alignedBin,
                                          /*wait=*/false, numOfLockedBins))
            return block;

    return NULL;
}

void Backend::requestBootstrapMem()
{
    if (bootsrapMemDone == FencedLoad(bootsrapMemStatus))
        return;
    MallocMutex::scoped_lock lock( bootsrapMemStatusMutex );
    if (bootsrapMemDone == bootsrapMemStatus)
        return;
    MALLOC_ASSERT(bootsrapMemNotDone == bootsrapMemStatus, ASSERT_TEXT);
    bootsrapMemStatus = bootsrapMemInitializing;
    // request some rather big region during bootstrap in advance
    // ok to get NULL here, as later we re-do a request with more modest size
    addNewRegion(2*1024*1024, MEMREG_FLEXIBLE_SIZE, /*addToBin=*/true);
    bootsrapMemStatus = bootsrapMemDone;
}

// try to allocate size Byte block in available bins
// needAlignedRes is true if result must be slab-aligned
FreeBlock *Backend::genericGetBlock(int num, size_t size, bool needAlignedBlock)
{
    FreeBlock *block = NULL;
    const size_t totalReqSize = num*size;
    // no splitting after requesting new region, asks exact size
    const int nativeBin = sizeToBin(totalReqSize);

    requestBootstrapMem();
    // If we found 2 or less locked bins, it's time to ask more memory from OS.
    // But nothing can be asked from fixed pool. And we prefer wait, not ask
    // for more memory, if block is quite large.
    int lockedBinsThreshold = extMemPool->fixedPool || size>=maxBinned_SmallPage? 0 : 2;

    // Find maximal requested size limited by getMaxBinnedSize()
    AtomicUpdate(maxRequestedSize, totalReqSize, MaxRequestComparator(this));
    scanCoalescQ(/*forceCoalescQDrop=*/false);

    bool splittable = true;
    for (;;) {
        const intptr_t startModifiedCnt = bkndSync.getNumOfMods();
        int numOfLockedBins;

        do {
            numOfLockedBins = 0;

            // TODO: try different bin search order
            if (needAlignedBlock) {
                block = freeAlignedBins.findBlock(nativeBin, &bkndSync, num*size,
                                    /*needAlignedBlock=*/true, /*alignedBin=*/true,
                                    &numOfLockedBins);
                if (!block)
                    block = freeLargeBins.findBlock(nativeBin, &bkndSync, num*size,
                                    /*needAlignedBlock=*/true, /*alignedBin=*/false,
                                    &numOfLockedBins);
            } else {
                block = freeLargeBins.findBlock(nativeBin, &bkndSync, num*size,
                                    /*needAlignedBlock=*/false, /*alignedBin=*/false,
                                    &numOfLockedBins);
                if (!block)
                    block = freeAlignedBins.findBlock(nativeBin, &bkndSync, num*size,
                                    /*needAlignedBlock=*/false, /*alignedBin=*/true,
                                    &numOfLockedBins);
            }
        } while (!block && numOfLockedBins>lockedBinsThreshold);

        if (block)
            break;

        if (!(scanCoalescQ(/*forceCoalescQDrop=*/true)
              | extMemPool->softCachesCleanup())) {
            // bins are not updated,
            // only remaining possibility is to ask for more memory
            block =
                askMemFromOS(totalReqSize, startModifiedCnt, &lockedBinsThreshold,
                             numOfLockedBins, &splittable);
            if (!block)
                return NULL;
            if (block != (FreeBlock*)VALID_BLOCK_IN_BIN) {
                // size can be increased in askMemFromOS, that's why >=
                MALLOC_ASSERT(block->sizeTmp >= size, ASSERT_TEXT);
                break;
            }
            // valid block somewhere in bins, let's find it
            block = NULL;
        }
    }
    MALLOC_ASSERT(block, ASSERT_TEXT);
    if (splittable)
        block = toAlignedBin(block, block->sizeTmp)?
            splitAlignedBlock(block, num, size, needAlignedBlock) :
            splitUnalignedBlock(block, num, size, needAlignedBlock);
    // matched blockConsumed() from startUseBlock()
    bkndSync.blockReleased();

    return block;
}

LargeMemoryBlock *Backend::getLargeBlock(size_t size)
{
    LargeMemoryBlock *lmb =
        (LargeMemoryBlock*)genericGetBlock(1, size, /*needAlignedRes=*/false);
    if (lmb) {
        lmb->unalignedSize = size;
        if (extMemPool->userPool())
            extMemPool->lmbList.add(lmb);
    }
    return lmb;
}

void *Backend::getBackRefSpace(size_t size, bool *rawMemUsed)
{
    // This block is released only at shutdown, so it can prevent
    // a entire region releasing when it's received from the backend,
    // so prefer getRawMemory using.
    if (void *ret = getRawMemory(size, /*hugePages=*/false)) {
        *rawMemUsed = true;
        return ret;
    }
    void *ret = genericGetBlock(1, size, /*needAlignedRes=*/false);
    if (ret) *rawMemUsed = false;
    return ret;
}

void Backend::putBackRefSpace(void *b, size_t size, bool rawMemUsed)
{
    if (rawMemUsed)
        freeRawMemory(b, size);
    // ignore not raw mem, as it released on region releasing
}

void Backend::removeBlockFromBin(FreeBlock *fBlock)
{
    if (fBlock->myBin != Backend::NO_BIN) {
        if (fBlock->aligned)
            freeAlignedBins.lockRemoveBlock(fBlock->myBin, fBlock);
        else
            freeLargeBins.lockRemoveBlock(fBlock->myBin, fBlock);
    }
}

void Backend::genericPutBlock(FreeBlock *fBlock, size_t blockSz)
{
    bkndSync.blockConsumed();
    coalescAndPut(fBlock, blockSz);
    bkndSync.blockReleased();
}

void AllLargeBlocksList::add(LargeMemoryBlock *lmb)
{
    MallocMutex::scoped_lock scoped_cs(largeObjLock);
    lmb->gPrev = NULL;
    lmb->gNext = loHead;
    if (lmb->gNext)
        lmb->gNext->gPrev = lmb;
    loHead = lmb;
}

void AllLargeBlocksList::remove(LargeMemoryBlock *lmb)
{
    MallocMutex::scoped_lock scoped_cs(largeObjLock);
    if (loHead == lmb)
        loHead = lmb->gNext;
    if (lmb->gNext)
        lmb->gNext->gPrev = lmb->gPrev;
    if (lmb->gPrev)
        lmb->gPrev->gNext = lmb->gNext;
}

void Backend::putLargeBlock(LargeMemoryBlock *lmb)
{
    if (extMemPool->userPool())
        extMemPool->lmbList.remove(lmb);
    genericPutBlock((FreeBlock *)lmb, lmb->unalignedSize);
}

void Backend::returnLargeObject(LargeMemoryBlock *lmb)
{
    removeBackRef(lmb->backRefIdx);
    putLargeBlock(lmb);
    STAT_increment(getThreadId(), ThreadCommonCounters, freeLargeObj);
}

#if BACKEND_HAS_MREMAP
void *Backend::remap(void *ptr, size_t oldSize, size_t newSize, size_t alignment)
{
    // no remap for user pools and for object too small that living in bins
    if (inUserPool() || min(oldSize, newSize)<maxBinned_SmallPage
        // during remap, can't guarantee alignment more strict than current or
        // more strict than page alignment
        || !isAligned(ptr, alignment) || alignment>extMemPool->granularity)
        return NULL;
    const LargeMemoryBlock* lmbOld = ((LargeObjectHdr *)ptr - 1)->memoryBlock;
    const size_t oldUnalignedSize = lmbOld->unalignedSize;
    FreeBlock *oldFBlock = (FreeBlock *)lmbOld;
    FreeBlock *right = oldFBlock->rightNeig(oldUnalignedSize);
    // in every region only one block can have LAST_REGION_BLOCK on right,
    // so don't need no synchronization
    if (!right->isLastRegionBlock())
        return NULL;

    MemRegion *oldRegion = static_cast<LastFreeBlock*>(right)->memRegion;
    MALLOC_ASSERT( oldRegion < ptr, ASSERT_TEXT );
    const size_t oldRegionSize = oldRegion->allocSz;
    if (oldRegion->type != MEMREG_ONE_BLOCK)
        return NULL;  // we are not single in the region
    const size_t userOffset = (uintptr_t)ptr - (uintptr_t)oldRegion;
    const size_t requestSize =
        alignUp(userOffset + newSize + sizeof(LastFreeBlock), extMemPool->granularity);
    if (requestSize < newSize) // is wrapped around?
        return NULL;
    regionList.remove(oldRegion);

    void *ret = mremap(oldRegion, oldRegion->allocSz, requestSize, MREMAP_MAYMOVE);
    if (MAP_FAILED == ret) { // can't remap, revert and leave
        regionList.add(oldRegion);
        return NULL;
    }
    MemRegion *region = (MemRegion*)ret;
    MALLOC_ASSERT(region->type == MEMREG_ONE_BLOCK, ASSERT_TEXT);
    region->allocSz = requestSize;

    FreeBlock *fBlock = (FreeBlock *)alignUp((uintptr_t)region + sizeof(MemRegion),
                                             largeObjectAlignment);
    // put LastFreeBlock at the very end of region
    const uintptr_t fBlockEnd = (uintptr_t)region + requestSize - sizeof(LastFreeBlock);
    region->blockSz = fBlockEnd - (uintptr_t)fBlock;

    regionList.add(region);
    startUseBlock(region, fBlock, /*addToBin=*/false);
    MALLOC_ASSERT(fBlock->sizeTmp == region->blockSz, ASSERT_TEXT);
    // matched blockConsumed() in startUseBlock().
    // TODO: get rid of useless pair blockConsumed()/blockReleased()
    bkndSync.blockReleased();

    // object must start at same offest from region's start
    void *object = (void*)((uintptr_t)region + userOffset);
    MALLOC_ASSERT(isAligned(object, alignment), ASSERT_TEXT);
    LargeObjectHdr *header = (LargeObjectHdr*)object - 1;
    setBackRef(header->backRefIdx, header);

    LargeMemoryBlock *lmb = (LargeMemoryBlock*)fBlock;
    lmb->unalignedSize = region->blockSz;
    lmb->objectSize = newSize;
    lmb->backRefIdx = header->backRefIdx;
    header->memoryBlock = lmb;
    MALLOC_ASSERT((uintptr_t)lmb + lmb->unalignedSize >=
                  (uintptr_t)object + lmb->objectSize, "An object must fit to the block.");

    usedAddrRange.registerFree((uintptr_t)oldRegion, (uintptr_t)oldRegion + oldRegionSize);
    usedAddrRange.registerAlloc((uintptr_t)region, (uintptr_t)region + requestSize);
    AtomicAdd((intptr_t&)totalMemSize, region->allocSz - oldRegionSize);

    return object;
}
#endif /* BACKEND_HAS_MREMAP */

void Backend::releaseRegion(MemRegion *memRegion)
{
    regionList.remove(memRegion);
    freeRawMem(memRegion, memRegion->allocSz);
}

// coalesce fBlock with its neighborhood
FreeBlock *Backend::doCoalesc(FreeBlock *fBlock, MemRegion **mRegion)
{
    FreeBlock *resBlock = fBlock;
    size_t resSize = fBlock->sizeTmp;
    MemRegion *memRegion = NULL;

    fBlock->markCoalescing(resSize);
    resBlock->blockInBin = false;

    // coalescing with left neighbor
    size_t leftSz = fBlock->trySetLeftUsed(GuardedSize::COAL_BLOCK);
    if (leftSz != GuardedSize::LOCKED) {
        if (leftSz == GuardedSize::COAL_BLOCK) {
            coalescQ.putBlock(fBlock);
            return NULL;
        } else {
            FreeBlock *left = fBlock->leftNeig(leftSz);
            size_t lSz = left->trySetMeUsed(GuardedSize::COAL_BLOCK);
            if (lSz <= GuardedSize::MAX_LOCKED_VAL) {
                fBlock->setLeftFree(leftSz); // rollback
                coalescQ.putBlock(fBlock);
                return NULL;
            } else {
                MALLOC_ASSERT(lSz == leftSz, "Invalid header");
                left->blockInBin = true;
                resBlock = left;
                resSize += leftSz;
                resBlock->sizeTmp = resSize;
            }
        }
    }
    // coalescing with right neighbor
    FreeBlock *right = fBlock->rightNeig(fBlock->sizeTmp);
    size_t rightSz = right->trySetMeUsed(GuardedSize::COAL_BLOCK);
    if (rightSz != GuardedSize::LOCKED) {
        // LastFreeBlock is on the right side
        if (GuardedSize::LAST_REGION_BLOCK == rightSz) {
            right->setMeFree(GuardedSize::LAST_REGION_BLOCK);
            memRegion = static_cast<LastFreeBlock*>(right)->memRegion;
        } else if (GuardedSize::COAL_BLOCK == rightSz) {
            if (resBlock->blockInBin) {
                resBlock->blockInBin = false;
                removeBlockFromBin(resBlock);
            }
            coalescQ.putBlock(resBlock);
            return NULL;
        } else {
            size_t rSz = right->rightNeig(rightSz)->
                trySetLeftUsed(GuardedSize::COAL_BLOCK);
            if (rSz <= GuardedSize::MAX_LOCKED_VAL) {
                right->setMeFree(rightSz);  // rollback
                if (resBlock->blockInBin) {
                    resBlock->blockInBin = false;
                    removeBlockFromBin(resBlock);
                }
                coalescQ.putBlock(resBlock);
                return NULL;
            } else {
                MALLOC_ASSERT(rSz == rightSz, "Invalid header");
                removeBlockFromBin(right);
                resSize += rightSz;

                // Is LastFreeBlock on the right side of right?
                FreeBlock *nextRight = right->rightNeig(rightSz);
                size_t nextRightSz = nextRight->
                    trySetMeUsed(GuardedSize::COAL_BLOCK);
                if (nextRightSz > GuardedSize::MAX_LOCKED_VAL) {
                    if (nextRightSz == GuardedSize::LAST_REGION_BLOCK)
                        memRegion = static_cast<LastFreeBlock*>(nextRight)->memRegion;

                    nextRight->setMeFree(nextRightSz);
                }
            }
        }
    }
    if (memRegion) {
        MALLOC_ASSERT((uintptr_t)memRegion + memRegion->allocSz >=
                      (uintptr_t)right + sizeof(LastFreeBlock), ASSERT_TEXT);
        MALLOC_ASSERT((uintptr_t)memRegion < (uintptr_t)resBlock, ASSERT_TEXT);
        *mRegion = memRegion;
    } else
        *mRegion = NULL;
    resBlock->sizeTmp = resSize;
    return resBlock;
}

bool Backend::coalescAndPutList(FreeBlock *list, bool forceCoalescQDrop,
                                bool reportBlocksProcessed)
{
    bool regionReleased = false;

    for (FreeBlock *helper; list;
         list = helper,
             // matches block enqueue in CoalRequestQ::putBlock()
             reportBlocksProcessed? coalescQ.blockWasProcessed() : (void)0) {
        MemRegion *memRegion;
        bool addToTail = false;

        helper = list->nextToFree;
        FreeBlock *toRet = doCoalesc(list, &memRegion);
        if (!toRet)
            continue;

        if (memRegion && memRegion->blockSz == toRet->sizeTmp
            && !extMemPool->fixedPool) {
            if (extMemPool->regionsAreReleaseable()) {
                // release the region, because there is no used blocks in it
                if (toRet->blockInBin)
                    removeBlockFromBin(toRet);
                releaseRegion(memRegion);
                regionReleased = true;
                continue;
            } else // add block from empty region to end of bin,
                addToTail = true; // preserving for exact fit
        }
        size_t currSz = toRet->sizeTmp;
        int bin = sizeToBin(currSz);
        bool toAligned = toAlignedBin(toRet, currSz);
        bool needAddToBin = true;

        if (toRet->blockInBin) {
            // Does it stay in same bin?
            if (toRet->myBin == bin && toRet->aligned == toAligned)
                needAddToBin = false;
            else {
                toRet->blockInBin = false;
                removeBlockFromBin(toRet);
            }
        }

        // Does not stay in same bin, or bin-less; add it
        if (needAddToBin) {
            toRet->prev = toRet->next = toRet->nextToFree = NULL;
            toRet->myBin = NO_BIN;

            // If the block is too small to fit in any bin, keep it bin-less.
            // It's not a leak because the block later can be coalesced.
            if (currSz >= minBinnedSize) {
                toRet->sizeTmp = currSz;
                IndexedBins *target = toAligned? &freeAlignedBins : &freeLargeBins;
                if (forceCoalescQDrop) {
                    target->addBlock(bin, toRet, toRet->sizeTmp, addToTail);
                } else if (!target->tryAddBlock(bin, toRet, addToTail)) {
                    coalescQ.putBlock(toRet);
                    continue;
                }
            }
            toRet->sizeTmp = 0;
        }
        // Free (possibly coalesced) free block.
        // Adding to bin must be done before this point,
        // because after a block is free it can be coalesced, and
        // using its pointer became unsafe.
        // Remember that coalescing is not done under any global lock.
        toRet->setMeFree(currSz);
        toRet->rightNeig(currSz)->setLeftFree(currSz);
    }
    return regionReleased;
}

// Coalesce fBlock and add it back to a bin;
// processing delayed coalescing requests.
void Backend::coalescAndPut(FreeBlock *fBlock, size_t blockSz)
{
    fBlock->sizeTmp = blockSz;
    fBlock->nextToFree = NULL;

    coalescAndPutList(fBlock, /*forceCoalescQDrop=*/false, /*reportBlocksProcessed=*/false);
}

bool Backend::scanCoalescQ(bool forceCoalescQDrop)
{
    FreeBlock *currCoalescList = coalescQ.getAll();

    if (currCoalescList)
        // reportBlocksProcessed=true informs that the blocks leave coalescQ,
        // matches blockConsumed() from CoalRequestQ::putBlock()
        coalescAndPutList(currCoalescList, forceCoalescQDrop,
                          /*reportBlocksProcessed=*/true);
    // returns status of coalescQ.getAll(), as an indication of possibe changes in backend
    // TODO: coalescAndPutList() may report is some new free blocks became available or not
    return currCoalescList;
}

FreeBlock *Backend::findBlockInRegion(MemRegion *region, size_t exactBlockSize)
{
    FreeBlock *fBlock;
    size_t blockSz;
    uintptr_t fBlockEnd,
        lastFreeBlock = (uintptr_t)region + region->allocSz - sizeof(LastFreeBlock);

    MALLOC_STATIC_ASSERT(sizeof(LastFreeBlock) % sizeof(uintptr_t) == 0,
        "Atomic applied on LastFreeBlock, and we put it at the end of region, that"
        " is uintptr_t-aligned, so no unaligned atomic operations are possible.");
     // right bound is slab-aligned, keep LastFreeBlock after it
    if (region->type==MEMREG_FLEXIBLE_SIZE) {
        fBlock = (FreeBlock *)alignUp((uintptr_t)region + sizeof(MemRegion),
                                      sizeof(uintptr_t));
        fBlockEnd = alignDown(lastFreeBlock, slabSize);
    } else {
        fBlock = (FreeBlock *)alignUp((uintptr_t)region + sizeof(MemRegion),
                                      largeObjectAlignment);
        fBlockEnd = (uintptr_t)fBlock + exactBlockSize;
        MALLOC_ASSERT(fBlockEnd <= lastFreeBlock, ASSERT_TEXT);
    }
    if (fBlockEnd <= (uintptr_t)fBlock)
        return NULL; // allocSz is too small
    blockSz = fBlockEnd - (uintptr_t)fBlock;
    // TODO: extend getSlabBlock to support degradation, i.e. getting less blocks
    // then requested, and then relax this check
    // (now all or nothing is implemented, check according to this)
    if (blockSz < numOfSlabAllocOnMiss*slabSize)
        return NULL;

    region->blockSz = blockSz;
    return fBlock;
}

// startUseBlock adds free block to a bin, the block can be used and
// even released after this, so the region must be added to regionList already
void Backend::startUseBlock(MemRegion *region, FreeBlock *fBlock, bool addToBin)
{
    size_t blockSz = region->blockSz;
    fBlock->initHeader();
    fBlock->setMeFree(blockSz);

    LastFreeBlock *lastBl = static_cast<LastFreeBlock*>(fBlock->rightNeig(blockSz));
    // to not get unaligned atomics during LastFreeBlock access
    MALLOC_ASSERT(isAligned(lastBl, sizeof(uintptr_t)), NULL);
    lastBl->initHeader();
    lastBl->setMeFree(GuardedSize::LAST_REGION_BLOCK);
    lastBl->setLeftFree(blockSz);
    lastBl->myBin = NO_BIN;
    lastBl->memRegion = region;

    if (addToBin) {
        unsigned targetBin = sizeToBin(blockSz);
        // during adding advance regions, register bin for a largest block in region
        advRegBins.registerBin(targetBin);
        if (region->type!=MEMREG_ONE_BLOCK && toAlignedBin(fBlock, blockSz)) {
            freeAlignedBins.addBlock(targetBin, fBlock, blockSz, /*addToTail=*/false);
        } else {
            freeLargeBins.addBlock(targetBin, fBlock, blockSz, /*addToTail=*/false);
        }
    } else {
        // to match with blockReleased() in genericGetBlock
        bkndSync.blockConsumed();
        fBlock->sizeTmp = fBlock->tryLockBlock();
        MALLOC_ASSERT(fBlock->sizeTmp >= FreeBlock::minBlockSize,
                      "Locking must be successful");
    }
}

void MemRegionList::add(MemRegion *r)
{
    r->prev = NULL;
    MallocMutex::scoped_lock lock(regionListLock);
    r->next = head;
    head = r;
    if (head->next)
        head->next->prev = head;
}

void MemRegionList::remove(MemRegion *r)
{
    MallocMutex::scoped_lock lock(regionListLock);
    if (head == r)
        head = head->next;
    if (r->next)
        r->next->prev = r->prev;
    if (r->prev)
        r->prev->next = r->next;
}

#if __TBB_MALLOC_BACKEND_STAT
int MemRegionList::reportStat(FILE *f)
{
    int regNum = 0;
    MallocMutex::scoped_lock lock(regionListLock);
    for (MemRegion *curr = head; curr; curr = curr->next) {
        fprintf(f, "%p: max block %lu B, ", curr, curr->blockSz);
        regNum++;
    }
    return regNum;
}
#endif

FreeBlock *Backend::addNewRegion(size_t size, MemRegionType memRegType, bool addToBin)
{
    MALLOC_STATIC_ASSERT(sizeof(BlockMutexes) <= sizeof(BlockI),
                 "Header must be not overwritten in used blocks");
    MALLOC_ASSERT(FreeBlock::minBlockSize > GuardedSize::MAX_SPEC_VAL,
          "Block length must not conflict with special values of GuardedSize");
    // If the region is not "flexible size" we should reserve some space for
    // a region header, the worst case alignment and the last block mark.
    const size_t requestSize = memRegType == MEMREG_FLEXIBLE_SIZE ? size :
        size + sizeof(MemRegion) + largeObjectAlignment
             +  FreeBlock::minBlockSize + sizeof(LastFreeBlock);

    size_t rawSize = requestSize;
    MemRegion *region = (MemRegion*)allocRawMem(rawSize);
    if (!region) {
        MALLOC_ASSERT(rawSize==requestSize, "getRawMem has not allocated memory but changed the allocated size.");
        return NULL;
    }
    if (rawSize < sizeof(MemRegion)) {
        if (!extMemPool->fixedPool)
            freeRawMem(region, rawSize);
        return NULL;
    }

    region->type = memRegType;
    region->allocSz = rawSize;
    FreeBlock *fBlock = findBlockInRegion(region, size);
    if (!fBlock) {
        if (!extMemPool->fixedPool)
            freeRawMem(region, rawSize);
        return NULL;
    }
    regionList.add(region);
    startUseBlock(region, fBlock, addToBin);
    bkndSync.binsModified();
    return addToBin? (FreeBlock*)VALID_BLOCK_IN_BIN : fBlock;
}

void Backend::init(ExtMemoryPool *extMemoryPool)
{
    extMemPool = extMemoryPool;
    usedAddrRange.init();
    coalescQ.init(&bkndSync);
    bkndSync.init(this);
}

void Backend::reset()
{
    MALLOC_ASSERT(extMemPool->userPool(), "Only user pool can be reset.");
    // no active threads are allowed in backend while reset() called
    verify();

    freeLargeBins.reset();
    freeAlignedBins.reset();
    advRegBins.reset();

    for (MemRegion *curr = regionList.head; curr; curr = curr->next) {
        FreeBlock *fBlock = findBlockInRegion(curr, curr->blockSz);
        MALLOC_ASSERT(fBlock, "A memory region unexpectedly got smaller");
        startUseBlock(curr, fBlock, /*addToBin=*/true);
    }
}

bool Backend::destroy()
{
    bool noError = true;
    // no active threads are allowed in backend while destroy() called
    verify();
    if (!inUserPool()) {
        freeLargeBins.reset();
        freeAlignedBins.reset();
    }
    while (regionList.head) {
        MemRegion *helper = regionList.head->next;
        noError &= freeRawMem(regionList.head, regionList.head->allocSz);
        regionList.head = helper;
    }
    return noError;
}

bool Backend::clean()
{
    scanCoalescQ(/*forceCoalescQDrop=*/false);

    bool res = false;
    // We can have several blocks occupying a whole region,
    // because such regions are added in advance (see askMemFromOS() and reset()),
    // and never used. Release them all.
    for (int i = advRegBins.getMinUsedBin(0); i != -1; i = advRegBins.getMinUsedBin(i+1)) {
        if (i == freeAlignedBins.getMinNonemptyBin(i))
            res |= freeAlignedBins.tryReleaseRegions(i, this);
        if (i == freeLargeBins.getMinNonemptyBin(i))
            res |= freeLargeBins.tryReleaseRegions(i, this);
    }

    return res;
}

void Backend::IndexedBins::verify()
{
    for (int i=0; i<freeBinsNum; i++) {
        for (FreeBlock *fb = freeBins[i].head; fb; fb=fb->next) {
            uintptr_t mySz = fb->myL.value;
            MALLOC_ASSERT(mySz>GuardedSize::MAX_SPEC_VAL, ASSERT_TEXT);
            FreeBlock *right = (FreeBlock*)((uintptr_t)fb + mySz);
            suppress_unused_warning(right);
            MALLOC_ASSERT(right->myL.value<=GuardedSize::MAX_SPEC_VAL, ASSERT_TEXT);
            MALLOC_ASSERT(right->leftL.value==mySz, ASSERT_TEXT);
            MALLOC_ASSERT(fb->leftL.value<=GuardedSize::MAX_SPEC_VAL, ASSERT_TEXT);
        }
    }
}

// For correct operation, it must be called when no other threads
// is changing backend.
void Backend::verify()
{
#if MALLOC_DEBUG
    scanCoalescQ(/*forceCoalescQDrop=*/false);

    freeLargeBins.verify();
    freeAlignedBins.verify();
#endif // MALLOC_DEBUG
}

#if __TBB_MALLOC_BACKEND_STAT
size_t Backend::Bin::countFreeBlocks()
{
    size_t cnt = 0;
    {
        MallocMutex::scoped_lock lock(tLock);
        for (FreeBlock *fb = head; fb; fb = fb->next)
            cnt++;
    }
    return cnt;
}

size_t Backend::Bin::reportFreeBlocks(FILE *f)
{
    size_t totalSz = 0;
    MallocMutex::scoped_lock lock(tLock);
    for (FreeBlock *fb = head; fb; fb = fb->next) {
        size_t sz = fb->tryLockBlock();
        fb->setMeFree(sz);
        fprintf(f, " [%p;%p]", fb, (void*)((uintptr_t)fb+sz));
        totalSz += sz;
    }
    return totalSz;
}

void Backend::IndexedBins::reportStat(FILE *f)
{
    size_t totalSize = 0;

    for (int i=0; i<Backend::freeBinsNum; i++)
        if (size_t cnt = freeBins[i].countFreeBlocks()) {
            totalSize += freeBins[i].reportFreeBlocks(f);
            fprintf(f, " %d:%lu, ", i, cnt);
        }
    fprintf(f, "\ttotal size %lu KB", totalSize/1024);
}

void Backend::reportStat(FILE *f)
{
    scanCoalescQ(/*forceCoalescQDrop=*/false);

    fprintf(f, "\n  regions:\n");
    int regNum = regionList.reportStat(f);
    fprintf(f, "\n%d regions, %lu KB in all regions\n  free bins:\nlarge bins: ",
            regNum, totalMemSize/1024);
    freeLargeBins.reportStat(f);
    fprintf(f, "\naligned bins: ");
    freeAlignedBins.reportStat(f);
    fprintf(f, "\n");
}
#endif // __TBB_MALLOC_BACKEND_STAT

} } // namespaces
