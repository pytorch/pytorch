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
#include <new>        /* for placement new */

namespace rml {
namespace internal {


/********* backreferences ***********************/
/* Each slab block and each large memory object header contains BackRefIdx
 * that points out in some BackRefBlock which points back to this block or header.
 */
struct BackRefBlock : public BlockI {
    BackRefBlock *nextForUse;     // the next in the chain of blocks with free items
    FreeObject   *bumpPtr;        // bump pointer moves from the end to the beginning of the block
    FreeObject   *freeList;
    // list of all blocks that were allocated from raw mem (i.e., not from backend)
    BackRefBlock *nextRawMemBlock;
    int           allocatedCount; // the number of objects allocated
    BackRefIdx::master_t myNum;   // the index in the master
    MallocMutex   blockMutex;
    // true if this block has been added to the listForUse chain,
    // modifications protected by masterMutex
    bool          addedToForUse;

    BackRefBlock(const BackRefBlock *blockToUse, intptr_t num) :
        nextForUse(NULL), bumpPtr((FreeObject*)((uintptr_t)blockToUse + slabSize - sizeof(void*))),
        freeList(NULL), nextRawMemBlock(NULL), allocatedCount(0), myNum(num),
        addedToForUse(false) {
        memset(&blockMutex, 0, sizeof(MallocMutex));

        MALLOC_ASSERT(!(num >> CHAR_BIT*sizeof(BackRefIdx::master_t)),
                      "index in BackRefMaster must fit to BackRefIdx::master");
    }
    // clean all but header
    void zeroSet() { memset(this+1, 0, BackRefBlock::bytes-sizeof(BackRefBlock)); }
    static const int bytes = slabSize;
};

// max number of backreference pointers in slab block
static const int BR_MAX_CNT = (BackRefBlock::bytes-sizeof(BackRefBlock))/sizeof(void*);

struct BackRefMaster {
/* On 64-bit systems a slab block can hold up to ~2K back pointers to slab blocks
 * or large objects, so it can address at least 32MB. The master array of 256KB
 * holds 32K pointers to such blocks, addressing ~1 TB.
 * On 32-bit systems there is ~4K back pointers in a slab block, so ~64MB can be addressed.
 * The master array of 8KB holds 2K pointers to leaves, so ~128 GB can addressed.
 */
    static const size_t bytes = sizeof(uintptr_t)>4? 256*1024 : 8*1024;
    static const int dataSz;
/* space is reserved for master table and 4 leaves
   taking into account VirtualAlloc allocation granularity */
    static const int leaves = 4;
    static const size_t masterSize = BackRefMaster::bytes+leaves*BackRefBlock::bytes;
    // The size of memory request for a few more leaf blocks;
    // selected to match VirtualAlloc granularity
    static const size_t blockSpaceSize = 64*1024;

    Backend       *backend;
    BackRefBlock  *active;         // if defined, use it for allocations
    BackRefBlock  *listForUse;     // the chain of data blocks with free items
    BackRefBlock  *allRawMemBlocks;
    intptr_t       lastUsed;       // index of the last used block
    bool           rawMemUsed;
    MallocMutex    requestNewSpaceMutex;
    BackRefBlock  *backRefBl[1];   // the real size of the array is dataSz

    BackRefBlock *findFreeBlock();
    void          addToForUseList(BackRefBlock *bl);
    void          initEmptyBackRefBlock(BackRefBlock *newBl);
    bool          requestNewSpace();
};

const int BackRefMaster::dataSz
    = 1+(BackRefMaster::bytes-sizeof(BackRefMaster))/sizeof(BackRefBlock*);

static MallocMutex masterMutex;
static BackRefMaster *backRefMaster;

bool initBackRefMaster(Backend *backend)
{
    bool rawMemUsed;
    BackRefMaster *master =
        (BackRefMaster*)backend->getBackRefSpace(BackRefMaster::masterSize,
                                                 &rawMemUsed);
    if (! master)
        return false;
    master->backend = backend;
    master->listForUse = master->allRawMemBlocks = NULL;
    master->rawMemUsed = rawMemUsed;
    master->lastUsed = -1;
    memset(&master->requestNewSpaceMutex, 0, sizeof(MallocMutex));
    for (int i=0; i<BackRefMaster::leaves; i++) {
        BackRefBlock *bl = (BackRefBlock*)((uintptr_t)master + BackRefMaster::bytes + i*BackRefBlock::bytes);
        bl->zeroSet();
        master->initEmptyBackRefBlock(bl);
        if (i)
            master->addToForUseList(bl);
        else // active leaf is not needed in listForUse
            master->active = bl;
    }
    // backRefMaster is read in getBackRef, so publish it in consistent state
    FencedStore((intptr_t&)backRefMaster, (intptr_t)master);
    return true;
}

void destroyBackRefMaster(Backend *backend)
{
    if (backRefMaster) { // Is initBackRefMaster() called?
        for (BackRefBlock *curr=backRefMaster->allRawMemBlocks; curr; ) {
            BackRefBlock *next = curr->nextRawMemBlock;
            // allRawMemBlocks list is only for raw mem blocks
            backend->putBackRefSpace(curr, BackRefMaster::blockSpaceSize,
                                     /*rawMemUsed=*/true);
            curr = next;
        }
        backend->putBackRefSpace(backRefMaster, BackRefMaster::masterSize,
                                 backRefMaster->rawMemUsed);
    }
}

void BackRefMaster::addToForUseList(BackRefBlock *bl)
{
    bl->nextForUse = listForUse;
    listForUse = bl;
    bl->addedToForUse = true;
}

void BackRefMaster::initEmptyBackRefBlock(BackRefBlock *newBl)
{
    intptr_t nextLU = lastUsed+1;
    new (newBl) BackRefBlock(newBl, nextLU);
    MALLOC_ASSERT(nextLU < dataSz, NULL);
    backRefBl[nextLU] = newBl;
    // lastUsed is read in getBackRef, and access to backRefBl[lastUsed]
    // is possible only after checking backref against current lastUsed
    FencedStore(lastUsed, nextLU);
}

bool BackRefMaster::requestNewSpace()
{
    bool rawMemUsed;
    MALLOC_STATIC_ASSERT(!(blockSpaceSize % BackRefBlock::bytes),
                         "Must request space for whole number of blocks.");

    if (backRefMaster->dataSz <= lastUsed + 1) // no space in master
        return false;

    // only one thread at a time may add blocks
    MallocMutex::scoped_lock newSpaceLock(requestNewSpaceMutex);

    if (listForUse) // double check that only one block is available
        return true;
    BackRefBlock *newBl =
        (BackRefBlock*)backend->getBackRefSpace(blockSpaceSize, &rawMemUsed);
    if (!newBl) return false;

    // touch a page for the 1st time without taking masterMutex ...
    for (BackRefBlock *bl = newBl; (uintptr_t)bl < (uintptr_t)newBl + blockSpaceSize;
         bl = (BackRefBlock*)((uintptr_t)bl + BackRefBlock::bytes))
        bl->zeroSet();

    MallocMutex::scoped_lock lock(masterMutex); // ... and share under lock

    const size_t numOfUnusedIdxs = backRefMaster->dataSz - lastUsed - 1;
    if (numOfUnusedIdxs <= 0) { // no space in master under lock, roll back
        backend->putBackRefSpace(newBl, blockSpaceSize, rawMemUsed);
        return false;
    }
    // It's possible that only part of newBl is used, due to lack of indices in master.
    // This is OK as such underutilization is possible only once for backreferneces table.
    int blocksToUse = min(numOfUnusedIdxs, blockSpaceSize / BackRefBlock::bytes);

    // use the first block in the batch to maintain the list of "raw" memory
    // to be released at shutdown
    if (rawMemUsed) {
        newBl->nextRawMemBlock = backRefMaster->allRawMemBlocks;
        backRefMaster->allRawMemBlocks = newBl;
    }
    for (BackRefBlock *bl = newBl; blocksToUse>0;
         bl = (BackRefBlock*)((uintptr_t)bl + BackRefBlock::bytes), blocksToUse--) {
        initEmptyBackRefBlock(bl);
        if (active->allocatedCount == BR_MAX_CNT)
            active = bl; // active leaf is not needed in listForUse
        else
            addToForUseList(bl);
    }
    return true;
}

BackRefBlock *BackRefMaster::findFreeBlock()
{
    if (active->allocatedCount < BR_MAX_CNT)
        return active;

    if (listForUse) {                                   // use released list
        MallocMutex::scoped_lock lock(masterMutex);

        if (active->allocatedCount == BR_MAX_CNT && listForUse) {
            active = listForUse;
            listForUse = listForUse->nextForUse;
            MALLOC_ASSERT(active->addedToForUse, ASSERT_TEXT);
            active->addedToForUse = false;
        }
    } else // allocate new data node
        if (!requestNewSpace())
            return NULL;
    return active;
}

void *getBackRef(BackRefIdx backRefIdx)
{
    // !backRefMaster means no initialization done, so it can't be valid memory
    // see addEmptyBackRefBlock for fences around lastUsed
    if (!FencedLoad((intptr_t&)backRefMaster)
        || backRefIdx.getMaster() > FencedLoad(backRefMaster->lastUsed)
        || backRefIdx.getOffset() >= BR_MAX_CNT)
        return NULL;
    return *(void**)((uintptr_t)backRefMaster->backRefBl[backRefIdx.getMaster()]
                     + sizeof(BackRefBlock)+backRefIdx.getOffset()*sizeof(void*));
}

void setBackRef(BackRefIdx backRefIdx, void *newPtr)
{
    MALLOC_ASSERT(backRefIdx.getMaster()<=backRefMaster->lastUsed && backRefIdx.getOffset()<BR_MAX_CNT,
                  ASSERT_TEXT);
    *(void**)((uintptr_t)backRefMaster->backRefBl[backRefIdx.getMaster()]
              + sizeof(BackRefBlock) + backRefIdx.getOffset()*sizeof(void*)) = newPtr;
}

BackRefIdx BackRefIdx::newBackRef(bool largeObj)
{
    BackRefBlock *blockToUse;
    void **toUse;
    BackRefIdx res;
    bool lastBlockFirstUsed = false;

    do {
        MALLOC_ASSERT(backRefMaster, ASSERT_TEXT);
        blockToUse = backRefMaster->findFreeBlock();
        if (!blockToUse)
            return BackRefIdx();
        toUse = NULL;
        { // the block is locked to find a reference
            MallocMutex::scoped_lock lock(blockToUse->blockMutex);

            if (blockToUse->freeList) {
                toUse = (void**)blockToUse->freeList;
                blockToUse->freeList = blockToUse->freeList->next;
                MALLOC_ASSERT(!blockToUse->freeList ||
                              ((uintptr_t)blockToUse->freeList>=(uintptr_t)blockToUse
                               && (uintptr_t)blockToUse->freeList <
                               (uintptr_t)blockToUse + slabSize), ASSERT_TEXT);
            } else if (blockToUse->allocatedCount < BR_MAX_CNT) {
                toUse = (void**)blockToUse->bumpPtr;
                blockToUse->bumpPtr =
                    (FreeObject*)((uintptr_t)blockToUse->bumpPtr - sizeof(void*));
                if (blockToUse->allocatedCount == BR_MAX_CNT-1) {
                    MALLOC_ASSERT((uintptr_t)blockToUse->bumpPtr
                                  < (uintptr_t)blockToUse+sizeof(BackRefBlock),
                                  ASSERT_TEXT);
                    blockToUse->bumpPtr = NULL;
                }
            }
            if (toUse) {
                if (!blockToUse->allocatedCount && !backRefMaster->listForUse)
                    lastBlockFirstUsed = true;
                blockToUse->allocatedCount++;
            }
        } // end of lock scope
    } while (!toUse);
    // The first thread that uses the last block requests new space in advance;
    // possible failures are ignored.
    if (lastBlockFirstUsed)
        backRefMaster->requestNewSpace();

    res.master = blockToUse->myNum;
    uintptr_t offset =
        ((uintptr_t)toUse - ((uintptr_t)blockToUse + sizeof(BackRefBlock)))/sizeof(void*);
    // Is offset too big?
    MALLOC_ASSERT(!(offset >> 15), ASSERT_TEXT);
    res.offset = offset;
    if (largeObj) res.largeObj = largeObj;

    return res;
}

void removeBackRef(BackRefIdx backRefIdx)
{
    MALLOC_ASSERT(!backRefIdx.isInvalid(), ASSERT_TEXT);
    MALLOC_ASSERT(backRefIdx.getMaster()<=backRefMaster->lastUsed
                  && backRefIdx.getOffset()<BR_MAX_CNT, ASSERT_TEXT);
    BackRefBlock *currBlock = backRefMaster->backRefBl[backRefIdx.getMaster()];
    FreeObject *freeObj = (FreeObject*)((uintptr_t)currBlock + sizeof(BackRefBlock)
                                        + backRefIdx.getOffset()*sizeof(void*));
    MALLOC_ASSERT(((uintptr_t)freeObj>(uintptr_t)currBlock &&
                   (uintptr_t)freeObj<(uintptr_t)currBlock + slabSize), ASSERT_TEXT);
    {
        MallocMutex::scoped_lock lock(currBlock->blockMutex);

        freeObj->next = currBlock->freeList;
        MALLOC_ASSERT(!freeObj->next ||
                      ((uintptr_t)freeObj->next > (uintptr_t)currBlock
                       && (uintptr_t)freeObj->next <
                       (uintptr_t)currBlock + slabSize), ASSERT_TEXT);
        currBlock->freeList = freeObj;
        currBlock->allocatedCount--;
    }
    // TODO: do we need double-check here?
    if (!currBlock->addedToForUse && currBlock!=backRefMaster->active) {
        MallocMutex::scoped_lock lock(masterMutex);

        if (!currBlock->addedToForUse && currBlock!=backRefMaster->active)
            backRefMaster->addToForUseList(currBlock);
    }
}

/********* End of backreferences ***********************/

} // namespace internal
} // namespace rml

