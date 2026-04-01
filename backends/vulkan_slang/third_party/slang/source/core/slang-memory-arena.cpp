
#include "slang-memory-arena.h"

namespace Slang
{

MemoryArena::MemoryArena()
{
    // Mark as invalid so any alloc call will fail
    m_blockAlignment = 0;
    m_blockAllocSize = 0;

    // Set up as empty
    m_usedBlocks = nullptr;
    m_availableBlocks = nullptr;

    _resetCurrentBlock();

    m_blockFreeList.init(sizeof(Block), sizeof(void*), 16);
}

MemoryArena::~MemoryArena()
{
    reset();
}


MemoryArena::MemoryArena(size_t blockPayloadSize, size_t blockAlignment)
{
    _initialize(blockPayloadSize, blockAlignment);
}

void MemoryArena::init(size_t blockPayloadSize, size_t blockAlignment)
{
    reset();
    _initialize(blockPayloadSize, blockAlignment);
}

void MemoryArena::_initialize(size_t blockPayloadSize, size_t alignment)
{
    // Alignment must be a power of 2
    assert(((alignment - 1) & alignment) == 0);

    // Ensure it's alignment is at least kMinAlignment
    alignment = (alignment < kMinAlignment) ? kMinAlignment : alignment;

    const size_t alignMask = alignment - 1;

    // Make sure the payload is rounded up to the alignment
    blockPayloadSize = (blockPayloadSize + alignMask) & ~alignMask;

    m_blockPayloadSize = blockPayloadSize;

    // If alignment required is larger then the backing allocators then
    // make larger to ensure when alignment correction takes place it will be aligned
    const size_t blockAllocSize =
        (alignment > kMinAlignment) ? (blockPayloadSize + alignment) : blockPayloadSize;

    m_blockAllocSize = blockAllocSize;
    m_blockAlignment = alignment;
    m_availableBlocks = nullptr;

    m_blockFreeList.init(sizeof(Block), sizeof(void*), 16);

    _resetCurrentBlock();
}

void MemoryArena::swapWith(ThisType& rhs)
{
    Swap(m_start, rhs.m_start);
    Swap(m_end, rhs.m_end);
    Swap(m_current, rhs.m_current);

    Swap(m_blockPayloadSize, rhs.m_blockPayloadSize);
    Swap(m_blockAllocSize, rhs.m_blockAllocSize);
    Swap(m_blockAlignment, rhs.m_blockAlignment);

    Swap(m_availableBlocks, rhs.m_availableBlocks);
    Swap(m_usedBlocks, rhs.m_usedBlocks);

    m_blockFreeList.swapWith(rhs.m_blockFreeList);
}

void MemoryArena::_resetCurrentBlock()
{
    m_start = nullptr;
    m_end = nullptr;
    m_current = nullptr;

    m_usedBlocks = nullptr;
}

void MemoryArena::_addCurrentBlock(Block* block)
{
    // Set up for allocation from
    m_end = block->m_end;
    m_start = block->m_start;
    m_current = m_start;

    // Add to linked list of used block, making it the top used block
    block->m_next = m_usedBlocks;
    m_usedBlocks = block;
}

void MemoryArena::_setCurrentBlock(Block* block)
{
    // Set up for allocation from
    m_end = block->m_end;
    m_start = block->m_start;
    m_current = m_start;

    assert(m_usedBlocks == block);
}

void MemoryArena::_deallocateBlocksPayload(Block* start)
{
    Block* cur = start;
    while (cur)
    {
        // Deallocate the block
        ::free(cur->m_alloc);
        cur = cur->m_next;
    }
}

void MemoryArena::_deallocateBlocks(Block* start)
{
    Block* cur = start;
    while (cur)
    {
        Block* next = cur->m_next;
        // Deallocate the block
        ::free(cur->m_alloc);

        m_blockFreeList.deallocate(cur);
        cur = next;
    }
}

bool MemoryArena::_isNormalBlock(Block* block)
{
    // The size of the block in total is from m_alloc to the m_end (ie the size that is passed into
    // _newBlock)
    const size_t blockSize = size_t(block->m_end - block->m_alloc);
    return (blockSize == m_blockAllocSize) &&
           ((size_t(block->m_start) & (m_blockAlignment - 1)) == 0);
}

void MemoryArena::_deallocateBlock(Block* block)
{
    // If it's a normal block then make it available
    if (_isNormalBlock(block))
    {
        block->m_next = m_availableBlocks;
        m_availableBlocks = block;
    }
    else
    {
        // Must be odd sized so free it
        ::free(block->m_alloc);
        // Free it in the block list
        m_blockFreeList.deallocate(block);
    }
}

void MemoryArena::deallocateAll()
{
    // we need to rewind through m_usedBlocks -> seeing it the are normal sized or not
    Block* block = m_usedBlocks;
    while (block)
    {
        Block* next = block->m_next;
        _deallocateBlock(block);
        block = next;
    }

    // Reset current block
    _resetCurrentBlock();
}

void MemoryArena::reset()
{
    _deallocateBlocksPayload(m_usedBlocks);
    _deallocateBlocksPayload(m_availableBlocks);

    m_blockFreeList.reset();

    m_availableBlocks = nullptr;

    _resetCurrentBlock();
}

MemoryArena::Block* MemoryArena::_findNonCurrent(const void* data, size_t size) const
{
    return m_usedBlocks ? _findInBlocks(m_usedBlocks->m_next, data, size) : nullptr;
}

MemoryArena::Block* MemoryArena::_findNonCurrent(const void* data) const
{
    return m_usedBlocks ? _findInBlocks(m_usedBlocks->m_next, data) : nullptr;
}

MemoryArena::Block* MemoryArena::_findInBlocks(Block* block, const void* data, size_t size) const
{
    const uint8_t* ptr = (const uint8_t*)data;
    while (block)
    {
        if (ptr >= block->m_start && ptr + size <= block->m_end)
        {
            return block;
        }
        block = block->m_next;
    }
    return nullptr;
}

MemoryArena::Block* MemoryArena::_findInBlocks(Block* block, const void* data) const
{
    const uint8_t* ptr = (const uint8_t*)data;
    while (block)
    {
        if (ptr >= block->m_start && ptr <= block->m_end)
        {
            return block;
        }
        block = block->m_next;
    }
    return nullptr;
}

MemoryArena::Block* MemoryArena::_newNormalBlock()
{
    if (m_availableBlocks)
    {
        // We have an available block..
        Block* block = m_availableBlocks;
        m_availableBlocks = block->m_next;
        return block;
    }

    Block* block = _newBlock(m_blockAllocSize, m_blockAlignment);
    // Check that every normal block has m_blockPayloadSize space
    assert(size_t(block->m_end - block->m_start) >= m_blockPayloadSize);
    return block;
}

MemoryArena::Block* MemoryArena::_newBlock(size_t allocSize, size_t alignment)
{
    assert(alignment >= m_blockAlignment);
    // Alignment must be a power of 2
    assert(((alignment - 1) & alignment) == 0);

    // Allocate block
    Block* block = (Block*)m_blockFreeList.allocate();
    if (!block)
    {
        return nullptr;
    }

    // Allocate the memory
    uint8_t* alloc = (uint8_t*)::malloc(allocSize);
    if (!alloc)
    {
        m_blockFreeList.deallocate(block);
        return nullptr;
    }

    const size_t alignMask = alignment - 1;

    // Do the alignment on the allocation
    uint8_t* const start = (uint8_t*)((size_t(alloc) + alignMask) & ~alignMask);

    // Setup the block
    block->m_alloc = alloc;
    block->m_start = start;
    block->m_end = alloc + allocSize;
    block->m_next = nullptr;

    return block;
}

void MemoryArena::addExternalBlock(void* inData, size_t size)
{
    // Allocate block
    Block* block = (Block*)m_blockFreeList.allocate();
    if (!block)
    {
        return;
    }

    uint8_t* alloc = (uint8_t*)inData;

    const size_t alignMask = m_blockAlignment - 1;

    // Do the alignment on the allocation
    uint8_t* const start = (uint8_t*)((size_t(alloc) + alignMask) & ~alignMask);

    // Setup the block
    block->m_alloc = alloc;
    block->m_start = start;
    block->m_end = alloc + size;
    block->m_next = nullptr;

    // We don't want to place at start, if there is any used blocks - as that is the one
    // that is being split from and can be rewound. So we place just behind in that case
    if (m_usedBlocks)
    {
        block->m_next = m_usedBlocks->m_next;
        m_usedBlocks->m_next = block;
    }
    else
    {
        // There aren't any blocks, so just place at the front
        m_usedBlocks = block;
    }
}

void* MemoryArena::_allocateAlignedFromNewBlockAndZero(size_t sizeInBytes, size_t alignment)
{
    void* mem = _allocateAlignedFromNewBlock(sizeInBytes, alignment);
    if (mem)
    {
        ::memset(mem, 0, sizeInBytes);
    }
    return mem;
}

void* MemoryArena::_allocateAlignedFromNewBlock(size_t size, size_t alignment)
{
    // Make sure init has been called (or has been set up in parameterized constructor)
    assert(m_blockAllocSize > 0);
    // Alignment must be a power of 2
    assert(((alignment - 1) & alignment) == 0);

    // Alignment must at a minimum be block alignment (such if reused the constraints hold)
    alignment = (alignment < m_blockAlignment) ? m_blockAlignment : alignment;

    const size_t alignMask = alignment - 1;

    // The size of the block must be at least large enough to take into account alignment
    size_t allocSize = (alignment <= kMinAlignment) ? size : (size + alignment);

    Block* block;

    // There are two scenarios
    // a) Allocate a new normal block and make current
    // b) Allocate a new 'odd-sized' block and make current
    //
    // That by always allocating a new block if odd-sized, we lose more efficiency in terms of
    // storage (the previous block may not have been used much). BUT doing so makes it easy to
    // rewind - as the blocks are always in order of allocation.
    //
    // An improvement might be to have some abstraction that sits on top that can do this tracking
    // (or have the blocks themselves record if they alias over a previously used block - but we
    // don't bother with this here. If the alignment is greater than regular alignment we need to
    // handle specially
    if (allocSize > m_blockPayloadSize ||
        (alignment > m_blockAlignment && allocSize + alignment > m_blockPayloadSize))
    {
        // This is an odd-sized block so just allocate the whole thing.
        block = _newBlock(allocSize, alignment);
    }
    else
    {
        // Must be allocatable within a normal block
        assert(allocSize <= m_blockAllocSize);
        block = _newNormalBlock();
    }

    // If not allocated we are done
    if (!block)
    {
        return nullptr;
    }

    // Make the current block
    _addCurrentBlock(block);

    // Align the memory
    uint8_t* memory = (uint8_t*)((size_t(m_current) + alignMask) & ~alignMask);

    // It must be aligned
    assert((size_t(memory) & alignMask) == 0);

    // Do the aligned allocation (which must fit) by aligning the pointer
    // It must fit if the previous code is correct...
    assert(memory + size <= m_end);
    // Move the current pointer
    m_current = memory + size;
    return memory;
}

size_t MemoryArena::_calcBlocksUsedMemory(const Block* block) const
{
    size_t total = 0;
    while (block)
    {
        total += size_t(block->m_end - block->m_start);
        block = block->m_next;
    }
    return total;
}

size_t MemoryArena::_calcBlocksAllocatedMemory(const Block* block) const
{
    size_t total = 0;
    while (block)
    {
        total += size_t(block->m_end - block->m_alloc);
        block = block->m_next;
    }
    return total;
}

void MemoryArena::_rewindToCursor(const void* cursorIn)
{
    // If it's nullptr, then there are no allocation so free all
    if (cursorIn == nullptr)
    {
        deallocateAll();
        return;
    }

    // Find the block that contains the allocation
    Block* cursorBlock = _findNonCurrent(cursorIn);
    assert(cursorBlock);
    if (!cursorBlock)
    {
        // If not found it means this address is NOT part any of the active used heap!
        // Probably an invalid cursor
        return;
    }

    // Deallocate all of the blocks up to the cursor block
    {
        Block* block = m_usedBlocks;
        while (block != cursorBlock)
        {
            Block* next = block->m_next;
            _deallocateBlock(block);
            block = next;
        }
    }

    // The cursor block is now the current block
    m_usedBlocks = cursorBlock;
    _setCurrentBlock(cursorBlock);

    const uint8_t* cursor = (const uint8_t*)cursorIn;
    // Must be in the range of the currently set block
    assert(cursor >= m_start && cursor <= m_end);

    // Set the current position where the cursor is
    m_current = const_cast<uint8_t*>(cursor);
}

size_t MemoryArena::calcTotalMemoryUsed() const
{
    return (m_usedBlocks ? _calcBlocksUsedMemory(m_usedBlocks->m_next) : 0) +
           size_t(m_current - m_start);
}

size_t MemoryArena::calcTotalMemoryAllocated() const
{
    return _calcBlocksAllocatedMemory(m_usedBlocks) + _calcBlocksAllocatedMemory(m_availableBlocks);
}


} // namespace Slang
