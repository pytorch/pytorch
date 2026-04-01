#ifndef SLANG_VIRTUAL_POOL_ALLOCATOR_H
#define SLANG_VIRTUAL_POOL_ALLOCATOR_H

namespace Slang
{

/// A virtual free-list allocater.
/// This class doesn't actually allocates memory, instead it operates on a
/// virtual integer space. Can be used to implement various types of object pools
/// that needs to support contiguous allocations of more than one elements.
class VirtualObjectPool
{
public:
    struct FreeListNode
    {
        int Offset;
        int Length;
        FreeListNode* prev;
        FreeListNode* next;
    };
    FreeListNode* freeListHead = nullptr;

public:
    void destroy()
    {
        auto list = freeListHead;
        while (list)
        {
            auto next = list->next;
            delete list;
            list = next;
        }
        freeListHead = nullptr;
    }

    ~VirtualObjectPool() { destroy(); }

    void initPool(int numElements)
    {
        freeListHead = new FreeListNode();
        freeListHead->prev = freeListHead->next = nullptr;
        freeListHead->Offset = 0;
        freeListHead->Length = numElements;
    }

    int alloc(int size)
    {
        if (!freeListHead)
            return -1;
        auto freeBlock = freeListHead;
        while (freeBlock && freeBlock->Length < size)
            freeBlock = freeBlock->next;
        if (!freeBlock || freeBlock->Length < size)
            return -1;
        int result = freeBlock->Offset;
        freeBlock->Offset += size;
        freeBlock->Length -= size;
        if (freeBlock->Length == 0)
        {
            if (freeBlock->prev)
                freeBlock->prev->next = freeBlock->next;
            if (freeBlock->next)
                freeBlock->next->prev = freeBlock->prev;
            if (freeBlock == freeListHead)
                freeListHead = freeBlock->next;
            delete freeBlock;
        }
        return result;
    }
    void free(int offset, int size)
    {
        if (!freeListHead)
        {
            freeListHead = new FreeListNode();
            freeListHead->next = freeListHead->prev = nullptr;
            freeListHead->Length = size;
            freeListHead->Offset = offset;
            return;
        }
        auto freeListNode = freeListHead;
        FreeListNode* prevFreeNode = nullptr;
        while (freeListNode && freeListNode->Offset < offset + size)
        {
            prevFreeNode = freeListNode;
            freeListNode = freeListNode->next;
        }
        FreeListNode* newNode = new FreeListNode();
        newNode->Offset = offset;
        newNode->Length = size;
        newNode->prev = prevFreeNode;
        newNode->next = freeListNode;
        if (freeListNode)
            freeListNode->prev = newNode;
        if (prevFreeNode)
            prevFreeNode->next = newNode;
        if (freeListNode == freeListHead)
            freeListHead = newNode;
        if (prevFreeNode && prevFreeNode->Offset + prevFreeNode->Length == newNode->Offset)
        {
            prevFreeNode->Length += newNode->Length;
            prevFreeNode->next = freeListNode;
            if (freeListNode)
                freeListNode->prev = prevFreeNode;
            delete newNode;
            newNode = prevFreeNode;
        }
        if (freeListNode && newNode->Offset + newNode->Length == freeListNode->Offset)
        {
            newNode->Length += freeListNode->Length;
            newNode->next = freeListNode->next;
            if (freeListNode->next)
                freeListNode->next->prev = newNode;
            delete freeListNode;
        }
    }
};

} // namespace Slang
#endif
