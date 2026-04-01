#include "core/slang-basic.h"
#include "renderer-shared.h"

namespace gfx
{
template<typename TDevice, typename TBufferResource>
class StagingBufferPool
{
public:
    struct StagingBufferPage
    {
        Slang::RefPtr<TBufferResource> resource;
        size_t size;
    };

    struct Allocation
    {
        TBufferResource* resource;
        size_t offset;
    };

    TDevice* m_device;
    MemoryType m_memoryType;
    uint32_t m_alignment;
    ResourceStateSet m_allowedStates;

    Slang::List<StagingBufferPage> m_pages;
    Slang::List<Slang::RefPtr<TBufferResource>> m_largeAllocations;

    Slang::Index m_pageAllocCounter = 0;
    size_t m_offsetAllocCounter = 0;

    const size_t kStagingBufferDefaultPageSize = 16 * 1024 * 1024;

    void init(
        TDevice* device,
        MemoryType memoryType,
        uint32_t alignment,
        ResourceStateSet allowedStates)
    {
        m_device = device;
        m_memoryType = memoryType;
        m_alignment = alignment;
        m_allowedStates = allowedStates;
    }

    static size_t alignUp(size_t value, uint32_t alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
    }

    void reset()
    {
        m_pageAllocCounter = 0;
        m_offsetAllocCounter = 0;
        m_largeAllocations.clearAndDeallocate();
    }

    Result newStagingBufferPage()
    {
        StagingBufferPage page;
        size_t pageSize = kStagingBufferDefaultPageSize;

        Slang::ComPtr<IBufferResource> bufferPtr;
        IBufferResource::Desc bufferDesc;
        bufferDesc.type = IResource::Type::Buffer;
        bufferDesc.defaultState = ResourceState::General;
        bufferDesc.allowedStates = m_allowedStates;
        bufferDesc.memoryType = m_memoryType;
        bufferDesc.sizeInBytes = pageSize;
        SLANG_RETURN_ON_FAIL(
            m_device->createBufferResource(bufferDesc, nullptr, bufferPtr.writeRef()));

        page.resource = static_cast<TBufferResource*>(bufferPtr.get());
        page.size = pageSize;
        m_pages.add(page);
        return SLANG_OK;
    }

    Result newLargeBuffer(size_t size)
    {
        Slang::ComPtr<IBufferResource> bufferPtr;
        IBufferResource::Desc bufferDesc;
        bufferDesc.type = IResource::Type::Buffer;
        bufferDesc.defaultState = ResourceState::General;
        bufferDesc.allowedStates = m_allowedStates;
        bufferDesc.memoryType = m_memoryType;
        bufferDesc.sizeInBytes = size;
        SLANG_RETURN_ON_FAIL(
            m_device->createBufferResource(bufferDesc, nullptr, bufferPtr.writeRef()));
        auto bufferImpl = static_cast<TBufferResource*>(bufferPtr.get());
        m_largeAllocations.add(bufferImpl);
        return SLANG_OK;
    }

    Allocation allocate(size_t size, bool forceLargePage)
    {
        if (forceLargePage || size >= (kStagingBufferDefaultPageSize >> 2))
        {
            newLargeBuffer(size);
            Allocation result;
            result.resource = m_largeAllocations.getLast();
            result.offset = 0;
            return result;
        }

        size_t bufferAllocOffset = alignUp(m_offsetAllocCounter, m_alignment);
        Slang::Index bufferId = -1;
        for (Slang::Index i = m_pageAllocCounter; i < m_pages.getCount(); i++)
        {
            auto cb = m_pages[i].resource.Ptr();
            if (bufferAllocOffset + size <= cb->getDesc()->sizeInBytes)
            {
                bufferId = i;
                break;
            }
            bufferAllocOffset = 0;
        }
        // If we cannot find an existing page with sufficient free space,
        // create a new page.
        if (bufferId == -1)
        {
            newStagingBufferPage();
            bufferId = m_pages.getCount() - 1;
        }
        // Sub allocate from current page.
        Allocation result;
        result.resource = m_pages[bufferId].resource.Ptr();
        result.offset = bufferAllocOffset;
        m_pageAllocCounter = bufferId;
        m_offsetAllocCounter = bufferAllocOffset + size;
        return result;
    }
};

template<typename TDevice, typename TBufferResource>
class TransientResourceHeapBaseImpl : public TransientResourceHeapBase
{
public:
    void breakStrongReferenceToDevice() { m_device.breakStrongReference(); }

public:
    BreakableReference<TDevice> m_device;
    StagingBufferPool<TDevice, TBufferResource> m_constantBufferPool;
    StagingBufferPool<TDevice, TBufferResource> m_uploadBufferPool;
    StagingBufferPool<TDevice, TBufferResource> m_readbackBufferPool;

    Result init(const ITransientResourceHeap::Desc& desc, uint32_t alignment, TDevice* device)
    {
        m_device = device;

        m_constantBufferPool.init(
            device,
            MemoryType::Upload,
            256,
            ResourceStateSet(
                ResourceState::ConstantBuffer,
                ResourceState::CopySource,
                ResourceState::CopyDestination));

        m_uploadBufferPool.init(
            device,
            MemoryType::Upload,
            256,
            ResourceStateSet(ResourceState::CopySource, ResourceState::CopyDestination));

        m_readbackBufferPool.init(
            device,
            MemoryType::ReadBack,
            256,
            ResourceStateSet(ResourceState::CopySource, ResourceState::CopyDestination));

        m_version = getVersionCounter();
        getVersionCounter()++;
        return SLANG_OK;
    }

    Result allocateStagingBuffer(
        size_t size,
        IBufferResource*& outBufferWeakPtr,
        size_t& offset,
        MemoryType memoryType,
        bool forceLargePage = false)
    {
        switch (memoryType)
        {
        case MemoryType::ReadBack:
            {
                auto allocation = m_readbackBufferPool.allocate(size, forceLargePage);
                outBufferWeakPtr = allocation.resource;
                offset = allocation.offset;
            }
            break;
        default:
            {
                auto allocation = m_uploadBufferPool.allocate(size, forceLargePage);
                outBufferWeakPtr = allocation.resource;
                offset = allocation.offset;
            }
            break;
        }
        return SLANG_OK;
    }

    Result allocateConstantBuffer(
        size_t size,
        IBufferResource*& outBufferWeakPtr,
        size_t& outOffset)
    {
        auto allocation = m_constantBufferPool.allocate(size, false);
        outBufferWeakPtr = allocation.resource;
        outOffset = allocation.offset;
        return SLANG_OK;
    }

    void reset()
    {
        m_constantBufferPool.reset();
        m_uploadBufferPool.reset();
        m_readbackBufferPool.reset();
        m_version = getVersionCounter();
        getVersionCounter()++;
    }
};

} // namespace gfx
