// metal-device.cpp
#include "metal-device.h"

#include "../resource-desc-utils.h"
#include "metal-buffer.h"
#include "metal-render-pass.h"
#include "metal-shader-program.h"
#include "metal-swap-chain.h"
#include "metal-texture.h"
#include "metal-util.h"
#include "metal-vertex-layout.h"
// #include "metal-command-queue.h"
#include "metal-fence.h"
#include "metal-query.h"
// #include "metal-resource-views.h"
#include "metal-sampler.h"
#include "metal-shader-object-layout.h"
#include "metal-shader-object.h"
// #include "metal-shader-table.h"
#include "metal-transient-heap.h"
// #include "metal-pipeline-dump-layer.h"
// #include "metal-helper-functions.h"

#include "core/slang-platform.h"
namespace gfx
{

using namespace Slang;

namespace metal
{

static bool shouldDumpPipeline()
{
    StringBuilder dumpPipelineSettings;
    PlatformUtil::getEnvironmentVariable(toSlice("SLANG_GFX_DUMP_PIPELINE"), dumpPipelineSettings);
    return dumpPipelineSettings.produceString() == "1";
}

DeviceImpl::~DeviceImpl() {}

Result DeviceImpl::getNativeDeviceHandles(InteropHandles* outHandles)
{
    outHandles->handles[0].api = InteropHandleAPI::Metal;
    outHandles->handles[0].handleValue = reinterpret_cast<intptr_t>(m_device.get());
    return SLANG_OK;
}

SlangResult DeviceImpl::initialize(const Desc& desc)
{
    AUTORELEASEPOOL

    // Initialize device info.
    {
        m_info.apiName = "Metal";
        m_info.bindingStyle = BindingStyle::Metal;
        m_info.projectionStyle = ProjectionStyle::Metal;
        m_info.deviceType = DeviceType::Metal;
        m_info.adapterName = "default";
        static const float kIdentity[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        ::memcpy(m_info.identityProjectionMatrix, kIdentity, sizeof(kIdentity));
    }

    m_desc = desc;

    SLANG_RETURN_ON_FAIL(RendererBase::initialize(desc));
    SlangResult initDeviceResult = SLANG_OK;

    m_device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    m_commandQueue = NS::TransferPtr(m_device->newCommandQueue(64));
    m_hasArgumentBufferTier2 = m_device->argumentBuffersSupport() >= MTL::ArgumentBuffersTier2;

    if (m_hasArgumentBufferTier2)
    {
        m_features.add("argument-buffer-tier-2");
    }

    SLANG_RETURN_ON_FAIL(slangContext.initialize(
        desc.slang,
        desc.extendedDescCount,
        desc.extendedDescs,
        SLANG_METAL_LIB,
        "",
        makeArray(slang::PreprocessorMacroDesc{"__METAL__", "1"}).getView()));

    // TODO: expose via some other means
    if (captureEnabled())
    {
        MTL::CaptureManager* captureManager = MTL::CaptureManager::sharedCaptureManager();
        MTL::CaptureDescriptor* d = MTL::CaptureDescriptor::alloc()->init();
        MTL::CaptureDestination captureDest =
            MTL::CaptureDestination::CaptureDestinationGPUTraceDocument;
        if (!captureManager->supportsDestination(MTL::CaptureDestinationGPUTraceDocument))
        {
            std::cout << "Cannot capture MTL calls to document; ensure that Info.plist exists with "
                         "'MetalCaptureEnabled' set to 'true'."
                      << std::endl;
            exit(1);
        }
        d->setDestination(MTL::CaptureDestinationGPUTraceDocument);
        d->setCaptureObject(m_device.get());
        NS::SharedPtr<NS::String> path = MetalUtil::createString("frame.gputrace");
        NS::SharedPtr<NS::URL> url =
            NS::TransferPtr(NS::URL::alloc()->initFileURLWithPath(path.get()));
        d->setOutputURL(url.get());
        NS::Error* errorCode = NS::Error::alloc();
        if (!captureManager->startCapture(d, &errorCode))
        {
            NS::String* errorString = errorCode->description();
            std::string estr(errorString->cString(NS::UTF8StringEncoding));
            std::cout << "Start capture failure: " << estr << std::endl;
            exit(1);
        }
    }
    return SLANG_OK;
}

// void DeviceImpl::waitForGpu() { m_deviceQueue.flushAndWait(); }


const DeviceInfo& DeviceImpl::getDeviceInfo() const
{
    return m_info;
}

Result DeviceImpl::createTransientResourceHeap(
    const ITransientResourceHeap::Desc& desc,
    ITransientResourceHeap** outHeap)
{
    AUTORELEASEPOOL

    RefPtr<TransientResourceHeapImpl> result = new TransientResourceHeapImpl();
    SLANG_RETURN_ON_FAIL(result->init(desc, this));
    returnComPtr(outHeap, result);
    return SLANG_OK;
}

Result DeviceImpl::createCommandQueue(const ICommandQueue::Desc& desc, ICommandQueue** outQueue)
{
    AUTORELEASEPOOL

    if (m_queueAllocCount != 0)
        return SLANG_FAIL;

    RefPtr<CommandQueueImpl> result = new CommandQueueImpl;
    result->init(this, m_commandQueue);
    returnComPtr(outQueue, result);
    m_queueAllocCount++;
    return SLANG_OK;
}

Result DeviceImpl::createSwapchain(
    const ISwapchain::Desc& desc,
    WindowHandle window,
    ISwapchain** outSwapchain)
{
    AUTORELEASEPOOL

    RefPtr<SwapchainImpl> swapchainImpl = new SwapchainImpl();
    SLANG_RETURN_ON_FAIL(swapchainImpl->init(this, desc, window));
    returnComPtr(outSwapchain, swapchainImpl);
    return SLANG_OK;
}

Result DeviceImpl::createFramebufferLayout(
    const IFramebufferLayout::Desc& desc,
    IFramebufferLayout** outLayout)
{
    AUTORELEASEPOOL

    RefPtr<FramebufferLayoutImpl> layoutImpl = new FramebufferLayoutImpl;
    SLANG_RETURN_ON_FAIL(layoutImpl->init(desc));
    returnComPtr(outLayout, layoutImpl);
    return SLANG_OK;
}

Result DeviceImpl::createRenderPassLayout(
    const IRenderPassLayout::Desc& desc,
    IRenderPassLayout** outRenderPassLayout)
{
    AUTORELEASEPOOL

    RefPtr<RenderPassLayoutImpl> renderPassLayoutImpl = new RenderPassLayoutImpl;
    SLANG_RETURN_ON_FAIL(renderPassLayoutImpl->init(this, desc));
    returnComPtr(outRenderPassLayout, renderPassLayoutImpl);
    return SLANG_OK;
}

Result DeviceImpl::createFramebuffer(const IFramebuffer::Desc& desc, IFramebuffer** outFramebuffer)
{
    AUTORELEASEPOOL

    RefPtr<FramebufferImpl> framebufferImpl = new FramebufferImpl;
    SLANG_RETURN_ON_FAIL(framebufferImpl->init(this, desc));
    returnComPtr(outFramebuffer, framebufferImpl);
    return SLANG_OK;
}

SlangResult DeviceImpl::readTextureResource(
    ITextureResource* texture,
    ResourceState state,
    ISlangBlob** outBlob,
    Size* outRowPitch,
    Size* outPixelSize)
{
    AUTORELEASEPOOL

    TextureResourceImpl* textureImpl = static_cast<TextureResourceImpl*>(texture);

    if (textureImpl->getDesc()->sampleDesc.numSamples > 1)
    {
        return SLANG_E_NOT_IMPLEMENTED;
    }

    NS::SharedPtr<MTL::Texture> srcTexture = textureImpl->m_texture;

    const ITextureResource::Desc& desc = *textureImpl->getDesc();
    Count width = Math::Max(desc.size.width, 1);
    Count height = Math::Max(desc.size.height, 1);
    Count depth = Math::Max(desc.size.depth, 1);
    FormatInfo formatInfo;
    gfxGetFormatInfo(desc.format, &formatInfo);
    Size bytesPerPixel = formatInfo.blockSizeInBytes / formatInfo.pixelsPerBlock;
    Size bytesPerRow = Size(width) * bytesPerPixel;
    Size bytesPerSlice = Size(height) * bytesPerRow;
    Size bufferSize = Size(depth) * bytesPerSlice;
    if (outRowPitch)
        *outRowPitch = bytesPerRow;
    if (outPixelSize)
        *outPixelSize = bytesPerPixel;

    // create staging buffer
    NS::SharedPtr<MTL::Buffer> stagingBuffer =
        NS::TransferPtr(m_device->newBuffer(bufferSize, MTL::StorageModeShared));
    if (!stagingBuffer)
    {
        return SLANG_FAIL;
    }

    MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
    MTL::BlitCommandEncoder* encoder = commandBuffer->blitCommandEncoder();
    encoder->copyFromTexture(
        srcTexture.get(),
        0,
        0,
        MTL::Origin(0, 0, 0),
        MTL::Size(width, height, depth),
        stagingBuffer.get(),
        0,
        bytesPerRow,
        bytesPerSlice);
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    List<uint8_t> blobData;
    blobData.setCount(bufferSize);
    ::memcpy(blobData.getBuffer(), stagingBuffer->contents(), bufferSize);
    auto blob = ListBlob::moveCreate(blobData);

    returnComPtr(outBlob, blob);
    return SLANG_OK;
}

SlangResult DeviceImpl::readBufferResource(
    IBufferResource* buffer,
    Offset offset,
    Size size,
    ISlangBlob** outBlob)
{
    AUTORELEASEPOOL

    // create staging buffer
    NS::SharedPtr<MTL::Buffer> stagingBuffer =
        NS::TransferPtr(m_device->newBuffer(size, MTL::StorageModeShared));
    if (!stagingBuffer)
    {
        return SLANG_FAIL;
    }

    MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
    MTL::BlitCommandEncoder* blitEncoder = commandBuffer->blitCommandEncoder();
    blitEncoder->copyFromBuffer(
        static_cast<BufferResourceImpl*>(buffer)->m_buffer.get(),
        offset,
        stagingBuffer.get(),
        0,
        size);
    blitEncoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    List<uint8_t> blobData;
    blobData.setCount(size);
    ::memcpy(blobData.getBuffer(), stagingBuffer->contents(), size);
    auto blob = ListBlob::moveCreate(blobData);

    returnComPtr(outBlob, blob);
    return SLANG_OK;
}

Result DeviceImpl::getAccelerationStructurePrebuildInfo(
    const IAccelerationStructure::BuildInputs& buildInputs,
    IAccelerationStructure::PrebuildInfo* outPrebuildInfo)
{
    AUTORELEASEPOOL

    return SLANG_E_NOT_IMPLEMENTED;
}

Result DeviceImpl::createAccelerationStructure(
    const IAccelerationStructure::CreateDesc& desc,
    IAccelerationStructure** outAS)
{
    AUTORELEASEPOOL

    return SLANG_E_NOT_IMPLEMENTED;
}

Result DeviceImpl::getTextureAllocationInfo(
    const ITextureResource::Desc& descIn,
    Size* outSize,
    Size* outAlignment)
{
    AUTORELEASEPOOL

    auto alignTo = [&](Size size, Size alignment) -> Size
    { return ((size + alignment - 1) / alignment) * alignment; };

    TextureResource::Desc desc = fixupTextureDesc(descIn);
    FormatInfo formatInfo;
    gfxGetFormatInfo(desc.format, &formatInfo);
    MTL::PixelFormat pixelFormat = MetalUtil::translatePixelFormat(desc.format);
    bool isCompressed = gfxIsCompressedFormat(desc.format);
    Size alignment =
        isCompressed ? 1 : m_device->minimumLinearTextureAlignmentForPixelFormat(pixelFormat);
    Size size = 0;
    ITextureResource::Extents extents = desc.size;
    extents.width = extents.width ? extents.width : 1;
    extents.height = extents.height ? extents.height : 1;
    extents.depth = extents.depth ? extents.depth : 1;

    for (Int i = 0; i < desc.numMipLevels; ++i)
    {
        Size rowSize = ((extents.width + formatInfo.blockWidth - 1) / formatInfo.blockWidth) *
                       formatInfo.blockSizeInBytes;
        rowSize = alignTo(rowSize, alignment);
        Size sliceSize = rowSize * alignTo(extents.height, formatInfo.blockHeight);
        size += sliceSize * extents.depth;
        extents.width = Math::Max(1, extents.width / 2);
        extents.height = Math::Max(1, extents.height / 2);
        extents.depth = Math::Max(1, extents.depth / 2);
    }
    size *= desc.arraySize ? desc.arraySize : 1;

    *outSize = size;
    *outAlignment = alignment;

    return SLANG_OK;
}

Result DeviceImpl::getTextureRowAlignment(Size* outAlignment)
{
    AUTORELEASEPOOL

    *outAlignment = 1;
    return SLANG_E_NOT_IMPLEMENTED;
}

Result DeviceImpl::createTextureResource(
    const ITextureResource::Desc& descIn,
    const ITextureResource::SubresourceData* initData,
    ITextureResource** outResource)
{
    AUTORELEASEPOOL

    TextureResource::Desc desc = fixupTextureDesc(descIn);

    // Metal doesn't support mip-mapping for 1D textures
    // However, we still need to use the provided mip level count when initializing the texture
    Count initMipLevels = desc.numMipLevels;
    desc.numMipLevels = desc.type == IResource::Type::Texture1D ? 1 : desc.numMipLevels;

    const MTL::PixelFormat pixelFormat = MetalUtil::translatePixelFormat(desc.format);
    if (pixelFormat == MTL::PixelFormat::PixelFormatInvalid)
    {
        assert(!"Unsupported texture format");
        return SLANG_FAIL;
    }

    RefPtr<TextureResourceImpl> textureImpl(new TextureResourceImpl(desc, this));

    NS::SharedPtr<MTL::TextureDescriptor> textureDesc =
        NS::TransferPtr(MTL::TextureDescriptor::alloc()->init());
    switch (desc.memoryType)
    {
    case MemoryType::DeviceLocal:
        textureDesc->setStorageMode(MTL::StorageModePrivate);
        break;
    case MemoryType::Upload:
        textureDesc->setStorageMode(MTL::StorageModeShared);
        textureDesc->setCpuCacheMode(MTL::CPUCacheModeWriteCombined);
        break;
    case MemoryType::ReadBack:
        textureDesc->setStorageMode(MTL::StorageModeShared);
        break;
    }

    bool isArray = desc.arraySize > 0;

    switch (desc.type)
    {
    case IResource::Type::Texture1D:
        textureDesc->setTextureType(isArray ? MTL::TextureType1DArray : MTL::TextureType1D);
        textureDesc->setWidth(desc.size.width);
        break;
    case IResource::Type::Texture2D:
        if (desc.sampleDesc.numSamples > 1)
        {
            textureDesc->setTextureType(
                isArray ? MTL::TextureType2DMultisampleArray : MTL::TextureType2DMultisample);
            textureDesc->setSampleCount(desc.sampleDesc.numSamples);
        }
        else
        {
            textureDesc->setTextureType(isArray ? MTL::TextureType2DArray : MTL::TextureType2D);
        }
        textureDesc->setWidth(descIn.size.width);
        textureDesc->setHeight(descIn.size.height);
        break;
    case IResource::Type::TextureCube:
        textureDesc->setTextureType(isArray ? MTL::TextureTypeCubeArray : MTL::TextureTypeCube);
        textureDesc->setWidth(descIn.size.width);
        textureDesc->setHeight(descIn.size.height);
        break;
    case IResource::Type::Texture3D:
        textureDesc->setTextureType(MTL::TextureType::TextureType3D);
        textureDesc->setWidth(descIn.size.width);
        textureDesc->setHeight(descIn.size.height);
        textureDesc->setDepth(descIn.size.depth);
        break;
    default:
        assert("!Unsupported texture type");
        return SLANG_FAIL;
    }

    MTL::TextureUsage textureUsage = MTL::TextureUsageUnknown;
    if (desc.allowedStates.contains(ResourceState::RenderTarget))
    {
        textureUsage |= MTL::TextureUsageRenderTarget;
    }
    if (desc.allowedStates.contains(ResourceState::ShaderResource))
    {
        textureUsage |= MTL::TextureUsageShaderRead;
    }
    if (desc.allowedStates.contains(ResourceState::UnorderedAccess))
    {
        textureUsage |= MTL::TextureUsageShaderRead;
        textureUsage |= MTL::TextureUsageShaderWrite;

        // Request atomic access if the format allows it.
        switch (desc.format)
        {
        case Format::R32_UINT:
        case Format::R32_SINT:
        case Format::R32G32_UINT:
        case Format::R32G32_SINT:
            textureUsage |= MTL::TextureUsageShaderAtomic;
            break;
        }
    }

    textureDesc->setMipmapLevelCount(desc.numMipLevels);
    textureDesc->setArrayLength(isArray ? desc.arraySize : 1);
    textureDesc->setPixelFormat(pixelFormat);
    textureDesc->setUsage(textureUsage);
    textureDesc->setSampleCount(desc.sampleDesc.numSamples);
    textureDesc->setAllowGPUOptimizedContents(desc.memoryType == MemoryType::DeviceLocal);

    textureImpl->m_texture = NS::TransferPtr(m_device->newTexture(textureDesc.get()));
    if (!textureImpl->m_texture)
    {
        return SLANG_FAIL;
    }
    textureImpl->m_textureType = textureDesc->textureType();
    textureImpl->m_pixelFormat = textureDesc->pixelFormat();

    // TODO: handle initData
    if (initData)
    {
        textureDesc->setStorageMode(MTL::StorageModeManaged);
        textureDesc->setCpuCacheMode(MTL::CPUCacheModeDefaultCache);
        NS::SharedPtr<MTL::Texture> stagingTexture =
            NS::TransferPtr(m_device->newTexture(textureDesc.get()));

        MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
        MTL::BlitCommandEncoder* encoder = commandBuffer->blitCommandEncoder();
        if (!stagingTexture || !commandBuffer || !encoder)
        {
            return SLANG_FAIL;
        }

        Count sliceCount = isArray ? desc.arraySize : 1;
        if (desc.type == IResource::Type::TextureCube)
        {
            sliceCount *= 6;
        }

        for (Index slice = 0; slice < sliceCount; ++slice)
        {
            MTL::Region region;
            region.origin = MTL::Origin(0, 0, 0);
            region.size = MTL::Size(desc.size.width, desc.size.height, desc.size.depth);
            for (Index level = 0; level < initMipLevels; ++level)
            {
                if (level >= desc.numMipLevels)
                    continue;
                const ITextureResource::SubresourceData& subresourceData =
                    initData[slice * initMipLevels + level];
                stagingTexture->replaceRegion(
                    region,
                    level,
                    slice,
                    subresourceData.data,
                    subresourceData.strideY,
                    subresourceData.strideZ);
                encoder->synchronizeTexture(stagingTexture.get(), slice, level);
                region.size.width =
                    region.size.width > 0 ? Math::Max(1ul, region.size.width >> 1) : 0;
                region.size.height =
                    region.size.height > 0 ? Math::Max(1ul, region.size.height >> 1) : 0;
                region.size.depth =
                    region.size.depth > 0 ? Math::Max(1ul, region.size.depth >> 1) : 0;
            }
        }

        encoder->copyFromTexture(stagingTexture.get(), textureImpl->m_texture.get());
        encoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
    }

    returnComPtr(outResource, textureImpl);
    return SLANG_OK;
}

Result DeviceImpl::createBufferResource(
    const IBufferResource::Desc& descIn,
    const void* initData,
    IBufferResource** outResource)
{
    AUTORELEASEPOOL

    BufferResource::Desc desc = fixupBufferDesc(descIn);

    const Size bufferSize = desc.sizeInBytes;

    MTL::ResourceOptions resourceOptions = MTL::ResourceOptions(0);
    switch (desc.memoryType)
    {
    case MemoryType::DeviceLocal:
        resourceOptions = MTL::ResourceStorageModePrivate;
        break;
    case MemoryType::Upload:
        resourceOptions = MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeWriteCombined;
        break;
    case MemoryType::ReadBack:
        resourceOptions = MTL::ResourceStorageModeShared;
        break;
    }
    resourceOptions |= (desc.memoryType == MemoryType::DeviceLocal)
                           ? MTL::ResourceStorageModePrivate
                           : MTL::ResourceStorageModeShared;

    RefPtr<BufferResourceImpl> bufferImpl(new BufferResourceImpl(desc, this));
    bufferImpl->m_buffer = NS::TransferPtr(m_device->newBuffer(bufferSize, resourceOptions));
    if (!bufferImpl->m_buffer)
    {
        return SLANG_FAIL;
    }

    if (initData)
    {
        NS::SharedPtr<MTL::Buffer> stagingBuffer = NS::TransferPtr(m_device->newBuffer(
            initData,
            bufferSize,
            MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeWriteCombined));
        MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
        MTL::BlitCommandEncoder* encoder = commandBuffer->blitCommandEncoder();
        if (!stagingBuffer || !commandBuffer || !encoder)
        {
            return SLANG_FAIL;
        }
        encoder->copyFromBuffer(stagingBuffer.get(), 0, bufferImpl->m_buffer.get(), 0, bufferSize);
        encoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
    }

    returnComPtr(outResource, bufferImpl);
    return SLANG_OK;
}

Result DeviceImpl::createBufferFromNativeHandle(
    InteropHandle handle,
    const IBufferResource::Desc& srcDesc,
    IBufferResource** outResource)
{
    AUTORELEASEPOOL

    return SLANG_E_NOT_IMPLEMENTED;
}

Result DeviceImpl::createSamplerState(ISamplerState::Desc const& desc, ISamplerState** outSampler)
{
    AUTORELEASEPOOL

    RefPtr<SamplerStateImpl> samplerImpl = new SamplerStateImpl();
    SLANG_RETURN_ON_FAIL(samplerImpl->init(this, desc));
    returnComPtr(outSampler, samplerImpl);
    return SLANG_OK;
}

Result DeviceImpl::createTextureView(
    ITextureResource* texture,
    IResourceView::Desc const& desc,
    IResourceView** outView)
{
    AUTORELEASEPOOL

    auto textureImpl = static_cast<TextureResourceImpl*>(texture);
    RefPtr<TextureResourceViewImpl> viewImpl = new TextureResourceViewImpl(this);
    viewImpl->m_desc = desc;
    viewImpl->m_device = this;
    viewImpl->m_texture = textureImpl;
    if (textureImpl == nullptr)
    {
        returnComPtr(outView, viewImpl);
        return SLANG_OK;
    }

    const ITextureResource::Desc& textureDesc = *textureImpl->getDesc();
    SubresourceRange sr = desc.subresourceRange;
    sr.mipLevelCount =
        sr.mipLevelCount == 0 ? textureDesc.numMipLevels - sr.mipLevel : sr.mipLevelCount;
    sr.layerCount = sr.layerCount == 0 ? textureDesc.arraySize - sr.baseArrayLayer : sr.layerCount;
    if (sr.mipLevel == 0 && sr.mipLevelCount == textureDesc.numMipLevels &&
        sr.baseArrayLayer == 0 && sr.layerCount == textureDesc.arraySize)
    {
        viewImpl->m_textureView = textureImpl->m_texture;
        returnComPtr(outView, viewImpl);
        return SLANG_OK;
    }

    MTL::PixelFormat pixelFormat = desc.format == Format::Unknown
                                       ? textureImpl->m_pixelFormat
                                       : MetalUtil::translatePixelFormat(desc.format);
    NS::Range sliceRange(sr.baseArrayLayer, sr.layerCount);
    NS::Range levelRange(sr.mipLevel, sr.mipLevelCount);

    viewImpl->m_textureView = NS::TransferPtr(textureImpl->m_texture->newTextureView(
        pixelFormat,
        textureImpl->m_textureType,
        levelRange,
        sliceRange));
    if (!viewImpl->m_textureView)
    {
        return SLANG_FAIL;
    }

    returnComPtr(outView, viewImpl);
    return SLANG_OK;
}

Result DeviceImpl::getFormatSupportedResourceStates(Format format, ResourceStateSet* outStates)
{
    AUTORELEASEPOOL

    // TODO - add table based on https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
    ResourceStateSet allowedStates;
    allowedStates.add(ResourceState::VertexBuffer);
    allowedStates.add(ResourceState::IndexBuffer);
    allowedStates.add(ResourceState::ConstantBuffer);
    allowedStates.add(ResourceState::ShaderResource);
    allowedStates.add(ResourceState::UnorderedAccess);
    allowedStates.add(ResourceState::RenderTarget);
    allowedStates.add(ResourceState::DepthRead);
    allowedStates.add(ResourceState::DepthWrite);
    allowedStates.add(ResourceState::Present);
    allowedStates.add(ResourceState::IndirectArgument);
    allowedStates.add(ResourceState::CopySource);
    allowedStates.add(ResourceState::ResolveSource);
    allowedStates.add(ResourceState::CopyDestination);
    allowedStates.add(ResourceState::ResolveDestination);
    allowedStates.add(ResourceState::AccelerationStructure);
    allowedStates.add(ResourceState::AccelerationStructureBuildInput);

    *outStates = allowedStates;
    return SLANG_OK;
}

Result DeviceImpl::createBufferView(
    IBufferResource* buffer,
    IBufferResource* counterBuffer,
    IResourceView::Desc const& desc,
    IResourceView** outView)
{
    AUTORELEASEPOOL

    // Counter buffers are not supported on metal.
    if (counterBuffer)
    {
        return SLANG_FAIL;
    }

    if (desc.type != IResourceView::Type::UnorderedAccess &&
        desc.type != IResourceView::Type::ShaderResource)
    {
        return SLANG_FAIL;
    }

    auto bufferImpl = static_cast<BufferResourceImpl*>(buffer);

    RefPtr<BufferResourceViewImpl> viewImpl = new BufferResourceViewImpl(this);
    viewImpl->m_desc = desc;
    viewImpl->m_buffer = bufferImpl;
    viewImpl->m_offset = desc.bufferRange.offset;
    viewImpl->m_size =
        desc.bufferRange.size == 0 ? bufferImpl->getDesc()->sizeInBytes : desc.bufferRange.size;
    returnComPtr(outView, viewImpl);
    return SLANG_OK;
}

Result DeviceImpl::createInputLayout(IInputLayout::Desc const& desc, IInputLayout** outLayout)
{
    AUTORELEASEPOOL

    RefPtr<InputLayoutImpl> layoutImpl(new InputLayoutImpl);
    SLANG_RETURN_ON_FAIL(layoutImpl->init(desc));
    returnComPtr(outLayout, layoutImpl);
    return SLANG_OK;
}

Result DeviceImpl::createProgram(
    const IShaderProgram::Desc& desc,
    IShaderProgram** outProgram,
    ISlangBlob** outDiagnosticBlob)
{
    AUTORELEASEPOOL

    RefPtr<ShaderProgramImpl> shaderProgram = new ShaderProgramImpl(this);
    shaderProgram->init(desc);

    RootShaderObjectLayoutImpl::create(
        this,
        shaderProgram->linkedProgram,
        shaderProgram->linkedProgram->getLayout(),
        shaderProgram->m_rootObjectLayout.writeRef());

    if (!shaderProgram->isSpecializable())
    {
        SLANG_RETURN_ON_FAIL(shaderProgram->compileShaders(this));
    }

    returnComPtr(outProgram, shaderProgram);
    return SLANG_OK;
}

Result DeviceImpl::createShaderObjectLayout(
    slang::ISession* session,
    slang::TypeLayoutReflection* typeLayout,
    ShaderObjectLayoutBase** outLayout)
{
    AUTORELEASEPOOL

    RefPtr<ShaderObjectLayoutImpl> layout;
    SLANG_RETURN_ON_FAIL(
        ShaderObjectLayoutImpl::createForElementType(this, session, typeLayout, layout.writeRef()));
    returnRefPtrMove(outLayout, layout);
    return SLANG_OK;
}

Result DeviceImpl::createShaderObject(ShaderObjectLayoutBase* layout, IShaderObject** outObject)
{
    AUTORELEASEPOOL

    RefPtr<ShaderObjectImpl> shaderObject;
    SLANG_RETURN_ON_FAIL(ShaderObjectImpl::create(
        this,
        static_cast<ShaderObjectLayoutImpl*>(layout),
        shaderObject.writeRef()));
    returnComPtr(outObject, shaderObject);
    return SLANG_OK;
}

Result DeviceImpl::createMutableShaderObject(
    ShaderObjectLayoutBase* layout,
    IShaderObject** outObject)
{
    AUTORELEASEPOOL

    return SLANG_E_NOT_IMPLEMENTED;
}

Result DeviceImpl::createMutableRootShaderObject(IShaderProgram* program, IShaderObject** outObject)
{
    AUTORELEASEPOOL

    return SLANG_E_NOT_IMPLEMENTED;
}

Result DeviceImpl::createShaderTable(const IShaderTable::Desc& desc, IShaderTable** outShaderTable)
{
    AUTORELEASEPOOL

    return SLANG_E_NOT_IMPLEMENTED;
}

Result DeviceImpl::createGraphicsPipelineState(
    const GraphicsPipelineStateDesc& desc,
    IPipelineState** outState)
{
    AUTORELEASEPOOL

    RefPtr<PipelineStateImpl> pipelineStateImpl = new PipelineStateImpl(this);
    pipelineStateImpl->init(desc);
    returnComPtr(outState, pipelineStateImpl);
    return SLANG_OK;
}

Result DeviceImpl::createComputePipelineState(
    const ComputePipelineStateDesc& desc,
    IPipelineState** outState)
{
    AUTORELEASEPOOL

    RefPtr<PipelineStateImpl> pipelineStateImpl = new PipelineStateImpl(this);
    pipelineStateImpl->init(desc);
    m_deviceObjectsWithPotentialBackReferences.add(pipelineStateImpl);
    returnComPtr(outState, pipelineStateImpl);
    return SLANG_OK;
}

Result DeviceImpl::createRayTracingPipelineState(
    const RayTracingPipelineStateDesc& desc,
    IPipelineState** outState)
{
    AUTORELEASEPOOL

    return SLANG_E_NOT_IMPLEMENTED;
}

Result DeviceImpl::createQueryPool(const IQueryPool::Desc& desc, IQueryPool** outPool)
{
    AUTORELEASEPOOL

    RefPtr<QueryPoolImpl> poolImpl = new QueryPoolImpl();
    SLANG_RETURN_ON_FAIL(poolImpl->init(this, desc));
    returnComPtr(outPool, poolImpl);
    return SLANG_OK;
}

Result DeviceImpl::createFence(const IFence::Desc& desc, IFence** outFence)
{
    AUTORELEASEPOOL

    RefPtr<FenceImpl> fenceImpl = new FenceImpl();
    SLANG_RETURN_ON_FAIL(fenceImpl->init(this, desc));
    returnComPtr(outFence, fenceImpl);
    return SLANG_OK;
}

Result DeviceImpl::waitForFences(
    GfxCount fenceCount,
    IFence** fences,
    uint64_t* fenceValues,
    bool waitForAll,
    uint64_t timeout)
{
    // return SLANG_E_NOT_IMPLEMENTED;
    for (GfxCount i = 0; i < fenceCount; ++i)
    {
        FenceImpl* fenceImpl = static_cast<FenceImpl*>(fences[i]);
        if (!fenceImpl->waitForFence(fenceValues[i], timeout))
        {
            return SLANG_FAIL;
        }
    }
    return SLANG_OK;
}

} // namespace metal
} // namespace gfx
