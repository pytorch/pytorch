#pragma once

#define SLANG_GFX_FORWARD_RESOURCE_COMMAND_ENCODER_IMPL(ResourceCommandEncoderBase)              \
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(                               \
        SlangUUID const& uuid,                                                                   \
        void** outObject) override                                                               \
    {                                                                                            \
        return ResourceCommandEncoderBase::queryInterface(uuid, outObject);                      \
    }                                                                                            \
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override                                \
    {                                                                                            \
        return ResourceCommandEncoderBase::addRef();                                             \
    }                                                                                            \
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() override                               \
    {                                                                                            \
        return ResourceCommandEncoderBase::release();                                            \
    }                                                                                            \
    virtual SLANG_NO_THROW void SLANG_MCALL copyBuffer(                                          \
        IBufferResource* dst,                                                                    \
        Offset dstOffset,                                                                        \
        IBufferResource* src,                                                                    \
        Offset srcOffset,                                                                        \
        Size size) override                                                                      \
    {                                                                                            \
        ResourceCommandEncoderBase::copyBuffer(dst, dstOffset, src, srcOffset, size);            \
    }                                                                                            \
    virtual SLANG_NO_THROW void SLANG_MCALL copyTexture(                                         \
        ITextureResource* dst,                                                                   \
        ResourceState dstState,                                                                  \
        SubresourceRange dstSubresource,                                                         \
        ITextureResource::Offset3D dstOffset,                                                    \
        ITextureResource* src,                                                                   \
        ResourceState srcState,                                                                  \
        SubresourceRange srcSubresource,                                                         \
        ITextureResource::Offset3D srcOffset,                                                    \
        ITextureResource::Extents extent) override                                               \
    {                                                                                            \
        ResourceCommandEncoderBase::copyTexture(                                                 \
            dst,                                                                                 \
            dstState,                                                                            \
            dstSubresource,                                                                      \
            dstOffset,                                                                           \
            src,                                                                                 \
            srcState,                                                                            \
            srcSubresource,                                                                      \
            srcOffset,                                                                           \
            extent);                                                                             \
    }                                                                                            \
    virtual SLANG_NO_THROW void SLANG_MCALL copyTextureToBuffer(                                 \
        IBufferResource* dst,                                                                    \
        Offset dstOffset,                                                                        \
        Size dstSize,                                                                            \
        Size dstRowStride,                                                                       \
        ITextureResource* src,                                                                   \
        ResourceState srcState,                                                                  \
        SubresourceRange srcSubresource,                                                         \
        ITextureResource::Offset3D srcOffset,                                                    \
        ITextureResource::Extents extent) override                                               \
    {                                                                                            \
        ResourceCommandEncoderBase::copyTextureToBuffer(                                         \
            dst,                                                                                 \
            dstOffset,                                                                           \
            dstSize,                                                                             \
            dstRowStride,                                                                        \
            src,                                                                                 \
            srcState,                                                                            \
            srcSubresource,                                                                      \
            srcOffset,                                                                           \
            extent);                                                                             \
    }                                                                                            \
    virtual SLANG_NO_THROW void SLANG_MCALL uploadTextureData(                                   \
        ITextureResource* dst,                                                                   \
        SubresourceRange subResourceRange,                                                       \
        ITextureResource::Offset3D offset,                                                       \
        ITextureResource::Extents extent,                                                        \
        ITextureResource::SubresourceData* subResourceData,                                      \
        GfxCount subResourceDataCount) override                                                  \
    {                                                                                            \
        ResourceCommandEncoderBase::uploadTextureData(                                           \
            dst,                                                                                 \
            subResourceRange,                                                                    \
            offset,                                                                              \
            extent,                                                                              \
            subResourceData,                                                                     \
            subResourceDataCount);                                                               \
    }                                                                                            \
    virtual SLANG_NO_THROW void SLANG_MCALL                                                      \
    uploadBufferData(IBufferResource* dst, Offset offset, Size size, void* data) override        \
    {                                                                                            \
        ResourceCommandEncoderBase::uploadBufferData(dst, offset, size, data);                   \
    }                                                                                            \
    virtual SLANG_NO_THROW void SLANG_MCALL textureBarrier(                                      \
        GfxCount count,                                                                          \
        ITextureResource* const* textures,                                                       \
        ResourceState src,                                                                       \
        ResourceState dst) override                                                              \
    {                                                                                            \
        ResourceCommandEncoderBase::textureBarrier(count, textures, src, dst);                   \
    }                                                                                            \
    virtual SLANG_NO_THROW void SLANG_MCALL textureSubresourceBarrier(                           \
        ITextureResource* texture,                                                               \
        SubresourceRange subresourceRange,                                                       \
        ResourceState src,                                                                       \
        ResourceState dst) override                                                              \
    {                                                                                            \
        ResourceCommandEncoderBase::textureSubresourceBarrier(                                   \
            texture,                                                                             \
            subresourceRange,                                                                    \
            src,                                                                                 \
            dst);                                                                                \
    }                                                                                            \
    virtual SLANG_NO_THROW void SLANG_MCALL bufferBarrier(                                       \
        GfxCount count,                                                                          \
        IBufferResource* const* buffers,                                                         \
        ResourceState src,                                                                       \
        ResourceState dst) override                                                              \
    {                                                                                            \
        ResourceCommandEncoderBase::bufferBarrier(count, buffers, src, dst);                     \
    }                                                                                            \
    virtual SLANG_NO_THROW void SLANG_MCALL clearResourceView(                                   \
        IResourceView* view,                                                                     \
        ClearValue* clearValue,                                                                  \
        ClearResourceViewFlags::Enum flags) override                                             \
    {                                                                                            \
        ResourceCommandEncoderBase::clearResourceView(view, clearValue, flags);                  \
    }                                                                                            \
    virtual SLANG_NO_THROW void SLANG_MCALL resolveResource(                                     \
        ITextureResource* source,                                                                \
        ResourceState sourceState,                                                               \
        SubresourceRange sourceRange,                                                            \
        ITextureResource* dest,                                                                  \
        ResourceState destState,                                                                 \
        SubresourceRange destRange) override                                                     \
    {                                                                                            \
        ResourceCommandEncoderBase::resolveResource(                                             \
            source,                                                                              \
            sourceState,                                                                         \
            sourceRange,                                                                         \
            dest,                                                                                \
            destState,                                                                           \
            destRange);                                                                          \
    }                                                                                            \
    virtual SLANG_NO_THROW void SLANG_MCALL resolveQuery(                                        \
        IQueryPool* queryPool,                                                                   \
        GfxIndex index,                                                                          \
        GfxCount count,                                                                          \
        IBufferResource* buffer,                                                                 \
        Offset offset) override                                                                  \
    {                                                                                            \
        ResourceCommandEncoderBase::resolveQuery(queryPool, index, count, buffer, offset);       \
    }                                                                                            \
    virtual SLANG_NO_THROW void SLANG_MCALL writeTimestamp(IQueryPool* pool, GfxIndex index)     \
        override                                                                                 \
    {                                                                                            \
        ResourceCommandEncoderBase::writeTimestamp(pool, index);                                 \
    }                                                                                            \
    virtual SLANG_NO_THROW void SLANG_MCALL beginDebugEvent(const char* name, float rgbColor[3]) \
        override                                                                                 \
    {                                                                                            \
        ResourceCommandEncoderBase::beginDebugEvent(name, rgbColor);                             \
    }                                                                                            \
    virtual SLANG_NO_THROW void SLANG_MCALL endDebugEvent() override                             \
    {                                                                                            \
        ResourceCommandEncoderBase::endDebugEvent();                                             \
    }
