// metal-util.h
#pragma once

#include "core/slang-basic.h"
#include "metal-api.h"
#include "slang-gfx.h"

namespace gfx
{

// Utility functions for Metal
struct MetalUtil
{
    static NS::SharedPtr<NS::String> createString(
        const char* str,
        NS::StringEncoding encoding = NS::UTF8StringEncoding)
    {
        NS::SharedPtr<NS::String> nsString =
            NS::TransferPtr(NS::String::alloc()->init(str, encoding));
        return nsString;
    }

    static NS::SharedPtr<NS::String> createStringView(
        void* bytes,
        size_t len,
        NS::StringEncoding encoding = NS::UTF8StringEncoding)
    {
        NS::SharedPtr<NS::String> nsString =
            NS::TransferPtr(NS::String::alloc()->init(bytes, len, encoding, false));
        return nsString;
    }

    static MTL::PixelFormat translatePixelFormat(Format format);
    static MTL::VertexFormat translateVertexFormat(Format format);

    static bool isDepthFormat(MTL::PixelFormat format);
    static bool isStencilFormat(MTL::PixelFormat format);

    static MTL::SamplerMinMagFilter translateSamplerMinMagFilter(TextureFilteringMode mode);
    static MTL::SamplerMipFilter translateSamplerMipFilter(TextureFilteringMode mode);
    static MTL::SamplerAddressMode translateSamplerAddressMode(TextureAddressingMode mode);
    static MTL::CompareFunction translateCompareFunction(ComparisonFunc func);
    static MTL::StencilOperation translateStencilOperation(StencilOp op);

    static MTL::VertexStepFunction translateVertexStepFunction(InputSlotClass slotClass);

    static MTL::PrimitiveType translatePrimitiveType(PrimitiveTopology topology);
    static MTL::PrimitiveTopologyClass translatePrimitiveTopologyClass(PrimitiveType type);

    static MTL::BlendFactor translateBlendFactor(BlendFactor factor);
    static MTL::BlendOperation translateBlendOperation(BlendOp op);
    static MTL::ColorWriteMask translateColorWriteMask(RenderTargetWriteMask::Type mask);

    static MTL::Winding translateWinding(FrontFaceMode mode);
    static MTL::CullMode translateCullMode(CullMode mode);
    static MTL::TriangleFillMode translateTriangleFillMode(FillMode mode);
};

struct ScopedAutoreleasePool
{
    ScopedAutoreleasePool() { m_pool = NS::AutoreleasePool::alloc()->init(); }
    ~ScopedAutoreleasePool() { m_pool->drain(); }
    NS::AutoreleasePool* m_pool;
};

#define AUTORELEASEPOOL ScopedAutoreleasePool _pool_;

} // namespace gfx
