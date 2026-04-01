// shader-renderer-util.h
#pragma once

#include "shader-input-layout.h"

#include <slang-rhi.h>

namespace renderer_test
{

using namespace Slang;

ComPtr<ISampler> _createSampler(IDevice* device, const InputSamplerDesc& srcDesc);

/// Utility class containing functions that construct items on the renderer using the
/// ShaderInputLayout representation
struct ShaderRendererUtil
{
    /// Generate a texture using the InputTextureDesc and construct a Texture using the Renderer
    /// with the contents
    static Slang::Result generateTexture(
        const InputTextureDesc& inputDesc,
        ResourceState defaultState,
        IDevice* device,
        ComPtr<ITexture>& textureOut);

    /// Create texture resource using inputDesc, and texData to describe format, and contents
    static Slang::Result createTexture(
        const InputTextureDesc& inputDesc,
        const TextureData& texData,
        ResourceState defaultState,
        IDevice* device,
        ComPtr<ITexture>& textureOut);

    /// Create the BufferResource using the renderer from the contents of inputDesc
    static Slang::Result createBuffer(
        const InputBufferDesc& inputDesc,
        size_t bufferSize,
        const void* initData,
        IDevice* device,
        ComPtr<IBuffer>& bufferOut);
};

} // namespace renderer_test
