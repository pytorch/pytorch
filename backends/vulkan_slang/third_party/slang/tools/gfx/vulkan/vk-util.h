// vk-util.h
#pragma once

#include "core/slang-basic.h"
#include "slang-gfx.h"
#include "vk-api.h"

// Macros to make testing vulkan return codes simpler

/// SLANG_VK_RETURN_ON_FAIL can be used in a similar way to SLANG_RETURN_ON_FAIL macro, except it
/// will turn a vulkan failure into Slang::Result in the process Calls handleFail which on debug
/// builds asserts
#define SLANG_VK_RETURN_ON_FAIL(x)               \
    {                                            \
        VkResult _res = x;                       \
        if (_res != VK_SUCCESS)                  \
        {                                        \
            return VulkanUtil::handleFail(_res); \
        }                                        \
    }

#define SLANG_VK_RETURN_NULL_ON_FAIL(x)   \
    {                                     \
        VkResult _res = x;                \
        if (_res != VK_SUCCESS)           \
        {                                 \
            VulkanUtil::handleFail(_res); \
            return nullptr;               \
        }                                 \
    }

/// Is similar to SLANG_VK_RETURN_ON_FAIL, but does not return. Will call checkFail on failure -
/// which asserts on debug builds.
#define SLANG_VK_CHECK(x)                \
    {                                    \
        VkResult _res = x;               \
        if (_res != VK_SUCCESS)          \
        {                                \
            VulkanUtil::checkFail(_res); \
        }                                \
    }

namespace gfx
{

// Utility functions for Vulkan
struct VulkanUtil
{
    /// Get the equivalent VkFormat from the format
    /// Returns VK_FORMAT_UNDEFINED if a match is not found
    static VkFormat getVkFormat(Format format);

    static VkImageAspectFlags getAspectMask(TextureAspect aspect, VkFormat format);

    /// Called by SLANG_VK_RETURN_FAIL if a res is a failure.
    /// On debug builds this will cause an assertion on failure.
    static Slang::Result handleFail(VkResult res);
    /// Called when a failure has occurred with SLANG_VK_CHECK - will typically assert.
    static void checkFail(VkResult res);

    /// Get the VkPrimitiveTopology for the given topology.
    /// Returns VK_PRIMITIVE_TOPOLOGY_MAX_ENUM on failure
    static VkPrimitiveTopology getVkPrimitiveTopology(PrimitiveTopology topology);

    static VkImageLayout mapResourceStateToLayout(ResourceState state);

    /// Returns Slang::Result equivalent of a VkResult
    static Slang::Result toSlangResult(VkResult res);

    static VkShaderStageFlags getShaderStage(SlangStage stage);

    static VkImageLayout getImageLayoutFromState(ResourceState state);

    /// Calculate size taking into account alignment. Alignment must be a power of 2
    static UInt calcAligned(UInt size, UInt alignment)
    {
        return (size + alignment - 1) & ~(alignment - 1);
    }

    static inline bool isDepthFormat(VkFormat format)
    {
        switch (format)
        {
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return true;
        }
        return false;
    }

    static inline bool isStencilFormat(VkFormat format)
    {
        switch (format)
        {
        case VK_FORMAT_S8_UINT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return true;
        }
        return false;
    }

    static VkSampleCountFlagBits translateSampleCount(uint32_t sampleCount);

    static VkCullModeFlags translateCullMode(CullMode cullMode);

    static VkFrontFace translateFrontFaceMode(FrontFaceMode frontFaceMode);

    static VkPolygonMode translateFillMode(FillMode fillMode);

    static VkBlendFactor translateBlendFactor(BlendFactor blendFactor);

    static VkBlendOp translateBlendOp(BlendOp op);

    static VkPrimitiveTopology translatePrimitiveTypeToListTopology(PrimitiveType primitiveType);

    static VkStencilOp translateStencilOp(StencilOp op);

    static VkFilter translateFilterMode(TextureFilteringMode mode);

    static VkSamplerMipmapMode translateMipFilterMode(TextureFilteringMode mode);

    static VkSamplerAddressMode translateAddressingMode(TextureAddressingMode mode);

    static VkCompareOp translateComparisonFunc(ComparisonFunc func);

    static VkStencilOpState translateStencilState(DepthStencilOpDesc desc);

    static VkSamplerReductionMode translateReductionOp(TextureReductionOp op);

    static CooperativeVectorComponentType translateCooperativeVectorComponentType(
        VkComponentTypeKHR type);
};

struct AccelerationStructureBuildGeometryInfoBuilder
{
public:
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    Slang::List<uint32_t> primitiveCounts;

    Slang::Result build(
        const IAccelerationStructure::BuildInputs& buildInputs,
        IDebugCallback* debugCallback);

private:
    Slang::List<VkAccelerationStructureGeometryKHR> m_geometryInfos;
    VkAccelerationStructureGeometryKHR m_vkInstanceInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
};


} // namespace gfx
