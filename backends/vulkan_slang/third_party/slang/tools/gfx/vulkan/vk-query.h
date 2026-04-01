// vk-query.h
#pragma once

#include "vk-base.h"
#include "vk-device.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

class QueryPoolImpl : public QueryPoolBase
{
public:
    Result init(const IQueryPool::Desc& desc, DeviceImpl* device);
    ~QueryPoolImpl();

public:
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getResult(GfxIndex index, GfxCount count, uint64_t* data) override;

public:
    VkQueryPool m_pool;
    RefPtr<DeviceImpl> m_device;
};

void _writeTimestamp(
    VulkanApi* api,
    VkCommandBuffer vkCmdBuffer,
    IQueryPool* queryPool,
    SlangInt index);

} // namespace vk
} // namespace gfx
