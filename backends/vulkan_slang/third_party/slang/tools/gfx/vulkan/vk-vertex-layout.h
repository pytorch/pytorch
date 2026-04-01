// vk-vertex-layout.h
#pragma once

#include "vk-base.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

class InputLayoutImpl : public InputLayoutBase
{
public:
    List<VkVertexInputAttributeDescription> m_attributeDescs;
    List<VkVertexInputBindingDescription> m_streamDescs;
};

} // namespace vk
} // namespace gfx
