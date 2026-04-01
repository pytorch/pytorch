// vk-api.cpp
#include "core/slang-string.h"
#include "vk-api.h"

namespace gfx
{

void installPipelineDumpLayer(VulkanApi& api);
void writePipelineDump(Slang::UnownedStringSlice path);

} // namespace gfx
