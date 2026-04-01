// png-serialize-util.h
#pragma once

#include "core/slang-blob.h"

namespace renderer_test
{

struct PngSerializeUtil
{
    static Slang::Result write(
        const char* filename,
        ISlangBlob* pixels,
        uint32_t width,
        uint32_t height);
};

} // namespace renderer_test
