// png-serialize-util.cpp
#define _CRT_SECURE_NO_WARNINGS

#include "png-serialize-util.h"

#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace renderer_test
{
using namespace Slang;

/* static */ Slang::Result PngSerializeUtil::write(
    const char* filename,
    ISlangBlob* pixels,
    uint32_t width,
    uint32_t height)
{
    int stbResult =
        stbi_write_png(filename, width, height, 4, pixels->getBufferPointer(), width * 4);

    return stbResult ? SLANG_OK : SLANG_FAIL;
}

} // namespace renderer_test
