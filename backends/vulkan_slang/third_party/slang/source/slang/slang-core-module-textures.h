#pragma once

#include "../core/slang-string.h"
#include "slang-ir.h"
#include "slang-type-system-shared.h"

namespace Slang
{

static const struct BaseTextureShapeInfo
{
    char const* shapeName;
    SlangResourceShape baseShape;
    int coordCount;
} kBaseTextureShapes[] = {
    {"1D", SLANG_TEXTURE_1D, 1},
    {"2D", SLANG_TEXTURE_2D, 2},
    {"3D", SLANG_TEXTURE_3D, 3},
    {"Cube", SLANG_TEXTURE_CUBE, 3},
};

static const struct BaseTextureAccessInfo
{
    char const* name;
    SlangResourceAccess access;
} kBaseTextureAccessLevels[] = {
    {"", SLANG_RESOURCE_ACCESS_READ},
    {"RW", SLANG_RESOURCE_ACCESS_READ_WRITE},
    {"RasterizerOrdered", SLANG_RESOURCE_ACCESS_RASTER_ORDERED},
    {"Feedback", SLANG_RESOURCE_ACCESS_FEEDBACK},
};

struct TextureTypeInfo
{
    TextureTypeInfo(
        BaseTextureShapeInfo const& base,
        bool isArray,
        bool isMultisample,
        bool isShadow,
        StringBuilder& inSB,
        String const& inPath);

    BaseTextureShapeInfo const& base;
    bool isArray;
    bool isMultisample;
    bool isShadow;
    StringBuilder& sb;
    String path;

    void emitTypeDecl();

public:
    //
    // Functions for writing specific parts of a definition
    //
    void writeGetDimensionFunctions();

    //
    // More general utilities
    //
    enum class ReadNoneMode
    {
        Never,
        Always
    };

    void writeFuncBody(
        const char* funcName,
        const String& glsl,
        const String& cuda,
        const String& spirvDefault,
        const String& spirvRWDefault,
        const String& spirvCombined,
        const String& metal,
        const String& wgsl);
    void writeFuncWithSig(
        const char* funcName,
        const String& sig,
        const String& glsl = String{},
        const String& spirvDefault = String{},
        const String& spirvRWDefault = String{},
        const String& spirvCombined = String{},
        const String& cuda = String{},
        const String& metal = String{},
        const String& wgsl = String{},
        const ReadNoneMode readNoneMode = ReadNoneMode::Never);
    void writeFunc(
        const char* returnType,
        const char* funcName,
        const String& params,
        const String& glsl = String{},
        const String& spirvDefault = String{},
        const String& spirvRWDefault = String{},
        const String& spirvCombined = String{},
        const String& cuda = String{},
        const String& metal = String{},
        const String& wgsl = String{},
        const ReadNoneMode readNoneMode = ReadNoneMode::Never);

    // A pointer to a string representing the current level of indentation
    const char* i;
};

} // namespace Slang
