// model.h
#pragma once

#include "platform-api.h"
#include "slang-com-ptr.h"
#include "slang-gfx.h"
#include "vector-math.h"

#include <string>
#include <vector>

namespace platform
{

struct ModelLoader
{
    struct MaterialData
    {
        glm::vec3 diffuseColor;
        glm::vec3 specularColor;
        float specularity;

        Slang::ComPtr<gfx::ITextureResource> diffuseMap;
    };

    struct Vertex
    {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
    };

    typedef uint32_t Index;

    struct MeshData
    {
        int firstIndex;
        int indexCount;

        void* material;
    };

    struct ModelData
    {
        Slang::ComPtr<gfx::IBufferResource> vertexBuffer;
        Slang::ComPtr<gfx::IBufferResource> indexBuffer;
        gfx::PrimitiveTopology primitiveTopology;
        int vertexCount;
        int indexCount;
        int meshCount;
        void* const* meshes;
    };

    struct ICallbacks
    {
        typedef ModelLoader::MaterialData MaterialData;
        typedef ModelLoader::MeshData MeshData;
        typedef ModelLoader::ModelData ModelData;

        virtual void* createMaterial(MaterialData const& data) = 0;
        virtual void* createMesh(MeshData const& data) = 0;
        virtual void* createModel(ModelData const& data) = 0;
    };

    typedef uint32_t LoadFlags;
    enum LoadFlag : LoadFlags
    {
        FlipWinding = 1 << 0,
    };

    ICallbacks* callbacks = nullptr;
    gfx::IDevice* device;
    LoadFlags loadFlags = 0;
    float scale = 1.0f;

    SLANG_PLATFORM_API SlangResult load(char const* inputPath, void** outModel);
};


} // namespace platform
