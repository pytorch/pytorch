// model.cpp
#include "model.h"

#include "window.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "../../external/tinyobjloader/tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../external/glm/glm/glm.hpp"
#include "../../external/glm/glm/gtc/constants.hpp"
#include "../../external/glm/glm/gtc/matrix_transform.hpp"
#include "stb_image_resize.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace platform
{

using namespace gfx;
using namespace Slang;

// TinyObj provides a tuple type that bundles up indices, but doesn't
// provide equality comparison or hashing for that type. We'd like
// to have a hash function so that we can unique indices.
//
// In the simplest case, we could define hashing and operator== operations
// directly on `tinobj::index_t`, but that would create problems if they
// revise their API.
//
// We will instead define our own wrapper type that supports equality
// comparisons.
//
struct ObjIndexKey
{
    tinyobj::index_t index;
};

bool operator==(ObjIndexKey const& left, ObjIndexKey const& right)
{
    return left.index.vertex_index == right.index.vertex_index &&
           left.index.normal_index == right.index.normal_index &&
           left.index.texcoord_index == right.index.texcoord_index;
}

struct Hasher
{
    template<typename T>
    void add(T const& v)
    {
        state ^= std::hash<T>()(v) + 0x9e3779b9 + (state << 6) + (state >> 2);
    }
    size_t state = 0;
};

struct SmoothingGroupVertexID
{
    size_t smoothingGroup;
    size_t positionID;
};
bool operator==(SmoothingGroupVertexID const& left, SmoothingGroupVertexID const& right)
{
    return left.smoothingGroup == right.smoothingGroup && left.positionID == right.positionID;
}

} // namespace platform

namespace std
{
template<>
struct hash<platform::ObjIndexKey>
{
    size_t operator()(platform::ObjIndexKey const& key) const
    {
        platform::Hasher hasher;
        hasher.add(key.index.vertex_index);
        hasher.add(key.index.normal_index);
        hasher.add(key.index.texcoord_index);
        return hasher.state;
    }
};

template<>
struct hash<platform::SmoothingGroupVertexID>
{
    size_t operator()(platform::SmoothingGroupVertexID const& id) const
    {
        platform::Hasher hasher;
        hasher.add(id.smoothingGroup);
        hasher.add(id.positionID);
        return hasher.state;
    }
};
} // namespace std

namespace platform
{

ComPtr<ITextureResource> loadTextureImage(IDevice* device, char const* path)
{
    int extentX = 0;
    int extentY = 0;
    int originalChannelCount = 0;
    int requestedChannelCount = 4; // force to 4-component result
    stbi_uc* data =
        stbi_load(path, &extentX, &extentY, &originalChannelCount, requestedChannelCount);
    if (!data)
        return nullptr;

    int channelCount = requestedChannelCount ? requestedChannelCount : originalChannelCount;

    Format format;
    switch (channelCount)
    {
    default:
        return nullptr;

    case 4:
        format = Format::R8G8B8A8_UNORM;

        // TODO: handle other cases here if/when we stop forcing 4-component
        // results when loading the image with stb_image.
    }

    std::vector<ITextureResource::SubresourceData> subresourceInitData;

    ptrdiff_t stride = extentX * channelCount * sizeof(stbi_uc);

    ITextureResource::SubresourceData baseInitData;
    baseInitData.data = data;
    baseInitData.strideY = stride;
    baseInitData.strideZ = 0;

    subresourceInitData.push_back(baseInitData);

    // create down-sampled images for the different mip levels
    bool generateMips = true;
    if (generateMips)
    {
        int prevExtentX = extentX;
        int prevExtentY = extentY;
        stbi_uc* prevData = data;
        int prevStride = int(stride);

        for (;;)
        {
            if (prevExtentX == 1 && prevExtentY == 1)
                break;

            int newExtentX = prevExtentX / 2;
            int newExtentY = prevExtentY / 2;

            if (!newExtentX)
                newExtentX = 1;
            if (!newExtentY)
                newExtentY = 1;

            stbi_uc* newData =
                (stbi_uc*)malloc(newExtentX * newExtentY * channelCount * sizeof(stbi_uc));
            int newStride = int(newExtentX * channelCount * sizeof(stbi_uc));

            stbir_resize_uint8_srgb(
                prevData,
                prevExtentX,
                prevExtentY,
                prevStride,
                newData,
                newExtentX,
                newExtentY,
                newStride,
                channelCount,
                STBIR_ALPHA_CHANNEL_NONE,
                STBIR_FLAG_ALPHA_PREMULTIPLIED);


            ITextureResource::SubresourceData mipInitData;
            mipInitData.data = newData;
            mipInitData.strideY = newStride;
            mipInitData.strideZ = 0;

            subresourceInitData.push_back(mipInitData);

            prevExtentX = newExtentX;
            prevExtentY = newExtentY;
            prevData = newData;
            prevStride = newStride;
        }
    }

    int mipCount = (int)subresourceInitData.size();

    ITextureResource::Desc desc = {};
    desc.type = IResource::Type::Texture2D;
    desc.defaultState = ResourceState::ShaderResource;
    desc.allowedStates = ResourceStateSet(ResourceState::ShaderResource);
    desc.format = format;
    desc.size.width = extentX;
    desc.size.height = extentY;
    desc.size.depth = 1;
    desc.numMipLevels = mipCount;
    auto texture = device->createTextureResource(desc, subresourceInitData.data());
    free(data);

    return texture;
}

static std::string makeString(const char* start, const char* end)
{
    return std::string(start, size_t(end - start));
}

SlangResult ModelLoader::load(char const* inputPath, void** outModel)
{
    // TODO: need to actually allocate/load the data

    tinyobj::attrib_t objVertexAttributes;
    std::vector<tinyobj::shape_t> objShapes;
    std::vector<tinyobj::material_t> objMaterials;

    std::string baseDir;
    if (auto lastSlash = strrchr(inputPath, '/'))
    {
        baseDir = makeString(inputPath, lastSlash);
    }

    std::string diagnostics;
    bool shouldTriangulate = true;
    bool success = tinyobj::LoadObj(
        &objVertexAttributes,
        &objShapes,
        &objMaterials,
        &diagnostics,
        inputPath,
        baseDir.size() ? baseDir.c_str() : nullptr,
        shouldTriangulate);

    if (!diagnostics.empty())
    {
        printf("%s", diagnostics.c_str());
    }
    if (!success)
    {
        return SLANG_FAIL;
    }

    // Translate each material imported by TinyObj into a format that
    // we can actually use for rendering.
    //
    std::vector<void*> materials;
    for (auto& objMaterial : objMaterials)
    {
        MaterialData materialData;

        materialData.diffuseColor =
            glm::vec3(objMaterial.diffuse[0], objMaterial.diffuse[1], objMaterial.diffuse[2]);

        materialData.specularColor =
            glm::vec3(objMaterial.specular[0], objMaterial.specular[1], objMaterial.specular[2]);

        materialData.specularity = objMaterial.shininess;

        // load any referenced textures here
        if (objMaterial.diffuse_texname.length())
        {
            materialData.diffuseMap = loadTextureImage(device, objMaterial.diffuse_texname.c_str());
        }

        auto material = callbacks->createMaterial(materialData);
        materials.push_back(material);
    }

    // Flip the winding order on all faces if we are asked to...
    //
    if (loadFlags & LoadFlag::FlipWinding)
    {
        for (auto& objShape : objShapes)
        {
            size_t objIndexCounter = 0;
            size_t objFaceCounter = 0;
            for (auto objFaceVertexCount : objShape.mesh.num_face_vertices)
            {
                size_t beginIndex = objIndexCounter;
                size_t endIndex = beginIndex + objFaceVertexCount;
                objIndexCounter = endIndex;

                size_t halfCount = objFaceVertexCount / 2;
                for (size_t ii = 0; ii < halfCount; ++ii)
                {
                    std::swap(
                        objShape.mesh.indices[beginIndex + ii],
                        objShape.mesh.indices[endIndex - (ii + 1)]);
                }
            }
        }
    }

    // Identify cases where a face has a vertex without a normal, and in that
    // case remember that the given vertex needs to be "smoothed" as part of
    // the smoothing group for that face. Note that it is possible for the
    // same vertex (position) to be part of faces in distinct smoothing groups.
    //
    std::unordered_map<SmoothingGroupVertexID, size_t> smoothedVertexNormals;
    size_t firstSmoothedNormalID = objVertexAttributes.normals.size() / 3;
    size_t flatFaceCounter = 0;
    for (auto& objShape : objShapes)
    {
        size_t objIndexCounter = 0;
        size_t objFaceCounter = 0;
        for (auto objFaceVertexCount : objShape.mesh.num_face_vertices)
        {
            const size_t flatFaceIndex = flatFaceCounter++;
            const size_t objFaceIndex = objFaceCounter++;
            size_t smoothingGroup = objShape.mesh.smoothing_group_ids[objFaceIndex];
            if (!smoothingGroup)
            {
                smoothingGroup = ~flatFaceIndex;
            }

            for (size_t objFaceVertex = 0; objFaceVertex < objFaceVertexCount; ++objFaceVertex)
            {
                tinyobj::index_t& objIndex = objShape.mesh.indices[objIndexCounter++];

                if (objIndex.normal_index < 0)
                {
                    SmoothingGroupVertexID smoothVertexID;
                    smoothVertexID.positionID = objIndex.vertex_index;
                    smoothVertexID.smoothingGroup = smoothingGroup;

                    if (smoothedVertexNormals.find(smoothVertexID) == smoothedVertexNormals.end())
                    {
                        size_t normalID = objVertexAttributes.normals.size() / 3;
                        objVertexAttributes.normals.push_back(0);
                        objVertexAttributes.normals.push_back(0);
                        objVertexAttributes.normals.push_back(0);

                        smoothedVertexNormals.insert(std::make_pair(smoothVertexID, normalID));

                        objIndex.normal_index = int(normalID);
                    }
                }
            }
        }
    }
    //
    // Having identified which vertices we need to smooth, we will make another
    // pass to compute face normals and apply them to the vertices that belong
    // to the same smoothing group.
    //
    flatFaceCounter = 0;
    for (auto& objShape : objShapes)
    {
        size_t objIndexCounter = 0;
        size_t objFaceCounter = 0;
        for (auto objFaceVertexCount : objShape.mesh.num_face_vertices)
        {
            const size_t flatFaceIndex = flatFaceCounter++;
            const size_t objFaceIndex = objFaceCounter++;
            size_t smoothingGroup = objShape.mesh.smoothing_group_ids[objFaceIndex];
            if (!smoothingGroup)
            {
                smoothingGroup = ~flatFaceIndex;
            }

            glm::vec3 faceNormal;
            if (objFaceVertexCount >= 3)
            {
                glm::vec3 v[3];
                for (size_t objFaceVertex = 0; objFaceVertex < 3; ++objFaceVertex)
                {
                    tinyobj::index_t objIndex =
                        objShape.mesh.indices[objIndexCounter + objFaceVertex];
                    if (objIndex.vertex_index >= 0)
                    {
                        v[objFaceVertex] = glm::vec3(
                            objVertexAttributes.vertices[3 * objIndex.vertex_index + 0],
                            objVertexAttributes.vertices[3 * objIndex.vertex_index + 1],
                            objVertexAttributes.vertices[3 * objIndex.vertex_index + 2]);
                    }
                }
                faceNormal = cross(v[1] - v[0], v[2] - v[0]);
            }

            // Add this face normal to any to-be-smoothed vertex on the face.
            for (size_t objFaceVertex = 0; objFaceVertex < objFaceVertexCount; ++objFaceVertex)
            {
                tinyobj::index_t objIndex = objShape.mesh.indices[objIndexCounter++];

                SmoothingGroupVertexID smoothVertexID;
                smoothVertexID.positionID = objIndex.vertex_index;
                smoothVertexID.smoothingGroup = smoothingGroup;

                auto ii = smoothedVertexNormals.find(smoothVertexID);
                if (ii != smoothedVertexNormals.end())
                {
                    size_t normalID = ii->second;
                    objVertexAttributes.normals[normalID * 3 + 0] += faceNormal.x;
                    objVertexAttributes.normals[normalID * 3 + 1] += faceNormal.y;
                    objVertexAttributes.normals[normalID * 3 + 2] += faceNormal.z;
                }
            }
        }
    }
    //
    // Once we've added all contributions from each smoothing group,
    // we can normalize the normals to compute the area-weighted average.
    //
    size_t normalCount = objVertexAttributes.normals.size() / 3;
    for (size_t ii = firstSmoothedNormalID; ii < normalCount; ++ii)
    {
        glm::vec3 normal = glm::vec3(
            objVertexAttributes.normals[3 * ii + 0],
            objVertexAttributes.normals[3 * ii + 1],
            objVertexAttributes.normals[3 * ii + 2]);

        normal = normalize(normal);

        objVertexAttributes.normals[3 * ii + 0] = normal.x;
        objVertexAttributes.normals[3 * ii + 1] = normal.y;
        objVertexAttributes.normals[3 * ii + 2] = normal.z;
    }

    // TODO: we should sort the faces to group faces with
    // the same material ID together, in case they weren't
    // grouped in the original file.

    // We need to undo the .obj indexing stuff so that we have
    // standard position/normal/etc. data in a single flat array

    std::unordered_map<ObjIndexKey, Index> mapObjIndexToFlatIndex;
    std::vector<Vertex> flatVertices;
    std::vector<Index> flatIndices;

    MeshData* currentMesh = nullptr;
    MeshData currentMeshStorage;

    std::vector<void*> meshes;

    void* defaultMaterial = nullptr;

    for (auto& objShape : objShapes)
    {
        size_t objIndexCounter = 0;
        size_t objFaceCounter = 0;
        for (auto objFaceVertexCount : objShape.mesh.num_face_vertices)
        {
            size_t objFaceIndex = objFaceCounter++;
            int faceMaterialID = objShape.mesh.material_ids[objFaceIndex];
            void* faceMaterial = nullptr;
            if (faceMaterialID < 0)
            {
                if (!defaultMaterial)
                {
                    MaterialData defaultMaterialData;
                    defaultMaterialData.diffuseColor = glm::vec3(0.5, 0.5, 0.5);
                    defaultMaterial = callbacks->createMaterial(defaultMaterialData);
                }
                faceMaterial = defaultMaterial;
            }
            else
            {
                faceMaterial = materials[faceMaterialID];
            }

            if (!currentMesh || (faceMaterial != currentMesh->material))
            {
                // finish old mesh.
                if (currentMesh)
                {
                    meshes.push_back(callbacks->createMesh(*currentMesh));
                }

                // Need to start a new mesh.
                currentMesh = &currentMeshStorage;
                currentMesh->material = faceMaterial;
                currentMesh->firstIndex = (int)flatIndices.size();
                currentMesh->indexCount = 0;
            }

            for (size_t objFaceVertex = 0; objFaceVertex < objFaceVertexCount; ++objFaceVertex)
            {
                tinyobj::index_t objIndex = objShape.mesh.indices[objIndexCounter++];
                ObjIndexKey objIndexKey;
                objIndexKey.index = objIndex;


                Index flatIndex = Index(-1);
                auto iter = mapObjIndexToFlatIndex.find(objIndexKey);
                if (iter != mapObjIndexToFlatIndex.end())
                {
                    flatIndex = iter->second;
                }
                else
                {
                    Vertex flatVertex;
                    if (objIndex.vertex_index >= 0)
                    {
                        flatVertex.position =
                            scale *
                            glm::vec3(
                                objVertexAttributes.vertices[3 * objIndex.vertex_index + 0],
                                objVertexAttributes.vertices[3 * objIndex.vertex_index + 1],
                                objVertexAttributes.vertices[3 * objIndex.vertex_index + 2]);
                    }
                    if (objIndex.normal_index >= 0)
                    {
                        flatVertex.normal = glm::vec3(
                            objVertexAttributes.normals[3 * objIndex.normal_index + 0],
                            objVertexAttributes.normals[3 * objIndex.normal_index + 1],
                            objVertexAttributes.normals[3 * objIndex.normal_index + 2]);
                    }
                    if (objIndex.texcoord_index >= 0)
                    {
                        flatVertex.uv = glm::vec2(
                            objVertexAttributes.texcoords[2 * objIndex.texcoord_index + 0],
                            objVertexAttributes.texcoords[2 * objIndex.texcoord_index + 1]);
                    }

                    flatIndex = uint32_t(flatVertices.size());
                    mapObjIndexToFlatIndex.insert(std::make_pair(objIndexKey, flatIndex));
                    flatVertices.push_back(flatVertex);
                }

                flatIndices.push_back(flatIndex);
                currentMesh->indexCount++;
            }
        }
    }

    // finish last mesh.
    if (currentMesh)
    {
        meshes.push_back(callbacks->createMesh(*currentMesh));
    }

    ModelData modelData;

    modelData.vertexCount = (int)flatVertices.size();
    modelData.indexCount = (int)flatIndices.size();

    modelData.meshCount = int(meshes.size());
    modelData.meshes = meshes.data();

    IBufferResource::Desc vertexBufferDesc;
    vertexBufferDesc.type = IResource::Type::Buffer;
    vertexBufferDesc.sizeInBytes = modelData.vertexCount * sizeof(Vertex);
    vertexBufferDesc.allowedStates =
        ResourceStateSet(ResourceState::VertexBuffer, ResourceState::CopyDestination);
    vertexBufferDesc.defaultState = ResourceState::VertexBuffer;

    modelData.vertexBuffer = device->createBufferResource(vertexBufferDesc, flatVertices.data());
    if (!modelData.vertexBuffer)
        return SLANG_FAIL;

    IBufferResource::Desc indexBufferDesc;
    indexBufferDesc.type = IResource::Type::Buffer;
    indexBufferDesc.sizeInBytes = modelData.indexCount * sizeof(Index);
    indexBufferDesc.allowedStates =
        ResourceStateSet(ResourceState::IndexBuffer, ResourceState::CopyDestination);
    indexBufferDesc.defaultState = ResourceState::IndexBuffer;

    modelData.indexBuffer = device->createBufferResource(indexBufferDesc, flatIndices.data());
    if (!modelData.indexBuffer)
        return SLANG_FAIL;

    *outModel = callbacks->createModel(modelData);

    return SLANG_OK;
}

} // namespace platform
