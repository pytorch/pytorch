#pragma once

#include "slang-gfx.h"

namespace gfx
{

/// Represents a "pointer" to the storage for a shader parameter of a (dynamically) known type.
///
/// A `ShaderCursor` serves as a pointer-like type for things stored inside a `ShaderObject`.
///
/// A cursor that points to the entire content of a shader object can be formed as
/// `ShaderCursor(someObject)`. A cursor pointing to a structure field or array element can be
/// formed from another cursor using `getField` or `getElement` respectively.
///
/// Given a cursor pointing to a value of some "primitive" type, we can set or get the value
/// using operations like `setResource`, `getResource`, etc.
///
/// Because type information for shader parameters is being reflected dynamically, all type
/// checking for shader cursors occurs at runtime, and errors may occur when attempting to
/// set a parameter using a value of an inappropriate type. As much as possible, `ShaderCursor`
/// attempts to protect against these cases and return an error `Result` or an invalid
/// cursor, rather than allowing operations to proceed with incorrect types.
///
struct ShaderCursor
{
    IShaderObject* m_baseObject = nullptr;
    slang::TypeLayoutReflection* m_typeLayout = nullptr;
    ShaderObjectContainerType m_containerType = ShaderObjectContainerType::None;
    ShaderOffset m_offset;

    /// Get the type (layout) of the value being pointed at by the cursor
    slang::TypeLayoutReflection* getTypeLayout() const { return m_typeLayout; }

    /// Is this cursor valid (that is, does it seem to point to an actual location)?
    ///
    /// This check is equivalent to checking whether a pointer is null, so it is
    /// a very weak sense of "valid." In particular, it is possible to form a
    /// `ShaderCursor` for which `isValid()` is true, but attempting to get or
    /// set the value would be an error (like dereferencing a garbage pointer).
    ///
    bool isValid() const { return m_baseObject != nullptr; }

    Result getDereferenced(ShaderCursor& outCursor) const;

    ShaderCursor getDereferenced() const
    {
        ShaderCursor result;
        getDereferenced(result);
        return result;
    }

    /// Form a cursor pointing to the field with the given `name` within the value this cursor
    /// points at.
    ///
    /// If the operation succeeds, then the field cursor is written to `outCursor`.
    Result getField(const char* nameBegin, const char* nameEnd, ShaderCursor& outCursor) const;

    ShaderCursor getField(const char* name) const
    {
        ShaderCursor cursor;
        getField(name, nullptr, cursor);
        return cursor;
    }

    /// Some resources such as RWStructuredBuffer, AppendStructuredBuffer and
    /// ConsumeStructuredBuffer need to have their counter explicitly bound on
    /// APIs other than DirectX, this will return a valid ShaderCursor pointing
    /// to that resource if that is the case.
    /// Otherwise, this returns an invalid cursor.
    ShaderCursor getExplicitCounter() const;

    ShaderCursor getElement(GfxIndex index) const;

    static Result followPath(const char* path, ShaderCursor& ioCursor);

    ShaderCursor getPath(const char* path) const
    {
        ShaderCursor result(*this);
        followPath(path, result);
        return result;
    }

    ShaderCursor() {}

    ShaderCursor(IShaderObject* object)
        : m_baseObject(object)
        , m_typeLayout(object->getElementTypeLayout())
        , m_containerType(object->getContainerType())
    {
    }

    SlangResult setData(void const* data, Size size) const
    {
        return m_baseObject->setData(m_offset, data, size);
    }

    template<typename T>
    SlangResult setData(T const& data) const
    {
        return setData(&data, sizeof(data));
    }

    SlangResult setObject(IShaderObject* object) const
    {
        return m_baseObject->setObject(m_offset, object);
    }

    SlangResult setSpecializationArgs(const slang::SpecializationArg* args, GfxCount count) const
    {
        return m_baseObject->setSpecializationArgs(m_offset, args, count);
    }

    SlangResult setResource(IResourceView* resourceView) const
    {
        return m_baseObject->setResource(m_offset, resourceView);
    }

    SlangResult setSampler(ISamplerState* sampler) const
    {
        return m_baseObject->setSampler(m_offset, sampler);
    }

    SlangResult setCombinedTextureSampler(IResourceView* textureView, ISamplerState* sampler) const
    {
        return m_baseObject->setCombinedTextureSampler(m_offset, textureView, sampler);
    }

    /// Produce a cursor to the field with the given `name`.
    ///
    /// This is a convenience wrapper around `getField()`.
    ShaderCursor operator[](const char* name) const { return getField(name); }

    /// Produce a cursor to the element or field with the given `index`.
    ///
    /// This is a convenience wrapper around `getElement()`.
    ShaderCursor operator[](int64_t index) const { return getElement((GfxIndex)index); }
    ShaderCursor operator[](uint64_t index) const { return getElement((GfxIndex)index); }
    ShaderCursor operator[](int32_t index) const { return getElement((GfxIndex)index); }
    ShaderCursor operator[](uint32_t index) const { return getElement((GfxIndex)index); }
    ShaderCursor operator[](int16_t index) const { return getElement((GfxIndex)index); }
    ShaderCursor operator[](uint16_t index) const { return getElement((GfxIndex)index); }
    ShaderCursor operator[](int8_t index) const { return getElement((GfxIndex)index); }
    ShaderCursor operator[](uint8_t index) const { return getElement((GfxIndex)index); }
};
} // namespace gfx
