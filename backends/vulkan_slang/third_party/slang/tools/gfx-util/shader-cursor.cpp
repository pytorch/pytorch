#include "shader-cursor.h"

namespace gfx
{

Result gfx::ShaderCursor::getDereferenced(ShaderCursor& outCursor) const
{
    switch (m_typeLayout->getKind())
    {
    default:
        return SLANG_E_INVALID_ARG;

    case slang::TypeReflection::Kind::ConstantBuffer:
    case slang::TypeReflection::Kind::ParameterBlock:
        {
            auto subObject = m_baseObject->getObject(m_offset);
            outCursor = ShaderCursor(subObject);
            return SLANG_OK;
        }
    }
}

ShaderCursor ShaderCursor::getExplicitCounter() const
{
    // Similar to getField below

    // The alternative to handling this here would be to augment IResourceView
    // with a `getCounterResourceView()`, and set that also in `setResource`
    if (const auto counterVarLayout = m_typeLayout->getExplicitCounter())
    {
        ShaderCursor counterCursor;

        // The counter cursor will point into the same parent object.
        counterCursor.m_baseObject = m_baseObject;

        // The type being pointed to is the type of the field.
        counterCursor.m_typeLayout = counterVarLayout->getTypeLayout();

        // The byte offset is the current offset plus the relative offset of the counter.
        // The offset in binding ranges is computed similarly.
        counterCursor.m_offset.uniformOffset =
            m_offset.uniformOffset + SlangInt(counterVarLayout->getOffset());
        counterCursor.m_offset.bindingRangeIndex =
            m_offset.bindingRangeIndex +
            GfxIndex(m_typeLayout->getExplicitCounterBindingRangeOffset());

        // The index of the counter within any binding ranges will be the same
        // as the index computed for the parent structure.
        //
        // Note: this case would arise for an array of structured buffers
        //
        //      AppendStructuredBuffer g[4];
        //
        // In this scenario, `g` holds two binding ranges:
        //
        // * Range #0 comprises 4 element buffers, representing `g[...].elements`
        // * Range #1 comprises 4 counter buffers, representing `g[...].counter`
        //
        // A cursor for `g[2]` would have a `bindingRangeIndex` of zero but
        // a `bindingArrayIndex` of 2, indicating that we could end up
        // referencing either range, but no matter what we know the index
        // is 2. Thus when we form a cursor for `g[2].counter` we want to
        // apply the binding range offset to get a `bindingRangeIndex` of
        // 1, while the `bindingArrayIndex` is unmodified.
        //
        // The result is that `g[2].counter` is stored in range #1 at array index 2.
        //
        counterCursor.m_offset.bindingArrayIndex = m_offset.bindingArrayIndex;

        return counterCursor;
    }
    // Otherwise, return an invalid cursor
    return ShaderCursor{};
}

Result ShaderCursor::getField(const char* name, const char* nameEnd, ShaderCursor& outCursor) const
{
    // If this cursor is invalid, then can't possible fetch a field.
    //
    if (!isValid())
        return SLANG_E_INVALID_ARG;

    // If the cursor is valid, we want to consider the type of data
    // it is referencing.
    //
    switch (m_typeLayout->getKind())
    {
        // The easy/expected case is when the value has a structure type.
        //
    case slang::TypeReflection::Kind::Struct:
        {
            // We start by looking up the index of a field matching `name`.
            //
            // If there is no such field, we have an error.
            //
            SlangInt fieldIndex = m_typeLayout->findFieldIndexByName(name, nameEnd);
            if (fieldIndex == -1)
                break;

            // Once we know the index of the field being referenced,
            // we create a cursor to point at the field, based on
            // the offset information already in this cursor, plus
            // offsets derived from the field's layout.
            //
            slang::VariableLayoutReflection* fieldLayout =
                m_typeLayout->getFieldByIndex((unsigned int)fieldIndex);
            ShaderCursor fieldCursor;

            // The field cursorwill point into the same parent object.
            //
            fieldCursor.m_baseObject = m_baseObject;

            // The type being pointed to is the tyep of the field.
            //
            fieldCursor.m_typeLayout = fieldLayout->getTypeLayout();

            // The byte offset is the current offset plus the relative offset of the field.
            // The offset in binding ranges is computed similarly.
            //
            fieldCursor.m_offset.uniformOffset = m_offset.uniformOffset + fieldLayout->getOffset();
            fieldCursor.m_offset.bindingRangeIndex =
                m_offset.bindingRangeIndex +
                (GfxIndex)m_typeLayout->getFieldBindingRangeOffset(fieldIndex);

            // The index of the field within any binding ranges will be the same
            // as the index computed for the parent structure.
            //
            // Note: this case would arise for an array of structures with texture-type
            // fields. Suppose we have:
            //
            //      struct S { Texture2D t; Texture2D u; }
            //      S g[4];
            //
            // In this scenario, `g` holds two binding ranges:
            //
            // * Range #0 comprises 4 textures, representing `g[...].t`
            // * Range #1 comprises 4 textures, representing `g[...].u`
            //
            // A cursor for `g[2]` would have a `bindingRangeIndex` of zero but
            // a `bindingArrayIndex` of 2, iindicating that we could end up
            // referencing either range, but no matter what we know the index
            // is 2. Thus when we form a cursor for `g[2].u` we want to
            // apply the binding range offset to get a `bindingRangeIndex` of
            // 1, while the `bindingArrayIndex` is unmodified.
            //
            // The result is that `g[2].u` is stored in range #1 at array index 2.
            //
            fieldCursor.m_offset.bindingArrayIndex = m_offset.bindingArrayIndex;

            outCursor = fieldCursor;
            return SLANG_OK;
        }
        break;

    // In some cases the user might be trying to acess a field by name
    // from a cursor that references a constant buffer or parameter block,
    // and in these cases we want the access to Just Work.
    //
    case slang::TypeReflection::Kind::ConstantBuffer:
    case slang::TypeReflection::Kind::ParameterBlock:
        {
            // We basically need to "dereference" the current cursor
            // to go from a pointer to a constant buffer to a pointer
            // to the *contents* of the constant buffer.
            //
            ShaderCursor d = getDereferenced();
            return d.getField(name, nameEnd, outCursor);
        }
        break;
    }

    // If a cursor is pointing at a root shader object (created for a
    // program), then we will also iterate over the entry point shader
    // objects attached to it and look for a matching parameter name
    // on them.
    //
    // This is a bit of "do what I mean" logic and could potentially
    // lead to problems if there could be multiple entry points with
    // the same parameter name.
    //
    // TODO: figure out whether we should support this long-term.
    //
    auto entryPointCount = (GfxIndex)m_baseObject->getEntryPointCount();
    for (GfxIndex e = 0; e < entryPointCount; ++e)
    {
        ComPtr<IShaderObject> entryPoint;
        m_baseObject->getEntryPoint(e, entryPoint.writeRef());

        ShaderCursor entryPointCursor(entryPoint);

        auto result = entryPointCursor.getField(name, nameEnd, outCursor);
        if (SLANG_SUCCEEDED(result))
            return result;
    }

    return SLANG_E_INVALID_ARG;
}

ShaderCursor ShaderCursor::getElement(GfxIndex index) const
{
    if (m_containerType != ShaderObjectContainerType::None)
    {
        ShaderCursor elementCursor;
        elementCursor.m_baseObject = m_baseObject;
        elementCursor.m_typeLayout = m_typeLayout->getElementTypeLayout();
        elementCursor.m_containerType = m_containerType;
        elementCursor.m_offset.uniformOffset = index * m_typeLayout->getStride();
        elementCursor.m_offset.bindingRangeIndex = 0;
        elementCursor.m_offset.bindingArrayIndex = index;
        return elementCursor;
    }

    switch (m_typeLayout->getKind())
    {
    case slang::TypeReflection::Kind::Array:
        {
            ShaderCursor elementCursor;
            elementCursor.m_baseObject = m_baseObject;
            elementCursor.m_typeLayout = m_typeLayout->getElementTypeLayout();
            elementCursor.m_offset.uniformOffset =
                m_offset.uniformOffset +
                index * m_typeLayout->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);
            elementCursor.m_offset.bindingRangeIndex = m_offset.bindingRangeIndex;
            elementCursor.m_offset.bindingArrayIndex =
                m_offset.bindingArrayIndex * (GfxCount)m_typeLayout->getElementCount() + index;
            return elementCursor;
        }
        break;

    case slang::TypeReflection::Kind::Struct:
        {
            // The logic here is similar to `getField()` except that we don't
            // need to look up the field index based on a name first.
            //
            auto fieldIndex = index;
            slang::VariableLayoutReflection* fieldLayout =
                m_typeLayout->getFieldByIndex((unsigned int)fieldIndex);
            if (!fieldLayout)
                return ShaderCursor();

            ShaderCursor fieldCursor;
            fieldCursor.m_baseObject = m_baseObject;
            fieldCursor.m_typeLayout = fieldLayout->getTypeLayout();
            fieldCursor.m_offset.uniformOffset = m_offset.uniformOffset + fieldLayout->getOffset();
            fieldCursor.m_offset.bindingRangeIndex =
                m_offset.bindingRangeIndex +
                (GfxIndex)m_typeLayout->getFieldBindingRangeOffset(fieldIndex);
            fieldCursor.m_offset.bindingArrayIndex = m_offset.bindingArrayIndex;

            return fieldCursor;
        }
        break;

    case slang::TypeReflection::Kind::Vector:
    case slang::TypeReflection::Kind::Matrix:
        {
            ShaderCursor fieldCursor;
            fieldCursor.m_baseObject = m_baseObject;
            fieldCursor.m_typeLayout = m_typeLayout->getElementTypeLayout();
            fieldCursor.m_offset.uniformOffset =
                m_offset.uniformOffset +
                m_typeLayout->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM) * index;
            fieldCursor.m_offset.bindingRangeIndex = m_offset.bindingRangeIndex;
            fieldCursor.m_offset.bindingArrayIndex = m_offset.bindingArrayIndex;
            return fieldCursor;
        }
        break;
    }

    return ShaderCursor();
}


static int _peek(const char* slice)
{
    const char* b = slice;
    if (!b || !*b)
        return -1;
    return *b;
}

static int _get(const char*& slice)
{
    const char* b = slice;
    if (!b || !*b)
        return -1;
    auto result = *b++;
    slice = b;
    return result;
}

Result ShaderCursor::followPath(const char* path, ShaderCursor& ioCursor)
{
    ShaderCursor cursor = ioCursor;

    enum
    {
        ALLOW_NAME = 0x1,
        ALLOW_SUBSCRIPT = 0x2,
        ALLOW_DOT = 0x4,
    };
    int state = ALLOW_NAME | ALLOW_SUBSCRIPT;

    const char* rest = path;
    for (;;)
    {
        int c = _peek(rest);

        if (c == -1)
            break;
        else if (c == '.')
        {
            if (!(state & ALLOW_DOT))
                return SLANG_E_INVALID_ARG;

            _get(rest);
            state = ALLOW_NAME;
            continue;
        }
        else if (c == '[')
        {
            if (!(state & ALLOW_SUBSCRIPT))
                return SLANG_E_INVALID_ARG;

            _get(rest);
            GfxCount index = 0;
            while (_peek(rest) != ']')
            {
                int d = _get(rest);
                if (d >= '0' && d <= '9')
                {
                    index = index * 10 + (d - '0');
                }
                else
                {
                    return SLANG_E_INVALID_ARG;
                }
            }

            if (_peek(rest) != ']')
                return SLANG_E_INVALID_ARG;
            _get(rest);

            cursor = cursor.getElement(index);
            state = ALLOW_DOT | ALLOW_SUBSCRIPT;
            continue;
        }
        else
        {
            const char* nameBegin = rest;
            for (;;)
            {
                switch (_peek(rest))
                {
                default:
                    _get(rest);
                    continue;

                case -1:
                case '.':
                case '[':
                    break;
                }
                break;
            }
            char const* nameEnd = rest;
            ShaderCursor newCursor;
            cursor.getField(nameBegin, nameEnd, newCursor);
            cursor = newCursor;
            state = ALLOW_DOT | ALLOW_SUBSCRIPT;
            continue;
        }
    }

    ioCursor = cursor;
    return SLANG_OK;
}

} // namespace gfx
