#include "slang-json-native.h"

#include "../core/slang-rtti-util.h"
#include "slang-com-helper.h"
#include "slang-json-diagnostics.h"

namespace Slang
{

/* static */ RttiTypeFuncsMap JSONNativeUtil::getTypeFuncsMap()
{
    RttiTypeFuncsMap typeMap;
    typeMap.add(GetRttiInfo<JSONValue>::get(), GetRttiTypeFuncsForZeroPod<JSONValue>::getFuncs());
    return typeMap;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!! JSONToNativeConverter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* static */ Index JSONToNativeConverter::_getFieldCount(const StructRttiInfo* structRttiInfo)
{
    if (structRttiInfo->m_super)
    {
        return _getFieldCount(structRttiInfo->m_super) + structRttiInfo->m_fieldCount;
    }
    else
    {
        return structRttiInfo->m_fieldCount;
    }
}

/* static */ Index JSONToNativeConverter::_findFieldIndex(
    const StructRttiInfo* structRttiInfo,
    const UnownedStringSlice& fieldName)
{
    if (structRttiInfo->m_super)
    {
        const Index index = _findFieldIndex(structRttiInfo->m_super, fieldName);
        if (index >= 0)
        {
            return index + _getFieldCount(structRttiInfo->m_super);
        }
    }

    ConstArrayView<StructRttiInfo::Field> fields(
        structRttiInfo->m_fields,
        structRttiInfo->m_fieldCount);

    Index index = fields.findFirstIndex(
        [fieldName](const StructRttiInfo::Field& field) -> bool
        { return fieldName == field.m_name; });
    if (index >= 0 && structRttiInfo->m_super)
    {
        index += _getFieldCount(structRttiInfo->m_super);
    }

    return index;
}

SlangResult JSONToNativeConverter::_structToNative(
    const ConstArrayView<JSONKeyValue>& pairs,
    const StructRttiInfo* structRttiInfo,
    void* out,
    Index& outFieldCount)
{
    Index fieldCount = 0;

    if (structRttiInfo->m_super)
    {
        SLANG_RETURN_ON_FAIL(_structToNative(pairs, structRttiInfo->m_super, out, fieldCount));
    }

    Byte* dst = (Byte*)out;

    const Index count = structRttiInfo->m_fieldCount;

    for (Index i = 0; i < count; ++i)
    {
        const auto& field = structRttiInfo->m_fields[i];

        auto key = m_container->findKey(UnownedStringSlice(field.m_name));

        if (key == 0)
        {
            if (field.m_flags & StructRttiInfo::Flag::Optional)
            {
                continue;
            }

            m_sink->diagnose(
                SourceLoc(),
                JSONDiagnostics::fieldRequiredOnType,
                field.m_name,
                structRttiInfo->m_name);

            // Unable to find this key
            return SLANG_FAIL;
        }

        // If there are any of the pairs, that are not in the type.. it's an error
        const Index index = pairs.findFirstIndex(
            [key](const JSONKeyValue& pair) -> bool { return pair.key == key; });
        if (index < 0)
        {
            if (field.m_flags & StructRttiInfo::Flag::Optional)
            {
                continue;
            }

            m_sink->diagnose(
                SourceLoc(),
                JSONDiagnostics::fieldRequiredOnType,
                field.m_name,
                structRttiInfo->m_name);

            // Unable to find this key
            return SLANG_FAIL;
        }

        auto& pair = pairs[index];

        // Copy the field over
        SLANG_RETURN_ON_FAIL(convert(pair.value, field.m_type, dst + field.m_offset));

        // Field was handled
        ++fieldCount;
    }

    // Write off the amount of fields converted/handled.
    outFieldCount = fieldCount;
    return SLANG_OK;
}

SlangResult JSONToNativeConverter::convert(const JSONValue& in, const RttiInfo* rttiInfo, void* out)
{
    if (rttiInfo->isIntegral())
    {
        return RttiUtil::setInt(m_container->asInteger(in), rttiInfo, out);
    }
    else if (rttiInfo->isFloat())
    {
        return RttiUtil::setFromDouble(m_container->asFloat(in), rttiInfo, out);
    }

    switch (rttiInfo->m_kind)
    {
    case RttiInfo::Kind::Bool:
        {
            *(bool*)out = m_container->asBool(in);
            return SLANG_OK;
        }
    case RttiInfo::Kind::Struct:
        {
            if (in.getKind() != JSONValue::Kind::Object)
            {
                return SLANG_FAIL;
            }

            auto pairs = m_container->getObject(in);
            const StructRttiInfo* structRttiInfo = static_cast<const StructRttiInfo*>(rttiInfo);

            Index fieldCount = 0;
            SLANG_RETURN_ON_FAIL(_structToNative(pairs, structRttiInfo, out, fieldCount));

            if (fieldCount != pairs.getCount() && !structRttiInfo->m_ignoreUnknownFieldsInJson)
            {
                // We want to find the fields not found in the type

                for (auto& pair : pairs)
                {
                    UnownedStringSlice fieldName = m_container->getStringFromKey(pair.key);
                    const Index index =
                        _findFieldIndex(structRttiInfo, UnownedStringSlice(fieldName));

                    if (index < 0)
                    {
                        m_sink->diagnose(
                            pair.keyLoc,
                            JSONDiagnostics::fieldNotDefinedOnType,
                            fieldName,
                            structRttiInfo->m_name);
                    }
                }

                // If these are different then there are fields defined in the object that are *not*
                // defined in class definition
                return SLANG_FAIL;
            }

            return SLANG_OK;
        }
    case RttiInfo::Kind::Enum:
        {
            return SLANG_E_NOT_IMPLEMENTED;
        }
    case RttiInfo::Kind::String:
        {
            *(String*)out = m_container->getTransientString(in);
            return SLANG_OK;
        }
    case RttiInfo::Kind::UnownedStringSlice:
        {
            // Problem -> if the slice is a lexeme, then when we decode with getString, it will lose
            // scope. So we do something a bit odd and place the decoding string

            *(UnownedStringSlice*)out = m_container->getString(in);
            return SLANG_OK;
        }
    case RttiInfo::Kind::List:
        {
            if (in.getKind() == JSONValue::Kind::Null)
                return SLANG_OK;
            if (in.getKind() != JSONValue::Kind::Array)
            {
                return SLANG_FAIL;
            }

            typedef List<Byte> Type;
            Type& list = *(Type*)out;

            auto arr = m_container->getArray(in);

            const Index count = arr.getCount();

            const ListRttiInfo* listRttiInfo = static_cast<const ListRttiInfo*>(rttiInfo);
            auto elementType = listRttiInfo->m_elementType;

            SLANG_RETURN_ON_FAIL(
                RttiUtil::setListCount(m_typeMap, elementType, out, arr.getCount()));

            // Okay, we need to copy over one by one
            Byte* dstEles = list.getBuffer();
            for (Index i = 0; i < count; ++i, dstEles += elementType->m_size)
            {
                SLANG_RETURN_ON_FAIL(convert(arr[i], elementType, dstEles));
            }

            return SLANG_OK;
        }
    case RttiInfo::Kind::FixedArray:
        {
            if (in.getKind() != JSONValue::Kind::Array)
            {
                return SLANG_FAIL;
            }
            const FixedArrayRttiInfo* fixedArrayRttiInfo =
                static_cast<const FixedArrayRttiInfo*>(rttiInfo);
            const auto elementType = fixedArrayRttiInfo->m_elementType;
            const Index elementCount = Index(fixedArrayRttiInfo->m_elementCount);
            const auto elementSize = elementType->m_size;

            auto srcArray = m_container->getArray(in);

            if (srcArray.getCount() > elementCount)
            {
                m_sink->diagnose(
                    in.loc,
                    JSONDiagnostics::tooManyElementsForArray,
                    srcArray.getCount(),
                    elementCount);
                return SLANG_FAIL;
            }

            Byte* dstEles = (Byte*)out;
            for (Index i = 0; i < elementCount; ++i, dstEles += elementSize)
            {
                SLANG_RETURN_ON_FAIL(convert(srcArray[i], elementType, dstEles));
            }

            return SLANG_OK;
        }
    case RttiInfo::Kind::Dictionary:
        {
            // We can *only* serialize this into a straight JSON object iff the key is a string-like
            // type We could turn into (say) an array of keys and values
            break;
        }
    case RttiInfo::Kind::Other:
        {
            if (rttiInfo == GetRttiInfo<JSONValue>::get())
            {
                // Do we need to copy into the container?
                // As it stands we have to assume src is stored in container.
                *(JSONValue*)out = in;
                return SLANG_OK;
            }
            return SLANG_FAIL;
        }
    default:
        break;
    }
    return SLANG_FAIL;
}

SlangResult JSONToNativeConverter::convertArrayToStruct(
    const JSONValue& value,
    const RttiInfo* rttiInfo,
    void* out)
{
    // Check converting JSON array into a struct, as that's what this method supports
    if (!(rttiInfo->m_kind == RttiInfo::Kind::Struct && value.getKind() == JSONValue::Kind::Array))
    {
        // If they are the wrong types then just fail
        return SLANG_FAIL;
    }

    // Find the total amount of fields, and all the classes involved
    Index totalFieldCount = 0;

    ShortList<const StructRttiInfo*, 8> infos;
    for (const StructRttiInfo* cur = static_cast<const StructRttiInfo*>(rttiInfo); cur;
         cur = cur->m_super)
    {
        totalFieldCount += cur->m_fieldCount;
        infos.add(cur);
    }

    // Must have the same amount of fields
    auto array = m_container->getArray(value);
    if (array.getCount() != totalFieldCount)
    {
        return SLANG_FAIL;
    }

    Byte* dstBase = (Byte*)out;

    // We work in the order from the base class to the final type
    Index argIndex = 0;
    for (Index i = infos.getCount() - 1; i >= 0; --i)
    {
        auto info = infos[i];

        const Index fieldCount = info->m_fieldCount;
        for (Index j = 0; j < fieldCount; ++j)
        {
            // Convert the field
            const auto& field = info->m_fields[j];
            SLANG_RETURN_ON_FAIL(
                convert(array[argIndex++], field.m_type, dstBase + field.m_offset));
        }
    }

    return SLANG_OK;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!! NativeToJSONConverter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

SlangResult NativeToJSONConverter::_structToJSON(
    const StructRttiInfo* structRttiInfo,
    const void* src,
    List<JSONKeyValue>& outPairs)
{
    // Do the super class first
    if (structRttiInfo->m_super)
    {
        SLANG_RETURN_ON_FAIL(_structToJSON(structRttiInfo->m_super, src, outPairs));
    }

    const Byte* base = (const Byte*)src;
    const Index count = structRttiInfo->m_fieldCount;

    for (Index i = 0; i < count; ++i)
    {
        const auto& field = structRttiInfo->m_fields[i];

        if (field.m_flags & StructRttiInfo::Flag::Optional)
        {
            const RttiDefaultValue defaultValue =
                RttiDefaultValue(field.m_flags & uint8_t(RttiDefaultValue::Mask));
            if (RttiUtil::isDefault(defaultValue, field.m_type, base + field.m_offset))
            {
                // If it's a default, we don't bother writing it
                continue;
            }
        }

        JSONKeyValue pair;
        pair.key = m_container->getKey(UnownedStringSlice(field.m_name));
        auto res = convert(field.m_type, base + field.m_offset, pair.value);

        if (SLANG_FAILED(res))
        {
            m_sink->diagnose(
                SourceLoc(),
                JSONDiagnostics::unableToConvertField,
                field.m_name,
                structRttiInfo->m_name);
            return res;
        }

        outPairs.add(pair);
    }

    return SLANG_OK;
}


SlangResult NativeToJSONConverter::convert(const RttiInfo* rttiInfo, const void* in, JSONValue& out)
{
    if (rttiInfo->isIntegral())
    {
        out = JSONValue::makeInt(RttiUtil::getInt64(rttiInfo, in));
        return SLANG_OK;
    }
    else if (rttiInfo->isFloat())
    {
        out = JSONValue::makeFloat(RttiUtil::asDouble(rttiInfo, in));
        return SLANG_OK;
    }

    switch (rttiInfo->m_kind)
    {
    case RttiInfo::Kind::Invalid:
        return SLANG_FAIL;
    case RttiInfo::Kind::Bool:
        {
            out = JSONValue::makeBool(RttiUtil::asBool(rttiInfo, in));
            return SLANG_OK;
        }
    case RttiInfo::Kind::String:
        {
            const String& str = *(const String*)in;
            out = m_container->createString(str.getUnownedSlice());
            return SLANG_OK;
        }
    case RttiInfo::Kind::UnownedStringSlice:
        {
            const UnownedStringSlice& slice = *(const UnownedStringSlice*)in;
            out = m_container->createString(slice);
            return SLANG_OK;
        }
    case RttiInfo::Kind::Struct:
        {
            const StructRttiInfo* structRttiInfo = static_cast<const StructRttiInfo*>(rttiInfo);

            List<JSONKeyValue> pairs;
            SLANG_RETURN_ON_FAIL(_structToJSON(structRttiInfo, in, pairs));
            out = m_container->createObject(pairs.getBuffer(), pairs.getCount());
            return SLANG_OK;
        }
    case RttiInfo::Kind::Enum:
        {
            return SLANG_E_NOT_IMPLEMENTED;
        }
    case RttiInfo::Kind::List:
        {
            const ListRttiInfo* listRttiInfo = static_cast<const ListRttiInfo*>(rttiInfo);
            const auto elementRttiInfo = listRttiInfo->m_elementType;

            // The src probably *doesn't* contain bytes, but can cast like this because
            // we only need the count (which doesn't depend on <T>), and the backing buffer
            const List<Byte>& srcValuesList = *(const List<Byte>*)in;

            const Index count = srcValuesList.getCount();
            const Byte* srcValues = srcValuesList.getBuffer();

            List<JSONValue> dstValues;
            dstValues.setCount(count);

            const size_t elementStride = elementRttiInfo->m_size;

            for (Index i = 0; i < count; ++i, srcValues += elementStride)
            {
                SLANG_RETURN_ON_FAIL(convert(elementRttiInfo, srcValues, dstValues[i]));
            }

            out = m_container->createArray(dstValues.getBuffer(), count);
            return SLANG_OK;
        }
    case RttiInfo::Kind::FixedArray:
        {
            const FixedArrayRttiInfo* fixedArrayRttiInfo =
                static_cast<const FixedArrayRttiInfo*>(rttiInfo);
            const auto elementType = fixedArrayRttiInfo->m_elementType;
            const auto elementCount = Index(fixedArrayRttiInfo->m_elementCount);
            const auto elementSize = elementType->m_size;

            List<JSONValue> dstValues;
            dstValues.setCount(elementCount);

            const Byte* src = (const Byte*)in;
            for (Index i = 0; i < elementCount; ++i, src += elementSize)
            {
                SLANG_RETURN_ON_FAIL(convert(elementType, src, dstValues[i]));
            }

            out = m_container->createArray(dstValues.getBuffer(), elementCount);
            return SLANG_OK;
        }
    case RttiInfo::Kind::Dictionary:
        {
            const DictionaryRttiInfo* listRttiInfo =
                static_cast<const DictionaryRttiInfo*>(rttiInfo);
            const auto keyRttiInfo = listRttiInfo->m_keyType;
            const auto valueRttiInfo = listRttiInfo->m_valueType;

            SLANG_UNUSED(keyRttiInfo);
            SLANG_UNUSED(valueRttiInfo);

            // We can *only* serialize this into a straight JSON object iff the key is a string-like
            // type We could turn into (say) an array of keys and values

            break;
        }
    case RttiInfo::Kind::Other:
        {
            if (rttiInfo == GetRttiInfo<JSONValue>::get())
            {
                // Do we need to copy into the container?
                // As it stands we have to assume src is stored in container.
                const JSONValue& src = *(const JSONValue*)in;

                out = src;
                return SLANG_OK;
            }
            break;
        }
    default:
        break;
    }

    return SLANG_E_NOT_IMPLEMENTED;
}

SlangResult NativeToJSONConverter::convertStructToArray(
    const RttiInfo* rttiInfo,
    const void* in,
    JSONValue& out)
{
    if (rttiInfo->m_kind != RttiInfo::Kind::Struct)
    {
        // Must be a struct
        return SLANG_FAIL;
    }

    // Work out the total amount of fields, and all invloved struct types
    Index totalFieldsCount = 0;
    ShortList<const StructRttiInfo*, 8> infos;
    for (const StructRttiInfo* cur = static_cast<const StructRttiInfo*>(rttiInfo); cur;
         cur = cur->m_super)
    {
        totalFieldsCount += Index(cur->m_fieldCount);
        infos.add(cur);
    }

    // Convert the args/params
    List<JSONValue> argsArray;
    argsArray.setCount(totalFieldsCount);

    // NOTE! We do no special handling here around optional parameters.
    // All fields of the input args are output
    {
        Index argsArrayIndex = 0;
        const Byte* argsBase = (const Byte*)in;

        // Work in the order from the base class to the actual type
        for (Index i = infos.getCount() - 1; i >= 0; --i)
        {
            auto structRttiInfo = infos[i];
            const Index fieldCount = Index(structRttiInfo->m_fieldCount);

            for (Index j = 0; j < fieldCount; ++j)
            {
                const auto& field = structRttiInfo->m_fields[j];
                // Convert the field
                SLANG_RETURN_ON_FAIL(
                    convert(field.m_type, argsBase + field.m_offset, argsArray[argsArrayIndex++]));
            }
        }
    }

    // Okay now we convert the List to the output, which will just be a JSON array.
    SLANG_RETURN_ON_FAIL(convert(&argsArray, out));

    return SLANG_OK;
}

} // namespace Slang
