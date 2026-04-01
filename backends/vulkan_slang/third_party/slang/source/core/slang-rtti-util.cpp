#include "slang-rtti-util.h"

namespace Slang
{

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! RttiTypeFuncs Impls
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

struct ListFuncs
{
    static void ctorArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* inDst,
        Index count)
    {
        SLANG_UNUSED(typeMap);
        SLANG_UNUSED(rttiInfo);
        SLANG_ASSERT(rttiInfo->m_kind == RttiInfo::Kind::List);

        // We don't care about the element type, as we can just initialize them all as List<Byte>
        // const ListRttiInfo* listRttiInfo = static_cast<const ListRttiInfo*>(rttiInfo);
        typedef List<Byte> Type;

        Type* dst = (Type*)inDst;

        for (Index i = 0; i < count; ++i)
        {
            new (dst + i) Type;
        }
    }

    static void copyArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* inDst,
        const void* inSrc,
        Index count)
    {
        SLANG_ASSERT(rttiInfo->m_kind == RttiInfo::Kind::List);
        const ListRttiInfo* listRttiInfo = static_cast<const ListRttiInfo*>(rttiInfo);
        const auto elementType = listRttiInfo->m_elementType;

        // We need to get the type funcs
        auto typeFuncs = typeMap->getFuncsForType(elementType);
        SLANG_ASSERT(typeFuncs.isValid());

        // We need a type that we can get information from the list from - List<Byte> gives us the
        // functions we need.
        typedef List<Byte> Type;

        Type* dst = (Type*)inDst;
        const Type* src = (const Type*)inSrc;

        for (Index i = 0; i < count; ++i)
        {
            auto& dstList = dst[i];
            auto& srcList = src[i];

            const Index srcCount = srcList.getCount();

            if (srcCount > dstList.getCount())
            {
                // Allocate new memory
                const Index dstCapacity = dstList.getCapacity();
                void* oldBuffer = dstList.detachBuffer();

                void* newBuffer = ::malloc(count * elementType->m_size);
                // Initialize it all first
                typeFuncs.ctorArray(typeMap, elementType, newBuffer, count);
                typeFuncs.copyArray(typeMap, elementType, newBuffer, oldBuffer, count);

                // Attach the new buffer
                dstList.attachBuffer((Byte*)newBuffer, count, count);

                // Free the old buffer
                if (oldBuffer)
                {
                    typeFuncs.dtorArray(typeMap, elementType, oldBuffer, dstCapacity);

                    ::free(oldBuffer);
                }
            }
            else
            {
                typeFuncs.copyArray(
                    typeMap,
                    elementType,
                    dstList.getBuffer(),
                    srcList.getBuffer(),
                    srcCount);
                dstList.unsafeShrinkToCount(srcCount);
            }
        }
    }

    static void dtorArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* inDst,
        Index count)
    {
        SLANG_ASSERT(rttiInfo->m_kind == RttiInfo::Kind::List);
        const ListRttiInfo* listRttiInfo = static_cast<const ListRttiInfo*>(rttiInfo);

        const auto elementType = listRttiInfo->m_elementType;

        // We need to get the type funcs
        auto typeFuncs = typeMap->getFuncsForType(elementType);
        SLANG_ASSERT(typeFuncs.isValid());

        typedef List<Byte> Type;
        Type* dst = (Type*)inDst;

        for (Index i = 0; i < count; ++i)
        {
            auto& dstList = dst[i];

            const Index capacity = dstList.getCapacity();
            Byte* buffer = dstList.detachBuffer();

            if (buffer)
            {
                typeFuncs.dtorArray(typeMap, elementType, buffer, capacity);
                ::free(buffer);
            }
        }
    }

    static RttiTypeFuncs getFuncs()
    {
        RttiTypeFuncs funcs;
        funcs.copyArray = &copyArray;
        funcs.dtorArray = &dtorArray;
        funcs.ctorArray = &ctorArray;
        return funcs;
    }
};

struct StructFuncs
{
    static void ctorArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* inDst,
        Index count)
    {
        SLANG_UNUSED(typeMap);
        SLANG_UNUSED(rttiInfo);
        SLANG_ASSERT(rttiInfo->m_kind == RttiInfo::Kind::List);

        // We don't care about the element type, as we can just initialize them all as List<Byte>
        // const ListRttiInfo* listRttiInfo = static_cast<const ListRttiInfo*>(rttiInfo);
        typedef List<Byte> Type;

        Type* dst = (Type*)inDst;

        for (Index i = 0; i < count; ++i)
        {
            new (dst + i) Type;
        }
    }
    static void copyArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* inDst,
        const void* inSrc,
        Index count)
    {
        SLANG_ASSERT(rttiInfo->m_kind == RttiInfo::Kind::List);
        const ListRttiInfo* listRttiInfo = static_cast<const ListRttiInfo*>(rttiInfo);
        const auto elementType = listRttiInfo->m_elementType;

        // We need to get the type funcs
        auto typeFuncs = typeMap->getFuncsForType(elementType);
        SLANG_ASSERT(typeFuncs.isValid());

        // We need a type that we can get information from the list from - List<Byte> gives us the
        // functions we need.
        typedef List<Byte> Type;

        Type* dst = (Type*)inDst;
        const Type* src = (const Type*)inSrc;

        for (Index i = 0; i < count; ++i)
        {
            auto& dstList = dst[i];
            auto& srcList = src[i];

            const Index srcCount = srcList.getCount();

            if (srcCount > dstList.getCount())
            {
                // Allocate new memory
                const Index dstCapacity = dstList.getCapacity();
                void* oldBuffer = dstList.detachBuffer();

                void* newBuffer = ::malloc(count * elementType->m_size);
                // Initialize it all first
                typeFuncs.ctorArray(typeMap, elementType, newBuffer, count);
                typeFuncs.copyArray(typeMap, elementType, newBuffer, oldBuffer, count);

                // Attach the new buffer
                dstList.attachBuffer((Byte*)newBuffer, count, count);

                // Free the old buffer
                if (oldBuffer)
                {
                    typeFuncs.dtorArray(typeMap, elementType, oldBuffer, dstCapacity);

                    ::free(oldBuffer);
                }
            }
            else
            {
                typeFuncs.copyArray(
                    typeMap,
                    elementType,
                    dstList.getBuffer(),
                    srcList.getBuffer(),
                    srcCount);
                dstList.unsafeShrinkToCount(srcCount);
            }
        }
    }

    static void dtorArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* inDst,
        Index count)
    {
        SLANG_ASSERT(rttiInfo->m_kind == RttiInfo::Kind::List);
        const ListRttiInfo* listRttiInfo = static_cast<const ListRttiInfo*>(rttiInfo);

        const auto elementType = listRttiInfo->m_elementType;

        // We need to get the type funcs
        auto typeFuncs = typeMap->getFuncsForType(elementType);
        SLANG_ASSERT(typeFuncs.isValid());

        typedef List<Byte> Type;
        Type* dst = (Type*)inDst;

        for (Index i = 0; i < count; ++i)
        {
            auto& dstList = dst[i];

            const Index capacity = dstList.getCapacity();
            Byte* buffer = dstList.detachBuffer();

            if (buffer)
            {
                typeFuncs.dtorArray(typeMap, elementType, buffer, capacity);
                ::free(buffer);
            }
        }
    }

    static RttiTypeFuncs getFuncs()
    {
        RttiTypeFuncs funcs;
        funcs.copyArray = &copyArray;
        funcs.dtorArray = &dtorArray;
        funcs.ctorArray = &ctorArray;
        return funcs;
    }
};

struct StructArrayFuncs
{
    static void ctorArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* inDst,
        Index count)
    {
        return RttiUtil::ctorArray(typeMap, rttiInfo, inDst, rttiInfo->m_size, count);
    }

    static void copyArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* inDst,
        const void* inSrc,
        Index count)
    {
        return RttiUtil::copyArray(typeMap, rttiInfo, inDst, inSrc, rttiInfo->m_size, count);
    }

    static void dtorArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* inDst,
        Index count)
    {
        return RttiUtil::dtorArray(typeMap, rttiInfo, inDst, rttiInfo->m_size, count);
    }

    static RttiTypeFuncs getFuncs()
    {
        RttiTypeFuncs funcs;
        funcs.copyArray = copyArray;
        funcs.dtorArray = dtorArray;
        funcs.ctorArray = ctorArray;
        return funcs;
    }
};

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! RttiUtil !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

RttiTypeFuncs RttiUtil::getDefaultTypeFuncs(const RttiInfo* rttiInfo)
{
    if (rttiInfo->isBuiltIn())
    {
        switch (rttiInfo->m_size)
        {
        case 1:
            return GetRttiTypeFuncsForZeroPod<uint8_t>::getFuncs();
        case 2:
            return GetRttiTypeFuncsForZeroPod<uint16_t>::getFuncs();
        case 4:
            return GetRttiTypeFuncsForZeroPod<uint32_t>::getFuncs();
        case 8:
            return GetRttiTypeFuncsForZeroPod<uint64_t>::getFuncs();
        }
        return RttiTypeFuncs::makeEmpty();
    }

    switch (rttiInfo->m_kind)
    {
    case RttiInfo::Kind::String:
        return GetRttiTypeFuncs<String>::getFuncs();
    case RttiInfo::Kind::UnownedStringSlice:
        return GetRttiTypeFuncs<UnownedStringSlice>::getFuncs();
    case RttiInfo::Kind::List:
        return ListFuncs::getFuncs();
    case RttiInfo::Kind::Struct:
        return StructArrayFuncs::getFuncs();
    default:
        break;
    }

    return RttiTypeFuncs::makeEmpty();
}

/* static */ SlangResult RttiUtil::setInt(int64_t value, const RttiInfo* rttiInfo, void* dst)
{
    SLANG_ASSERT(rttiInfo->isIntegral());

    // We could check ranges are appropriate, but for now we just write.
    // Passing in rttiInfo allows for other more complex types to be econverted
    switch (rttiInfo->m_kind)
    {
    case RttiInfo::Kind::I32:
        *(int32_t*)dst = int32_t(value);
        break;
    case RttiInfo::Kind::U32:
        *(uint32_t*)dst = uint32_t(value);
        break;
    case RttiInfo::Kind::I64:
        *(int64_t*)dst = int64_t(value);
        break;
    case RttiInfo::Kind::U64:
        *(uint64_t*)dst = uint64_t(value);
        break;
    default:
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

/* static */ int64_t RttiUtil::getInt64(const RttiInfo* rttiInfo, const void* src)
{
    SLANG_ASSERT(rttiInfo->isIntegral());

    switch (rttiInfo->m_kind)
    {
    case RttiInfo::Kind::I32:
        return *(const int32_t*)src;
    case RttiInfo::Kind::U32:
        return *(const uint32_t*)src;
    case RttiInfo::Kind::I64:
        return *(const int64_t*)src;
    case RttiInfo::Kind::U64:
        return *(const uint64_t*)src;
    default:
        break;
    }

    SLANG_ASSERT(!"Not integral!");
    return -1;
}

/* static */ double RttiUtil::asDouble(const RttiInfo* rttiInfo, const void* src)
{
    if (rttiInfo->isIntegral())
    {
        return (double)getInt64(rttiInfo, src);
    }
    else if (rttiInfo->isFloat())
    {
        switch (rttiInfo->m_kind)
        {
        case RttiInfo::Kind::F32:
            return *(const float*)src;
        case RttiInfo::Kind::F64:
            return *(const double*)src;
        default:
            break;
        }
    }

    SLANG_ASSERT(!"Cannot convert to float");
    return 0.0;
}

/* static */ SlangResult RttiUtil::setFromDouble(double v, const RttiInfo* rttiInfo, void* dst)
{
    if (rttiInfo->isIntegral())
    {
        return setInt(int64_t(v), rttiInfo, dst);
    }
    else if (rttiInfo->isFloat())
    {
        switch (rttiInfo->m_kind)
        {
        case RttiInfo::Kind::F32:
            *(float*)dst = float(v);
            return SLANG_OK;
        case RttiInfo::Kind::F64:
            *(double*)dst = v;
            return SLANG_OK;
        default:
            break;
        }
    }

    return SLANG_FAIL;
}

/* static */ bool RttiUtil::asBool(const RttiInfo* rttiInfo, const void* src)
{
    if (rttiInfo->m_kind == RttiInfo::Kind::Bool)
    {
        return *(const bool*)src;
    }

    if (rttiInfo->isIntegral())
    {
        return getInt64(rttiInfo, src) != 0;
    }
    else if (rttiInfo->isFloat())
    {
        return asDouble(rttiInfo, src) != 0.0;
    }

    SLANG_ASSERT(!"Cannot convert to bool");
    return false;
}

static int64_t _getIntDefaultValue(RttiDefaultValue value)
{
    switch (value)
    {
    default:
    case RttiDefaultValue::Normal:
        return 0;
    case RttiDefaultValue::One:
        return 1;
    case RttiDefaultValue::MinusOne:
        return -1;
    }
}

static bool _isStructDefault(const StructRttiInfo* type, const void* src)
{
    if (type->m_super)
    {
        if (!_isStructDefault(type->m_super, src))
        {
            return false;
        }
    }

    const Byte* base = (const Byte*)src;

    const Index count = type->m_fieldCount;
    for (Index i = 0; i < count; ++i)
    {
        const auto& field = type->m_fields[i];

        const RttiDefaultValue defaultValue =
            RttiDefaultValue(field.m_flags & uint8_t(RttiDefaultValue::Mask));

        if (!RttiUtil::isDefault(defaultValue, field.m_type, base + field.m_offset))
        {
            return false;
        }
    }

    return true;
}

/* static */ bool RttiUtil::isDefault(
    RttiDefaultValue defaultValue,
    const RttiInfo* rttiInfo,
    const void* src)
{
    if (rttiInfo->isIntegral())
    {
        const auto value = getInt64(rttiInfo, src);
        return _getIntDefaultValue(defaultValue) == value;
    }
    else if (rttiInfo->isFloat())
    {
        const auto value = asDouble(rttiInfo, src);
        return _getIntDefaultValue(defaultValue) == value;
    }

    switch (rttiInfo->m_kind)
    {
    case RttiInfo::Kind::Invalid:
        return true;
    case RttiInfo::Kind::Bool:
        return *(const bool*)src == (_getIntDefaultValue(defaultValue) != 0);
    case RttiInfo::Kind::String:
        {
            return ((const String*)src)->getLength() == 0;
        }
    case RttiInfo::Kind::UnownedStringSlice:
        {
            return ((const UnownedStringSlice*)src)->getLength() == 0;
        }
    case RttiInfo::Kind::Struct:
        {
            return _isStructDefault(static_cast<const StructRttiInfo*>(rttiInfo), src);
        }
    case RttiInfo::Kind::Enum:
        {
            SLANG_ASSERT(!"Not implemented yet");
            return false;
        }
    case RttiInfo::Kind::List:
        {
            const auto& v = *(const List<Byte>*)src;
            return v.getCount() == 0;
        }
    case RttiInfo::Kind::Dictionary:
        {
            const auto& v = *(const Dictionary<Byte, Byte>*)src;
            return v.getCount() == 0;
        }
    case RttiInfo::Kind::Other:
        {
            const OtherRttiInfo* otherRttiInfo = static_cast<const OtherRttiInfo*>(rttiInfo);
            return otherRttiInfo->m_isDefaultFunc && otherRttiInfo->m_isDefaultFunc(rttiInfo, src);
        }
    default:
        {
            return false;
        }
    }
}

/* static */ SlangResult RttiUtil::setListCount(
    RttiTypeFuncsMap* typeMap,
    const RttiInfo* elementType,
    void* dst,
    Index count)
{
    // NOTE! The following only works because List<T> has capacity initialized members, and
    // setting the count if it is <= capacity just sets the count (ie things aren't released(!)).

    List<Byte>& dstList = *(List<Byte>*)dst;
    const Index oldCount = dstList.getCount();
    if (oldCount == count)
    {
        return SLANG_OK;
    }
    if (count < oldCount)
    {
        dstList.unsafeShrinkToCount(count);
        return SLANG_OK;
    }

    const auto typeFuncs = typeMap->getFuncsForType(elementType);
    SLANG_ASSERT(typeFuncs.isValid());

    const Index dstCapacity = dstList.getCapacity();
    void* oldBuffer = dstList.detachBuffer();

    void* newBuffer = ::malloc(count * elementType->m_size);
    // Initialize it all first
    typeFuncs.ctorArray(typeMap, elementType, newBuffer, count);

    typeFuncs.copyArray(typeMap, elementType, newBuffer, oldBuffer, oldCount);

    // Attach the new buffer
    dstList.attachBuffer((Byte*)newBuffer, count, count);

    // Free the old buffer
    if (oldBuffer)
    {
        typeFuncs.dtorArray(typeMap, elementType, oldBuffer, dstCapacity);
        ::free(oldBuffer);
    }

    return SLANG_OK;
}

/* static */ bool RttiUtil::canMemCpy(const RttiInfo* type)
{
    switch (type->m_kind)
    {
    case RttiInfo::Kind::RefPtr:
    case RttiInfo::Kind::String:
    case RttiInfo::Kind::Invalid:
        {
            return false;
        }
    case RttiInfo::Kind::UnownedStringSlice:
    case RttiInfo::Kind::Ptr:
    case RttiInfo::Kind::Enum:
        {
            return true;
        }
    case RttiInfo::Kind::FixedArray:
        {
            const FixedArrayRttiInfo* fixedArrayRttiInfo =
                static_cast<const FixedArrayRttiInfo*>(type);
            return canMemCpy(fixedArrayRttiInfo->m_elementType);
        }
    case RttiInfo::Kind::Other:
    case RttiInfo::Kind::List:
    case RttiInfo::Kind::Dictionary:
        {
            return false;
        }
    case RttiInfo::Kind::Struct:
        {
            const StructRttiInfo* structRttiInfo = static_cast<const StructRttiInfo*>(type);

            do
            {
                // If all the fields can be zero inited, struct can be
                const auto fieldCount = structRttiInfo->m_fieldCount;
                const auto fields = structRttiInfo->m_fields;

                for (Index i = 0; i < fieldCount; ++i)
                {
                    const auto& field = fields[i];
                    if (!canMemCpy(field.m_type))
                    {
                        return false;
                    }
                }
                structRttiInfo = structRttiInfo->m_super;
            } while (structRttiInfo);

            return true;
        }
    default:
        {
            return type->isBuiltIn();
        }
    }
}

/* static */ bool RttiUtil::canZeroInit(const RttiInfo* type)
{
    switch (type->m_kind)
    {
    case RttiInfo::Kind::Invalid:
        {
            return true;
        }
    case RttiInfo::Kind::String:
        {
            // As it stands we can zero init String, but if impl changes that might not
            // be true
            return true;
        }
    case RttiInfo::Kind::UnownedStringSlice:
    case RttiInfo::Kind::Ptr:
    case RttiInfo::Kind::RefPtr:
    case RttiInfo::Kind::Enum:
        {
            return true;
        }
    case RttiInfo::Kind::FixedArray:
        {
            const FixedArrayRttiInfo* fixedArrayRttiInfo =
                static_cast<const FixedArrayRttiInfo*>(type);
            return canZeroInit(fixedArrayRttiInfo->m_elementType);
        }
    case RttiInfo::Kind::Other:
    case RttiInfo::Kind::List:
    case RttiInfo::Kind::Dictionary:
        {
            return false;
        }
    case RttiInfo::Kind::Struct:
        {
            const StructRttiInfo* structRttiInfo = static_cast<const StructRttiInfo*>(type);

            do
            {
                // If all the fields can be zero inited, struct can be
                const auto fieldCount = structRttiInfo->m_fieldCount;
                const auto fields = structRttiInfo->m_fields;

                for (Index i = 0; i < fieldCount; ++i)
                {
                    const auto& field = fields[i];
                    if (!canZeroInit(field.m_type))
                    {
                        return false;
                    }
                }
                structRttiInfo = structRttiInfo->m_super;
            } while (structRttiInfo);

            return true;
        }
    default:
        {
            return type->isBuiltIn();
        }
    }
}

/* static */ bool RttiUtil::hasDtor(const RttiInfo* type)
{
    switch (type->m_kind)
    {
    case RttiInfo::Kind::Invalid:
        {
            return false;
        }
    case RttiInfo::Kind::String:
    case RttiInfo::Kind::RefPtr:
        {
            return true;
        }
    case RttiInfo::Kind::UnownedStringSlice:
    case RttiInfo::Kind::Ptr:
    case RttiInfo::Kind::Enum:
        {
            return false;
        }
    case RttiInfo::Kind::FixedArray:
        {
            const FixedArrayRttiInfo* fixedArrayRttiInfo =
                static_cast<const FixedArrayRttiInfo*>(type);
            return hasDtor(fixedArrayRttiInfo->m_elementType);
        }
    case RttiInfo::Kind::Other:
    case RttiInfo::Kind::List:
    case RttiInfo::Kind::Dictionary:
        {
            return true;
        }
    case RttiInfo::Kind::Struct:
        {
            const StructRttiInfo* structRttiInfo = static_cast<const StructRttiInfo*>(type);

            do
            {
                // If all the fields can be zero inited, struct can be
                const auto fieldCount = structRttiInfo->m_fieldCount;
                const auto fields = structRttiInfo->m_fields;

                for (Index i = 0; i < fieldCount; ++i)
                {
                    const auto& field = fields[i];
                    if (hasDtor(field.m_type))
                    {
                        return true;
                    }
                }
                structRttiInfo = structRttiInfo->m_super;
            } while (structRttiInfo);
            return false;
        }
    default:
        {
            return !type->isBuiltIn();
        }
    }
}

/* static */ void RttiUtil::ctorArray(
    RttiTypeFuncsMap* typeMap,
    const RttiInfo* rttiInfo,
    void* inDst,
    ptrdiff_t stride,
    Index count)
{
    if (count <= 0)
    {
        return;
    }

    Byte* dst = (Byte*)inDst;
    if (canZeroInit(rttiInfo))
    {
        if (stride == rttiInfo->m_size)
        {
            ::memset(dst, 0, count * stride);
        }
        else
        {
            const size_t size = rttiInfo->m_size;
            for (Index i = 0; i < count; ++i, dst += stride)
            {
                ::memset(dst, 0, size);
            }
        }
        return;
    }

    switch (rttiInfo->m_kind)
    {
    case RttiInfo::Kind::FixedArray:
        {
            const FixedArrayRttiInfo* fixedArrayRttiInfo =
                static_cast<const FixedArrayRttiInfo*>(rttiInfo);

            if (fixedArrayRttiInfo->m_size == stride)
            {
                // It's contiguous do in one go
                ctorArray(
                    typeMap,
                    fixedArrayRttiInfo->m_elementType,
                    dst,
                    fixedArrayRttiInfo->m_elementType->m_size,
                    fixedArrayRttiInfo->m_elementCount * count);
            }
            else
            {
                // Do it in array runs
                for (Index i = 0; i < count; ++i, dst += stride)
                {
                    ctorArray(
                        typeMap,
                        fixedArrayRttiInfo->m_elementType,
                        dst,
                        fixedArrayRttiInfo->m_elementType->m_size,
                        fixedArrayRttiInfo->m_elementCount);
                }
            }
            return;
        }
    case RttiInfo::Kind::List:
    case RttiInfo::Kind::Dictionary:
    case RttiInfo::Kind::Other:
        {
            auto funcs = typeMap->getFuncsForType(rttiInfo);
            SLANG_ASSERT(funcs.isValid());

            const OtherRttiInfo* otherRttiInfo = static_cast<const OtherRttiInfo*>(rttiInfo);
            if (otherRttiInfo->m_size == stride)
            {
                funcs.ctorArray(typeMap, rttiInfo, dst, count);
            }
            else
            {
                // Do it in array runs
                for (Index i = 0; i < count; ++i, dst += stride)
                {
                    funcs.ctorArray(typeMap, rttiInfo, dst, 1);
                }
            }
            return;
        }
    case RttiInfo::Kind::Struct:
        {
            const StructRttiInfo* structRttiInfo = static_cast<const StructRttiInfo*>(rttiInfo);

            do
            {
                // If all the fields can be zero inited, struct can be
                const auto fieldCount = structRttiInfo->m_fieldCount;
                const auto fields = structRttiInfo->m_fields;

                for (Index i = 0; i < fieldCount; ++i)
                {
                    const auto& field = fields[i];
                    ctorArray(typeMap, field.m_type, dst + field.m_offset, stride, count);
                }
                structRttiInfo = structRttiInfo->m_super;
            } while (structRttiInfo);

            return;
        }
    }

    SLANG_ASSERT(!"Unexpected");
}

/* static */ void RttiUtil::copyArray(
    RttiTypeFuncsMap* typeMap,
    const RttiInfo* rttiInfo,
    void* inDst,
    const void* inSrc,
    ptrdiff_t stride,
    Index count)
{
    if (count <= 0)
    {
        return;
    }

    const size_t size = rttiInfo->m_size;

    Byte* dst = (Byte*)inDst;
    const Byte* src = (const Byte*)inSrc;
    if (canMemCpy(rttiInfo))
    {
        if (stride == ptrdiff_t(size))
        {
            ::memcpy(dst, src, count * stride);
        }
        else
        {

            for (Index i = 0; i < count; ++i, dst += stride, src += stride)
            {
                ::memcpy(dst, src, size);
            }
        }
        return;
    }

    switch (rttiInfo->m_kind)
    {
    case RttiInfo::Kind::FixedArray:
        {
            const FixedArrayRttiInfo* fixedArrayRttiInfo =
                static_cast<const FixedArrayRttiInfo*>(rttiInfo);
            const auto elementType = fixedArrayRttiInfo->m_elementType;
            const auto elementSize = elementType->m_size;
            const auto elementCount = fixedArrayRttiInfo->m_elementCount;

            if (ptrdiff_t(size) == stride)
            {
                // It's contiguous do in one go
                copyArray(typeMap, elementType, dst, src, elementSize, elementCount * count);
            }
            else
            {
                // Do it in array runs
                for (Index i = 0; i < count; ++i, dst += stride, src += stride)
                {
                    copyArray(typeMap, elementType, dst, src, elementSize, elementCount);
                }
            }
            return;
        }
    case RttiInfo::Kind::List:
    case RttiInfo::Kind::Dictionary:
    case RttiInfo::Kind::Other:
        {
            auto funcs = typeMap->getFuncsForType(rttiInfo);
            SLANG_ASSERT(funcs.isValid());

            const OtherRttiInfo* otherRttiInfo = static_cast<const OtherRttiInfo*>(rttiInfo);
            if (otherRttiInfo->m_size == stride)
            {
                funcs.copyArray(typeMap, rttiInfo, dst, src, count);
            }
            else
            {
                for (Index i = 0; i < count; ++i, dst += stride, src += stride)
                {
                    funcs.copyArray(typeMap, rttiInfo, dst, src, 1);
                }
            }
            return;
        }
    case RttiInfo::Kind::Struct:
        {
            const StructRttiInfo* structRttiInfo = static_cast<const StructRttiInfo*>(rttiInfo);

            do
            {
                // If all the fields can be zero inited, struct can be
                const auto fieldCount = structRttiInfo->m_fieldCount;
                const auto fields = structRttiInfo->m_fields;

                for (Index i = 0; i < fieldCount; ++i)
                {
                    const auto& field = fields[i];
                    copyArray(
                        typeMap,
                        field.m_type,
                        dst + field.m_offset,
                        src + field.m_offset,
                        stride,
                        count);
                }
                structRttiInfo = structRttiInfo->m_super;
            } while (structRttiInfo);

            return;
        }
    }

    SLANG_ASSERT(!"Unexpected");
}

/* static */ void RttiUtil::dtorArray(
    RttiTypeFuncsMap* typeMap,
    const RttiInfo* rttiInfo,
    void* inDst,
    ptrdiff_t stride,
    Index count)
{
    if (count <= 0 || !hasDtor(rttiInfo))
    {
        return;
    }

    const size_t size = rttiInfo->m_size;
    Byte* dst = (Byte*)inDst;

    switch (rttiInfo->m_kind)
    {
    case RttiInfo::Kind::FixedArray:
        {
            const FixedArrayRttiInfo* fixedArrayRttiInfo =
                static_cast<const FixedArrayRttiInfo*>(rttiInfo);
            const auto elementType = fixedArrayRttiInfo->m_elementType;
            const auto elementSize = elementType->m_size;
            const auto elementCount = fixedArrayRttiInfo->m_elementCount;

            if (ptrdiff_t(size) == stride)
            {
                // It's contiguous do in one go
                dtorArray(typeMap, elementType, dst, elementSize, elementCount * count);
            }
            else
            {
                // Do it in array runs
                for (Index i = 0; i < count; ++i, dst += stride)
                {
                    dtorArray(typeMap, elementType, dst, elementSize, elementCount);
                }
            }
            return;
        }
    case RttiInfo::Kind::List:
    case RttiInfo::Kind::Dictionary:
    case RttiInfo::Kind::Other:
        {
            auto funcs = typeMap->getFuncsForType(rttiInfo);
            SLANG_ASSERT(funcs.isValid());

            const OtherRttiInfo* otherRttiInfo = static_cast<const OtherRttiInfo*>(rttiInfo);
            if (otherRttiInfo->m_size == stride)
            {
                funcs.dtorArray(typeMap, rttiInfo, dst, count);
            }
            else
            {
                for (Index i = 0; i < count; ++i, dst += stride)
                {
                    funcs.dtorArray(typeMap, rttiInfo, dst, 1);
                }
            }
            return;
        }
    case RttiInfo::Kind::Struct:
        {
            const StructRttiInfo* structRttiInfo = static_cast<const StructRttiInfo*>(rttiInfo);

            do
            {
                // If all the fields can be zero inited, struct can be
                const auto fieldCount = structRttiInfo->m_fieldCount;
                const auto fields = structRttiInfo->m_fields;

                for (Index i = 0; i < fieldCount; ++i)
                {
                    const auto& field = fields[i];
                    dtorArray(typeMap, field.m_type, dst + field.m_offset, stride, count);
                }
                structRttiInfo = structRttiInfo->m_super;
            } while (structRttiInfo);

            return;
        }
    }

    SLANG_ASSERT(!"Unexpected");
}

} // namespace Slang
