#ifndef SLANG_CORE_RTTI_INFO_H
#define SLANG_CORE_RTTI_INFO_H

#include "slang-basic.h"
#include "slang-dictionary.h"
#include "slang-list.h"
#include "slang-memory-arena.h"

namespace Slang
{

struct RttiInfo;
struct RttiTypeFuncsMap;

struct RttiTypeFuncs
{
    typedef void (
        *CtorArray)(RttiTypeFuncsMap* typeMap, const RttiInfo* rttiInfo, void* dst, Index count);
    typedef void (
        *DtorArray)(RttiTypeFuncsMap* typeMap, const RttiInfo* rttiInfo, void* dst, Index count);
    typedef void (*CopyArray)(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* dst,
        const void* src,
        Index count);

    bool isValid() const { return ctorArray && dtorArray && copyArray; }

    static RttiTypeFuncs makeEmpty() { return RttiTypeFuncs{nullptr, nullptr, nullptr}; }

    CtorArray ctorArray;
    DtorArray dtorArray;
    CopyArray copyArray;
};

/* Provides a mechanism to map a type to it's RttiFuncs */
struct RttiTypeFuncsMap
{
    /// For a given type returns the funcs.
    /// If not found returns funcs that return 'isValid' as false.
    RttiTypeFuncs getFuncsForType(const RttiInfo* rttiInfo);

    /// Add funcs for a type
    void add(const RttiInfo* rttiInfo, const RttiTypeFuncs& funcs);

protected:
    Dictionary<const RttiInfo*, RttiTypeFuncs> m_map;
};

/* Template to get funcs for any arbitrary type */
template<typename T>
struct GetRttiTypeFuncs
{
    static void ctorArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* in,
        Index count)
    {
        SLANG_UNUSED(typeMap);
        SLANG_UNUSED(rttiInfo);
        T* dst = (T*)in;
        for (Index i = 0; i < count; ++i)
        {
            new (dst + i) T;
        }
    }
    static void dtorArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* in,
        Index count)
    {
        SLANG_UNUSED(typeMap);
        SLANG_UNUSED(rttiInfo);
        T* dst = (T*)in;
        for (Index i = 0; i < count; ++i)
        {
            (dst + i)->~T();
        }
    }
    static void copyArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* inDst,
        const void* inSrc,
        Index count)
    {
        SLANG_UNUSED(rttiInfo);
        SLANG_UNUSED(typeMap);

        T* dst = (T*)inDst;
        const T* src = (T*)inSrc;
        for (Index i = 0; i < count; ++i)
        {
            dst[i] = src[i];
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

/* An implementation of funcs, for a type that is POD *and* can be zero initialized.
Built in types generally fall into this catagory, but so do raw pointers and other types,
such as structs that only contain "ZeroPod" types */
template<typename T>
struct GetRttiTypeFuncsForZeroPod
{
    static void ctorArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* dst,
        Index count)
    {
        SLANG_UNUSED(typeMap);
        SLANG_UNUSED(rttiInfo);
        ::memset(dst, 0, sizeof(T) * count);
    }
    static void dtorArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* dst,
        Index count)
    {
        SLANG_UNUSED(typeMap);
        SLANG_UNUSED(rttiInfo);
        SLANG_UNUSED(dst);
        SLANG_UNUSED(count);
    }
    static void copyArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* dst,
        const void* src,
        Index count)
    {
        SLANG_UNUSED(typeMap);
        SLANG_UNUSED(rttiInfo);
        ::memcpy(dst, src, sizeof(T) * count);
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

struct RttiInfo
{
    typedef uint8_t AlignmentType;
    typedef uint16_t SizeType;

    enum class Kind : uint8_t
    {
        Invalid,
        I32,
        U32,
        I64,
        U64,
        F32,
        F64,
        Bool,
        String,
        UnownedStringSlice,
        Ptr,
        RefPtr,
        FixedArray,
        Struct,
        Other,
        Enum,
        List,
        Dictionary,

        CountOf,
    };

    Kind m_kind;
    AlignmentType m_alignment;
    SizeType m_size;

    void init(Kind kind, size_t alignment, size_t size)
    {
        m_kind = kind;
        m_alignment = AlignmentType(alignment);
        m_size = SizeType(size);
    }

    template<typename T>
    void init(Kind kind)
    {
        init(kind, SLANG_ALIGN_OF(T), sizeof(T));
    }

    /// Allocate memory for RttiInfo types.
    /// Is thread safe, and doesn't require the memory to be freed explicitly
    /// Will be freed at shutdown (via global dtor)
    static void* allocate(size_t size);
    /// Will free up any allocations. Can only be called at shutdown, and there are guarenteed no
    /// uses of RttiInfo - otherwise contents may be undefined. NOTE! Memory *will* be freed with
    /// final dtors, but if memory check functions are used they can report this memory.
    static void deallocateAll();

    static bool isIntegral(RttiInfo::Kind kind)
    {
        return Index(kind) >= Index(RttiInfo::Kind::I32) &&
               Index(kind) <= Index(RttiInfo::Kind::U64);
    }
    static bool isFloat(RttiInfo::Kind kind)
    {
        return kind == RttiInfo::Kind::F32 || kind == RttiInfo::Kind::F64;
    }
    static bool isBuiltIn(RttiInfo::Kind kind)
    {
        return Index(kind) >= Index(RttiInfo::Kind::I32) &&
               Index(kind) <= Index(RttiInfo::Kind::Bool);
    }
    static bool isNamed(RttiInfo::Kind kind)
    {
        return Index(kind) >= Index(RttiInfo::Kind::Struct) &&
               Index(kind) <= Index(RttiInfo::Kind::Enum);
    }

    bool isIntegral() const { return isIntegral(m_kind); }
    bool isFloat() const { return isFloat(m_kind); }
    bool isBuiltIn() const { return isBuiltIn(m_kind); }
    bool isNamed() const { return isNamed(m_kind); }

    static void append(const RttiInfo* info, StringBuilder& out);

    static const RttiInfo g_basicTypes[Index(Kind::CountOf)];
};

// Can combine into flags on a field. Could store default value with a field,
// but this works fine for most purposes
enum class RttiDefaultValue : uint8_t
{
    Normal, ///< Zero for integral/float types/false for bool
    One,
    MinusOne,

    Mask = 0x7,
};

struct NamedRttiInfo : public RttiInfo
{
    const char* m_name; ///< Name
};

struct StructRttiInfo : public NamedRttiInfo
{
    typedef uint8_t Flags;
    struct Flag
    {
        enum Enum : Flags
        {
            // We use low bits for 'RttiDefaultValue' value
            Optional = 0x8,
        };
    };

    struct Field
    {
        const char* m_name;     ///< Name of this field
        const RttiInfo* m_type; ///< The type of this field
        uint32_t m_offset;      ///< Offset from object type in bytes
        Flags m_flags;          ///< Field flags
    };

    const StructRttiInfo* m_super; ///< Super class or nullptr if not defined

    Index m_fieldCount;    ///< Amount of fields
    const Field* m_fields; ///< Fields
    bool m_ignoreUnknownFieldsInJson = false;
};

struct EnumRttiInfo : public NamedRttiInfo
{
    // TODO(JS):
};

SLANG_FORCE_INLINE StructRttiInfo::Flags combine(
    StructRttiInfo::Flags flags,
    RttiDefaultValue defaultValue)
{
    return StructRttiInfo::Flags(defaultValue) | flags;
}

struct ListRttiInfo : public RttiInfo
{
    const RttiInfo* m_elementType;
};

struct DictionaryRttiInfo : public RttiInfo
{
    const RttiInfo* m_keyType;
    const RttiInfo* m_valueType;
};

struct PtrRttiInfo : public RttiInfo
{
    const RttiInfo* m_targetType;
};

struct RefPtrRttiInfo : public RttiInfo
{
    const RttiInfo* m_targetType;
};

struct FixedArrayRttiInfo : public RttiInfo
{
    const RttiInfo* m_elementType;
    size_t m_elementCount;
};

struct OtherRttiInfo : public NamedRttiInfo
{
    typedef bool (*IsDefaultFunc)(const RttiInfo* rttiInfo, const void* in);
    IsDefaultFunc m_isDefaultFunc;
    RttiTypeFuncs m_typeFuncs;
};

// The default is to just get the info from a global held inside the type.
template<typename T>
struct GetRttiInfo
{
    SLANG_FORCE_INLINE static const RttiInfo* get() { return &T::g_rttiInfo; }
};

template<>
struct GetRttiInfo<bool>
{
    static const RttiInfo* get() { return &RttiInfo::g_basicTypes[Index(RttiInfo::Kind::Bool)]; }
};
template<>
struct GetRttiInfo<int32_t>
{
    static const RttiInfo* get() { return &RttiInfo::g_basicTypes[Index(RttiInfo::Kind::I32)]; }
};
template<>
struct GetRttiInfo<int64_t>
{
    static const RttiInfo* get() { return &RttiInfo::g_basicTypes[Index(RttiInfo::Kind::I64)]; }
};
template<>
struct GetRttiInfo<uint32_t>
{
    static const RttiInfo* get() { return &RttiInfo::g_basicTypes[Index(RttiInfo::Kind::U32)]; }
};
template<>
struct GetRttiInfo<uint64_t>
{
    static const RttiInfo* get() { return &RttiInfo::g_basicTypes[Index(RttiInfo::Kind::U64)]; }
};
template<>
struct GetRttiInfo<float>
{
    static const RttiInfo* get() { return &RttiInfo::g_basicTypes[Index(RttiInfo::Kind::F32)]; }
};
template<>
struct GetRttiInfo<double>
{
    static const RttiInfo* get() { return &RttiInfo::g_basicTypes[Index(RttiInfo::Kind::F64)]; }
};
template<>
struct GetRttiInfo<String>
{
    static const RttiInfo* get() { return &RttiInfo::g_basicTypes[Index(RttiInfo::Kind::String)]; }
};
template<>
struct GetRttiInfo<UnownedStringSlice>
{
    static const RttiInfo* get()
    {
        return &RttiInfo::g_basicTypes[Index(RttiInfo::Kind::UnownedStringSlice)];
    }
};

template<typename T>
struct GetRttiInfo<List<T>>
{
    static const ListRttiInfo _make()
    {
        ListRttiInfo info;
        info.init<List<Byte>>(RttiInfo::Kind::List);
        info.m_elementType = GetRttiInfo<T>::get();
        return info;
    }
    static const RttiInfo* get()
    {
        static const ListRttiInfo g_info = _make();
        return &g_info;
    }
};

// Strip const
template<typename T>
struct GetRttiInfo<const T>
{
    static const RttiInfo* get() { return GetRttiInfo<T>::get(); }
};

template<typename K, typename V>
struct GetRttiInfo<Dictionary<K, V>>
{
    static const DictionaryRttiInfo _make()
    {
        DictionaryRttiInfo info;
        info.init<Dictionary<Byte, Byte>>(RttiInfo::Kind::Dictionary);
        info.m_keyType = GetRttiInfo<K>::get();
        info.m_valueType = GetRttiInfo<V>::get();
        return info;
    }
    static const RttiInfo* get()
    {
        static const DictionaryRttiInfo g_info = _make();
        return &g_info;
    }
};

template<typename TARGET>
struct GetRttiInfo<TARGET*>
{
    static const PtrRttiInfo _make()
    {
        PtrRttiInfo info;
        info.init<void*>(RttiInfo::Kind::Ptr);
        info.m_targetType = GetRttiInfo<TARGET>::get();
        return info;
    }
    static const RttiInfo* get()
    {
        static const PtrRttiInfo g_info = _make();
        return &g_info;
    }
};

template<typename TARGET>
struct GetRttiInfo<RefPtr<TARGET>>
{
    static const RefPtrRttiInfo _make()
    {
        RefPtrRttiInfo info;
        info.init<RefPtr<StringRepresentation>>(RttiInfo::Kind::RefPtr);
        info.m_targetType = GetRttiInfo<TARGET>::get();
        return info;
    }
    static const RttiInfo* get()
    {
        static const RefPtrRttiInfo g_info = _make();
        return &g_info;
    }
};

template<typename T, size_t COUNT>
struct GetRttiInfo<T[COUNT]>
{
    static const FixedArrayRttiInfo _make()
    {
        FixedArrayRttiInfo info;
        info.m_kind = RttiInfo::Kind::FixedArray;
        info.m_alignment = RttiInfo::AlignmentType(SLANG_ALIGN_OF(T));
        info.m_size = RttiInfo::SizeType(sizeof(T) * COUNT);
        info.m_elementType = GetRttiInfo<T>::get();
        info.m_elementCount = COUNT;
        return info;
    }
    static const RttiInfo* get()
    {
        static const FixedArrayRttiInfo g_info = _make();
        return &g_info;
    }
};

struct StructRttiBuilder
{
    template<typename T>
    StructRttiBuilder(T* obj, const char* name, const StructRttiInfo* super)
    {
        m_rttiInfo.init<T>(RttiInfo::Kind::Struct);
        _init(name, super, (const Byte*)obj);
    }

    template<typename T>
    void addField(const char* name, const T* fieldPtr, StructRttiInfo::Flags flags = 0)
    {
        StructRttiInfo::Field field;

        field.m_name = name;
        field.m_type = GetRttiInfo<T>::get();
        field.m_offset = uint32_t(ptrdiff_t((const Byte*)fieldPtr - m_base));
        field.m_flags = flags;
        m_fields.add(field);
    }

    void ignoreUnknownFields() { m_rttiInfo.m_ignoreUnknownFieldsInJson = true; }

    StructRttiInfo make();

    void _init(const char* name, const StructRttiInfo* super, const Byte* base);

    StructRttiInfo m_rttiInfo;

    List<StructRttiInfo::Field> m_fields;
    const Byte* m_base;
};


} // namespace Slang

#endif // SLANG_CORE_RTTI_INFO_H
