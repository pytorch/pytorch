#ifndef SLANG_CORE_RTTI_UTIL_H
#define SLANG_CORE_RTTI_UTIL_H

#include "slang-rtti-info.h"

// Some macros to help generate StructRttiInfo for structs without too much
// boilerplate, use as so:
//
// struct MyStruct
// {
//     int an_optional_field;
//     List<UnownedStringSlice> a_list_of_strings_field;
// }
// SLANG_MAKE_STRUCT_RTTI_INFO(
//     MyStruct,
//     SLANG_OPTIONAL_RTTI_FIELD(an_optional_field),
//     SLANG_RTTI_FIELD(a_list_of_strings_field)
//  );
//
// This allows parsing JSON objects like
// {
//     "an_optional_field": 10,
//     "a_list_of_strings_field": ["hello", "world"]
// }
//
// Convert from such JSON objects using JSONToNativeConverter::convert

#define SLANG_MAKE_STRUCT_RTTI_INFO(S, ...)                            \
    template<>                                                         \
    struct GetRttiInfo<S>                                              \
    {                                                                  \
        static const RttiInfo* get()                                   \
        {                                                              \
            using S_ = S;                                              \
            const static StructRttiInfo::Field fs[] = {__VA_ARGS__};   \
            const auto ignoreUnknownFields = true;                     \
            const static auto ret = StructRttiInfo{                    \
                {{RttiInfo::Kind::Struct, alignof(S), sizeof(S)}, #S}, \
                nullptr,                                               \
                SLANG_COUNT_OF(fs),                                    \
                fs,                                                    \
                ignoreUnknownFields};                                  \
            return &ret;                                               \
        }                                                              \
    };
#define SLANG_RTTI_FIELD_IMPL(m, name, flags)                             \
    {                                                                     \
        name, GetRttiInfo<decltype(S_::m)>::get(), offsetof(S_, m), flags \
    }
#define SLANG_RTTI_FIELD(m) SLANG_RTTI_FIELD_IMPL(m, #m, 0)
#define SLANG_OPTIONAL_RTTI_FIELD(m) SLANG_RTTI_FIELD_IMPL(m, #m, StructRttiInfo::Flag::Optional)

namespace Slang
{

struct RttiUtil
{
    static SlangResult setInt(int64_t value, const RttiInfo* rttiInfo, void* dst);
    static int64_t getInt64(const RttiInfo* rttiInfo, const void* src);

    static double asDouble(const RttiInfo* rttiInfo, const void* src);

    static SlangResult setFromDouble(double v, const RttiInfo* rttiInfo, void* dst);

    static bool asBool(const RttiInfo* rttiInfo, const void* src);

    static bool isDefault(RttiDefaultValue defaultValue, const RttiInfo* rttiInfo, const void* src);

    /// Gets funcs for default scenarios
    static RttiTypeFuncs getDefaultTypeFuncs(const RttiInfo* rttiInfo);

    /// Set a list count
    static SlangResult setListCount(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* elementType,
        void* dst,
        Index count);

    /// Returns if the type can be zero initialized
    static bool canZeroInit(const RttiInfo* type);
    /// Returns true if the type needs dtor
    static bool hasDtor(const RttiInfo* type);
    /// Returns true if we can memcpy to copy
    static bool canMemCpy(const RttiInfo* type);

    static void ctorArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* inDst,
        ptrdiff_t stride,
        Index count);
    static void copyArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* inDst,
        const void* inSrc,
        ptrdiff_t stride,
        Index count);
    static void dtorArray(
        RttiTypeFuncsMap* typeMap,
        const RttiInfo* rttiInfo,
        void* inDst,
        ptrdiff_t stride,
        Index count);
};

} // namespace Slang

#endif // SLANG_CORE_RTTI_UTIL_H
