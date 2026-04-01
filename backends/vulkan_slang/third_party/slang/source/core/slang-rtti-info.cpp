#include "slang-rtti-info.h"

#include "slang-com-helper.h"
#include "slang-rtti-util.h"

#include <mutex>

namespace Slang
{

#define SLANG_RTTI_INFO_INVALID(name) \
    RttiInfo                          \
    {                                 \
        RttiInfo::Kind::Invalid, 0, 0 \
    }
#define SLANG_RTTI_INFO_BASIC(name, type)                                    \
    RttiInfo                                                                 \
    {                                                                        \
        RttiInfo::Kind::name, RttiInfo::AlignmentType(SLANG_ALIGN_OF(type)), \
            RttiInfo::SizeType(sizeof(type))                                 \
    }

/* static */ const RttiInfo RttiInfo::g_basicTypes[Index(Kind::CountOf)] = {
    SLANG_RTTI_INFO_INVALID(Invalid),
    SLANG_RTTI_INFO_BASIC(I32, int32_t),
    SLANG_RTTI_INFO_BASIC(U32, uint32_t),
    SLANG_RTTI_INFO_BASIC(I64, int64_t),
    SLANG_RTTI_INFO_BASIC(U64, uint64_t),
    SLANG_RTTI_INFO_BASIC(F32, float),
    SLANG_RTTI_INFO_BASIC(F64, double),
    SLANG_RTTI_INFO_BASIC(Bool, bool),
    SLANG_RTTI_INFO_BASIC(String, String),
    SLANG_RTTI_INFO_BASIC(UnownedStringSlice, UnownedStringSlice),
    SLANG_RTTI_INFO_BASIC(Ptr, void*),
    SLANG_RTTI_INFO_BASIC(RefPtr, RefPtr<StringRepresentation>),
    SLANG_RTTI_INFO_INVALID(FixedArray),
    SLANG_RTTI_INFO_INVALID(Struct),
    SLANG_RTTI_INFO_INVALID(Other),
    SLANG_RTTI_INFO_INVALID(Enum),
    SLANG_RTTI_INFO_INVALID(List),
    SLANG_RTTI_INFO_INVALID(Dictionary),
};

struct RttiInfoManager
{
    void* allocate(size_t size)
    {
        std::lock_guard<std::recursive_mutex> guard(m_mutex);
        return m_arena.allocate(size);
    }
    void deallocateAll()
    {
        std::lock_guard<std::recursive_mutex> guard(m_mutex);
        m_arena.reset();
    }

    static RttiInfoManager& getSingleton()
    {
        static RttiInfoManager g_manager;
        return g_manager;
    }

protected:
    RttiInfoManager()
        : m_arena(1024)
    {
    }

    std::recursive_mutex m_mutex; ///< We need a mutex to guard access to m_arena
    MemoryArena m_arena;
};

/* static */ void* RttiInfo::allocate(size_t size)
{
    return RttiInfoManager::getSingleton().allocate(size);
}

/* static */ void RttiInfo::deallocateAll()
{
    return RttiInfoManager::getSingleton().deallocateAll();
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! StructRttiBuilder !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

static void _appendFixedArray(const FixedArrayRttiInfo* inFixedArray, StringBuilder& out)
{
    List<const FixedArrayRttiInfo*> fixedArrays;
    fixedArrays.add(inFixedArray);

    const RttiInfo* cur = inFixedArray->m_elementType;
    while (cur->m_kind == RttiInfo::Kind::FixedArray)
    {
        const FixedArrayRttiInfo* curArray = static_cast<const FixedArrayRttiInfo*>(cur);
        fixedArrays.add(curArray);
        cur = curArray->m_elementType;
    }

    // Append the 'target' which is in cur
    RttiInfo::append(cur, out);
    // Now all the fixed array values, in order
    for (auto fixedArray : fixedArrays)
    {
        out << "[" << int32_t(fixedArray->m_elementCount) << "]";
    }
}

/* static */ void RttiInfo::append(const RttiInfo* info, StringBuilder& out)
{
    switch (info->m_kind)
    {
    case RttiInfo::Kind::I32:
        out << "int32_t";
        break;
    case RttiInfo::Kind::U32:
        out << "uint32_t";
        break;
    case RttiInfo::Kind::I64:
        out << "int64_t";
        break;
    case RttiInfo::Kind::U64:
        out << "uint64_t";
        break;
    case RttiInfo::Kind::F32:
        out << "float";
        break;
    case RttiInfo::Kind::F64:
        out << "double";
        break;
    case RttiInfo::Kind::Bool:
        out << "bool";
        break;
    case RttiInfo::Kind::String:
        out << "String";
        break;
    case RttiInfo::Kind::UnownedStringSlice:
        out << "UnownedStringSlice";
        break;
    case RttiInfo::Kind::Ptr:
        {
            const PtrRttiInfo* ptrRttiInfo = static_cast<const PtrRttiInfo*>(info);
            append(ptrRttiInfo->m_targetType, out);
            out << "*";
            break;
        }
    case RttiInfo::Kind::RefPtr:
        {
            const RefPtrRttiInfo* ptrRttiInfo = static_cast<const RefPtrRttiInfo*>(info);
            out << "RefPtr<";
            append(ptrRttiInfo->m_targetType, out);
            out << ">";
            break;
        }
    case RttiInfo::Kind::FixedArray:
        {
            const FixedArrayRttiInfo* arrayRttiInfo = static_cast<const FixedArrayRttiInfo*>(info);
            _appendFixedArray(arrayRttiInfo, out);
            break;
        }
    case RttiInfo::Kind::List:
        {
            const ListRttiInfo* listRttiInfo = static_cast<const ListRttiInfo*>(info);
            out << "List<";
            append(listRttiInfo->m_elementType, out);
            out << ">";
            break;
        }
    case RttiInfo::Kind::Dictionary:
        {
            const DictionaryRttiInfo* dictionaryRttiInfo =
                static_cast<const DictionaryRttiInfo*>(info);

            out << "Dictionary<";
            append(dictionaryRttiInfo->m_keyType, out);
            out << ",";
            append(dictionaryRttiInfo->m_valueType, out);
            out << ">";
            break;
        }
    default:
        {
            if (info->isNamed())
            {
                const NamedRttiInfo* namedRttiInfo = static_cast<const NamedRttiInfo*>(info);
                out << namedRttiInfo->m_name;
                break;
            }

            out << "%Unknown%";
            break;
        }
    }
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! StructRttiBuilder !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

void StructRttiBuilder::_init(const char* name, const StructRttiInfo* super, const Byte* base)
{
    m_rttiInfo.m_name = name;
    m_rttiInfo.m_super = super;
    m_base = base;

    m_rttiInfo.m_fieldCount = 0;
    m_rttiInfo.m_fields = nullptr;
}

StructRttiInfo StructRttiBuilder::make()
{
    const Index fieldCount = m_fields.getCount();

    if (fieldCount)
    {
        StructRttiInfo::Field* dstFields =
            (StructRttiInfo::Field*)RttiInfo::allocate(sizeof(StructRttiInfo::Field) * fieldCount);
        ::memcpy(dstFields, m_fields.getBuffer(), sizeof(StructRttiInfo::Field) * fieldCount);

        m_rttiInfo.m_fields = dstFields;
        m_rttiInfo.m_fieldCount = fieldCount;
    }

    return m_rttiInfo;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! RttiTypeFuncsMap !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

RttiTypeFuncs RttiTypeFuncsMap::getFuncsForType(const RttiInfo* rttiInfo)
{
    if (auto funcsPtr = m_map.tryGetValue(rttiInfo))
    {
        return *funcsPtr;
    }

    // Try to get the default impl
    // NOTE! funcs could be invalid if there is no default impl.
    const auto funcs = RttiUtil::getDefaultTypeFuncs(rttiInfo);

    // Add to the map
    m_map.add(rttiInfo, funcs);
    return funcs;
}

void RttiTypeFuncsMap::add(const RttiInfo* rttiInfo, const RttiTypeFuncs& funcs)
{
    if (auto funcsPtr = m_map.tryGetValueOrAdd(rttiInfo, funcs))
    {
        // If there are funcs set, they aren't valid otherwise this would be
        // replacing, so assert on that scenario.
        SLANG_ASSERT(!funcsPtr->isValid());

        // Replace the funcs
        *funcsPtr = funcs;
    }
}

} // namespace Slang
