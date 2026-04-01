// slang-artifact-desc-util.cpp
#include "slang-artifact-desc-util.h"

#include "../core/slang-io.h"
#include "../core/slang-type-text-util.h"
#include "slang-artifact-impl.h"
#include "slang-artifact-representation.h"

namespace Slang
{

namespace
{ // anonymous

struct HierarchicalEnumEntry
{
    Index value;
    Index parent;
    const char* name;
};

static bool _isHierarchicalEnumOk(ConstArrayView<HierarchicalEnumEntry> entries, Count countOf)
{
    // All values should be set
    if (entries.getCount() != countOf)
    {
        return false;
    }

    List<uint8_t> isUsed;
    isUsed.setCount(countOf);
    ::memset(isUsed.getBuffer(), 0, countOf);

    for (const auto& entry : entries)
    {
        const auto value = entry.value;
        // Must be in range
        if (value < 0 || value >= countOf)
        {
            return false;
        }

        if (isUsed[value] != 0)
        {
            return false;
        }
        // Mark as used
        isUsed[value]++;
    }

    // There can't be any gaps
    for (auto v : isUsed)
    {
        if (v == 0)
        {
            return false;
        }
    }

    // Okay, looks reasonable..
    return true;
}

template<typename T>
struct HierarchicalEnumTable
{
    HierarchicalEnumTable(ConstArrayView<HierarchicalEnumEntry> entries)
    {
        // Remove warnings around this not being used.
        {
            const auto unused = _isHierarchicalEnumOk;
            SLANG_UNUSED(unused);
        }

        SLANG_COMPILE_TIME_ASSERT(Index(T::Invalid) < Index(T::Base));
        SLANG_ASSERT(entries.getCount() == Count(T::CountOf));

        SLANG_ASSERT(_isHierarchicalEnumOk(entries, Count(T::CountOf)));

        ::memset(&m_parents, 0, sizeof(m_parents));

        for (const auto& entry : entries)
        {
            const auto value = entry.value;
            m_parents[value] = T(entry.parent);
            m_names[value] = UnownedStringSlice(entry.name);
        }

        // TODO(JS): NOTE! If we wanted to use parent to indicate if a value was *invalid*
        // we would want the Parent of Base to be Base.
        //
        // Base parent should be invalid
        SLANG_ASSERT(getParent(T::Base) == T::Invalid);
        // Invalids parent should be invalid
        SLANG_ASSERT(getParent(T::Invalid) == T::Invalid);
    }

    T getParent(T kind) const { return (kind >= T::CountOf) ? T::Invalid : m_parents[Index(kind)]; }
    UnownedStringSlice getName(T kind) const
    {
        return (kind >= T::CountOf) ? UnownedStringSlice() : m_names[Index(kind)];
    }

    bool isDerivedFrom(T type, T base) const
    {
        if (Index(type) >= Index(T::CountOf))
        {
            return false;
        }

        do
        {
            if (type == base)
            {
                return true;
            }
            type = m_parents[Index(type)];
        } while (Index(type) >= Index(T::Base));

        return false;
    }

protected:
    T m_parents[Count(T::CountOf)];
    UnownedStringSlice m_names[Count(T::CountOf)];
};

} // namespace

// Macro utils to create "enum hierarchy" tables

#define SLANG_HIERARCHICAL_ENUM_GET_VALUES(ENUM_TYPE, ENUM_TYPE_MACRO, ENUM_ENTRY_MACRO)   \
    static ConstArrayView<HierarchicalEnumEntry> _getEntries##ENUM_TYPE()                  \
    {                                                                                      \
        static const HierarchicalEnumEntry values[] = {ENUM_TYPE_MACRO(ENUM_ENTRY_MACRO)}; \
        return makeConstArrayView(values);                                                 \
    }

#define SLANG_HIERARCHICAL_ENUM(ENUM_TYPE, ENUM_TYPE_MACRO, ENUM_VALUE_MACRO)                   \
    SLANG_HIERARCHICAL_ENUM_GET_VALUES(ENUM_TYPE, ENUM_TYPE_MACRO, ENUM_VALUE_MACRO)            \
                                                                                                \
    static const HierarchicalEnumTable<ENUM_TYPE> g_table##ENUM_TYPE(_getEntries##ENUM_TYPE()); \
                                                                                                \
    ENUM_TYPE getParent(ENUM_TYPE kind)                                                         \
    {                                                                                           \
        return g_table##ENUM_TYPE.getParent(kind);                                              \
    }                                                                                           \
    UnownedStringSlice getName(ENUM_TYPE kind)                                                  \
    {                                                                                           \
        return g_table##ENUM_TYPE.getName(kind);                                                \
    }                                                                                           \
    bool isDerivedFrom(ENUM_TYPE kind, ENUM_TYPE base)                                          \
    {                                                                                           \
        return g_table##ENUM_TYPE.isDerivedFrom(kind, base);                                    \
    }

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!! ArtifactKind !!!!!!!!!!!!!!!!!!!!!!! */

// clang-format off
#define SLANG_ARTIFACT_KIND(x) \
    x(Invalid, Invalid) \
    x(Base, Invalid) \
        x(None, Base) \
        x(Unknown, Base) \
        x(BinaryFormat, Base) \
            x(Container, BinaryFormat) \
                x(Zip, Container) \
                x(RiffContainer, Container) \
                x(RiffLz4Container, Container) \
                x(RiffDeflateContainer, Container) \
            x(CompileBinary, BinaryFormat) \
                x(ObjectCode, CompileBinary) \
                x(Library, CompileBinary) \
                x(Executable, CompileBinary) \
                x(SharedLibrary, CompileBinary) \
                x(HostCallable, CompileBinary) \
        x(Text, Base) \
            x(HumanText, Text) \
            x(Source, Text) \
            x(Assembly, Text) \
            x(Json, Text) \
        x(Instance, Base)

#define SLANG_ARTIFACT_KIND_ENTRY(TYPE, PARENT) { Index(ArtifactKind::TYPE), Index(ArtifactKind::PARENT), #TYPE },

SLANG_HIERARCHICAL_ENUM(ArtifactKind, SLANG_ARTIFACT_KIND, SLANG_ARTIFACT_KIND_ENTRY)

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!! ArtifactPayload !!!!!!!!!!!!!!!!!!!!!!! */

#define SLANG_ARTIFACT_PAYLOAD(x) \
    x(Invalid, Invalid) \
    x(Base, Invalid) \
        x(None, Base) \
        x(Unknown, Base) \
        x(Source, Base) \
            x(C, Source) \
            x(Cpp, Source) \
            x(HLSL, Source) \
            x(GLSL, Source) \
            x(CUDA, Source) \
            x(Metal, Source) \
            x(Slang, Source) \
            x(WGSL, Source) \
        x(KernelLike, Base) \
            x(DXIL, KernelLike) \
            x(DXBC, KernelLike) \
            x(SPIRV, KernelLike) \
            x(PTX, KernelLike) \
            x(CuBin, KernelLike) \
            x(MetalAIR, KernelLike) \
            x(WGSL_SPIRV, KernelLike) \
        x(CPULike, Base) \
            x(UnknownCPU, CPULike) \
            x(X86, CPULike) \
            x(X86_64, CPULike) \
            x(Aarch, CPULike) \
            x(Aarch64, CPULike) \
            x(HostCPU, CPULike) \
            x(UniversalCPU, CPULike) \
        x(GeneralIR, Base) \
            x(SlangIR, GeneralIR) \
            x(LLVMIR, GeneralIR) \
        x(AST, Base) \
            x(SlangAST, AST) \
        x(CompileResults, Base) \
        x(Metadata, Base) \
            x(DebugInfo, Metadata) \
                x(PdbDebugInfo, DebugInfo) \
            x(Diagnostics, Metadata) \
            x(PostEmitMetadata, Metadata) \
        x(Miscellaneous, Base) \
            x(Log, Miscellaneous) \
            x(Lock, Miscellaneous) \
        x(SourceMap, Base)

#define SLANG_ARTIFACT_PAYLOAD_ENTRY(TYPE, PARENT) { Index(ArtifactPayload::TYPE), Index(ArtifactPayload::PARENT), #TYPE },

SLANG_HIERARCHICAL_ENUM(ArtifactPayload, SLANG_ARTIFACT_PAYLOAD, SLANG_ARTIFACT_PAYLOAD_ENTRY)

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!! ArtifactStyle !!!!!!!!!!!!!!!!!!!!!!! */

#define SLANG_ARTIFACT_STYLE(x) \
    x(Invalid, Invalid) \
    x(Base, Invalid) \
        x(None, Base) \
        x(Unknown, Base) \
        x(CodeLike, Base) \
            x(Kernel, CodeLike) \
            x(Host, CodeLike) \
        x(Obfuscated, Base)
// clang-format on

#define SLANG_ARTIFACT_STYLE_ENTRY(TYPE, PARENT) \
    {Index(ArtifactStyle::TYPE), Index(ArtifactStyle::PARENT), #TYPE},

SLANG_HIERARCHICAL_ENUM(ArtifactStyle, SLANG_ARTIFACT_STYLE, SLANG_ARTIFACT_STYLE_ENTRY)

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ArtifactDescUtil !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* static */ ArtifactDesc ArtifactDescUtil::makeDescForCompileTarget(SlangCompileTarget target)
{
    switch (target)
    {
    case SLANG_TARGET_UNKNOWN:
        return Desc::make(Kind::Unknown, Payload::None, Style::Unknown, 0);
    case SLANG_TARGET_NONE:
        return Desc::make(Kind::None, Payload::None, Style::Unknown, 0);
    case SLANG_GLSL:
        {
            // For the moment we Desc::make all just map to GLSL, but we could use flags
            // or some other mechanism to distinguish the types
            return Desc::make(Kind::Source, Payload::GLSL, Style::Kernel, 0);
        }
    case SLANG_HLSL:
        return Desc::make(Kind::Source, Payload::HLSL, Style::Kernel, 0);
    case SLANG_SPIRV:
        return Desc::make(Kind::ObjectCode, Payload::SPIRV, Style::Kernel, 0);
    case SLANG_SPIRV_ASM:
        return Desc::make(Kind::Assembly, Payload::SPIRV, Style::Kernel, 0);
    case SLANG_DXBC:
        return Desc::make(Kind::ObjectCode, Payload::DXBC, Style::Kernel, 0);
    case SLANG_DXBC_ASM:
        return Desc::make(Kind::Assembly, Payload::DXBC, Style::Kernel, 0);
    case SLANG_DXIL:
        return Desc::make(Kind::ObjectCode, Payload::DXIL, Style::Kernel, 0);
    case SLANG_DXIL_ASM:
        return Desc::make(Kind::Assembly, Payload::DXIL, Style::Kernel, 0);
    case SLANG_C_SOURCE:
        return Desc::make(Kind::Source, Payload::C, Style::Kernel, 0);
    case SLANG_CPP_SOURCE:
        return Desc::make(Kind::Source, Payload::Cpp, Style::Kernel, 0);
    case SLANG_HOST_CPP_SOURCE:
        return Desc::make(Kind::Source, Payload::Cpp, Style::Host, 0);
    case SLANG_CPP_PYTORCH_BINDING:
        return Desc::make(Kind::Source, Payload::Cpp, Style::Host, 0);
    case SLANG_HOST_EXECUTABLE:
        return Desc::make(Kind::Executable, Payload::HostCPU, Style::Host, 0);
    case SLANG_HOST_SHARED_LIBRARY:
        return Desc::make(Kind::SharedLibrary, Payload::HostCPU, Style::Host, 0);
    case SLANG_SHADER_SHARED_LIBRARY:
        return Desc::make(Kind::SharedLibrary, Payload::HostCPU, Style::Kernel, 0);
    case SLANG_SHADER_HOST_CALLABLE:
        return Desc::make(Kind::HostCallable, Payload::HostCPU, Style::Kernel, 0);
    case SLANG_CUDA_SOURCE:
        return Desc::make(Kind::Source, Payload::CUDA, Style::Kernel, 0);
        // TODO(JS):
        // Not entirely clear how best to represent PTX here. We could mark as 'Assembly'.
        // Saying it is 'Executable' implies it is Binary (which PTX isn't). Executable also
        // implies 'complete for executation', irrespective of it being text.
    case SLANG_PTX:
        return Desc::make(Kind::ObjectCode, Payload::PTX, Style::Kernel, 0);
    case SLANG_OBJECT_CODE:
        return Desc::make(Kind::ObjectCode, Payload::HostCPU, Style::Kernel, 0);
    case SLANG_HOST_HOST_CALLABLE:
        return Desc::make(Kind::HostCallable, Payload::HostCPU, Style::Host, 0);
    case SLANG_METAL:
        return Desc::make(Kind::Source, Payload::Metal, Style::Kernel, 0);
    case SLANG_METAL_LIB:
        return Desc::make(Kind::ObjectCode, Payload::MetalAIR, Style::Kernel, 0);
    case SLANG_METAL_LIB_ASM:
        return Desc::make(Kind::Assembly, Payload::MetalAIR, Style::Kernel, 0);
    case SLANG_WGSL:
        return Desc::make(Kind::Source, Payload::WGSL, Style::Kernel, 0);
    case SLANG_WGSL_SPIRV_ASM:
        return Desc::make(Kind::Assembly, Payload::WGSL_SPIRV, Style::Kernel, 0);
    case SLANG_WGSL_SPIRV:
        return Desc::make(Kind::ObjectCode, Payload::WGSL_SPIRV, Style::Kernel, 0);

    case SLANG_HOST_VM:
        return Desc::make(Kind::ObjectCode, Payload::UniversalCPU, Style::Host, 0);
    default:
        break;
    }

    SLANG_UNEXPECTED("Unhandled type");
}


/* static */ ArtifactPayload ArtifactDescUtil::getPayloadForSourceLanaguage(
    SlangSourceLanguage language)
{
    switch (language)
    {
    default:
    case SLANG_SOURCE_LANGUAGE_UNKNOWN:
        return Payload::Unknown;
    case SLANG_SOURCE_LANGUAGE_SLANG:
        return Payload::Slang;
    case SLANG_SOURCE_LANGUAGE_HLSL:
        return Payload::HLSL;
    case SLANG_SOURCE_LANGUAGE_GLSL:
        return Payload::GLSL;
    case SLANG_SOURCE_LANGUAGE_C:
        return Payload::C;
    case SLANG_SOURCE_LANGUAGE_CPP:
        return Payload::Cpp;
    case SLANG_SOURCE_LANGUAGE_CUDA:
        return Payload::CUDA;
    }
}

/* static */ ArtifactDesc ArtifactDescUtil::makeDescForSourceLanguage(SlangSourceLanguage language)
{
    return Desc::make(Kind::Source, getPayloadForSourceLanaguage(language), Style::Unknown, 0);
}

/* static */ SlangCompileTarget ArtifactDescUtil::getCompileTargetFromDesc(const ArtifactDesc& desc)
{
    switch (desc.kind)
    {
    case ArtifactKind::None:
        return SLANG_TARGET_NONE;
    case ArtifactKind::Source:
        {
            switch (desc.payload)
            {
            case Payload::HLSL:
                return SLANG_HLSL;
            case Payload::GLSL:
                return SLANG_GLSL;
            case Payload::C:
                return SLANG_C_SOURCE;
            case Payload::Cpp:
                return (desc.style == Style::Host) ? SLANG_HOST_CPP_SOURCE : SLANG_CPP_SOURCE;
            case Payload::CUDA:
                return SLANG_CUDA_SOURCE;
            case Payload::Metal:
                return SLANG_METAL;
            case Payload::WGSL:
                return SLANG_WGSL;
            default:
                break;
            }
            break;
        }
    case ArtifactKind::Assembly:
        {
            switch (desc.payload)
            {
            case Payload::SPIRV:
                return SLANG_SPIRV_ASM;
            case Payload::DXIL:
                return SLANG_DXIL_ASM;
            case Payload::DXBC:
                return SLANG_DXBC_ASM;
            case Payload::PTX:
                return SLANG_PTX;
            case Payload::MetalAIR:
                return SLANG_METAL_LIB_ASM;
            case Payload::WGSL_SPIRV:
                return SLANG_WGSL_SPIRV_ASM;
            default:
                break;
            }
        }
    default:
        break;
    }

    if (isDerivedFrom(desc.kind, ArtifactKind::CompileBinary))
    {
        if (isDerivedFrom(desc.payload, ArtifactPayload::CPULike))
        {
            switch (desc.kind)
            {
            case Kind::Executable:
                return SLANG_HOST_EXECUTABLE;
            case Kind::SharedLibrary:
                return desc.style == ArtifactStyle::Host ? SLANG_HOST_SHARED_LIBRARY
                                                         : SLANG_SHADER_SHARED_LIBRARY;
            case Kind::HostCallable:
                return desc.style == ArtifactStyle::Host ? SLANG_HOST_HOST_CALLABLE
                                                         : SLANG_SHADER_HOST_CALLABLE;
            case Kind::ObjectCode:
                return SLANG_OBJECT_CODE;
            default:
                break;
            }
        }
        else
        {
            switch (desc.payload)
            {
            case Payload::SPIRV:
                return SLANG_SPIRV;
            case Payload::DXIL:
                return SLANG_DXIL;
            case Payload::DXBC:
                return SLANG_DXBC;
            case Payload::PTX:
                return SLANG_PTX;
            case Payload::MetalAIR:
                return SLANG_METAL_LIB_ASM;
            case Payload::WGSL_SPIRV:
                return SLANG_WGSL_SPIRV;
            default:
                break;
            }
        }
    }

    return SLANG_TARGET_UNKNOWN;
}


namespace
{ // anonymous
struct KindExtension
{
    ArtifactKind kind;
    UnownedStringSlice ext;
};
} // namespace

#define SLANG_KIND_EXTENSION(kind, ext) {ArtifactKind::kind, toSlice(ext)},

static const KindExtension g_cpuKindExts[] = {
#if SLANG_WINDOWS_FAMILY
    SLANG_KIND_EXTENSION(Library, "lib") SLANG_KIND_EXTENSION(ObjectCode, "obj")
        SLANG_KIND_EXTENSION(Executable, "exe") SLANG_KIND_EXTENSION(SharedLibrary, "dll")
#else
    SLANG_KIND_EXTENSION(Library, "a") SLANG_KIND_EXTENSION(ObjectCode, "o")
        SLANG_KIND_EXTENSION(Executable, "")

#if __CYGWIN__
            SLANG_KIND_EXTENSION(SharedLibrary, "dll")
#elif SLANG_APPLE_FAMILY
            SLANG_KIND_EXTENSION(SharedLibrary, "dylib")
#else
            SLANG_KIND_EXTENSION(SharedLibrary, "so")
#endif

#endif
};

/* static */ bool ArtifactDescUtil::isCpuBinary(const ArtifactDesc& desc)
{
    return isDerivedFrom(desc.kind, ArtifactKind::CompileBinary) &&
           isDerivedFrom(desc.payload, ArtifactPayload::CPULike);
}

/* static */ bool ArtifactDescUtil::isText(const ArtifactDesc& desc)
{
    // If it's derived from text...
    if (isDerivedFrom(desc.kind, ArtifactKind::Text))
    {
        return true;
    }

    // Special case PTX...
    if (isDerivedFrom(desc.kind, ArtifactKind::CompileBinary))
    {
        return desc.payload == ArtifactPayload::PTX;
    }

    // Not text
    return false;
}

/* static */ bool ArtifactDescUtil::isGpuUsable(const ArtifactDesc& desc)
{
    if (isDerivedFrom(desc.kind, ArtifactKind::CompileBinary))
    {
        return isDerivedFrom(desc.payload, ArtifactPayload::KernelLike);
    }

    // PTX is a kind of special case, it's an 'assembly' (low level text represention) that can be
    // passed to CUDA runtime
    return desc.kind == ArtifactKind::Assembly && desc.payload == ArtifactPayload::PTX;
}

/* static */ bool ArtifactDescUtil::isKindBinaryLinkable(Kind kind)
{
    switch (kind)
    {
    case Kind::Library:
    case Kind::ObjectCode:
        {
            return true;
        }
    default:
        break;
    }
    return false;
}

/* static */ bool ArtifactDescUtil::isLinkable(const ArtifactDesc& desc)
{
    // If is a container with compile results *assume* that result is linkable
    if (isDerivedFrom(desc.kind, ArtifactKind::Container) &&
        isDerivedFrom(desc.payload, ArtifactPayload::CompileResults))
    {
        return true;
    }

    // if it's a compile binary or a container
    if (isDerivedFrom(desc.kind, ArtifactKind::CompileBinary))
    {
        if (isDerivedFrom(desc.payload, ArtifactPayload::KernelLike))
        {
            // It seems as if DXBC is potentially linkable from
            // https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-appendix-keywords#export

            // We can't *actually* link PTX or SPIR-V currently but it is in principal possible
            // so let's say we accept for now

            return true;
        }
        else if (isDerivedFrom(desc.payload, ArtifactPayload::CPULike))
        {
            // If kind is exe or shared library, linking will arguably not work
            if (desc.kind == ArtifactKind::SharedLibrary || desc.kind == ArtifactKind::Executable)
            {
                return false;
            }

            return true;
        }
        else if (isDerivedFrom(desc.payload, ArtifactPayload::GeneralIR))
        {
            // We'll *assume* IR is linkable
            return true;
        }
    }
    return false;
}

/* static */ bool ArtifactDescUtil::isCpuLikeTarget(const ArtifactDesc& desc)
{
    if (isDerivedFrom(desc.kind, ArtifactKind::CompileBinary))
    {
        return isDerivedFrom(desc.payload, ArtifactPayload::CPULike);
    }
    else if (isDerivedFrom(desc.kind, ArtifactKind::Source))
    {
        // We'll assume C/C++ are targetting CPU, although that is perhaps somewhat arguable.
        return desc.payload == Payload::C || desc.payload == Payload::Cpp;
    }

    return false;
}

/* static */ ArtifactDesc ArtifactDescUtil::getDescFromExtension(const UnownedStringSlice& slice)
{
    if (slice == "slang-module" || slice == "slang-lib")
    {
        return ArtifactDesc::make(ArtifactKind::Library, ArtifactPayload::SlangIR);
    }

    // Metal
    // https://developer.apple.com/documentation/metal/shader_libraries/building_a_library_with_metal_s_command-line_tools
    if (slice == toSlice("air"))
    {
        return ArtifactDesc::make(ArtifactKind::ObjectCode, ArtifactPayload::MetalAIR);
    }
    else if (slice == toSlice("metallib") || slice == toSlice("metalar"))
    {
        return ArtifactDesc::make(ArtifactKind::Library, ArtifactPayload::MetalAIR);
    }

    if (slice == toSlice("zip"))
    {
        return ArtifactDesc::make(ArtifactKind::Zip, ArtifactPayload::Unknown);
    }

    if (slice.startsWith(toSlice("riff")))
    {
        auto tail = slice.tail(4);
        if (tail.getLength() == 0)
        {
            return ArtifactDesc::make(ArtifactKind::RiffContainer, ArtifactPayload::Unknown);
        }
        else if (tail == "-lz4")
        {
            return ArtifactDesc::make(ArtifactKind::RiffLz4Container, ArtifactPayload::Unknown);
        }
        else if (tail == "-deflate")
        {
            return ArtifactDesc::make(ArtifactKind::RiffDeflateContainer, ArtifactPayload::Unknown);
        }
    }

    if (slice == toSlice("asm"))
    {
        // We'll assume asm means current CPU assembler..
        return ArtifactDesc::make(ArtifactKind::Assembly, ArtifactPayload::HostCPU);
    }

    // TODO(JS): Unfortunately map extension is also used from output for linkage from
    // Visual Studio. It's used here for source map.
    if (slice == toSlice("map"))
    {
        return ArtifactDesc::make(ArtifactKind::Json, ArtifactPayload::SourceMap);
    }

    if (slice == toSlice("pdb"))
    {
        // Program database
        return ArtifactDesc::make(ArtifactKind::Assembly, ArtifactPayload::PdbDebugInfo);
    }

    for (const auto& kindExt : g_cpuKindExts)
    {
        if (slice == kindExt.ext)
        {
            // We'll assume it's for the host CPU for now..
            return ArtifactDesc::make(kindExt.kind, Payload::HostCPU);
        }
    }

    const auto target = TypeTextUtil::findCompileTargetFromExtension(slice);

    return makeDescForCompileTarget(target);
}

/* static */ ArtifactDesc ArtifactDescUtil::getDescFromPath(const UnownedStringSlice& slice)
{
    auto extension = Path::getPathExt(slice);
    return getDescFromExtension(extension);
}

/* static*/ SlangResult ArtifactDescUtil::appendCpuExtensionForKind(Kind kind, StringBuilder& out)
{
    for (const auto& kindExt : g_cpuKindExts)
    {
        if (kind == kindExt.kind)
        {
            out << kindExt.ext;
            return SLANG_OK;
        }
    }
    return SLANG_E_NOT_FOUND;
}

static UnownedStringSlice _getPayloadExtension(ArtifactPayload payload)
{
    typedef ArtifactPayload Payload;
    switch (payload)
    {
    /* Source types */
    case Payload::HLSL:
        return toSlice("hlsl");
    case Payload::GLSL:
        return toSlice("glsl");

    case Payload::Cpp:
        return toSlice("cpp");
    case Payload::C:
        return toSlice("c");

    case Payload::Metal:
        return toSlice("metal");

    case Payload::CUDA:
        return toSlice("cu");

    case Payload::Slang:
        return toSlice("slang");

    /* Binary types */
    case Payload::DXIL:
        return toSlice("dxil");
    case Payload::DXBC:
        return toSlice("dxbc");
    case Payload::SPIRV:
        return toSlice("spv");

    case Payload::PTX:
        return toSlice("ptx");

    case Payload::LLVMIR:
        return toSlice("llvm-ir");

    case Payload::SlangIR:
        return toSlice("slang-ir");

    case Payload::MetalAIR:
        return toSlice("metallib");

    case Payload::PdbDebugInfo:
        return toSlice("pdb");
    case Payload::SourceMap:
        return toSlice("map");

    default:
        break;
    }
    return UnownedStringSlice();
}

SlangResult ArtifactDescUtil::appendDefaultExtension(const ArtifactDesc& desc, StringBuilder& out)
{
    switch (desc.kind)
    {
    case ArtifactKind::Library:
        {
            // Special cases
            if (desc.payload == Payload::SlangIR)
            {
                out << toSlice("slang-module");
                return SLANG_OK;
            }
            else if (desc.payload == Payload::MetalAIR)
            {
                // https://developer.apple.com/documentation/metal/shader_libraries/building_a_library_with_metal_s_command-line_tools
                out << toSlice("metallib");
                return SLANG_OK;
            }

            break;
        }
    case ArtifactKind::Zip:
        {
            out << toSlice("zip");
            return SLANG_OK;
        }
    case ArtifactKind::RiffContainer:
        {
            out << toSlice("riff");
            return SLANG_OK;
        }
    case ArtifactKind::RiffLz4Container:
        {
            out << toSlice("riff-lz4");
            return SLANG_OK;
        }
    case ArtifactKind::RiffDeflateContainer:
        {
            out << toSlice("riff-deflate");
            return SLANG_OK;
        }
    case ArtifactKind::Assembly:
        {
            // Special case PTX, because it is assembly
            if (desc.payload == Payload::PTX)
            {
                out << _getPayloadExtension(desc.payload);
                return SLANG_OK;
            }

            // We'll just use asm for all CPU assembly type
            if (isDerivedFrom(desc.payload, ArtifactPayload::CPULike))
            {
                out << toSlice("asm");
                return SLANG_OK;
            }

            // Use the payload extension "-asm"
            out << _getPayloadExtension(desc.payload);
            out << toSlice("-asm");
            return SLANG_OK;
        }
    case ArtifactKind::Source:
        {
            auto ext = _getPayloadExtension(desc.payload);
            if (ext.begin() != nullptr)
            {
                out << ext;
                return SLANG_OK;
            }
            // Don't know the extension for that
            return SLANG_E_NOT_FOUND;
        }
    case ArtifactKind::Json:
        {
            auto ext = _getPayloadExtension(desc.payload);
            if (ext.begin() != nullptr)
            {
                // TODO(JS):
                // Do we need to alter the extension or the name if it's an
                // obfuscated map?
                // if (isDerivedFrom(desc.style, ArtifactStyle::Obfuscated))
                //{
                //}

                out << ext;
                return SLANG_OK;
            }

            // Not really what kind of json, so just use 'generic' json extension
            out << "json";
            return SLANG_OK;
        }
    case ArtifactKind::CompileBinary:
        {
            if (isDerivedFrom(desc.payload, ArtifactPayload::SlangIR) ||
                isDerivedFrom(desc.payload, ArtifactPayload::SlangAST))
            {
                out << "slang-module";
                return SLANG_OK;
            }
            break;
        }
    default:
        break;
    }

    if (ArtifactDescUtil::isGpuUsable(desc))
    {
        auto ext = _getPayloadExtension(desc.payload);
        if (ext.getLength())
        {
            out << ext;
            return SLANG_OK;
        }
    }

    if (ArtifactDescUtil::isCpuLikeTarget(desc) &&
        !isDerivedFrom(desc.payload, ArtifactPayload::Source))
    {
        return appendCpuExtensionForKind(desc.kind, out);
    }

    return SLANG_E_NOT_FOUND;
}

/* static */ String ArtifactDescUtil::getBaseNameFromPath(
    const ArtifactDesc& desc,
    const UnownedStringSlice& path)
{
    const String name = Path::getFileName(path);
    return getBaseNameFromName(desc, name.getUnownedSlice());
}

/* static */ String ArtifactDescUtil::getBaseNameFromName(
    const ArtifactDesc& desc,
    const UnownedStringSlice& inName)
{
    String name(inName);

    const bool isSharedLibraryPrefixPlatform = SLANG_LINUX_FAMILY || SLANG_APPLE_FAMILY;
    if (isSharedLibraryPrefixPlatform)
    {
        // Strip lib prefix
        if (isCpuBinary(desc) &&
            (desc.kind == ArtifactKind::Library || desc.kind == ArtifactKind::SharedLibrary))
        {
            // If it starts with lib strip it
            if (name.startsWith("lib"))
            {
                const String stripLib = name.getUnownedSlice().tail(3);
                name = stripLib;
            }
        }
    }

    // Strip any extension
    {
        StringBuilder descExt;
        if (SLANG_SUCCEEDED(appendDefaultExtension(desc, descExt)) && descExt.getLength())
        {
            // TODO(JS):
            // It has an extension. We could check if they are the same
            // but if they are not that might be fine, because of case insensitivity
            // or perhaps there are multiple valid extensions. So for now we just strip
            // and don't bother confirming with something like..
            // if (Path::getPathExt(name) == descExt))

            name = Path::getFileNameWithoutExt(name);
        }
    }

    return name;
}

/* static */ String ArtifactDescUtil::getBaseName(
    const ArtifactDesc& desc,
    IPathArtifactRepresentation* pathRep)
{
    UnownedStringSlice path(pathRep->getPath());
    return getBaseNameFromPath(desc, path);
}

/* static */ SlangResult ArtifactDescUtil::hasDefinedNameForDesc(const ArtifactDesc& desc)
{
    StringBuilder buf;
    return SLANG_SUCCEEDED(appendDefaultExtension(desc, buf));
}

/* static */ SlangResult ArtifactDescUtil::calcNameForDesc(
    const ArtifactDesc& desc,
    const UnownedStringSlice& inBaseName,
    StringBuilder& outName)
{
    UnownedStringSlice baseName(inBaseName);

    // If there is no basename, set one
    if (baseName.getLength() == 0)
    {
        baseName = toSlice("unknown");
    }

    // Prefix
    if (isCpuBinary(desc) &&
        (desc.kind == ArtifactKind::SharedLibrary || desc.kind == ArtifactKind::Library))
    {
        const bool isSharedLibraryPrefixPlatform = SLANG_LINUX_FAMILY || SLANG_APPLE_FAMILY;
        if (isSharedLibraryPrefixPlatform)
        {
            outName << "lib";
        }
    }

    // Output the basename
    outName << baseName;

    // If there is an extension append it
    StringBuilder ext;
    if (SLANG_SUCCEEDED(appendDefaultExtension(desc, ext)))
    {
        if (ext.getLength())
        {
            outName.appendChar('.');
            outName.append(ext);
        }
    }
    else
    {
        // If we can't determine the type we can output with .unknown
        outName.append(toSlice(".unknown"));
    }

    return SLANG_OK;
}

/* static */ SlangResult ArtifactDescUtil::calcPathForDesc(
    const ArtifactDesc& desc,
    const UnownedStringSlice& basePath,
    StringBuilder& outPath)
{
    outPath.clear();

    // Append the directory
    Index pos = Path::findLastSeparatorIndex(basePath);
    if (pos >= 0)
    {
        // Keep the stem including the delimiter
        outPath.append(basePath.head(pos + 1));

        StringBuilder buf;
        const auto baseName = basePath.tail(pos + 1);

        SLANG_RETURN_ON_FAIL(calcNameForDesc(desc, baseName, buf));
        outPath.append(buf);

        return SLANG_OK;
    }
    else
    {
        return calcNameForDesc(desc, basePath, outPath);
    }
}

/* static */ bool ArtifactDescUtil::isDisassembly(const ArtifactDesc& from, const ArtifactDesc& to)
{
    // From must be a binary like type
    if (!isDerivedFrom(from.kind, ArtifactKind::CompileBinary))
    {
        return false;
    }


    // Target must be assembly, and the payload be the same type
    if (!(to.kind == ArtifactKind::Assembly && to.payload == from.payload))
    {
        return false;
    }

    const auto payload = from.payload;

    // Check the payload seems like something plausible to 'disassemble'
    if (!(isDerivedFrom(payload, ArtifactPayload::KernelLike) ||
          isDerivedFrom(payload, ArtifactPayload::CPULike) ||
          isDerivedFrom(payload, ArtifactPayload::GeneralIR)))
    {
        return false;
    }

    // If the flags or style are different, then it's something more than just disassembly
    if (!(from.style == to.style && from.flags == to.flags))
    {
        return false;
    }

    return true;
}

/* static */ void ArtifactDescUtil::appendText(const ArtifactDesc& desc, StringBuilder& out)
{
    out << getName(desc.kind) << "/" << getName(desc.payload) << "/" << getName(desc.style);
    // TODO(JS): Output flags? None currently used
}

/* static */ String ArtifactDescUtil::getText(const ArtifactDesc& desc)
{
    StringBuilder buf;
    appendText(desc, buf);
    return buf;
}

} // namespace Slang
