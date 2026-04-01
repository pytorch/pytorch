#ifndef SLANG_CORE_RENDER_API_UTIL_H
#define SLANG_CORE_RENDER_API_UTIL_H

#include "../../source/core/slang-string.h"
#include "slang-com-helper.h"

namespace Slang
{

enum class RenderApiType
{
    Unknown = -1,
    Vulkan = 0,
    D3D12,
    D3D11,
    Metal,
    CPU,
    CUDA,
    WebGPU,
    CountOf,
};

// Use a struct wrapped Enum instead of enum class, cos we want to be able to manipulate as
// integrals
struct RenderApiFlag
{
    enum Enum
    {
        Vulkan = 1 << int(RenderApiType::Vulkan),
        D3D12 = 1 << int(RenderApiType::D3D12),
        D3D11 = 1 << int(RenderApiType::D3D11),
        Metal = 1 << int(RenderApiType::Metal),
        CPU = 1 << int(RenderApiType::CPU),
        CUDA = 1 << int(RenderApiType::CUDA),
        WebGPU = 1 << int(RenderApiType::WebGPU),
        AllOf = (1 << int(RenderApiType::CountOf)) - 1 ///< All bits set
    };
};
typedef uint32_t RenderApiFlags;

struct RenderApiUtil
{
    struct Info
    {
        RenderApiType type;        ///< The type
        const char* names;         ///< Comma separated list of names associated with the type
        const char* languageNames; ///< Comma separated list of target language names associated
                                   ///< with the type
    };

    /// Returns true if the API is available.
    static bool calcHasApi(RenderApiType type);

    /// Returns a combination of RenderApiFlag bits which if set indicates that the API is
    /// available.
    static int getAvailableApis();

    /// Get the name
    static UnownedStringSlice getApiName(RenderApiType type);

    /// Returns RenderApiType::Unknown if not found
    static RenderApiType findApiTypeByName(const Slang::UnownedStringSlice& name);
    /// FlagsOut will have flag/flags specified by a name if returns with successful result code.
    static Slang::Result findApiFlagsByName(
        const Slang::UnownedStringSlice& name,
        RenderApiFlags* flagsOut);

    /// Parse api flags string, returning SLANG_OK on success.
    /// If first character is + or - the flags will be applied to initialFlags, else initialFlags is
    /// ignored. For example "all-dx12" would be all apis, except dx12 -vk would be whatever is in
    /// initial flags, but not vulkan.
    static Slang::Result parseApiFlags(
        const Slang::UnownedStringSlice& text,
        RenderApiFlags initialFlags,
        RenderApiFlags* apiBitsOut);

    /// Gets the API type from a string, or returns RenderApiType::Unknown if not found
    static RenderApiType findRenderApiType(const Slang::UnownedStringSlice& text);

    static RenderApiType findImplicitLanguageRenderApiType(const Slang::UnownedStringSlice& text);

    /// Get information about a render API
    static const Info& getInfo(RenderApiType type) { return s_infos[int(type)]; }

    /// Static information about each render api
    static const Info s_infos[int(RenderApiType::CountOf)];
};

} // namespace Slang

#endif // SLANG_RENDER_API_UTIL_H
