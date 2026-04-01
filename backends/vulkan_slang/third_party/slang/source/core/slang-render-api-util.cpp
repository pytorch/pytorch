
#include "slang-render-api-util.h"

#include "slang-list.h"
#include "slang-platform.h"
#include "slang-string-util.h"
#include "slang.h"

namespace Slang
{

// NOTE! Must keep in same order as RenderApiType and have same amount of entries
/* static */ const RenderApiUtil::Info RenderApiUtil::s_infos[] = {
    {RenderApiType::Vulkan, "vk,vulkan", ""},
    {RenderApiType::D3D12, "dx12,d3d12", ""},
    {RenderApiType::D3D11, "dx11,d3d11", "hlsl,hlsl-rewrite,glsl-rewrite,glsl-cross,slang"},
    {RenderApiType::Metal, "mtl,metal", ""},
    {RenderApiType::CPU, "cpu", ""},
    {RenderApiType::CUDA, "cuda", "cuda,ptx"},
    {RenderApiType::WebGPU, "wgpu,webgpu", "wgsl"},
};

static int _calcAvailableApis()
{
    int flags = 0;
    for (int i = 0; i < int(RenderApiType::CountOf); i++)
    {
        if (RenderApiUtil::calcHasApi(RenderApiType(i)))
        {
            flags |= (1 << i);
        }
    }

    return flags;
}

/* static */ int RenderApiUtil::getAvailableApis()
{
    static int s_availableApis = _calcAvailableApis();
    return s_availableApis;
}

UnownedStringSlice RenderApiUtil::getApiName(RenderApiType type)
{
    int index = int(type);
    if (index < 0 || index >= int(RenderApiType::CountOf))
    {
        return UnownedStringSlice();
    }
    SLANG_ASSERT(s_infos[index].type == type);
    return StringUtil::getAtInSplit(UnownedStringSlice(s_infos[index].names), ',', 0);
}

/* static */ RenderApiType RenderApiUtil::findApiTypeByName(const Slang::UnownedStringSlice& name)
{
    using namespace Slang;
    List<UnownedStringSlice> namesList;
    for (Index j = 0; j < SLANG_COUNT_OF(RenderApiUtil::s_infos); j++)
    {
        const auto& apiInfo = RenderApiUtil::s_infos[j];
        const UnownedStringSlice names(apiInfo.names);

        if (names.indexOf(',') >= 0)
        {
            StringUtil::split(names, ',', namesList);
            if (namesList.indexOf(name) != Index(-1))
            {
                return apiInfo.type;
            }
        }
        else if (names == name)
        {
            return apiInfo.type;
        }
    }
    return RenderApiType::Unknown;
}

/* static */ Slang::Result RenderApiUtil::findApiFlagsByName(
    const Slang::UnownedStringSlice& name,
    RenderApiFlags* flagsOut)
{
    // Special case 'all'
    if (name == "all")
    {
        *flagsOut = RenderApiFlags(RenderApiFlag::AllOf);
        return SLANG_OK;
    }
    if (name == "none")
    {
        *flagsOut = RenderApiFlags(0);
        return SLANG_OK;
    }
    RenderApiType type = findApiTypeByName(name);
    if (type == RenderApiType::Unknown)
    {
        return SLANG_FAIL;
    }
    *flagsOut = RenderApiFlags(1) << int(type);
    return SLANG_OK;
}

static bool isNameStartChar(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '_');
}

static bool isNameNextChar(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '_') || (c >= '0' && c <= '9');
}

namespace
{ // anonymous
enum class Token
{
    eError,
    eOp,
    eId,
    eEnd,
};
} // namespace

static Token nextToken(Slang::UnownedStringSlice& textInOut, Slang::UnownedStringSlice& lexemeOut)
{
    using namespace Slang;
    if (textInOut.getLength() <= 0)
    {
        return Token::eEnd;
    }
    const char* start = textInOut.begin();
    const char* end = textInOut.end();

    const char firstChar = start[0];
    if (firstChar == '-' || firstChar == '+')
    {
        lexemeOut = UnownedStringSlice(start, start + 1);
        textInOut = UnownedStringSlice(start + 1, end);
        return Token::eOp;
    }

    if (!isNameStartChar(firstChar))
    {
        lexemeOut = UnownedStringSlice(start, start + 1);
        return Token::eError;
    }
    const char* cur = start + 1;
    while (cur < end && isNameNextChar(*cur))
    {
        cur++;
    }

    lexemeOut = UnownedStringSlice(start, cur);
    textInOut = UnownedStringSlice(cur, end);
    return Token::eId;
}

/* static */ Slang::Result RenderApiUtil::parseApiFlags(
    const Slang::UnownedStringSlice& textIn,
    RenderApiFlags initialFlags,
    RenderApiFlags* apiFlagsOut)
{
    using namespace Slang;

    UnownedStringSlice text(textIn);
    UnownedStringSlice lexeme;

    RenderApiFlags apiFlags = 0;

    switch (nextToken(text, lexeme))
    {
    case Token::eOp:
        {
            // If we start with an op - we use the passed in values as the default
            // Rewind back to the start
            text = textIn;
            apiFlags = initialFlags;
            break;
        }
    case Token::eId:
        {
            // If we start with an Id - we use that as the starting state
            SLANG_RETURN_ON_FAIL(findApiFlagsByName(lexeme, &apiFlags));
            break;
        }
    default:
        return SLANG_FAIL;
    }

    while (true)
    {
        // Must have an op followed by an id unless we are at the end
        switch (nextToken(text, lexeme))
        {
        case Token::eEnd:
            {
                *apiFlagsOut = apiFlags;
                return SLANG_OK;
            }
        case Token::eOp:
            break;
        default:
            return SLANG_FAIL;
        }

        const char op = lexeme[0];
        if (nextToken(text, lexeme) != Token::eId)
        {
            return SLANG_FAIL;
        }

        RenderApiFlags flags;
        SLANG_RETURN_ON_FAIL(findApiFlagsByName(lexeme, &flags));

        if (op == '+')
        {
            apiFlags |= flags;
        }
        else
        {
            apiFlags &= ~flags;
        }
    }
}

/* static */ RenderApiType RenderApiUtil::findRenderApiType(const Slang::UnownedStringSlice& text)
{
    using namespace Slang;
    for (Index j = 0; j < SLANG_COUNT_OF(RenderApiUtil::s_infos); j++)
    {
        const auto& apiInfo = RenderApiUtil::s_infos[j];
        if (StringUtil::indexOfInSplit(UnownedStringSlice(apiInfo.names), ',', text) >= 0)
        {
            return apiInfo.type;
        }
    }
    // Didn't find any
    return RenderApiType::Unknown;
}

/* static */ RenderApiType RenderApiUtil::findImplicitLanguageRenderApiType(
    const Slang::UnownedStringSlice& text)
{
    using namespace Slang;
    for (Index j = 0; j < SLANG_COUNT_OF(RenderApiUtil::s_infos); j++)
    {
        const auto& apiInfo = RenderApiUtil::s_infos[j];
        if (StringUtil::indexOfInSplit(UnownedStringSlice(apiInfo.languageNames), ',', text) >= 0)
        {
            return apiInfo.type;
        }
    }
    // Didn't find any
    return RenderApiType::Unknown;
}

#if SLANG_ENABLE_DIRECTX
static bool _canLoadSharedLibrary(const char* libName)
{
    SharedLibrary::Handle handle;
    SlangResult res = SharedLibrary::load(libName, handle);
    if (SLANG_FAILED(res))
    {
        return false;
    }
    SharedLibrary::unload(handle);
    return true;
}
#endif

/* static */ bool RenderApiUtil::calcHasApi(RenderApiType type)
{
    switch (type)
    {
#if SLANG_WINDOWS_FAMILY
    case RenderApiType::Vulkan:
        return _canLoadSharedLibrary("vulkan-1") || _canLoadSharedLibrary("vk_swiftshader");
    case RenderApiType::WebGPU:
        return _canLoadSharedLibrary("webgpu_dawn") && _canLoadSharedLibrary("dxcompiler") &&
               _canLoadSharedLibrary("dxil");
#elif SLANG_APPLE_FAMILY
    case RenderApiType::Vulkan:
        return true;
    case RenderApiType::Metal:
        return true;
#elif SLANG_UNIX_FAMILY
    case RenderApiType::Vulkan:
        return true;
#endif

#if SLANG_ENABLE_DIRECTX
    case RenderApiType::D3D11:
        return _canLoadSharedLibrary(SLANG_ENABLE_DXVK ? "dxvk_d3d11" : "d3d11");
    case RenderApiType::D3D12:
        return _canLoadSharedLibrary(SLANG_ENABLE_VKD3D ? "vkd3d-proton-d3d12" : "d3d12");
#endif

    case RenderApiType::CPU:
        return true;
    // We'll assume CUDA is available, and if not, trying to create it will detect it
    case RenderApiType::CUDA:
        return true;
    default:
        break;
    }
    return false;
}

} // namespace Slang
