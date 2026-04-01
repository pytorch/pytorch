// slang-downstream-compiler.cpp
#include "slang-downstream-compiler-util.h"

#include "../core/slang-blob.h"
#include "../core/slang-char-util.h"
#include "../core/slang-common.h"
#include "../core/slang-io.h"
#include "../core/slang-shared-library.h"
#include "../core/slang-string-util.h"
#include "../core/slang-type-text-util.h"
#include "slang-com-helper.h"

#ifdef SLANG_VC
#include "windows/slang-win-visual-studio-util.h"
#endif

#include "slang-dxc-compiler.h"
#include "slang-fxc-compiler.h"
#include "slang-gcc-compiler-util.h"
#include "slang-glslang-compiler.h"
#include "slang-llvm-compiler.h"
#include "slang-metal-compiler.h"
#include "slang-nvrtc-compiler.h"
#include "slang-tint-compiler.h"
#include "slang-visual-studio-compiler-util.h"

namespace Slang
{

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DownstreamCompilerInfos !!!!!!!!!!!!!!!!!!!!!!*/

struct DownstreamCompilerInfos
{
    DownstreamCompilerInfo infos[int(SLANG_PASS_THROUGH_COUNT_OF)];

    static DownstreamCompilerInfos _calcInfos();
    static DownstreamCompilerInfos s_infos;
};

/* static */ DownstreamCompilerInfos DownstreamCompilerInfos::_calcInfos()
{
    typedef DownstreamCompilerInfo Info;
    typedef Info::SourceLanguageFlag SourceLanguageFlag;

    DownstreamCompilerInfos infos;

    infos.infos[int(SLANG_PASS_THROUGH_CLANG)] =
        Info(SourceLanguageFlag::CPP | SourceLanguageFlag::C);
    infos.infos[int(SLANG_PASS_THROUGH_VISUAL_STUDIO)] =
        Info(SourceLanguageFlag::CPP | SourceLanguageFlag::C);
    infos.infos[int(SLANG_PASS_THROUGH_GCC)] =
        Info(SourceLanguageFlag::CPP | SourceLanguageFlag::C);
    infos.infos[int(SLANG_PASS_THROUGH_LLVM)] =
        Info(SourceLanguageFlag::CPP | SourceLanguageFlag::C);

    infos.infos[int(SLANG_PASS_THROUGH_NVRTC)] = Info(SourceLanguageFlag::CUDA);

    infos.infos[int(SLANG_PASS_THROUGH_DXC)] = Info(SourceLanguageFlag::HLSL);
    infos.infos[int(SLANG_PASS_THROUGH_FXC)] = Info(SourceLanguageFlag::HLSL);
    infos.infos[int(SLANG_PASS_THROUGH_GLSLANG)] = Info(SourceLanguageFlag::GLSL);
    infos.infos[int(SLANG_PASS_THROUGH_SPIRV_OPT)] = Info(SourceLanguageFlag::SPIRV);

    return infos;
}

/* static */ DownstreamCompilerInfos DownstreamCompilerInfos::s_infos =
    DownstreamCompilerInfos::_calcInfos();

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DownstreamCompilerInfo !!!!!!!!!!!!!!!!!!!!!!*/

/* static */ const DownstreamCompilerInfo& DownstreamCompilerInfo::getInfo(
    SlangPassThrough compiler)
{
    return DownstreamCompilerInfos::s_infos.infos[int(compiler)];
}

/* static */ bool DownstreamCompilerInfo::canCompile(
    SlangPassThrough compiler,
    SlangSourceLanguage sourceLanguage)
{
    const auto& info = getInfo(compiler);
    return (info.sourceLanguageFlags & (SourceLanguageFlags(1) << int(sourceLanguage))) != 0;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!! DownstreamCompilerUtil !!!!!!!!!!!!!!!!!!!!!!*/

static DownstreamCompilerMatchVersion _calcCompiledVersion()
{
    DownstreamCompilerMatchVersion matchVersion;

#if SLANG_VC
    matchVersion = WinVisualStudioUtil::getCompiledVersion();
#elif SLANG_CLANG
    matchVersion.type = SLANG_PASS_THROUGH_CLANG;
    matchVersion.matchVersion.set(Index(__clang_major__), Index(__clang_minor__));
#elif SLANG_GCC
    matchVersion.type = SLANG_PASS_THROUGH_GCC;
    matchVersion.matchVersion.set(Index(__GNUC__), Index(__GNUC_MINOR__));
#else
    // TODO(JS): Hmmm None is not quite the same as unknown. It works for now, but we might want to
    // have a distinct enum for unknown.
    matchVersion.type = SLANG_PASS_THROUGH_NONE;
#endif

    return matchVersion;
}


DownstreamCompilerMatchVersion DownstreamCompilerUtil::getCompiledVersion()
{
    static DownstreamCompilerMatchVersion s_version = _calcCompiledVersion();
    return s_version;
}

/* static */ IDownstreamCompiler* DownstreamCompilerUtil::findCompiler(
    const DownstreamCompilerSet* set,
    MatchType matchType,
    const DownstreamCompilerDesc& desc)
{
    List<IDownstreamCompiler*> compilers;
    set->getCompilers(compilers);
    return findCompiler(compilers, matchType, desc);
}

/* static */ IDownstreamCompiler* DownstreamCompilerUtil::findCompiler(
    const List<IDownstreamCompiler*>& compilers,
    MatchType matchType,
    const DownstreamCompilerDesc& desc)
{
    if (compilers.getCount() <= 0)
    {
        return nullptr;
    }

    Int bestIndex = -1;

    const SlangPassThrough compilerType = desc.type;

    Int maxVersionValue = 0;
    Int minVersionDiff = 0x7fffffff;

    Int descVersionValue = desc.getVersionValue();

    // If we don't have version set, then anything 0 or above is good enough, and just take newest
    if (descVersionValue == 0)
    {
        maxVersionValue = -1;
        matchType = MatchType::Newest;
    }

    for (Index i = 0; i < compilers.getCount(); ++i)
    {
        IDownstreamCompiler* compiler = compilers[i];
        auto compilerDesc = compiler->getDesc();

        if (compilerType == compilerDesc.type)
        {
            const Int versionValue = compilerDesc.getVersionValue();
            switch (matchType)
            {
            case MatchType::MinGreaterEqual:
                {
                    auto diff = descVersionValue - versionValue;
                    if (diff >= 0 && diff < minVersionDiff)
                    {
                        bestIndex = i;
                        minVersionDiff = diff;
                    }
                    break;
                }
            case MatchType::MinAbsolute:
                {
                    auto diff = descVersionValue - versionValue;
                    diff = (diff >= 0) ? diff : -diff;
                    if (diff < minVersionDiff)
                    {
                        bestIndex = i;
                        minVersionDiff = diff;
                    }
                    break;
                }
            case MatchType::Newest:
                {
                    if (versionValue > maxVersionValue)
                    {
                        maxVersionValue = versionValue;
                        bestIndex = i;
                    }
                    break;
                }
            }
        }
    }

    return (bestIndex >= 0) ? compilers[bestIndex] : nullptr;
}

/* static */ IDownstreamCompiler* DownstreamCompilerUtil::findCompiler(
    const List<IDownstreamCompiler*>& compilers,
    const DownstreamCompilerDesc& desc)
{
    for (auto compiler : compilers)
    {
        if (compiler->getDesc() == desc)
        {
            return compiler;
        }
    }
    return nullptr;
}

/* static */ IDownstreamCompiler* DownstreamCompilerUtil::findCompiler(
    const List<IDownstreamCompiler*>& compilers,
    SlangPassThrough type,
    const SemanticVersion& version)
{
    DownstreamCompilerDesc desc;
    desc.type = type;
    desc.version = version;
    return findCompiler(compilers, desc);
}

/* static */ void DownstreamCompilerUtil::findVersions(
    const List<IDownstreamCompiler*>& compilers,
    SlangPassThrough type,
    List<SemanticVersion>& outVersions)
{
    for (auto compiler : compilers)
    {
        auto desc = compiler->getDesc();

        if (desc.type == type)
        {
            outVersions.add(desc.version);
        }
    }
}

/* static */ IDownstreamCompiler* DownstreamCompilerUtil::findClosestCompiler(
    const List<IDownstreamCompiler*>& compilers,
    const DownstreamCompilerMatchVersion& matchVersion)
{
    List<SemanticVersion> versions;

    findVersions(compilers, matchVersion.type, versions);

    if (versions.getCount() > 0)
    {
        if (versions.getCount() == 1)
        {
            // Must be that one
            return findCompiler(compilers, matchVersion.type, versions[0]);
        }

        // Okay lets find the best one
        auto bestVersion = MatchSemanticVersion::findAnyBest(
            versions.getBuffer(),
            versions.getCount(),
            matchVersion.matchVersion);

        // If one is found use it
        if (bestVersion.isSet())
        {
            return findCompiler(compilers, matchVersion.type, bestVersion);
        }
    }

    {
        // TODO(JS):
        // NOTE! This may not really be appropriate, because LLVM is *not* interchangable with
        // a 'normal' C++ compiler as cannot access standard libraries/headers.
        // So `slang-llvm` can't be used for 'host' code.

        // These compilers should be usable interchangably. The order is important, as the first one
        // that matches will be used, so LLVM is used before CLANG or GCC if appropriate
        const SlangPassThrough compatiblePassThroughs[] = {
            SLANG_PASS_THROUGH_LLVM,
            SLANG_PASS_THROUGH_CLANG,
            SLANG_PASS_THROUGH_GCC,
        };

        // Check the version is one of the compatible types
        if (makeConstArrayView(compatiblePassThroughs).indexOf(matchVersion.type) >= 0)
        {
            // Try each compatible type in turn
            for (auto passThrough : compatiblePassThroughs)
            {
                versions.clear();
                findVersions(compilers, passThrough, versions);

                if (versions.getCount() > 0)
                {
                    // Get the latest version (as we have no way to really compare)
                    auto latestVersion =
                        SemanticVersion::getLatest(versions.getBuffer(), versions.getCount());
                    return findCompiler(compilers, matchVersion.type, latestVersion);
                }
            }
        }
    }

    return nullptr;
}

/* static */ IDownstreamCompiler* DownstreamCompilerUtil::findClosestCompiler(
    const DownstreamCompilerSet* set,
    const DownstreamCompilerMatchVersion& matchVersion)
{
    List<IDownstreamCompiler*> compilers;
    set->getCompilers(compilers);
    return findClosestCompiler(compilers, matchVersion);
}

/* static */ void DownstreamCompilerUtil::updateDefault(
    DownstreamCompilerSet* set,
    SlangSourceLanguage sourceLanguage)
{
    IDownstreamCompiler* compiler = nullptr;

    switch (sourceLanguage)
    {
    case SLANG_SOURCE_LANGUAGE_CPP:
    case SLANG_SOURCE_LANGUAGE_C:
        {
            // Find the compiler closest to the compiler this was compiled with
            if (!compiler)
            {
                compiler = findClosestCompiler(set, getCompiledVersion());
            }
            break;
        }
    case SLANG_SOURCE_LANGUAGE_CUDA:
        {
            DownstreamCompilerDesc desc;
            desc.type = SLANG_PASS_THROUGH_NVRTC;
            compiler = findCompiler(set, MatchType::Newest, desc);
            break;
        }
    default:
        break;
    }

    set->setDefaultCompiler(sourceLanguage, compiler);
}

/* static */ void DownstreamCompilerUtil::updateDefaults(DownstreamCompilerSet* set)
{
    for (Index i = 0; i < Index(SLANG_SOURCE_LANGUAGE_COUNT_OF); ++i)
    {
        updateDefault(set, SlangSourceLanguage(i));
    }
}

/* static */ void DownstreamCompilerUtil::setDefaultLocators(
    DownstreamCompilerLocatorFunc outFuncs[int(SLANG_PASS_THROUGH_COUNT_OF)])
{
    outFuncs[int(SLANG_PASS_THROUGH_VISUAL_STUDIO)] = &VisualStudioCompilerUtil::locateCompilers;
    outFuncs[int(SLANG_PASS_THROUGH_CLANG)] = &GCCDownstreamCompilerUtil::locateClangCompilers;
    outFuncs[int(SLANG_PASS_THROUGH_GCC)] = &GCCDownstreamCompilerUtil::locateGCCCompilers;
    outFuncs[int(SLANG_PASS_THROUGH_NVRTC)] = &NVRTCDownstreamCompilerUtil::locateCompilers;
    outFuncs[int(SLANG_PASS_THROUGH_DXC)] = &DXCDownstreamCompilerUtil::locateCompilers;
    outFuncs[int(SLANG_PASS_THROUGH_FXC)] = &FXCDownstreamCompilerUtil::locateCompilers;
    outFuncs[int(SLANG_PASS_THROUGH_GLSLANG)] = &GlslangDownstreamCompilerUtil::locateCompilers;
    outFuncs[int(SLANG_PASS_THROUGH_SPIRV_OPT)] = &SpirvOptDownstreamCompilerUtil::locateCompilers;
    outFuncs[int(SLANG_PASS_THROUGH_LLVM)] = &LLVMDownstreamCompilerUtil::locateCompilers;
    outFuncs[int(SLANG_PASS_THROUGH_SPIRV_DIS)] = &SpirvDisDownstreamCompilerUtil::locateCompilers;
    outFuncs[int(SLANG_PASS_THROUGH_METAL)] = &MetalDownstreamCompilerUtil::locateCompilers;
    outFuncs[int(SLANG_PASS_THROUGH_TINT)] = &TintDownstreamCompilerUtil::locateCompilers;
}

static String _getParentPath(const String& path)
{
    // If we can get the canonical path, we'll do that before getting the parent
    String canonicalPath;
    if (SLANG_SUCCEEDED(Path::getCanonical(path, canonicalPath)))
    {
        return Path::getParentDirectory(canonicalPath);
    }
    else
    {
        return Path::getParentDirectory(path);
    }
}

static SlangResult _findPaths(
    const String& path,
    const char* libraryName,
    String& outParentPath,
    String& outLibraryPath)
{
    // Try to determine what the path is by looking up the path type
    SlangPathType pathType;
    if (SLANG_SUCCEEDED(Path::getPathType(path, &pathType)))
    {
        if (pathType == SLANG_PATH_TYPE_DIRECTORY)
        {
            outParentPath = path;
            outLibraryPath = Path::combine(outParentPath, libraryName);
        }
        else
        {
            SLANG_ASSERT(pathType == SLANG_PATH_TYPE_FILE);

            outParentPath = _getParentPath(path);
            outLibraryPath = path;
        }

        return SLANG_OK;
    }

    // If this failed the path could be to a shared library, but we may need to convert to the
    // shared library filename first
    const String sharedLibraryFilePath = SharedLibrary::calcPlatformPath(path.getUnownedSlice());
    if (SLANG_SUCCEEDED(Path::getPathType(sharedLibraryFilePath, &pathType)) &&
        pathType == SLANG_PATH_TYPE_FILE)
    {
        // We pass in the shared library path, as canonical paths can sometimes only apply to
        // pre-existing objects.
        outParentPath = _getParentPath(sharedLibraryFilePath);
        // The original path should work as is for the SharedLibrary load. Notably we don't use the
        // sharedLibraryFilePath as this is the wrong name to do a SharedLibrary load with.
        outLibraryPath = path;

        return SLANG_OK;
    }

    return SLANG_FAIL;
}

/* static */ SlangResult DownstreamCompilerUtil::loadSharedLibrary(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    const char* const* dependentNames,
    const char* inLibraryName,
    ComPtr<ISlangSharedLibrary>& outSharedLib)
{
    String parentPath;
    String libraryPath;

    // If a path is passed in lets, try and determine what kind of path it is.
    if (path.getLength())
    {
        if (SLANG_FAILED(_findPaths(path, inLibraryName, parentPath, libraryPath)))
        {
            // We have a few scenarios here.
            // 1) The path could be the shared library/dll filename, that will be found through some
            // operating system mechanism 2) That the shared library is *NOT* on the filesystem
            // directly (the loader does something different) 3) Permissions or some other mechanism
            // stops the lookup from working

            // We should probably assume that the path means something, else why set it.
            // It's probably less likely that it is a directory that we can't detect - as if it's a
            // directory as part of an app it's permissions should allow detection, or be made to
            // allow it.

            // All this being the case we should probably assume that it is the shared library name.
            libraryPath = path;

            // Attempt to get a parent. If there isn't one this will be empty, which will mean it
            // will be ignored, which is probably what we want if path is just a shared library name
            parentPath = Path::getParentDirectory(libraryPath);
        }
    }

    // Keep all dependent libs in scope, before we load the library we want
    List<ComPtr<ISlangSharedLibrary>> dependentLibs;

    // Try to load any dependent libs from the parent path
    if (dependentNames)
    {
        for (const char* const* cur = dependentNames; *cur; ++cur)
        {
            const char* dependentName = *cur;
            ComPtr<ISlangSharedLibrary> lib;
            if (parentPath.getLength())
            {
                String dependentPath = Path::combine(parentPath, dependentName);
                loader->loadSharedLibrary(dependentPath.getBuffer(), lib.writeRef());
            }
            else
            {
                loader->loadSharedLibrary(dependentName, lib.writeRef());
            }

            if (lib)
            {
                dependentLibs.add(lib);
            }
        }
    }

    if (libraryPath.getLength())
    {
        // If we hare a library path use that
        return loader->loadSharedLibrary(libraryPath.getBuffer(), outSharedLib.writeRef());
    }
    else
    {
        // Else just use the name that was passed in.
        return loader->loadSharedLibrary(inLibraryName, outSharedLib.writeRef());
    }
}

/* static */ void DownstreamCompilerUtil::appendAsText(
    const DownstreamCompilerDesc& desc,
    StringBuilder& out)
{
    out << TypeTextUtil::getPassThroughAsHumanText(desc.type);

    // Append the version if there is a version
    if (desc.version.isSet())
    {
        out << " ";
        out << desc.version.m_major;
        out << ".";
        out << desc.version.m_minor;
    }
}

} // namespace Slang
