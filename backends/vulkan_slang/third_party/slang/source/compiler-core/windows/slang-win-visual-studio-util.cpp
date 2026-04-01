#include "slang-win-visual-studio-util.h"

#include "../../core/slang-common.h"
#include "../../core/slang-process-util.h"
#include "../../core/slang-string-util.h"
#include "../slang-json-parser.h"
#include "../slang-json-value.h"
#include "../slang-visual-studio-compiler-util.h"

#ifdef _WIN32
#include <shlobj.h>
#include <windows.h>
#pragma comment(lib, "advapi32")
#pragma comment(lib, "Shell32")
#endif

// The method used to invoke VS was originally inspired by some ideas in
// https://github.com/RuntimeCompiledCPlusPlus/RuntimeCompiledCPlusPlus/

namespace Slang
{

// Information on VS versioning can be found here
// https://en.wikipedia.org/wiki/Microsoft_Visual_C%2B%2B#Internal_version_numbering


namespace
{ // anonymous

struct RegistryInfo
{
    const char* regName; ///< The name of the entry in the registry
    const char* pathFix; ///< With the value from the registry how to fix the path
};

struct VersionInfo
{
    SemanticVersion version; ///< The version
    const char* name;        ///< The name of the registry key
};

} // namespace

static SlangResult _readRegistryKey(const char* path, const char* keyName, String& outString)
{
    // https://docs.microsoft.com/en-us/windows/desktop/api/winreg/nf-winreg-regopenkeyexa
    HKEY key;
    LONG ret = RegOpenKeyExA(HKEY_LOCAL_MACHINE, path, 0, KEY_READ | KEY_WOW64_32KEY, &key);
    if (ret != ERROR_SUCCESS)
    {
        return SLANG_FAIL;
    }

    char value[MAX_PATH];
    DWORD size = MAX_PATH;

    // https://docs.microsoft.com/en-us/windows/desktop/api/winreg/nf-winreg-regqueryvalueexa
    ret = RegQueryValueExA(key, keyName, nullptr, nullptr, (LPBYTE)value, &size);
    RegCloseKey(key);

    if (ret != ERROR_SUCCESS)
    {
        return SLANG_FAIL;
    }

    outString = value;
    return SLANG_OK;
}

// Make easier to set up the array

[[maybe_unused]] static DownstreamCompilerMatchVersion _makeVersion(int main)
{
    DownstreamCompilerMatchVersion version;
    version.type = SLANG_PASS_THROUGH_VISUAL_STUDIO;
    version.matchVersion.set(main);
    return version;
}

[[maybe_unused]] static DownstreamCompilerMatchVersion _makeVersion(int main, int dot)
{
    DownstreamCompilerMatchVersion version;
    version.type = SLANG_PASS_THROUGH_VISUAL_STUDIO;
    version.matchVersion.set(main, dot);
    return version;
}

VersionInfo _makeVersionInfo(const char* name, int high, int dot = 0)
{
    VersionInfo info;
    info.name = name;
    info.version = SemanticVersion(high, dot);
    return info;
}

// https://en.wikipedia.org/wiki/Microsoft_Visual_Studio
static const VersionInfo s_versionInfos[] = {
    _makeVersionInfo("VS 2005", 8),
    _makeVersionInfo("VS 2008", 9),
    _makeVersionInfo("VS 2010", 10),
    _makeVersionInfo("VS 2012", 11),
    _makeVersionInfo("VS 2013", 12),
    _makeVersionInfo("VS 2015", 14),
    _makeVersionInfo("VS 2017", 15),
    _makeVersionInfo("VS 2019", 16),
    _makeVersionInfo("VS 2022", 17),
};

// When trying to figure out how this stuff works by running regedit - care is needed,
// because what regedit displays varies on which version of regedit is used.
// In order to use the registry paths used here it's necessary to use Start/Run with
// %systemroot%\syswow64\regedit to view 32 bit keys

static const RegistryInfo s_regInfos[] = {
    {"SOFTWARE\\Microsoft\\VisualStudio\\SxS\\VC7", ""},
    {"SOFTWARE\\Microsoft\\VisualStudio\\SxS\\VS7", "VC\\Auxiliary\\Build\\"},
};

static bool _canUseVSWhere(SemanticVersion version)
{
    // If greater than 15.0 we can use vswhere tool
    return version.m_major >= 15;
}

static int _getRegistryKeyIndex(const SemanticVersion& version)
{
    if (version.m_major >= 15)
    {
        return 1;
    }
    return 0;
}

static SlangResult _parseVersion(UnownedStringSlice versionString, SemanticVersion& outVersion)
{
    // We only want the first 2 semantic numbers as 3rd looks like a build number, and too large
    List<UnownedStringSlice> slices;
    StringUtil::split(versionString, '.', slices);
    if (slices.getCount() >= 2)
    {
        versionString = UnownedStringSlice(versionString.begin(), slices[1].end());
    }

    // Extract the version
    SemanticVersion semanticVersion;
    return SemanticVersion::parse(versionString, outVersion);
}


/* static */ DownstreamCompilerMatchVersion WinVisualStudioUtil::getCompiledVersion()
{
#ifdef _MSC_VER
    // Get the version of visual studio used to compile this source
    // Not const, because otherwise we get an warning/error about constant expression...
    uint32_t version = _MSC_VER;

    switch (version)
    {
    case 1400:
        return _makeVersion(8);
    case 1500:
        return _makeVersion(9);
    case 1600:
        return _makeVersion(10);
    case 1700:
        return _makeVersion(11);
    case 1800:
        return _makeVersion(12);
    default:
        break;
    }

    // Seems like versions go in runs of 10 at this point
    // https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros?view=msvc-170
    // https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros?redirectedfrom=MSDN&view=msvc-170
    if (version >= 1900 && version < 1910)
    {
        return _makeVersion(14);
    }
    else if (version >= 1910 && version < 1920)
    {
        switch (version)
        {
        case 1910:
            return _makeVersion(15, 0);
        case 1911:
            return _makeVersion(15, 3);
        case 1912:
            return _makeVersion(15, 5);
        case 1913:
            return _makeVersion(15, 6);
        case 1914:
            return _makeVersion(15, 7);
        case 1915:
            return _makeVersion(15, 8);
        case 1916:
            return _makeVersion(15, 9);
        default:
            return _makeVersion(15);
        }
    }
    else if (version >= 1920 && version < 1930)
    {
        switch (version)
        {
        case 1920:
            return _makeVersion(16, 0);
        case 1921:
            return _makeVersion(16, 1);
        case 1922:
            return _makeVersion(16, 2);
        case 1923:
            return _makeVersion(16, 3);
        case 1924:
            return _makeVersion(16, 4);
        case 1925:
            return _makeVersion(16, 5);
        case 1926:
            return _makeVersion(16, 6);
        case 1927:
            return _makeVersion(16, 7);
        case 1928:
            return _makeVersion(16, 9);
        case 1929:
            return _makeVersion(16, 11);
        default:
            return _makeVersion(16);
        }
    }
    else if (version >= 1930 && version < 1940)
    {
        switch (version)
        {
        case 1930:
            return _makeVersion(17, 0);
        case 1931:
            return _makeVersion(17, 1);
        case 1932:
            return _makeVersion(17, 2);
        default:
            return _makeVersion(17);
        }
    }
    else if (version >= 1940)
    {
        // Its an unknown newer version
        return DownstreamCompilerMatchVersion(
            SLANG_PASS_THROUGH_VISUAL_STUDIO,
            MatchSemanticVersion::makeFuture());
    }
#endif

    // Unknown version
    return DownstreamCompilerMatchVersion(SLANG_PASS_THROUGH_VISUAL_STUDIO, MatchSemanticVersion());
}

static SlangResult _parseJson(
    const String& contents,
    DiagnosticSink* sink,
    JSONContainer* container,
    JSONValue& outRoot)
{
    auto sourceManager = sink->getSourceManager();

    SourceFile* sourceFile =
        sourceManager->createSourceFileWithString(PathInfo::makeUnknown(), contents);
    SourceView* sourceView = sourceManager->createSourceView(sourceFile, nullptr, SourceLoc());

    JSONLexer lexer;
    lexer.init(sourceView, sink);

    JSONBuilder builder(container);

    JSONParser parser;
    SLANG_RETURN_ON_FAIL(parser.parse(&lexer, sourceView, &builder, sink));

    outRoot = builder.getRootValue();
    return SLANG_OK;
}

static void _orderVersions(List<WinVisualStudioUtil::VersionPath>& ioVersions)
{
    typedef WinVisualStudioUtil::VersionPath VersionPath;
    // Put into increasing version order, from oldest to newest
    ioVersions.sort(
        [&](const VersionPath& a, const VersionPath& b) -> bool { return a.version < b.version; });
}

static SlangResult _findVersionsWithVSWhere(
    const VersionInfo* versionInfo,
    List<WinVisualStudioUtil::VersionPath>& outVersions)
{
    typedef WinVisualStudioUtil::VersionPath VersionPath;

    CommandLine cmd;

    // Lookup directly %ProgramFiles(x86)% path
    // https://docs.microsoft.com/en-us/windows/desktop/api/shlobj_core/nf-shlobj_core-shgetfolderpatha
    HWND hwnd = GetConsoleWindow();

    char programFilesPath[_MAX_PATH];
    SHGetFolderPathA(hwnd, CSIDL_PROGRAM_FILESX86, NULL, 0, programFilesPath);

    String vswherePath = programFilesPath;
    vswherePath.append("\\Microsoft Visual Studio\\Installer\\vswhere");

    cmd.setExecutableLocation(ExecutableLocation(vswherePath));

    // Using -? we can find out vswhere options.

    // Previous args - works but returns multiple versions, without listing what version is
    // associated with which path or the order.
    // String args[] = { "-version", versionName, "-requires",
    // "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", ""-property", "installationPath" };

    // Use JSON parsing, we can verify the versions for a path, otherwise multiple versions are
    // returned not just the version specified. The ordering isn't defined (and -sort doesn't appear
    // to work)

    SemanticVersion requiredVersion;
    if (versionInfo)
    {
        StringBuilder versionName;
        versionInfo->version.append(versionName);

        cmd.addArg("-version");
        cmd.addArg(versionName);
    }

    // Add other args
    {
        // TODO(JS):
        // For arm targets will probably need something different for tooling
        String args[] = {
            "-format",
            "json",
            "-utf8",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"};
        cmd.addArgs(args, SLANG_COUNT_OF(args));
    }

    // We are going to use JSON parser to extract the info
    SourceManager sourceManager;
    sourceManager.initialize(nullptr, nullptr);
    DiagnosticSink sink(&sourceManager, nullptr);

    RefPtr<JSONContainer> container = new JSONContainer(&sourceManager);

    ExecuteResult exeRes;
    SLANG_RETURN_ON_FAIL(ProcessUtil::execute(cmd, exeRes));

    JSONValue jsonRoot;
    SLANG_RETURN_ON_FAIL(_parseJson(exeRes.standardOutput, &sink, container, jsonRoot));

    // Search through the array...
    if (jsonRoot.getKind() != JSONValue::Kind::Array)
    {
        return SLANG_FAIL;
    }

    auto arr = container->getArray(jsonRoot);

    const auto pathKey = container->getKey(UnownedStringSlice::fromLiteral("installationPath"));
    const auto versionKey =
        container->getKey(UnownedStringSlice::fromLiteral("installationVersion"));

    // Find all the versions, that match
    for (auto elem : arr)
    {
        // Get the path and the name
        if (elem.getKind() != JSONValue::Kind::Object)
        {
            continue;
        }

        auto pathJsonValue = container->findObjectValue(elem, pathKey);
        auto versionJsonValue = container->findObjectValue(elem, versionKey);

        if (!pathJsonValue.isValid() || !versionJsonValue.isValid())
        {
            continue;
        }

        auto pathString = container->getString(pathJsonValue);
        auto versionString = container->getString(versionJsonValue).trim();

        // Extract the version
        SemanticVersion semanticVersion;
        if (SLANG_SUCCEEDED(_parseVersion(versionString, semanticVersion)))
        {
            if (!requiredVersion.isSet() || requiredVersion.m_major == semanticVersion.m_major)
            {
                WinVisualStudioUtil::VersionPath versionPath;

                versionPath.vcvarsPath = pathString;
                versionPath.vcvarsPath.append("\\VC\\Auxiliary\\Build\\");
                versionPath.version = semanticVersion;

                outVersions.add(versionPath);
            }
        }
    }

    return SLANG_OK;
}

static SlangResult _findVersionsWithRegistery(List<WinVisualStudioUtil::VersionPath>& outVersions)
{
    typedef WinVisualStudioUtil::VersionPath VersionPath;

    const int versionCount = SLANG_COUNT_OF(s_versionInfos);

    for (int i = versionCount - 1; i >= 0; --i)
    {
        const auto versionInfo = s_versionInfos[i];

        auto version = versionInfo.version;

        // Try locating via the registry
        const Int keyIndex = _getRegistryKeyIndex(version);
        if (keyIndex >= 0)
        {
            SLANG_ASSERT(keyIndex < SLANG_COUNT_OF(s_regInfos));

            // Try reading the key
            const auto& keyInfo = s_regInfos[keyIndex];

            StringBuilder keyName;
            versionInfo.version.append(keyName);

            String value;
            if (SLANG_SUCCEEDED(_readRegistryKey(keyInfo.regName, keyName.getBuffer(), value)))
            {
                VersionPath versionPath;
                versionPath.version = versionInfo.version;
                versionPath.vcvarsPath = value;

                // Append
                if (keyInfo.pathFix && keyInfo.pathFix[0] != 0)
                {
                    versionPath.vcvarsPath.append(keyInfo.pathFix);
                }

                outVersions.add(versionPath);
            }
        }
    }

    return SLANG_OK;
}

/* static */ SlangResult WinVisualStudioUtil::find(List<VersionPath>& outVersionPaths)
{
    outVersionPaths.clear();

    List<VersionPath> regVersions;

    // Find all versions with vswhere
    _findVersionsWithVSWhere(nullptr, outVersionPaths);
    // Find all with the registry
    _findVersionsWithRegistery(regVersions);

    // Merge
    for (const auto& regVersion : regVersions)
    {
        Index foundIndex = -1;
        if (_canUseVSWhere(regVersion.version))
        {
            // If there is a major version already from vswhere, we don't need to merge
            const auto majorVersion = regVersion.version.m_major;
            foundIndex = outVersionPaths.findFirstIndex(
                [&](const VersionPath& cur) -> bool
                { return cur.version.m_major == majorVersion; });
        }
        else
        {
            // See if we can find the exact version
            foundIndex = outVersionPaths.findFirstIndex(
                [&](const VersionPath& cur) -> bool { return cur.version == regVersion.version; });
        }

        // If it wasn't found add it.
        if (foundIndex < 0)
        {
            outVersionPaths.add(regVersion);
        }
    }
    // Sort
    _orderVersions(outVersionPaths);
    return SLANG_OK;
}

/* static */ SlangResult WinVisualStudioUtil::find(DownstreamCompilerSet* set)
{
    List<VersionPath> versionPaths;
    SLANG_RETURN_ON_FAIL(find(versionPaths));

    for (const auto& versionPath : versionPaths)
    {
        // Turn into a desc
        const DownstreamCompilerDesc desc(SLANG_PASS_THROUGH_VISUAL_STUDIO, versionPath.version);

        // If not in set add it
        if (!set->getCompiler(desc))
        {
            auto compiler = new VisualStudioDownstreamCompiler(desc);
            ComPtr<IDownstreamCompiler> compilerIntf(compiler);
            calcExecuteCompilerArgs(versionPath, compiler->m_cmdLine);
            set->addCompiler(compilerIntf);
        }
    }

    return SLANG_OK;
}

/* static */ void WinVisualStudioUtil::calcExecuteCompilerArgs(
    const VersionPath& versionPath,
    CommandLine& outCmdLine)
{
    // To invoke cl we need to run the suitable vcvars. In order to run this we have to have MS
    // CommandLine. So here we build up a cl command line that is run by first running vcvars, and
    // then executing cl with the parameters as passed to commandLine

    // https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessa
    // To run a batch file, you must start the command interpreter; set lpApplicationName to cmd.exe
    // and set lpCommandLine to the following arguments: /c plus the name of the batch file.

    CommandLine cmdLine;
    cmdLine.setExecutableLocation(ExecutableLocation(ExecutableLocation::Type::Name, "cmd.exe"));

    {
        String options[] = {"/q", "/c", "@prompt", "$"};
        cmdLine.addArgs(options, SLANG_COUNT_OF(options));
    }

    cmdLine.addArg("&&");
    cmdLine.addArg(Path::combine(versionPath.vcvarsPath, "vcvarsall.bat"));

#if SLANG_PTR_IS_32
    cmdLine.addArg("x86");
#else
    cmdLine.addArg("x86_amd64");
#endif

    cmdLine.addArg("&&");
    cmdLine.addArg("cl");

    outCmdLine = cmdLine;
}

/* static */ SlangResult WinVisualStudioUtil::executeCompiler(
    const VersionPath& versionPath,
    const CommandLine& commandLine,
    ExecuteResult& outResult)
{
    CommandLine cmdLine;
    calcExecuteCompilerArgs(versionPath, cmdLine);
    // Append the command line options
    cmdLine.addArgs(commandLine.m_args.getBuffer(), commandLine.m_args.getCount());
    return ProcessUtil::execute(cmdLine, outResult);
}

} // namespace Slang
