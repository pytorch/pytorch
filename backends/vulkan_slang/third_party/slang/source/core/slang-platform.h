// slang-platform.h
#ifndef SLANG_CORE_PLATFORM_H
#define SLANG_CORE_PLATFORM_H

#include "../core/slang-string.h"
#include "slang.h"

namespace Slang
{
enum class PlatformKind : uint8_t
{
    Unknown,
    WinRT,
    XBoxOne,
    Win64,
    Win32,
    X360,
    Android,
    Linux,
    IOS,
    OSX,
    PS3,
    PS4,
    PSP2,
    WIIU,
    CountOf,
};

typedef uint32_t PlatformFlags;
struct PlatformFlag
{
    enum Enum
    {
        Unknown = 1 << int(PlatformKind::Unknown),
        WinRT = 1 << int(PlatformKind::WinRT),
        XBoxOne = 1 << int(PlatformKind::XBoxOne),
        Win64 = 1 << int(PlatformKind::Win64),
        Win32 = 1 << int(PlatformKind::Win32),
        X360 = 1 << int(PlatformKind::X360),
        Android = 1 << int(PlatformKind::Android),
        Linux = 1 << int(PlatformKind::Linux),
        IOS = 1 << int(PlatformKind::IOS),
        OSX = 1 << int(PlatformKind::OSX),
        PS3 = 1 << int(PlatformKind::PS3),
        PS4 = 1 << int(PlatformKind::PS4),
        PSP2 = 1 << int(PlatformKind::PSP2),
        WIIU = 1 << int(PlatformKind::WIIU),
    };
};

enum class PlatformFamily : uint8_t
{
    Unknown,
    Windows,
    Microsoft,
    Linux,
    Apple,
    Unix,
    CountOf,
};

// Interface for working with shared libraries
// in a platform-independent fashion.
struct SharedLibrary
{
    typedef struct SharedLibraryImpl* Handle;

    typedef void (*FuncPtr)(void);

    /// Load via an unadorned path/filename.
    ///
    /// The unadorned path here means without a path without any platform specific filename
    /// elements. This typically means no extension and no prefix. On windows this means without the
    /// '.dll' extension. On linux this means without the 'lib' prefix and '.so' extension. To load
    /// with a platform specific filename use the 'loadWithPlatformFilename' API Most platforms have
    /// a built in mechanism to search for shared libraries, such that shared libraries are often
    /// just passed by filename.
    ///
    /// @param the unadorned filename/path
    /// @return Returns a non null handle for the shared library on success. nullptr indicated
    /// failure
    static SlangResult load(const char* path, Handle& handleOut);

    /// Attempt to load a shared library for
    /// the current platform. Returns null handle on failure
    /// The platform specific filename can be generated from a call to appendPlatformFileName
    ///
    /// @param platform the platform specific file name/ or path
    /// @return Returns a non null handle for the shared library on success. nullptr indicated
    /// failure
    static SlangResult loadWithPlatformPath(char const* platformPath, Handle& handleOut);

    /// Unload the library that was returned from load as handle
    /// @param The valid handle returned from load
    static void unload(Handle handle);

    /// Given a shared library handle and a name, return the associated object
    /// Return nullptr if object is not found
    /// @param The shared library handle as returned by loadPlatformLibrary
    static void* findSymbolAddressByName(Handle handle, char const* name);

    /// Append to the end of dst, the name, with any platform specific additions
    /// The input name should be unadorned with any 'lib' prefix or extension
    static void appendPlatformFileName(const UnownedStringSlice& name, StringBuilder& dst);

    /// Given a path, calculate that path with the filename replaced with the platform filename
    /// (using appendPlatformFilename)
    static String calcPlatformPath(const UnownedStringSlice& path);
    static void calcPlatformPath(const UnownedStringSlice& path, StringBuilder& outBuilder);


private:
    /// Not constructible!
    SharedLibrary();
};

struct PlatformUtil
{
    /// Appends a text interpretation of a result (as defined by supporting OS)
    /// @param res Result to produce a string for
    /// @param builderOut Append the string produced to builderOut
    /// @return SLANG_OK if string is found and appended. Fail otherwise. SLANG_E_NOT_IMPLEMENTED if
    /// there is no impl for this platform.
    static SlangResult appendResult(SlangResult res, StringBuilder& builderOut);

    /// Get the platform kind as determined at compile time
    static PlatformKind getPlatformKind();

    /// Get the platforms that make up a family
    static PlatformFlags getPlatformFlags(PlatformFamily family);

    /// True if the kind is part of the family
    static bool isFamily(PlatformFamily family, PlatformKind kind)
    {
        return (getPlatformFlags(family) & (PlatformFlags(1) << int(kind))) != 0;
    }

    /// Given an environment name returns the set system variable.
    /// Will return SLANG_E_NOT_FOUND if the variable is not set
    static SlangResult getEnvironmentVariable(const UnownedStringSlice& name, StringBuilder& out);

    /// Get the path to this instance (the path to the dll/executable/shared library the call is in)
    /// NOTE! This is not supported on all platforms, and will return SLANG_E_NOT_IMPLEMENTED in
    /// that scenario
    static SlangResult getInstancePath(StringBuilder& out);

    /// Outputs message to a debug stream. Not all platforms support
    /// this feature.
    ///
    /// @param text Text to be displayed in 'debugger output'
    /// @return SLANG_E_NOT_AVAILABLE if not on this platform, and potentially other errors
    static SlangResult outputDebugMessage(const char* text);
};

#ifndef _MSC_VER
#define _fileno fileno
#define _isatty isatty
#define _setmode setmode
#define _O_BINARY O_BINARY
#endif
} // namespace Slang

#endif
