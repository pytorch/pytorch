# Only used for PyTorch open source BUCK build

CXX = "Default"

ANDROID = "Android"

APPLE = "Apple"

FBCODE = "Fbcode"

WINDOWS = "Windows"

UNIFIED = "Unified"

# Apple SDK Definitions
IOS = "ios"

WATCHOS = "watchos"

MACOSX = "macosx"

APPLETVOS = "appletvos"

xplat_platforms = struct(
    ANDROID = ANDROID,
    APPLE = APPLE,
    CXX = CXX,
    FBCODE = FBCODE,
    WINDOWS = WINDOWS,
    UNIFIED = UNIFIED,
)

apple_sdks = struct(
    IOS = IOS,
    WATCHOS = WATCHOS,
    MACOSX = MACOSX,
    APPLETVOS = APPLETVOS,
)
