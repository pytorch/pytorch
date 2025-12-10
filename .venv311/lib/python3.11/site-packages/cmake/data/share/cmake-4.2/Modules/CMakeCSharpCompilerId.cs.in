using System;

namespace CSharp
{
    public class CSharpApp
    {
        const string InfoCompiler = "INFO:compiler[Microsoft "
#if PlatformToolsetv100
        + "Visual Studio"
#elif PlatformToolsetv110
        + "Visual Studio"
#elif PlatformToolsetv120
        + "Visual Studio"
#elif PlatformToolsetv140
        + "Visual Studio"
#elif PlatformToolsetv141
        + "Visual Studio"
#elif PlatformToolsetv142
        + "Visual Studio"
#elif PlatformToolsetv143
        + "Visual Studio"
#elif PlatformToolsetv145
        + "Visual Studio"
#else
        + "unknown"
#endif
        + "]";

        const string InfoPlatform = "INFO:platform[Windows]";

        const string InfoArchitecture = "INFO:arch["
#if Platformx64
        + "x64"
#elif Platformx86
        + "x86"
#elif PlatformxWin32
        + "Win32]"
#else
        + "unknown"
#endif
        + "]";

        const string InfoCompilerVersion = "INFO:compiler_version["
#if PlatformToolsetv100
        + "2010"
#elif PlatformToolsetv110
        + "2012"
#elif PlatformToolsetv120
        + "2013"
#elif PlatformToolsetv140
        + "2015"
#elif PlatformToolsetv141
        + "2017"
#elif PlatformToolsetv142
        + "2019"
#elif PlatformToolsetv143
        + "2022"
#elif PlatformToolsetv145
        + "2026"
#else
        + "9999"
#endif
        + "]";

        static void Main(string[] args)
        {
            // we have to print the lines to make sure
            // the compiler does not optimize them away ...
            System.Console.WriteLine(InfoCompiler);
            System.Console.WriteLine(InfoPlatform);
            System.Console.WriteLine(InfoArchitecture);
            System.Console.WriteLine(InfoCompilerVersion);
        }
    }
}
