@echo off

set VS_DOWNLOAD_LINK=https://aka.ms/vs/15/release/vs_buildtools.exe
IF "%VS_LATEST%" == "1" (
   set VS_INSTALL_ARGS= --nocache --norestart --quiet --wait --add Microsoft.VisualStudio.Workload.VCTools
   set VSDEVCMD_ARGS=
) ELSE (
   set VS_INSTALL_ARGS=--nocache --quiet --wait --add Microsoft.VisualStudio.Workload.VCTools ^
                                                --add Microsoft.VisualStudio.Component.VC.Tools.14.11 ^
                                                --add Microsoft.Component.MSBuild ^
                                                --add Microsoft.VisualStudio.Component.Roslyn.Compiler ^
                                                --add Microsoft.VisualStudio.Component.TextTemplating ^
                                                --add Microsoft.VisualStudio.Component.VC.CoreIde ^
                                                --add Microsoft.VisualStudio.Component.VC.Redist.14.Latest ^
                                                --add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Core ^
                                                --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
                                                --add Microsoft.VisualStudio.Component.VC.Tools.14.11 ^
                                                --add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Win81
   set VSDEVCMD_ARGS=-vcvars_ver=14.11
)

curl -k -L %VS_DOWNLOAD_LINK% --output vs_installer.exe
if errorlevel 1 exit /b 1

start /wait .\vs_installer.exe %VS_INSTALL_ARGS%
if not errorlevel 0 exit /b 1
if errorlevel 1 if not errorlevel 3010 exit /b 1
if errorlevel 3011 exit /b 1
