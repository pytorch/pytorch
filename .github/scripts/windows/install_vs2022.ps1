#Requires -RunAsAdministrator

# Enable long paths on Windows
Set-ItemProperty -Path "HKLM:\\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1

$VC_VERSION_major = [int] ${env:VC_VERSION}.split(".")[0]
$VC_DOWNLOAD_LINK = "https://aka.ms/vs/$VC_VERSION_major/release/vs_BuildTools.exe"
$VC_INSTALL_ARGS = @("--nocache","--quiet","--norestart","--wait", "--add Microsoft.VisualStudio.Workload.VCTools",
                                                    "--add Microsoft.Component.MSBuild",
                                                    "--add Microsoft.VisualStudio.Component.Roslyn.Compiler",
                                                    "--add Microsoft.VisualStudio.Component.TextTemplating",
                                                    "--add Microsoft.VisualStudio.Component.VC.CoreBuildTools",
                                                    "--add Microsoft.VisualStudio.Component.VC.CoreIde",
                                                    "--add Microsoft.VisualStudio.Component.VC.Redist.14.Latest",
                                                    "--add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Core",
                                                    "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                                    "--add Microsoft.VisualStudio.Component.Windows11SDK.22621")


echo "Downloading Visual Studio installer from $VC_DOWNLOAD_LINK."
curl.exe --retry 3 -kL $VC_DOWNLOAD_LINK --output vs_installer.exe
if ($LASTEXITCODE -ne 0) {
    echo "Download of the VS ${env:VC_YEAR} Version ${env:VC_VERSION} installer failed"
    exit 1
}
$InstallationPath = ${env:VC_INSTALL_PATH}
$VC_INSTALL_ARGS = "--installPath `"$InstallationPath`"" + " " + $VC_INSTALL_ARGS
echo "Installing Visual Studio version ${env:VC_VERSION} in $InstallationPath."
$process = Start-Process "${PWD}\vs_installer.exe" -ArgumentList $VC_INSTALL_ARGS -NoNewWindow -Wait -PassThru
Remove-Item -Path vs_installer.exe -Force
$exitCode = $process.ExitCode
if (($exitCode -ne 0) -and ($exitCode -ne 3010)) {
    echo "VS ${env:VC_YEAR} installer exited with code $exitCode, which should be one of [0, 3010]."
    exit 1
}
