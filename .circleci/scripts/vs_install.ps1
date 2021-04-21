# https://developercommunity.visualstudio.com/t/install-specific-version-of-vs-component/1142479
# https://docs.microsoft.com/en-us/visualstudio/releases/2019/history#release-dates-and-build-numbers

# 16.8.5 BuildTools
$VS_DOWNLOAD_LINK = "https://download.visualstudio.microsoft.com/download/pr/20130c62-1bc8-43d6-b4f0-c20bb7c79113/145a319d79a83376915d8f855605e152ef5f6fa2b2f1d2dca411fb03722eea72/vs_BuildTools.exe"
$COLLECT_DOWNLOAD_LINK = "https://aka.ms/vscollect.exe"
$VS_INSTALL_ARGS = @("--nocache","--quiet","--wait", "--add Microsoft.VisualStudio.Workload.VCTools",
                                                     "--add Microsoft.Component.MSBuild",
                                                     "--add Microsoft.VisualStudio.Component.Roslyn.Compiler",
                                                     "--add Microsoft.VisualStudio.Component.TextTemplating",
                                                     "--add Microsoft.VisualStudio.Component.VC.CoreIde",
                                                     "--add Microsoft.VisualStudio.Component.VC.Redist.14.Latest",
                                                     "--add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Core",
                                                     "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                                     "--add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Win81")

curl.exe --retry 3 -kL $VS_DOWNLOAD_LINK --output vs_installer.exe
if ($LASTEXITCODE -ne 0) {
    echo "Download of the VS 2019 Version 16.8.5 installer failed"
    exit 1
}

if (Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe") {
    $existingPath = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -products "Microsoft.VisualStudio.Product.BuildTools" -version "[16, 17)" -property installationPath
    if ($existingPath -ne $null) {
        echo "Found existing BuildTools installation in $existingPath"
        $VS_UNINSTALL_ARGS = @("uninstall", "--installPath", "`"$existingPath`"", "--quiet","--wait")
        $process = Start-Process "${PWD}\vs_installer.exe" -ArgumentList $VS_UNINSTALL_ARGS -NoNewWindow -Wait -PassThru
        $exitCode = $process.ExitCode
        if (($exitCode -ne 0) -and ($exitCode -ne 3010)) {
            echo "Original BuildTools uninstall failed with code $exitCode"
            exit 1
        }
        echo "Original BuildTools uninstalled"
    }
}

$process = Start-Process "${PWD}\vs_installer.exe" -ArgumentList $VS_INSTALL_ARGS -NoNewWindow -Wait -PassThru
Remove-Item -Path vs_installer.exe -Force
$exitCode = $process.ExitCode
if (($exitCode -ne 0) -and ($exitCode -ne 3010)) {
    echo "VS 2017 installer exited with code $exitCode, which should be one of [0, 3010]."
    curl.exe --retry 3 -kL $COLLECT_DOWNLOAD_LINK --output Collect.exe
    if ($LASTEXITCODE -ne 0) {
        echo "Download of the VS Collect tool failed."
        exit 1
    }
    Start-Process "${PWD}\Collect.exe" -NoNewWindow -Wait -PassThru
    New-Item -Path "C:\w\build-results" -ItemType "directory" -Force
    Copy-Item -Path "C:\Users\circleci\AppData\Local\Temp\vslogs.zip" -Destination "C:\w\build-results\"
    exit 1
}
