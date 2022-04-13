# https://developercommunity.visualstudio.com/t/install-specific-version-of-vs-component/1142479
# Where to find the links: https://docs.microsoft.com/en-us/visualstudio/releases/2019/history#release-dates-and-build-numbers

# BuildTools from S3
$VS_DOWNLOAD_LINK = "https://s3.amazonaws.com/ossci-windows/vs${env:VS_VERSION}_BuildTools.exe"
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

if (${env:INSTALL_WINDOWS_SDK} -eq "1") {
    $VS_INSTALL_ARGS += "--add Microsoft.VisualStudio.Component.Windows10SDK.19041"
}

if (Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe") {
    $VS_VERSION_major = [int] ${env:VS_VERSION}.split(".")[0]
    $existingPath = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -products "Microsoft.VisualStudio.Product.BuildTools" -version "[${env:VS_VERSION}, ${env:VS_VERSION_major + 1})" -property installationPath
    if (($existingPath -ne $null) -and (!${env:CIRCLECI})) {
        echo "Found correctly versioned existing BuildTools installation in $existingPath"
        exit 0
    }
    $pathToRemove = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -products "Microsoft.VisualStudio.Product.BuildTools" -property installationPath
}

echo "Downloading VS installer from S3."
curl.exe --retry 3 -kL $VS_DOWNLOAD_LINK --output vs_installer.exe
if ($LASTEXITCODE -ne 0) {
    echo "Download of the VS 2019 Version ${env:VS_VERSION} installer failed"
    exit 1
}

if ($pathToRemove -ne $null) {
    echo "Uninstalling $pathToRemove."
    $VS_UNINSTALL_ARGS = @("uninstall", "--installPath", "`"$pathToRemove`"", "--quiet","--wait")
    $process = Start-Process "${PWD}\vs_installer.exe" -ArgumentList $VS_UNINSTALL_ARGS -NoNewWindow -Wait -PassThru
    $exitCode = $process.ExitCode
    if (($exitCode -ne 0) -and ($exitCode -ne 3010)) {
        echo "Original BuildTools uninstall failed with code $exitCode"
        exit 1
    }
    echo "Other versioned BuildTools uninstalled."
}

echo "Installing Visual Studio version ${env:VS_VERSION}."
$process = Start-Process "${PWD}\vs_installer.exe" -ArgumentList $VS_INSTALL_ARGS -NoNewWindow -Wait -PassThru
Remove-Item -Path vs_installer.exe -Force
$exitCode = $process.ExitCode
if (($exitCode -ne 0) -and ($exitCode -ne 3010)) {
    echo "VS 2019 installer exited with code $exitCode, which should be one of [0, 3010]."
    curl.exe --retry 3 -kL $COLLECT_DOWNLOAD_LINK --output Collect.exe
    if ($LASTEXITCODE -ne 0) {
        echo "Download of the VS Collect tool failed."
        exit 1
    }
    Start-Process "${PWD}\Collect.exe" -NoNewWindow -Wait -PassThru
    New-Item -Path "C:\w\build-results" -ItemType "directory" -Force
    Copy-Item -Path "${env:TEMP}\vslogs.zip" -Destination "C:\w\build-results\"
    exit 1
}
