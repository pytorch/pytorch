# https://developercommunity.visualstudio.com/t/install-specific-version-of-vs-component/1142479
# https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history#evergreen-bootstrappers

# latest BuildTools
$VS_DOWNLOAD_LINK = "https://aka.ms/vs/17/release/vs_BuildTools.exe"
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
    echo "Download of the VS $VC_YEAR Version $VS_VERSION installer failed"
    exit 1
}

echo "VS download worked"

echo $PATH

if (Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe") {
    $existingPath = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -products "Microsoft.VisualStudio.Product.BuildTools" -version "[17.10, 18)" -property installationPath
    Write-Output "Existing BuildTools installation found at: $existingPath"
    if ($existingPath -ne $null -or $existingPath -ne "") {
        if (!${env:CIRCLECI}) {
            echo "Found correctly versioned existing BuildTools installation in $existingPath"
            exit 0
        }
        echo "Found existing BuildTools installation in $existingPath, keeping it"
    }
}

$process = Start-Process "${PWD}\vs_installer.exe" -ArgumentList $VS_INSTALL_ARGS -NoNewWindow -Wait -PassThru
Remove-Item -Path vs_installer.exe -Force
$exitCode = $process.ExitCode
if (($exitCode -ne 0) -and ($exitCode -ne 3010)) {
    echo "VS $VC_YEAR installer exited with code $exitCode, which should be one of [0, 3010]."
    curl.exe --retry 3 -kL $COLLECT_DOWNLOAD_LINK --output Collect.exe
    if ($LASTEXITCODE -ne 0) {
        echo "Download of the VS Collect tool failed."
        exit 1
    }
    Start-Process "${PWD}\Collect.exe" -NoNewWindow -Wait -PassThru
    New-Item -Path "C:\w\build-results" -ItemType "directory" -Force
    Copy-Item -Path "C:\Users\${env:USERNAME}\AppData\Local\Temp\vslogs.zip" -Destination "C:\w\build-results\"
    exit 1
}
