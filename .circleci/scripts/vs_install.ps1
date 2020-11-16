$VS_DOWNLOAD_LINK = "https://aka.ms/vs/15/release/vs_buildtools.exe"
$COLLECT_DOWNLOAD_LINK = "https://aka.ms/vscollect.exe"
$VS_INSTALL_ARGS = @("--nocache","--quiet","--wait", "--add Microsoft.VisualStudio.Workload.VCTools",
                                                     "--add Microsoft.VisualStudio.Component.VC.Tools.14.13",
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
    echo "Download of the VS 2017 installer failed"
    exit 1
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
