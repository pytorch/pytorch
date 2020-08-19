$VS_INSTALLER = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe"
$COLLECT_DOWNLOAD_LINK = "https://aka.ms/vscollect.exe"
$VS_INSTALL_ARGS = @("modify", "--nocache","--quiet",
                                           "--installPath ""C:\Program Files (x86)\Microsoft Visual Studio\2019\Community""",
                                           "--add Microsoft.VisualStudio.Component.VC.14.26.x86.x64")

$process = Start-Process "$VS_INSTALLER" -ArgumentList $VS_INSTALL_ARGS -NoNewWindow -Wait -PassThru
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
    Copy-Item -Path "C:\Users\circleci\AppData\Local\Temp\vslogs.zip" -Destination "C:\w\build-results\"
    exit 1
}
