# Source install utils
. $PSScriptRoot\windows_install_utils.ps1

# Install Windows 10 SDK version 10.0.14393.795
$sdkUrl = "https://go.microsoft.com/fwlink/p/?LinkId=838916"
$sdkFileName = "sdksetup14393.exe"
$argumentList = ("/q", "/norestart", "/ceip off", "/features OptionId.WindowsSoftwareDevelopmentKit")
Write-Output "Installing $sdkUrl"
Install-Binary -Url $sdkUrl -Name $sdkFileName -ArgumentList $argumentList
