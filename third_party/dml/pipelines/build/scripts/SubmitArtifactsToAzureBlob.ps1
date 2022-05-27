<#
.SYNOPSIS
Submit a folder containing artifacts to azure blob storage
#>
Param(
    [Parameter(Position=1,mandatory=$true)]
    [string]$AzureBlobLink,
    [Parameter(Position=2,mandatory=$true)]
    [string]$AzureBlobLinkToken,
    [Parameter(Position=3,mandatory=$true)]
    [string]$ArtifactsFolder,
    [Parameter(Position=4,mandatory=$true)]
    [string]$ArtifactName
)
$azcopyPath = "./azcopy.exe"

# download azcopy
if(-not (Test-Path $azcopyPath)) {
    $tempZipPath = "./temp.zip"
    try {
        Invoke-WebRequest -Uri "https://aka.ms/downloadazcopy-v10-windows" -OutFile $tempZipPath
    }
    catch {
        Write-Host "Failed to fetch azcopy from https://aka.ms/downloadazcopy-v10-windows . Make sure that the link is working correctly."
    }

    # Expand the Zip file
    Expand-Archive $tempZipPath . -Force

    # Move to $InstallPath
    Get-ChildItem "./azcopy*windows*\azcopy.exe" | Move-Item -Destination "."

    # Clean up
    Remove-Item $tempZipPath -Force -Confirm:$false
    Remove-Item "./azcopy*windows*\" -Recurse
}

# compress the artifacts folder into zip
$compress = @{
    Path = "$($ArtifactsFolder)\*"
    CompressionLevel = "Fastest"
    DestinationPath = "./$($ArtifactName).zip"
  }
Compress-Archive @compress

# Call azcopy and copy to blob storage
try {
    Start-Process $azcopyPath -ArgumentList "copy ./$($ArtifactName).zip $($AzureBlobLink)$($AzureBlobLinkToken)" -Wait -NoNewWindow
}
catch {
    Write-Host "Failed to upload ./$($ArtifactName).zip to $($AzureBlobLink) please ensure that the token isn't expired."
}

# Clean up azcopy
Remove-Item $azcopyPath