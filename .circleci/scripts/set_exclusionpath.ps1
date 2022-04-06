Set-MpPreference -ExclusionPath $(Get-Location).tostring()
$preference = Get-MpPreference
echo "show ExclusionPath"
Write-Host $preference.ExclusionPath
