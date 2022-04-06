Set-MpPreference -ExclusionPath $(Get-Location).tostring()
$preference = Get-MpPreference
echo "show ExclusionPath"
echo "path is "$preference.ExclusionPath
