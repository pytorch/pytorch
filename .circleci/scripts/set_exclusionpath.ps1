$pytorch_root=[string](Get-Location)
echo $pytorch_root
Add-MpPreference -ExclusionPath $pytorch_root
$preference = Get-MpPreference
echo "show ExclusionPath"
echo $preference.ExclusionPath
