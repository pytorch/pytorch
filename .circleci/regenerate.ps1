cd $PSScriptRoot;
$NewFile = New-TemporaryFile;
python generate_config_yml.py > $NewFile.name
(Get-Content $NewFile.name -Raw).TrimEnd().Replace("`r`n","`n") | Set-Content config.yml -Force
Remove-Item $NewFile.name
