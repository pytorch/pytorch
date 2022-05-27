<#
.SYNOPSIS
Compile machine information along with performance results.
#>
Param(
    [Parameter(Position=1,mandatory=$true)]
    [string]$BenchmarkPerformanceFile,
    [Parameter(Position=2,mandatory=$true)]
    [string]$OutputFilePath,
    [Parameter(Position=3,mandatory=$true)]
    [string]$gpuModel,
    [Parameter(Position=4,mandatory=$true)]
    [string]$gpuDriverVersion
)

$machineInfo = New-Object -TypeName psobject
$machineInfo | Add-Member -NotePropertyName cpu_model -NotePropertyValue (Get-WmiObject -Class Win32_Processor | Select-Object -Property Name).Name
$machineInfo | Add-Member -NotePropertyName gpu_model -NotePropertyValue $gpuModel
$machineInfo | Add-Member -NotePropertyName gpu_driver_version -NotePropertyValue $gpuDriverVersion

$schema = "model_name",
"test_mode",
"device",
"performance(seconds)",
"device_model",
"gpu_driver_version"

if(Test-Path $BenchmarkPerformanceFile) {
    Import-Csv $BenchmarkPerformanceFile |
    Select-Object $schema | ForEach-Object {
        if ($_."device" -eq "cpu") {
            $_."device_model" = $machineInfo.cpu_model.trim()
        }
        elseif($_."device" -eq "dml") {
            $_."device_model" = $machineInfo.gpu_model.trim()
            $_.gpu_driver_version = $machineInfo.gpu_driver_version.trim()
        }
        $_
    } | Export-Csv $OutputFilePath -NoTypeInformation -Encoding ascii
}
else {
    throw "BenchmarkPerformanceFile not found, check path."
}