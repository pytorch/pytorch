# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

<#
.SYNOPSIS
Performs all steps necessary to build Pytorch from a fresh clone after submodule update.
#>

Param
(
    # Build configuration.
    [ValidateSet('Debug', 'Release', 'RelWithDebInfo')][string]$Configuration = 'RelWithDebInfo',

    # CMake generator.
    [ValidateSet( 
        'Visual Studio 16 2019', 
        'Ninja')]
    [string]$Generator='Visual Studio 16 2019',

    # Cleans build files before proceeding.
    [switch]$Clean,

    # Build the python wheel
    [switch]$BuildWheel
)

$env:CMAKE_BUILD_TYPE = "$($configuration)"

# Set the cmake generator
if($Generator -ne "Ninja") {
    $env:USE_NINJA=0
}
else 
{
    $env:USE_NINJA=1
}
$env:CMAKE_GENERATOR= $Generator

$env:USE_DISTRIBUTED=0
$env:USE_MKLDNN=0
$env:USE_CUDA=0
$env:BUILD_TEST=1
$env:USE_FBGEMM=1
$env:USE_NNPACK=0
$env:USE_QNNPACK=0
$env:USE_XNNPACK=0
$env:USE_DIRECTML=1

$GitRoot = git rev-parse --show-toplevel
$BuildDirectory = "$($GitRoot)\build"
Write-Host "Build Directory : $($BuildDirectory)"
if ($Clean -and (Test-Path $BuildDirectory))
{
    python "$($gitroot)\setup.py" clean 
}

if($BuildWheel)
{
    python "$($gitroot)\setup.py" develop bdist_wheel
} 
else 
{
    python "$($gitroot)\setup.py" develop
}