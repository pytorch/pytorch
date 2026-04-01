# PowerShell script to fetch, build, and install LLVM for Slang

# Set script to fail on errors
$ErrorActionPreference = "Stop"

function Show-Help {
    Write-Host "Fetch, build and install LLVM for Slang

Options:
  --repo: The source git repo, default: $repo
  --branch: The branch (or tag) to fetch, default: $branch
  --source-dir: Unpack and build in this directory: default $source_dir
  --config: The configuration to build, default $config
  --install-prefix: Install under this prefix
  --: Any following arguments will be passed to the CMake configuration command
"
}

function Msg {
    Write-Host "$args" -ForegroundColor Yellow
}

function Fail {
    Msg "$args"
    exit 1
}

function New-TemporaryDirectory {
  $tmp = [System.IO.Path]::GetTempPath() # Not $env:TEMP, see https://stackoverflow.com/a/946017
  $name = (New-Guid).ToString("N")
  New-Item -ItemType Directory -Path (Join-Path $tmp $name)
}

# Check if required programs are available
$requiredPrograms = "cmake", "git", "ninja"
foreach ($prog in $requiredPrograms) {
    if (-not (Get-Command $prog -ErrorAction SilentlyContinue)) {
        Msg "This script needs $prog, but it isn't in PATH"
        $missingBin = $true
    }
}
if ($missingBin) {
    exit 1
}

# Temp directory with cleanup on exit
$tempDir = New-TemporaryDirectory
$cleanup = {
    if ($tempDir) {
        Remove-Item -Recurse -Force $tempDir
    }
    exit $lastExitCode
}
$null = Register-EngineEvent PowerShell.Exiting -Action $cleanup

# Default values
$repo = "https://github.com/llvm/llvm-project"
$branch = "llvmorg-13.0.1"
$sourceDir = $tempDir.FullName
$installPrefix = ""
$config = "Release"
$extraArguments = @()

# Argument parsing
for ($i = 0; $i -lt $args.Length; $i++) {
    switch ($args[$i]) {
        "--help" {
            Show-Help
            exit
        }
        "--repo" {
            $repo = $args[++$i]
        }
        "--branch" {
            $branch = $args[++$i]
        }
        "--source-dir" {
            $sourceDir = $args[++$i]
        }
        "--config" {
            $config = $args[++$i]
        }
        "--install-prefix" {
            $installPrefix = $args[++$i]
        }
        "--" {
            $extraArguments = $args[$i + 1..$args.Length]
            break
        }
        default {
            Msg "Unknown parameter passed: $($args[$i])"
            Show-Help
            exit 1
        }
    }
}

if (-not $repo) { Fail "please set --repo" }
if (-not $branch) { Fail "please set --branch" }
if (-not $sourceDir) { Fail "please set --source-dir" }
if (-not $config) { Fail "please set --config" }
if (-not $installPrefix) { Fail "please set --install-prefix" }

# Fetch LLVM from the repo
Msg "##########################################################"
Msg "# Fetching LLVM from $repo at $branch"
Msg "##########################################################"
git clone --depth 1 --branch $branch $repo $sourceDir

# Configure LLVM with CMake
Msg "##########################################################"
Msg "# Configuring LLVM in $sourceDir"
Msg "##########################################################"
$msvcRuntimeLib = "MultiThreaded"
if ($config -eq 'Debug')
{
    $msvcRuntimeLib = "MultiThreadedDebug"
}
$cmakeArgumentsForSlang = @(
    "-DLLVM_BUILD_LLVM_C_DYLIB=0"
    "-DLLVM_INCLUDE_BENCHMARKS=0"
    "-DLLVM_INCLUDE_DOCS=0"
    "-DLLVM_INCLUDE_EXAMPLES=0"
    "-DLLVM_INCLUDE_TESTS=0"
    "-DLLVM_ENABLE_TERMINFO=0"
    "-DCLANG_BUILD_TOOLS=0"
    "-DCLANG_ENABLE_STATIC_ANALYZER=0"
    "-DCLANG_ENABLE_ARCMT=0"
    "-DCLANG_INCLUDE_DOCS=0"
    "-DCLANG_INCLUDE_TESTS=0"
    "-DLLVM_ENABLE_PROJECTS=clang"
    "-DLLVM_TARGETS_TO_BUILD=X86;ARM;AArch64"
    "-DLLVM_BUILD_TOOLS=0"
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>"
    "-DLLVM_USE_CRT_RELEASE=MT"
    "-DLLVM_USE_CRT_DEBUG=MTd"
)

$buildDir = Join-Path $sourceDir "build"
New-Item -Path $buildDir -ItemType Directory -Force
$myScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$toolchainFile = Join-Path $myScriptDir "WindowsToolchain\Windows.MSVC.toolchain.cmake"
cmake -S $sourceDir\llvm -B $buildDir $cmakeArgumentsForSlang + $extraArguments -G "Ninja" --toolchain $toolchainFile

# Build LLVM
Msg "##########################################################"
Msg "# Building LLVM in $buildDir"
Msg "##########################################################"
cmake --build $buildDir -j --config $config

# Install LLVM
Msg "##########################################################"
Msg "# Installing LLVM to $installPrefix"
Msg "##########################################################"
cmake --install $buildDir --prefix $installPrefix --config $config

Msg "##########################################################"
Msg "LLVM installed in $installPrefix"
Msg "Please add $installPrefix to CMAKE_PREFIX_PATH"
Msg "##########################################################"
