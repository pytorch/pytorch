param(
  [string]$protoc,
  [string]$srcdir,
  [string]$unprocessed,
  [string]$processed,
  [string]$out
)
$ErrorActionPreference = "Stop"

try {
    # Check if input files exist before processing
    if (-not (Test-Path $unprocessed)) { throw "Input file not found: $unprocessed" }
    if (-not (Test-Path "$srcdir/caffe2/proto/caffe2.proto")) { throw "Source file not found at $srcdir/caffe2/proto/caffe2.proto" }

    Get-Content $unprocessed | % {$_ -Replace "caffe2/proto/caffe2.proto", "caffe2.proto"} | Set-Content $processed
    Add-Content -Path $processed -Value "option optimize_for = LITE_RUNTIME;`n" -NoNewline
    $dir = (Get-Item $processed).DirectoryName

    copy $srcdir/caffe2/proto/caffe2.proto $srcdir/caffe2.proto
    Add-Content -Path $srcdir/caffe2.proto -Value "option optimize_for = LITE_RUNTIME;`n" -NoNewline

    $processed = (Get-Item $processed).Name
    $cmd = "$protoc -I${dir} --cpp_out=$out $processed"
    
    Invoke-Expression $cmd
    
    # Catch compiler errors (External executables don't trigger PowerShell's try/catch automatically)
    if ($LASTEXITCODE -ne 0) {
        throw "protoc execution failed with exit code $LASTEXITCODE"
    }
}
catch {
    Write-Error "Script Error: $_"
    exit 1
}
