param(
  [string]$protoc,
  [string]$srcdir,
  [string]$unprocessed,
  [string]$processed,
  [string]$out
)
$ErrorActionPreference = "Stop"
Get-Content $unprocessed | % {$_ -Replace "caffe2/proto/caffe2.proto", "caffe2.proto"} | Set-Content $processed
Add-Content -Path $processed -Value "option optimize_for = LITE_RUNTIME;`n" -NoNewline
$dir = (Get-Item $processed).DirectoryName

copy $srcdir/caffe2/proto/caffe2.proto $srcdir/caffe2.proto
Add-Content -Path $srcdir/caffe2.proto -Value "option optimize_for = LITE_RUNTIME;`n" -NoNewline

$processed = (Get-Item $processed).Name
$cmd = "$protoc -I${dir} --cpp_out=$out $processed"
Invoke-Expression $cmd
