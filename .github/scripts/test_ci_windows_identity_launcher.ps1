# Synthetic child-process tests; no metadata, cloud request, or real credential.
param([Parameter(Mandatory=$true)][string]$AssemblyPath,[Parameter(Mandatory=$true)][string]$PythonPath)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
Add-Type -Path $AssemblyPath
$script:Checks=0
function Check([bool]$Value) { $script:Checks++; if(!$Value){throw "collector check $script:Checks failed"} }
function Child([string]$Code,[bool]$Input=$false) {
    $encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Code))
    $start=[Diagnostics.ProcessStartInfo]::new()
    $start.FileName=$PythonPath
    $start.Arguments='-I -B -S -c "import base64;exec(base64.b64decode('''+$encoded+'''))"'
    $start.RedirectStandardInput=$Input
    return $start
}
function Run-Case([string]$Code,[int]$Timeout=5000,[bool]$Input=$false) {
    $start=Child $Code $Input
    $report=[PyTorch.Ci.WindowsIdentityLauncher]::CollectForTest($start,$Timeout)
    Check $report.ChildStopped
    return $report
}
$failure=[PyTorch.Ci.WindowsIdentityLauncher]::Failure().Json
$good=$failure.Replace('"windows_administrator":false','"windows_administrator":true').Replace('"probe_error":true','"probe_error":false')
$code='import sys;sys.stdout.write('+(($good+"`n")|ConvertTo-Json -Compress)+')'
$report=Run-Case $code
Check ($report.Json -ceq $good -and $report.ExitCode -eq 0)
$report=Run-Case ('import sys;sys.stdout.write('+(($failure+"`r`n")|ConvertTo-Json -Compress)+');sys.exit(1)')
Check ($report.Json -ceq $failure -and $report.ExitCode -eq 1)
$marker='SYNTHETIC-UNEXPECTED-TOKEN-9351'
$bad=@(
    ('{"token":"'+$marker+'"}'),
    $good.Replace('"windows_administrator":true','"windows_administrator":"'+$marker+'"'),
    $good.Replace('"windows_administrator":true','"windows_administrator":true,"windows_administrator":true'),
    $good.Replace('{','{"extra":false,'),
    $good.Replace('"windows_administrator":true,',''),
    $good.Replace(':true',': true'),
    ($good+"`n"+$good)
)
Check ($bad.Count -eq 7)
foreach($text in $bad) {
    $report=Run-Case ('import sys;sys.stdout.write('+(($text+"`n")|ConvertTo-Json -Compress)+')')
    Check ($report.Json -ceq $failure -and $report.ExitCode -eq 1)
}
$codes=@(
    'import os;os.write(1,b"x"*1048576)',
    'import os;os.write(2,b"x"*1048576)',
    'import os;os.write(1,b"\xff")',
    ('import sys;sys.stderr.write("'+$marker+'");sys.stdout.write('+(($good+"`n")|ConvertTo-Json -Compress)+')')
)
Check ($codes.Count -eq 4)
foreach($code in $codes) {
    $report=Run-Case $code
    Check ($report.Json -ceq $failure -and $report.ExitCode -eq 1)
}
$report=Run-Case 'import sys;sys.stdin.buffer.read(1)' 100 $true
Check ($report.Json -ceq $failure -and $report.ExitCode -eq 1)
# Trace the actual non-cmdlet collector and only then hand its safe result to a cmdlet.
$start=Child ('import sys;sys.stdout.write('+(('{"token":"'+$marker+'"}'+"`n")|ConvertTo-Json -Compress)+')')
$trace=[IO.Path]::GetTempFileName()
try {
    Trace-Command -Name ParameterBinding -Expression {
        $safe=[PyTorch.Ci.WindowsIdentityLauncher]::CollectForTest($start,5000)
        $safe.Json | ConvertFrom-Json | Out-Null
    } -FilePath $trace
    Check (!([IO.File]::ReadAllText($trace)).Contains($marker))
} finally {Remove-Item -LiteralPath $trace -Force}
[Console]::WriteLine("PASS: $script:Checks actual collector process/trace checks")
