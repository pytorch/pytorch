# Run only in the authorized Windows CI canary. Do not print tokens/metadata bodies.
$ErrorActionPreference = 'Stop'
function Emit($Key, $Value) { Write-Output "CI_INSTANCE_AUDIT $Key=$Value" }
function Invoke-LinkLocalMetadata($Method, $Path, $Headers) {
    # Windows PowerShell 5.1 has no Invoke-RestMethod -NoProxy. Disable the
    # proxy on the individual request, not in process/global configuration.
    $request = [Net.HttpWebRequest]::Create("http://169.254.169.254/$Path")
    $request.Proxy = $null
    $request.AllowAutoRedirect = $false
    $request.Method = $Method
    $request.Timeout = 2000
    $request.ReadWriteTimeout = 2000
    foreach ($name in $Headers.Keys) { $request.Headers[$name] = $Headers[$name] }
    $response = $null
    $reader = $null
    try {
        $response = $request.GetResponse()
        $reader = [IO.StreamReader]::new($response.GetResponseStream())
        return $reader.ReadToEnd()
    } finally {
        if ($null -ne $reader) { $reader.Dispose() }
        if ($null -ne $response) { $response.Dispose() }
    }
}
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
Emit 'windows_admin_token' ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator).ToString().ToLowerInvariant())
Emit 'aws_env_credentials' (([bool]$env:AWS_ACCESS_KEY_ID -and [bool]$env:AWS_SECRET_ACCESS_KEY).ToString().ToLowerInvariant())
$token = $null
$info = $null
try {
    $token = Invoke-LinkLocalMetadata 'PUT' 'latest/api/token' @{'X-aws-ec2-metadata-token-ttl-seconds'='60'}
    $info = (Invoke-LinkLocalMetadata 'GET' 'latest/meta-data/iam/info' @{'X-aws-ec2-metadata-token'=$token}) | ConvertFrom-Json
    Emit 'aws_imds_iam_info' (([bool]$info.InstanceProfileArn).ToString().ToLowerInvariant())
} catch {
    Emit 'aws_imds_iam_info' 'false'
} finally {
    $token = $null
    $info = $null
}
# Standard provider-chain identity probe only; no assumed role or resource write.
$oldAttempts = $env:AWS_MAX_ATTEMPTS
$oldPager = $env:AWS_PAGER
$account = $null
try {
    $env:AWS_MAX_ATTEMPTS = '1'
    $env:AWS_PAGER = ''
    if (Get-Command aws -ErrorAction SilentlyContinue) {
        $account = & aws sts get-caller-identity --query Account --output text --cli-connect-timeout 2 --cli-read-timeout 3 2>$null
        Emit 'aws_default_provider_sts' (($LASTEXITCODE -eq 0 -and $account -match '^[0-9]{12}$').ToString().ToLowerInvariant())
    } else {
        Emit 'aws_default_provider_sts' 'false'
    }
} catch {
    Emit 'aws_default_provider_sts' 'false'
} finally {
    $env:AWS_MAX_ATTEMPTS = $oldAttempts
    $env:AWS_PAGER = $oldPager
    $account = $null
}
Emit 'completed' 'true'
