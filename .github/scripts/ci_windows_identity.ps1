# Existing-runtime launcher. Only validated canonical booleans reach PowerShell.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$json = '{"runtime_matched":false,"windows_administrator":false,"ambient_credential_inputs_present":false,"metadata_reachable":false,"runner_name_well_formed":false,"document_instance_matches_runner":false,"metadata_instance_matches_runner":false,"document_account_matches_expected":false,"document_region_matches_expected":false,"metadata_region_matches_expected":false,"availability_zone_matches_expected":false,"image_id_format_valid":false,"profile_metadata_success":false,"profile_arn_matches_expected":false,"profile_id_format_valid":false,"role_name_matches_expected":false,"probe_error":true}'
$exitCode = 1
try {
    Add-Type -Path (Join-Path $PSScriptRoot 'ci_windows_identity_launcher.cs')
    $result = [PyTorch.Ci.WindowsIdentityLauncher]::Run($PSScriptRoot)
    $json = $result.Json
    $exitCode = $result.ExitCode
} catch {
    # Compilation/runtime failures expose no exception, path, or child output.
}
[Console]::WriteLine($json)
exit $exitCode
