# Existing-runtime launcher. Only validated canonical booleans reach PowerShell.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$json = '{"windows_administrator":false,"ambient_credential_inputs_present":false,"metadata_reachable":false,"runner_instance_identity_matched":false,"instance_profile_matched":false,"instance_role_matched":false,"temporary_credentials_present":false,"sts_authenticated":false,"own_instance_describe_allowed":false,"own_instance_describe_denied":false,"own_instance_identity_matched":false,"own_instance_environment_matched":false,"own_instance_dynamic":false,"own_root_device_ebs":false,"own_root_delete_on_termination":false,"own_other_ebs_mapping_present":false,"own_all_ebs_delete_on_termination":false,"own_ephemeral_finished_tag_present":false,"own_root_replacement_tag_present":false,"probe_error":true}'
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
