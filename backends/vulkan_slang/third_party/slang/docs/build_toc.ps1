$job = Start-Job -ArgumentList $PSScriptRoot -ScriptBlock {
    Set-Location $args[0]
    $code = (Get-Content -Raw -Path "scripts/Program.cs").ToString()
    $assemblies = ("System.Core", "System.IO", "System.Collections")
    Add-Type -ReferencedAssemblies $assemblies -TypeDefinition $code -Language CSharp
    $path = Join-Path -Path $args[0] -ChildPath "user-guide"
    [toc.Builder]::Run($path);
    $path = Join-Path -Path $args[0] -ChildPath "gfx-user-guide"
    [toc.Builder]::Run($path);
}
Wait-Job $job
Receive-Job -Job $job
