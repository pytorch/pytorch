# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

<#
.SYNOPSIS
Outputs a JSON-formatted matrix to test all agents in an agent pool.

.DESCRIPTION
Azure pipeline jobs can use a "matrix strategy" to spawn multiple jobs from a single job template.
The matrix is typically written inline in the job YAML, and this technique can be used to test multiple
agents with different variables; however, the drawback to this approach is that the YAML needs to be 
updated whenever an agent is added or removed. A way around this limitation is to generate the matrix 
at pipeline runtime. This script uses the REST API to scan an agent pool and create a JSON-formatted 
matrix that includes agents that are online, enabled, and have the required capabilities.

The JSON stored in the output variable is expanded, at runtime, in the pipeline job that uses it. See: 
https://docs.microsoft.com/en-us/azure/devops/pipelines/process/phases?view=azure-devops&tabs=yaml#multi-job-configuration
#>
Param
(
    # Personal access token to authenticate with ADO REST API.
    [parameter(Mandatory)][string]$AccessToken,

    # Names of the agent pool to test.
    [Parameter(Mandatory)][string[]]$AgentPoolNames,

    # List of all possible build artifacts to test.
    [Parameter(Mandatory)][string[]]$Artifacts,

    # List of all possible python versions to test.
    [Parameter(Mandatory)][string[]]$PythonVersions,

    # Path to a file to store the full matrix (includes all agents in the pool).
    [Parameter(Mandatory)][string]$OutputFilePath,

    # Name of pipeline variable to store the pruned matrix (includes only agents that can be tested).
    [Parameter(Mandatory)][string]$OutputVariableName
)

."$PSScriptRoot\ADOHelper.ps1"
$Ado = [ADOHelper]::CreateFromPipeline($AccessToken)

$Matrix = @{}

foreach ($AgentPoolName in $AgentPoolNames)
{
    $AgentPool = $Ado.GetAgentPool($AgentPoolName)
    $Agents = $Ado.GetAgents($AgentPool.id)

    foreach ($Agent in $Agents)
    {
        foreach($Artifact in $Artifacts)
        {
            foreach ($PythonVersion in $PythonVersions)
            {
                $SupportedArtifactsOnAgent = $Artifacts -match $Agent.UserCapabilities.'AP.SupportedArtifacts'
                if (($Artifact -match "linux" -and $SupportedArtifactsOnAgent -match "linux") -or
                    (-not($Artifact -match "linux")))
                {
                    $Matrix[$Agent.Name + "_"+ $Artifact + "_Python" + $PythonVersion] = [ordered]@{
                        agentName = $Agent.Name;
                        agentPool = $AgentPoolName;
                        agentStatus = $Agent.Status;
                        agentEnabled = $Agent.Enabled;
                        agentTestPythonVersion = $PythonVersion;
                        agentTestArtifact = $Artifact;
                    }
                }
            }
        }
    }
}

# Write the full matrix into the output file.
$Matrix.Values | ConvertTo-Json | Out-File $OutputFilePath -Encoding utf8

# Write the pruned matrix into the pipeline variable.
$PrunedMatrix = @{}
foreach ($Key in $Matrix.Keys)
{
    $Value = $Matrix[$Key]
    if ($Value.agentEnabled -and ($Value.agentStatus -eq 'online'))
    {
        $PrunedMatrix[$Key] = [ordered]@{
            agentName = $Value.agentName;
            agentPool = $Value.agentPool;
            agentTestPythonVersion = $Value.agentTestPythonVersion;
            agentTestArtifact = $Value.agentTestArtifact;
        }
    }
}
$PrunedMatrixJson = $PrunedMatrix | ConvertTo-Json -Compress
Write-Host "##vso[task.setVariable variable=$OutputVariableName;isOutput=true]$PrunedMatrixJson"