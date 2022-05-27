# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

<#
Helper class for invoking ADO REST APIs.
#>
class ADOHelper
{
    [string]$Account
    [string]$Project
    $AuthHeaders

    ADOHelper($PersonalAccessToken)
    {
        $this.Project = $env:SYSTEM_TEAMPROJECT
    
        $AuthInfo = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes(":${PersonalAccessToken}"))
        $this.AuthHeaders = @{'Authorization' = "Basic $AuthInfo"}
    }

    ADOHelper($Account, $Project, $PersonalAccessToken)
    {
        $this.Account = $Account
        $this.Project = $Project

        # Use a personal access token.
        $AuthInfo = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes(":${PersonalAccessToken}"))
        $this.AuthHeaders = @{'Authorization' = "Basic $AuthInfo"}
    }

    ADOHelper($Account, $Project)
    {
        $this.Account = $Account
        $this.Project = $Project

        # Use the OAuth token. Requires the script to run in a pipeline with OAuth token access enabled.
        $this.AuthHeaders = @{'Authorization' = "Bearer $env:SYSTEM_ACCESSTOKEN"}
    }

    [ADOHelper] static CreateFromPipeline($PersonalAccessToken)
    {
        if (($env:SYSTEM_COLLECTIONURI -match "^https://(\w+)\.visualstudio\.com") -or
            ($env:SYSTEM_COLLECTIONURI -match "^https://dev.azure.com/(\w+)/"))
        {
            $AdoAccount = $Matches[1]
        }
        else
        {
            throw "Cannot parse ADO account from '`$env:SYSTEM_COLLECTIONURI' ($env:SYSTEM_COLLECTIONURI)"
        }

        if (!$env:SYSTEM_TEAMPROJECT)
        {
            throw "Cannot parse ADO project from '`$env:SYSTEM_TEAMPROJECT' ($env:SYSTEM_TEAMPROJECT)"
        }

        return [ADOHelper]::new($AdoAccount, $env:SYSTEM_TEAMPROJECT, $PersonalAccessToken)
    }

    [object] InvokeApi($Url, $Method, $Body)
    {
        $Arguments = @{}
        $Arguments.Uri = $Url
        $Arguments.Method = $Method
        $Arguments.ContentType = 'application/json'
        $Arguments.Headers = $this.AuthHeaders
        if ($Body)
        {
            $Arguments.Body = $Body
        }
        
        return Invoke-RestMethod @Arguments
    }

    [object] InvokeAccountApi($Api, $Method, $Body)
    {
        return $this.InvokeApi("https://$($this.Account).visualstudio.com/_apis/$Api", $Method, $Body)
    }

    [object] InvokeProjectApi($Api, $Method, $Body)
    {
        return $this.InvokeApi("https://$($this.Account).visualstudio.com/$($this.Project)/_apis/$Api", $Method, $Body)
    }

    [object] GetAgentPool($Name)
    {
        $Response = $this.InvokeAccountApi("distributedtask/pools?api-version=4.1", 'GET', $null)
        return $Response.Value | Where-Object Name -eq $Name
    }

    [object] GetAgentQueue($Name)
    {
        $Response = $this.InvokeProjectApi("distributedtask/queues?queueName=$Name&api-version=5.1-preview", 'GET', $null)
        return $Response.Value | Where-Object Name -eq $Name
    }

    [object] GetAgents($PoolId)
    {
        $Response = $this.InvokeAccountApi("distributedtask/pools/$PoolId/agents?includeCapabilities=true&api-version=4.1", 'GET', $null)
        return $Response.Value
    }

    [object] GetBuildDefinition($Name)
    {
        $Response = $this.InvokeProjectApi('build/definitions?api-version=4.1', 'GET', $null)
        return $Response.Value | Where-Object Name -eq $Name
    }

    [object] GetBuild($BuildId)
    {
        $Response = $this.InvokeProjectApi("build/builds/${BuildId}?api-version=4.1", 'GET', $null)
        return $Response
    }

    [object] GetBuilds($BuildIDs)
    {
        $Response = $this.InvokeProjectApi("build/builds?buildIds=$($BuildIDs -join ',')&api-version=5.0", 'GET', $null)
        return $Response.Value
    }

    [object] QueueBuild($BuildDefinitionId, $Demands, $Parameters, $SourceBranch, $SourceVersion)
    {
        $Body = @{}
        $Body.definition = @{'id' = $BuildDefinitionId}
        $Body.demands = @($Demands.Keys | ForEach-Object { "$_ -equals $($Demands[$_])" })
        $Body.parameters = $Parameters | ConvertTo-Json
        $Body.sourceBranch = $SourceBranch
        $Body.sourceVersion = $SourceVersion
        $Body = $Body | ConvertTo-Json
    
        $Response = $this.InvokeProjectApi('build/builds?api-version=4.1', 'POST', $Body)
        return $Response
    }

    [object] QueuePipeline($PipelineId, $AgentQueueId, $Parameters, $SourceBranch, $SourceVersion)
    {
        $Body = @{}
        $Body.definition = @{'id' = $PipelineId}
        $Body.parameters = $Parameters | ConvertTo-Json
        $Body.sourceBranch = $SourceBranch
        $Body.sourceVersion = $SourceVersion
        if ($AgentQueueId)
        {
            $Body.queue = @{'id' = $AgentQueueId}
        }
        $Body = $Body | ConvertTo-Json
    
        $Response = $this.InvokeProjectApi('build/builds?api-version=5.0', 'POST', $Body)
        return $Response
    }

    [int] GetLastBuildId($BuildDefinitionName)
    {
        $LastBuildId = $null
        $BuildDefId = $this.GetBuildDefinition($BuildDefinitionName).id

        if ($BuildDefId)
        {
            $ApiUrl = "build/builds?"
            $ApiUrl += "definitions=$BuildDefId"
            $ApiUrl += "&maxBuildsPerDefinition=1"
            $ApiUrl += "&statusFilter=completed"
            $ApiUrl += "&api-version=4.1"

            $LastBuildId = $this.InvokeProjectApi($ApiUrl, 'GET', $null).value.id
        }

        return $LastBuildId
    }

    [void] Download($Url, $Destination)
    {
        Invoke-WebRequest -Uri $Url -Headers $this.AuthHeaders -OutFile $Destination
    }

    [void] DownloadBuildArtifacts($BuildId, $Name, $Destination)
    {
        $progressPreference = 'silentlyContinue'
        $Artifacts = $this.InvokeProjectApi("build/builds/$BuildId/artifacts?artifactName=$Name&api-version=4.1", 'GET', $null)
        $ArtifactsUrl = $Artifacts.resource.downloadUrl
        if ($ArtifactsUrl)
        {
            $this.Download($ArtifactsUrl, $Destination)
        }
    }
}