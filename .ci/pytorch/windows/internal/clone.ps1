# The conda and wheels jobs are separated on Windows, so we don't need to clone again.
if (-not (Test-Path $env:NIGHTLIES_PYTORCH_ROOT)) {
    # Clone PyTorch
    git clone "https://github.com/$env:PYTORCH_REPO/$env:MODULE_NAME"
    Set-Location $env:MODULE_NAME
} else {
    Write-Host "Changing to NIGHTLIES_PYTORCH_ROOT"
    Set-Location $env:NIGHTLIES_PYTORCH_ROOT
}

# Handle latest branch logic
if ($env:PYTORCH_BRANCH -eq "latest") {

    # Set date if not already set
    if ([string]::IsNullOrEmpty($env:NIGHTLIES_DATE)) {
        # Get Pacific time dates
        $pacificTz = [System.TimeZoneInfo]::FindSystemTimeZoneById('Pacific Standard Time')
        $pacificTime = [System.TimeZoneInfo]::ConvertTimeFromUtc((Get-Date).ToUniversalTime(), $pacificTz)

        $env:NIGHTLIES_DATE = $pacificTime.ToString('yyyy_MM_dd')
        $env:NIGHTLIES_DATE_COMPACT = $pacificTime.ToString('yyyyMMdd')
    }

    # Set compact date format if not already set
    if ([string]::IsNullOrEmpty($env:NIGHTLIES_DATE_COMPACT)) {
        $env:NIGHTLIES_DATE_COMPACT = $env:NIGHTLIES_DATE.Replace('_', '')
    }

    # Switch to the latest commit by 11:59 yesterday
    Write-Host "PYTORCH_BRANCH is set to latest so I will find the last commit"
    Write-Host "before 0:00 midnight on $env:NIGHTLIES_DATE"

    $git_date = $env:NIGHTLIES_DATE.Replace('_', '-')
    $last_commit = git log --before $git_date -n 1 "--pretty=%H"

    Write-Host "Setting PYTORCH_BRANCH to $last_commit since that was the last"
    Write-Host "commit before $env:NIGHTLIES_DATE"

    $env:PYTORCH_BRANCH = $last_commit
}

# Set default branch if empty
if ([string]::IsNullOrEmpty($env:PYTORCH_BRANCH)) {
    $env:PYTORCH_BRANCH = "v$env:PYTORCH_BUILD_VERSION"
}

# Checkout the branch or tag
git checkout $env:PYTORCH_BRANCH
if ($LASTEXITCODE -ne 0) {
    git checkout "tags/$env:PYTORCH_BRANCH"
}

# Update submodules
git submodule update --init --recursive
if ($LASTEXITCODE -ne 0) {
    exit 1
}
