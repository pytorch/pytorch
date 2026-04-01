# This script uses `slangc` to generate the core module reference documentation and push the updated
# documents to shader-slang/stdlib-reference repository.
# The stdlib-reference repository has github-pages setup so that the markdown files we generate
# in this step will be rendered as html pages by Jekyll upon a commit to the repository.
# So we we need to do here is to pull the stdlib-reference repository, regenerate the markdown files
# and push the changes back to the repository.

# The generated markdown files will be located in three folders:
# - ./global-decls
# - ./interfaces
# - ./types
# In addition, slangc will generate a table of content file `toc.html` which will be copied to
# ./_includes/stdlib-reference-toc.html for Jekyll for consume it correctly.

# If stdlib-reference folder does not exist, clone from github repo
if (-not (Test-Path ".\stdlib-reference")) {
    git clone https://github.com/shader-slang/stdlib-reference/
}
else {
# If it already exist, just pull the latest changes.
    cd stdlib-reference
    git pull
    cd ../
}
# Remove the old generated files.
Remove-Item -Path ".\stdlib-reference\global-decls" -Recurse -Force
Remove-Item -Path ".\stdlib-reference\interfaces" -Recurse -Force
Remove-Item -Path ".\stdlib-reference\types" -Recurse -Force
Remove-Item -Path ".\stdlib-reference\attributes" -Recurse -Force

# Use git describe to produce a version string and write it to _includes/version.inc.
# This file will be included by the stdlib-reference Jekyll template.
git describe --tags | Out-File -FilePath ".\stdlib-reference\_includes\version.inc" -Encoding ASCII

cd stdlib-reference
$slangPaths = @(
    "../../build/RelWithDebInfo/bin/slangc.exe",
    "../../build/Release/bin/slangc.exe",
    "../../build/Debug/bin/slangc.exe"
)
$slangExe = $slangPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($slangExe) {
    & $slangExe -compile-core-module -doc
    Move-Item -Path ".\toc.html" -Destination ".\_includes\stdlib-reference-toc.html" -Force
    git config user.email "bot@shader-slang.com"
    git config user.name "Stdlib Reference Bot"
    git add .
    git commit -m "Update the core module reference"
    git push
} else {
    Write-Error "Could not find slangc executable in RelWithDebInfo or Release directories"
}
cd ../

# For local debugging only.
# Remove-Item -Path "D:\git_repo\stdlib-reference\global-decls" -Recurse -Force
# Remove-Item -Path "D:\git_repo\stdlib-reference\interfaces" -Recurse -Force
# Remove-Item -Path "D:\git_repo\stdlib-reference\types" -Recurse -Force
# Copy-Item -Path .\stdlib-reference\global-decls -Destination D:\git_repo\stdlib-reference\global-decls -Recurse -Force
# Copy-Item -Path .\stdlib-reference\interfaces -Destination D:\git_repo\stdlib-reference\interfaces -Recurse -Force
# Copy-Item -Path .\stdlib-reference\types -Destination D:\git_repo\stdlib-reference\types -Recurse -Force
# Copy-Item -Path .\stdlib-reference\_includes\stdlib-reference-toc.html -Destination D:\git_repo\stdlib-reference\_includes\stdlib-reference-toc.html -Force
