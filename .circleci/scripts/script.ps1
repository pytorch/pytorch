param(
    [string]$sp_id,
    [string]$sp_secret,
    [string]$tenant_id,
    [string]$pkg_name,
    [string]$pull_branch,
    [string]$commit_id
)

C:\pytorch_test\EnableCondaInPS.ps1
conda env list
pip list
d:
git clone https://github.com/gunandrose4u/pytorch.git

az login --service-principal -u $sp_id -p $sp_secret --tenant $tenant_id
az storage blob download --account-name pytorchbuildsto --container-name pytorch -n $pkg_name -f $pkg_name
7z x $pkg_name -otmp -r -aou
echo $env:CONDA_PREFIX
Copy-Item -Path d:\tmp\caffe2 -Destination $env:CONDA_PREFIX\Lib\site-packages\caffe2 -recurse -Force
Copy-Item -Path d:\tmp\torch -Destination $env:CONDA_PREFIX\Lib\site-packages\torch -recurse -Force
cd pytorch
git reset --hard $commit_id
git checkout -q -B $pull_branch
git reset --hard $commit_id

python test/run_test.py --verbose -i distributed/test_c10d
