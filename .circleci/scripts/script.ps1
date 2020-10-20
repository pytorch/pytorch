C:\pytorch_test\EnableCondaInPS.ps1
conda env list
pip list
d:
git clone https://github.com/gunandrose4u/pytorch.git
az storage blob download --container-name pytorch --account-key 69yY6ZGqj0AqLB4w7i8IbzDd/e6FbgUp3vhnhk5bkTp0LR6xS3WvOayIgy9SdYRdI6yKBoIWKmx5FtKsG4pUvg== --account-name pytorchteststorage -n torch_7z.7z -f torch.7z
7z x torch.7z -otmp -r -aou
echo $env:CONDA_PREFIX
Copy-Item -Path d:\tmp\caffe2 -Destination $env:CONDA_PREFIX\Lib\site-packages\caffe2 -recurse -Force
Copy-Item -Path d:\tmp\torch -Destination $env:CONDA_PREFIX\Lib\site-packages\torch -recurse -Force
cd pytorch
git checkout -b jozh/enable_2gpu_win_ci remotes/origin/jozh/enable_2gpu_win_ci
python test/run_test.py --verbose -i distributed/test_c10d