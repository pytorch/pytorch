git clone https://github.com/ROCmSoftwarePlatform/pytorch-micro-benchmarking.git
cd pytorch-micro-benchmarking
#1GPU --
# export AMD_LOG_LEVEL=3
# export HIP_LAUNCH_BLOCKING=1
python3.6 micro_benchmarking_pytorch.py --network resnet50 --batch-size 256 --iterations 10 --dataparallel --device_ids 0
