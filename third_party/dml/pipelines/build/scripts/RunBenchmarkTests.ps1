<#
.SYNOPSIS
Given a path of the Pytorch Benchmark repository, run the the benchmark model tests
#>
Param(
    [string]$OutputPerformanceFile
)

python $PSScriptRoot/pytorch-benchmark/run.py -d cpu -t eval -s $OutputPerformanceFile squeezenet1_1
python $PSScriptRoot/pytorch-benchmark/run.py -d dml -t eval -s $OutputPerformanceFile squeezenet1_1
python $PSScriptRoot/pytorch-benchmark/run.py -d cpu -t train -s $OutputPerformanceFile squeezenet1_1
python $PSScriptRoot/pytorch-benchmark/run.py -d dml -t train -s $OutputPerformanceFile squeezenet1_1

python $PSScriptRoot/pytorch-benchmark/run.py -d cpu -t eval -s $OutputPerformanceFile yolov3
python $PSScriptRoot/pytorch-benchmark/run.py -d dml -t eval -s $OutputPerformanceFile yolov3
# python $PSScriptRoot/pytorch-benchmark/run.py -d cpu -t train -s $OutputPerformanceFile yolov3 # CPU yolov3 evaluation takes too long, and causes timeouts in CI builds!
python $PSScriptRoot/pytorch-benchmark/run.py -d dml -t train -s $OutputPerformanceFile yolov3

# Disabling VGG due to widespread failures on multiple AP machines that are likey due to OOM
# Bug 38136443: Fix VGG16 failures on PyTorch-DirectML APTEST machines
#
python $PSScriptRoot/pytorch-benchmark/run.py -d cpu -t eval -s $OutputPerformanceFile vgg16
python $PSScriptRoot/pytorch-benchmark/run.py -d dml -t eval -s $OutputPerformanceFile vgg16
# python $PSScriptRoot/pytorch-benchmark/run.py -d cpu -t train -s $OutputPerformanceFile vgg16 # CPU VGG evaluation takes too long, and causes timeouts in CI builds!

# Bug 37711563: Pytorch directml queues up too d3d12 commands too fast for GPU to handle and causes GPU memory to fill up
# https://microsoft.visualstudio.com/OS/_workitems/edit/37711563
# python $PSScriptRoot/pytorch-benchmark/run.py -d dml -t train -s $OutputPerformanceFile vgg16

python $PSScriptRoot/pytorch-benchmark/run.py -d cpu -t eval -s $OutputPerformanceFile AlexNet
python $PSScriptRoot/pytorch-benchmark/run.py -d dml -t eval -s $OutputPerformanceFile AlexNet
python $PSScriptRoot/pytorch-benchmark/run.py -d cpu -t train -s $OutputPerformanceFile AlexNet
python $PSScriptRoot/pytorch-benchmark/run.py -d dml -t train -s $OutputPerformanceFile AlexNet

python $PSScriptRoot/pytorch-benchmark/run.py -d cpu -t eval -s $OutputPerformanceFile resnet50
python $PSScriptRoot/pytorch-benchmark/run.py -d dml -t eval -s $OutputPerformanceFile resnet50
python $PSScriptRoot/pytorch-benchmark/run.py -d cpu -t train -s $OutputPerformanceFile resnet50
python $PSScriptRoot/pytorch-benchmark/run.py -d dml -t train -s $OutputPerformanceFile resnet50

python $PSScriptRoot/pytorch-benchmark/run.py -d cpu -t eval -s $OutputPerformanceFile mobilenet_v2
python $PSScriptRoot/pytorch-benchmark/run.py -d dml -t eval -s $OutputPerformanceFile mobilenet_v2
python $PSScriptRoot/pytorch-benchmark/run.py -d cpu -t train -s $OutputPerformanceFile mobilenet_v2
python $PSScriptRoot/pytorch-benchmark/run.py -d dml -t train -s $OutputPerformanceFile mobilenet_v2

python $PSScriptRoot/pytorch-benchmark/run.py -d cpu -t eval -s $OutputPerformanceFile mobilenet_v3_large
python $PSScriptRoot/pytorch-benchmark/run.py -d dml -t eval -s $OutputPerformanceFile mobilenet_v3_large
python $PSScriptRoot/pytorch-benchmark/run.py -d cpu -t train -s $OutputPerformanceFile mobilenet_v3_large
python $PSScriptRoot/pytorch-benchmark/run.py -d dml -t train -s $OutputPerformanceFile mobilenet_v3_large

python $PSScriptRoot/pytorch-benchmark/run.py -d cpu -t eval -s $OutputPerformanceFile densenet121
python $PSScriptRoot/pytorch-benchmark/run.py -d dml -t eval -s $OutputPerformanceFile densenet121
python $PSScriptRoot/pytorch-benchmark/run.py -d cpu -t train -s $OutputPerformanceFile densenet121
python $PSScriptRoot/pytorch-benchmark/run.py -d dml -t train -s $OutputPerformanceFile densenet121