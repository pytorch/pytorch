#!/bin/bash
conda_installation=''
environment_name=''
source_directory_path=''
while getopts 'c:e:p:' flag; do
  case "${flag}" in
  c) conda_installation="${OPTARG}" ;;
  e) environment_name="${OPTARG}" ;;
  p) source_directory_path="${OPTARG}" ;;
  esac
done

eval "$($conda_installation/bin/conda shell.bash hook)"
export LD_LIBRARY_PATH=$conda_installation/envs/$environment_name/lib
conda activate $environment_name
# Run unit tests
python $source_directory_path/pytorch-directml/third_party/dml/tests/aten_unittests.py

# Change working directory into pytorch-benchmark
cd $source_directory_path/pytorch-benchmark
# This is needed to mitigate the error: ImportError: Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'headless' is currently running
export MPLBACKEND=Agg 

# Run Squeezenet
python $source_directory_path/pytorch-benchmark/run.py -d cpu -t eval squeezenet1_1
python $source_directory_path/pytorch-benchmark/run.py -d dml -t eval squeezenet1_1
python $source_directory_path/pytorch-benchmark/run.py -d cpu -t train squeezenet1_1
python $source_directory_path/pytorch-benchmark/run.py -d dml -t train squeezenet1_1

# Run YoloV3
python $source_directory_path/pytorch-benchmark/run.py -d cpu -t eval yolov3
python $source_directory_path/pytorch-benchmark/run.py -d dml -t eval yolov3
# Runs into matplotlib error : _tkinter.TclError: couldn't connect to display ":0"
# python $source_directory_path/pytorch-benchmark/run.py -d cpu -t train yolov3
python $source_directory_path/pytorch-benchmark/run.py -d dml -t train yolov3

# Run VGG16
python $source_directory_path/pytorch-benchmark/run.py -d cpu -t eval vgg16
python $source_directory_path/pytorch-benchmark/run.py -d dml -t eval vgg16
# Gets killed and doesn't work
# python $source_directory_path/pytorch-benchmark/run.py -d cpu -t train vgg16
python $source_directory_path/pytorch-benchmark/run.py -d dml -t train vgg16

# Run AlexNet
python $source_directory_path/pytorch-benchmark/run.py -d cpu -t eval AlexNet
python $source_directory_path/pytorch-benchmark/run.py -d dml -t eval AlexNet
python $source_directory_path/pytorch-benchmark/run.py -d cpu -t train AlexNet
python $source_directory_path/pytorch-benchmark/run.py -d dml -t train AlexNet

# Run Resnet50
python $source_directory_path/pytorch-benchmark/run.py -d cpu -t eval resnet50
python $source_directory_path/pytorch-benchmark/run.py -d dml -t eval resnet50
python $source_directory_path/pytorch-benchmark/run.py -d cpu -t train resnet50
python $source_directory_path/pytorch-benchmark/run.py -d dml -t train resnet50

# Run mobilenet_v2
python $source_directory_path/pytorch-benchmark/run.py -d cpu -t eval mobilenet_v2
python $source_directory_path/pytorch-benchmark/run.py -d dml -t eval mobilenet_v2
python $source_directory_path/pytorch-benchmark/run.py -d cpu -t train mobilenet_v2
python $source_directory_path/pytorch-benchmark/run.py -d dml -t train mobilenet_v2

# Run mobilenet_v3_large
python $source_directory_path/pytorch-benchmark/run.py -d cpu -t eval mobilenet_v3_large
python $source_directory_path/pytorch-benchmark/run.py -d dml -t eval mobilenet_v3_large
python $source_directory_path/pytorch-benchmark/run.py -d cpu -t train mobilenet_v3_large
python $source_directory_path/pytorch-benchmark/run.py -d dml -t train mobilenet_v3_large

# Run densenet121
python $source_directory_path/pytorch-benchmark/run.py -d cpu -t eval densenet121
python $source_directory_path/pytorch-benchmark/run.py -d dml -t eval densenet121
# Gets killed and doesn't work
# python $source_directory_path/pytorch-benchmark/run.py -d cpu -t train densenet121
python $source_directory_path/pytorch-benchmark/run.py -d dml -t train densenet121

# Run YoloV3 Ultralytics
cd $source_directory_path/directml/pytorch
# Bug 37159924: CPU yolov3 ultralytics has tensor problems https://microsoft.visualstudio.com/OS/_workitems/edit/37159842/
# python $source_directory_path/ultralytics-yolov3-dml/detect.py --source $source_directory_path/ultralytics-yolov3-dml/data/images --conf 0.25 --device 'cpu'
# Bug 37159842: silu.out not implemented. This is needed for yolov3 ultralytics dml eval https://microsoft.visualstudio.com/OS/_workitems/edit/37159842
python $source_directory_path/directml/pytorch/yolov3/detect.py --source $source_directory_path/directml/yolov3/data/images --conf 0.25 --device 'dml'
# Training on CPU takes forever, commenting out for now
# python $source_directory_path/ultralytics-yolov3-dml/train.py --data coco128.yaml --cfg yolov3.yaml --weights yolov3.pt --batch-size 8 --epochs 1 --device 'cpu'
python $source_directory_path/directml/pytorch/yolov3/train.py --batch-size 8 --epochs 1 --device 'dml'