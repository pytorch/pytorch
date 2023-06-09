#!/bin/bash
# Run this script under pytorch/benchmarks/dynamo/_onnx after 'bench.sh' completes.
# This script generates ONNX benchmark error summary report in markdown.
# It expects to find benchmark logs under pytorch/.logs/onnx_bench/benchmark_logs.
# It will generate markdown reports under that folder.
# When it's done, it will archive the benchmark logs and rename the folder
# to pytorch/.logs/onnx_bench/benchmark_logs_<timestamp>.

set -e

function show_usage {
    echo "Usage: 2_generate_report_and_archive.sh --device <cpu|cuda> [-h]"
    echo "Options:"
    echo "  --device <cpu|cuda>   Specify the device to use: cpu or cuda"
    echo "  -h                    Display this help message"
}

# Parse command-line arguments
quick=""
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        --device)
            device="$2"
            shift
            shift
            ;;
        -h)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [[ -z $device ]]; then
    echo "Missing required argument: --device"
    show_usage
    exit 1
fi

if [[ $device != "cpu" && $device != "cuda" ]]; then
    echo "Invalid device specified: $device"
    show_usage
    exit 1
fi

pushd "../../../"

log_folder=".logs/onnx_bench/benchmark_logs"

python benchmarks/dynamo/_onnx/reporter.py \
    --suites=torchbench \
    --suites=huggingface \
    --suites=timm_models \
    --compilers dynamo-onnx \
    --compilers torchscript-onnx \
    --device "$device" \
    --output-dir=./"$log_folder"

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
new_log_folder="$log_folder"_"$timestamp"

echo "Archiving benchmark logs to $new_log_folder"
echo "Final report is $new_log_folder/gh_report.txt"
mv "$log_folder" "$new_log_folder"

popd
