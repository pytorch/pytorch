#!/bin/bash
# Run this script under pytorch/benchmarks/dynamo/_onnx to start benchmarking onnx w/ torchbench.
# This script generates ONNX benchmark report logs under pytorch/.logs/onnx_bench.

# NOTE: use 'nohup' and add '&' to the end to prevent script stopping due to terminal timeout.

set -e

function show_usage {
    echo "Usage: 1_bench_and_report.sh --device <cpu|cuda> [--quick] [--filter <model_name>] [-h]"
    echo "Options:"
    echo "  --device <cpu|cuda>   Specify the device to use: cpu or cuda"
    echo "  --quick               Optional flag to run on small subset of ~3 models. Helps in debugging."
    echo "  --filter <model_name> Optional flag to filter benchmarks with regexp on model names. e.g. --filter resnet50"
    echo "  -h                    Display this help message"
}

# Parse command-line arguments
quick=""
filter=""
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        --device)
            device="$2"
            shift
            shift
            ;;
        --quick)
            quick="--quick"
            shift
            ;;
        --filter)
            filter="--filter $2"
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

# Run benchmark script
./bench.sh --device $device $quick $filter

# Generate report and move to archive
./generate_report_and_archive.sh --device $device

echo "Benchmarking and report generation completed."
