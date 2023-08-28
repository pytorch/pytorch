#!/bin/bash
# Run this script under pytorch/benchmarks/dynamo/_onnx to start benchmarking onnx w/ torchbench.
# This script generates ONNX benchmark report logs under pytorch/.logs/onnx_bench.
# It is expected to further run "generate_report_and_archive.sh" after this script completes.

# NOTE: use 'nohup' and add '&' to the end to prevent script stopping due to terminal timeout.

set -e

function show_usage {
    echo "Usage: bench.sh --device <cpu|cuda> [--quick] [--filter <model_name>] [-h]"
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

pushd "../../../"

log_folder=".logs/onnx_bench"

echo "Running benchmarking onnx w/ torchbench..."
echo "Benchmark logs will be saved under pytorch/$log_folder"

# NOTE: --quick is handy to run on small subset of ~3 models for quick sanity check.
(set -x; PATH=/usr/local/cuda/bin/:$PATH python benchmarks/dynamo/runner.py \
    --suites=torchbench \
    --suites=huggingface \
    --suites=timm_models \
    --devices "$device" \
    --inference \
    --batch_size 1 \
    --flag-compilers dynamo-onnx \
    --flag-compilers torchscript-onnx \
    --flag-compilers dort \
    --compilers dynamo-onnx \
    --compilers torchscript-onnx \
    --compilers dort \
    ${quick:+"$quick"} \
    ${filter:+--extra-args "$filter"} \
    --dashboard-image-uploader None \
    --dashboard-archive-path "$log_folder"/cron_logs \
    --dashboard-gh-cli-path None \
    --output-dir "$log_folder"/benchmark_logs \
    --update-dashboard \
    --no-gh-comment \
    --no-graphs )

popd
