#!/bin/bash

set -ex

UNKNOWN=()

# defaults
PARALLEL=1
export TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK=ERRORS

while [[ $# -gt 0 ]]
do
    arg="$1"
    case $arg in
        -p|--parallel)
            PARALLEL=1
            shift # past argument
            ;;
        *) # unknown option
            UNKNOWN+=("$1") # save it in an array for later
            shift # past argument
            ;;
    esac
done
set -- "${UNKNOWN[@]}" # leave UNKNOWN

# allows coverage to run w/o failing due to a missing plug-in
pip install -e tools/coverage_plugins_package

# realpath might not be available on MacOS
script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
top_dir=$(dirname $(dirname $(dirname "$script_path")))
test_paths=(
    "$top_dir/test/onnx"
)

args=()
args+=("-v")
args+=("--cov")
args+=("--cov-report")
args+=("xml:test/coverage.xml")
args+=("--cov-append")

time python "${top_dir}/test/run_test.py" --onnx --shard "$SHARD_NUMBER" 2 --verbose

if [[ "$SHARD_NUMBER" == "2" ]]; then
  # xdoctests on onnx
  xdoctest torch.onnx --style=google --options="+IGNORE_WHITESPACE"
fi

if [[ "$SHARD_NUMBER" == "2" ]]; then
  # Sanity check on torchbench w/ onnx
  pip install pandas
  log_folder="test/.torchbench_logs"
  device="cpu"
  modes=("accuracy" "performance")
  compilers=("dynamo-onnx" "torchscript-onnx")
  suites=("huggingface" "timm_models")

  mkdir -p "${log_folder}"
  for mode in "${modes[@]}"; do
    for compiler in "${compilers[@]}"; do
      for suite in "${suites[@]}"; do
        output_file="${log_folder}/${compiler}_${suite}_float32_inference_${device}_${mode}.csv"
        bench_file="benchmarks/dynamo/${suite}.py"
        bench_args=("--${mode}" --float32 "-d${device}" "--output=${output_file}" "--output-directory=${top_dir}" --inference -n5 "--${compiler}" --no-skip --dashboard --batch-size 1)
        # Run only selected model for each suite to quickly validate the benchmark suite works as expected.
        case "$suite" in
            "torchbench")
                bench_args+=(-k resnet18)
                ;;
            "huggingface")
                bench_args+=(-k ElectraForQuestionAnswering)
                ;;
            "timm_models")
                bench_args+=(-k lcnet_050)
                ;;
            *)
                echo "Unknown suite: ${suite}"
                exit 1
                ;;
        esac
        python "${top_dir}/${bench_file}" "${bench_args[@]}"
      done
    done
  done
fi

# Our CI expects both coverage.xml and .coverage to be within test/
if [ -d .coverage ]; then
  mv .coverage test/.coverage
fi
