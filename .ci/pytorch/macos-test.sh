#!/bin/bash
set -x

# shellcheck disable=SC2034
# shellcheck source=./macos-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/macos-common.sh"

# Test that OpenMP is enabled
pushd test
if [[ ! $(python -c "import torch; print(int(torch.backends.openmp.is_available()))") == "1" ]]; then
  echo "Build should have OpenMP enabled, but torch.backends.openmp.is_available() is False"
  exit 1
fi
popd

# enable debug asserts in serialization
export TORCH_SERIALIZATION_DEBUG=1

setup_test_python() {
  # The CircleCI worker hostname doesn't resolve to an address.
  # This environment variable makes ProcessGroupGloo default to
  # using the address associated with the loopback interface.
  export GLOO_SOCKET_IFNAME=lo0
  echo "Ninja version: $(ninja --version)"
  echo "Python version: $(which python) ($(python --version))"

  # Set the limit on open file handles to 16384
  # might help with intermittent compiler test failures
  ulimit -n 16384
}

test_python_all() {
  setup_test_python

  time python test/run_test.py --verbose --exclude-jit-executor

  assert_git_not_dirty
}

test_python_mps() {
  setup_test_python

  time PYTORCH_TEST_WITH_SLOW=1 python test/run_test.py --verbose --mps
  MTL_CAPTURE_ENABLED=1 python3 test/test_mps.py --verbose -k test_metal_capture

  assert_git_not_dirty
}

test_python_openreg() {
  setup_test_python

  time python test/run_test.py --openreg --verbose

  assert_git_not_dirty
}


test_python_shard() {
  if [[ -z "$NUM_TEST_SHARDS" ]]; then
    echo "NUM_TEST_SHARDS must be defined to run a Python test shard"
    exit 1
  fi

  setup_test_python

  time python test/run_test.py --verbose --exclude-jit-executor --exclude-distributed-tests --exclude-quantization-tests --shard "$1" "$NUM_TEST_SHARDS"

  assert_git_not_dirty
}

test_libtorch() {
  # C++ API

  if [[ "$BUILD_TEST_LIBTORCH" == "1" ]]; then
    # NB: Install outside of source directory (at the same level as the root
    # pytorch folder) so that it doesn't get cleaned away prior to docker push.
    # But still clean it before we perform our own build.

    echo "Testing libtorch"

    CPP_BUILD="$PWD/../cpp-build"
    rm -rf "$CPP_BUILD"
    mkdir -p "$CPP_BUILD"/caffe2

    BUILD_LIBTORCH_PY=$PWD/tools/build_libtorch.py
    pushd "$CPP_BUILD"/caffe2
    VERBOSE=1 DEBUG=1 python "$BUILD_LIBTORCH_PY"
    popd

    MNIST_DIR="${PWD}/test/cpp/api/mnist"
    python tools/download_mnist.py --quiet -d "${MNIST_DIR}"

    # Unfortunately it seems like the test can't load from miniconda3
    # without these paths being set
    export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$PWD/miniconda3/lib"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/miniconda3/lib"
    TORCH_CPP_TEST_MNIST_PATH="${MNIST_DIR}" CPP_TESTS_DIR="${CPP_BUILD}/caffe2/bin" python test/run_test.py --cpp --verbose -i cpp/test_api

    assert_git_not_dirty
  fi
}

test_custom_backend() {
  echo "Testing custom backends"
  pushd test/custom_backend
  rm -rf build && mkdir build
  pushd build
  SITE_PACKAGES="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
  CMAKE_PREFIX_PATH="$SITE_PACKAGES/torch" cmake ..
  make VERBOSE=1
  popd

  # Run Python tests and export a lowered module.
  python test_custom_backend.py -v
  python backend.py --export-module-to=model.pt
  # Run C++ tests using the exported module.
  build/test_custom_backend ./model.pt
  rm -f ./model.pt
  popd
  assert_git_not_dirty
}

test_custom_script_ops() {
  echo "Testing custom script operators"
  pushd test/custom_operator
  # Build the custom operator library.
  rm -rf build && mkdir build
  pushd build
  SITE_PACKAGES="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
  CMAKE_PREFIX_PATH="$SITE_PACKAGES/torch" cmake ..
  make VERBOSE=1
  popd

  # Run tests Python-side and export a script module.
  python test_custom_ops.py -v
  python model.py --export-script-module=model.pt
  # Run tests C++-side and load the exported script module.
  build/test_custom_ops ./model.pt
  popd
  assert_git_not_dirty
}

test_jit_hooks() {
  echo "Testing jit hooks in cpp"
  pushd test/jit_hooks
  # Build the custom operator library.
  rm -rf build && mkdir build
  pushd build
  SITE_PACKAGES="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
  CMAKE_PREFIX_PATH="$SITE_PACKAGES/torch" cmake ..
  make VERBOSE=1
  popd

  # Run tests Python-side and export a script module.
  python model.py --export-script-module=model
  # Run tests C++-side and load the exported script module.
  build/test_jit_hooks ./model
  popd
  assert_git_not_dirty
}

# Shellcheck doesn't like it when you pass no arguments to a function
# that can take args. See https://www.shellcheck.net/wiki/SC2120
# shellcheck disable=SC2120
checkout_install_torchbench() {
  local commit
  commit=$(cat .ci/docker/ci_commit_pins/torchbench.txt)
  git clone https://github.com/pytorch/benchmark torchbench
  pushd torchbench
  git checkout "$commit"

  if [ "$1" ]; then
    python install.py --continue_on_fail models "$@"
  else
    # Occasionally the installation may fail on one model but it is ok to continue
    # to install and test other models
    python install.py --continue_on_fail
  fi
  popd

  pip install -r .ci/docker/ci_commit_pins/huggingface-requirements.txt
  # https://github.com/pytorch/pytorch/issues/160689 to remove torchao because
  # its current version 0.12.0 doesn't work with transformers 4.54.0
  pip uninstall -y torchao

  echo "Print all dependencies after TorchBench is installed"
  python -mpip freeze
}

torchbench_setup_macos() {
  git clone --recursive https://github.com/pytorch/vision torchvision
  git clone --recursive https://github.com/pytorch/audio torchaudio
  brew install jpeg-turbo libpng

  pushd torchvision
  git fetch
  git checkout "$(cat ../.github/ci_commit_pins/vision.txt)"
  git submodule update --init --recursive
  python setup.py clean
  python -m pip install -e . -v --no-build-isolation
  popd

  pushd torchaudio
  git fetch
  git checkout "$(cat ../.github/ci_commit_pins/audio.txt)"
  git submodule update --init --recursive
  python setup.py clean
  #TODO: Remove me, when figure out how to make TorchAudio find brew installed openmp
  USE_OPENMP=0 python -m pip install -e . -v --no-build-isolation
  popd

  checkout_install_torchbench
}

pip_benchmark_deps() {
  python -mpip install --no-input requests cython scikit-learn six
}


test_torchbench_perf() {
  echo "Launching torchbench setup"
  pip_benchmark_deps
  torchbench_setup_macos

  TEST_REPORTS_DIR=$(pwd)/test/test-reports
  mkdir -p "$TEST_REPORTS_DIR"

  local backend=eager
  local dtype=notset
  local device=mps

  echo "Setup complete, launching torchbench training performance run"
  PYTHONPATH="$(pwd)"/torchbench python benchmarks/dynamo/torchbench.py \
    --performance --backend "$backend" --training --devices "$device" \
    --output "$TEST_REPORTS_DIR/inductor_${backend}_torchbench_${dtype}_training_${device}_performance.csv"

  echo "Launching torchbench inference performance run"
  PYTHONPATH="$(pwd)"/torchbench python benchmarks/dynamo/torchbench.py \
    --performance --backend "$backend" --inference --devices "$device" \
    --output "$TEST_REPORTS_DIR/inductor_${backend}_torchbench_${dtype}_inference_${device}_performance.csv"

  echo "Pytorch benchmark on mps device completed"
}

test_torchbench_smoketest() {
  echo "Launching torchbench setup"
  pip_benchmark_deps
  # shellcheck disable=SC2119,SC2120
  torchbench_setup_macos

  TEST_REPORTS_DIR=$(pwd)/test/test-reports
  mkdir -p "$TEST_REPORTS_DIR"

  local device=mps
  local dtypes=(undefined float16 bfloat16 notset)
  local dtype=${dtypes[$1]}
  local models=(llama BERT_pytorch dcgan yolov3 resnet152 sam sam_fast pytorch_unet stable_diffusion_text_encoder speech_transformer Super_SloMo doctr_det_predictor doctr_reco_predictor vgg16)

  for backend in eager inductor; do

    echo "Launching torchbench inference performance run for backend ${backend} and dtype ${dtype}"
    local dtype_arg="--${dtype}"
    if [ "$dtype" == notset ]; then
        dtype_arg="--float32"
    fi
    touch "$TEST_REPORTS_DIR/inductor_${backend}_torchbench_${dtype}_inference_${device}_performance.csv"
    for model in "${models[@]}"; do
      PYTHONPATH="$(pwd)"/torchbench python benchmarks/dynamo/torchbench.py \
        --performance --only "$model" --backend "$backend" --inference --devices "$device" "$dtype_arg" \
        --output "$TEST_REPORTS_DIR/inductor_${backend}_torchbench_${dtype}_inference_${device}_performance.csv" || true
      if [ "$backend" == "inductor" ]; then
        PYTHONPATH="$(pwd)"/torchbench python benchmarks/dynamo/torchbench.py \
          --accuracy --only "$model" --backend "$backend" --inference --devices "$device" "$dtype_arg" \
          --output "$TEST_REPORTS_DIR/inductor_${backend}_torchbench_${dtype}_inference_${device}_accuracy.csv" || true
      fi
    done
    if [ "$backend" == "inductor" ]; then
      PYTHONPATH="$(pwd)"/torchbench python benchmarks/dynamo/huggingface.py \
        --performance --backend "$backend" --inference --devices "$device" "$dtype_arg" \
        --output "$TEST_REPORTS_DIR/inductor_${backend}_huggingface_${dtype}_inference_${device}_performance.csv" || true
      PYTHONPATH="$(pwd)"/torchbench python benchmarks/dynamo/huggingface.py \
        --accuracy --backend "$backend" --inference --devices "$device" "$dtype_arg" \
        --output "$TEST_REPORTS_DIR/inductor_${backend}_huggingface_${dtype}_inference_${device}_accuracy.csv" || true
    fi

    if [ "$dtype" == notset ]; then
      for dtype_ in notset amp; do
        echo "Launching torchbench training performance run for backend ${backend} and dtype ${dtype_}"
        touch "$TEST_REPORTS_DIR/inductor_${backend}_torchbench_${dtype_}_training_${device}_performance.csv"
        local dtype_arg="--${dtype_}"
        if [ "$dtype_" == notset ]; then
          dtype_arg="--float32"
        fi
        for model in "${models[@]}"; do
          PYTHONPATH="$(pwd)"/torchbench python benchmarks/dynamo/torchbench.py \
            --performance --only "$model" --backend "$backend" --training --devices "$device" "$dtype_arg" \
            --output "$TEST_REPORTS_DIR/inductor_${backend}_torchbench_${dtype_}_training_${device}_performance.csv" || true
        done
      done
    fi

  done
  echo "Pytorch benchmark on mps device completed"
}

test_aoti_torchbench_smoketest() {
  echo "Launching AOTInductor torchbench setup"
  pip_benchmark_deps
  # shellcheck disable=SC2119,SC2120
  torchbench_setup_macos

  TEST_REPORTS_DIR=$(pwd)/test/test-reports
  mkdir -p "$TEST_REPORTS_DIR"

  local device=mps
  local dtypes=(undefined float16 bfloat16 notset)
  local dtype=${dtypes[$1]}
  local models=(llama BERT_pytorch dcgan yolov3 resnet152 sam sam_fast pytorch_unet stable_diffusion_text_encoder speech_transformer Super_SloMo doctr_det_predictor doctr_reco_predictor vgg16)

  echo "Launching torchbench inference performance run for AOT Inductor and dtype ${dtype}"
  local dtype_arg="--${dtype}"
  if [ "$dtype" == notset ]; then
      dtype_arg="--float32"
  fi
  touch "$TEST_REPORTS_DIR/aot_inductor_torchbench_${dtype}_inference_${device}_performance.csv"
  for model in "${models[@]}"; do
    PYTHONPATH="$(pwd)"/torchbench python benchmarks/dynamo/torchbench.py \
      --performance --only "$model" --export-aot-inductor --inference --devices "$device" "$dtype_arg" \
      --output "$TEST_REPORTS_DIR/aot_inductor_torchbench_${dtype}_inference_${device}_performance.csv" || true
    PYTHONPATH="$(pwd)"/torchbench python benchmarks/dynamo/torchbench.py \
      --accuracy --only "$model" --export-aot-inductor --inference --devices "$device" "$dtype_arg" \
      --output "$TEST_REPORTS_DIR/aot_inductor_torchbench_${dtype}_inference_${device}_accuracy.csv" || true
  done

  echo "Launching HuggingFace inference performance run for AOT Inductor and dtype ${dtype}"
  PYTHONPATH="$(pwd)"/torchbench python benchmarks/dynamo/huggingface.py \
    --performance --export-aot-inductor --inference --devices "$device" "$dtype_arg" \
    --output "$TEST_REPORTS_DIR/aot_inductor_huggingface_${dtype}_inference_${device}_performance.csv" || true
  PYTHONPATH="$(pwd)"/torchbench python benchmarks/dynamo/huggingface.py \
    --accuracy --export-aot-inductor --inference --devices "$device" "$dtype_arg" \
    --output "$TEST_REPORTS_DIR/aot_inductor_huggingface_${dtype}_inference_${device}_accuracy.csv" || true

  echo "Pytorch benchmark on mps device completed"
}

test_hf_perf() {
  TEST_REPORTS_DIR=$(pwd)/test/test-reports
  mkdir -p "$TEST_REPORTS_DIR"
  pip_benchmark_deps
  torchbench_setup_macos

  echo "Launching HuggingFace training perf run"
  python "$(pwd)"/benchmarks/dynamo/huggingface.py --backend eager --device mps --performance --training --output="${TEST_REPORTS_DIR}"/hf_training.csv

  echo "Launching HuggingFace inference perf run"
  python "$(pwd)"/benchmarks/dynamo/huggingface.py --backend eager --device mps --performance --training --output="${TEST_REPORTS_DIR}"/hf_inference.csv

  echo "HuggingFace benchmark on mps device completed"
}

test_timm_perf() {
  TEST_REPORTS_DIR=$(pwd)/test/test-reports
  mkdir -p "$TEST_REPORTS_DIR"
  pip_benchmark_deps
  torchbench_setup_macos

  echo "Launching timm training perf run"
  python "$(pwd)"/benchmarks/dynamo/timm_models.py --backend eager --device mps --performance --training --output="${TEST_REPORTS_DIR}"/timm_training.csv

  echo "Launching timm inference perf run"
  python "$(pwd)"/benchmarks/dynamo/timm_models.py --backend eager --device mps --performance --training --output="${TEST_REPORTS_DIR}"/timm_inference.csv

  echo "timm benchmark on mps device completed"
}

if [[ $TEST_CONFIG == *"perf_all"* ]]; then
  test_torchbench_perf
  test_hf_perf
  test_timm_perf
elif [[ $TEST_CONFIG == *"perf_torchbench"* ]]; then
  test_torchbench_perf
elif [[ $TEST_CONFIG == *"perf_hf"* ]]; then
  test_hf_perf
elif [[ $TEST_CONFIG == *"perf_timm"* ]]; then
  test_timm_perf
elif [[ $TEST_CONFIG == *"perf_smoketest"* ]]; then
  test_torchbench_smoketest "${SHARD_NUMBER}"
elif [[ $TEST_CONFIG == *"aot_inductor_perf_smoketest"* ]]; then
  test_aoti_torchbench_smoketest "${SHARD_NUMBER}"
elif [[ $TEST_CONFIG == *"openreg"* ]]; then
  test_python_openreg
elif [[ $TEST_CONFIG == *"mps"* ]]; then
  test_python_mps
elif [[ $NUM_TEST_SHARDS -gt 1 ]]; then
  test_python_shard "${SHARD_NUMBER}"
  if [[ "${SHARD_NUMBER}" == 1 ]]; then
    test_libtorch
    test_custom_script_ops
  elif [[ "${SHARD_NUMBER}" == 2 ]]; then
    test_jit_hooks
    test_custom_backend
  fi
else
  test_python_all
  test_libtorch
  test_custom_script_ops
  test_jit_hooks
  test_custom_backend
fi
