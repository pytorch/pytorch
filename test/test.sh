#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# FAIL
source ~/miniforge3/bin/activate oneapi_current_pt-3d6eb40_dle-2026.2.0.536_drv-1146.78
LOG_DIR="$SCRIPT_DIR/logs_fail"
# PASS
# source ~/miniforge3/bin/activate oneccl_nightly_20260828_218ba3b
# Tanima's suggestion 2026.2.0.466      1f2df61 9ddfa68c        842579a6
# source ~/miniforge3/bin/activate PYTORCHDGQ-9881-pt_9ddfa68c-xpu_842579a6-ccl_1f2df61
LOG_DIR="$SCRIPT_DIR/logs_pass"


DLE_ROOT=/home/sdp/DLEs/dle_2026.2.0.536
mkdir -p "$LOG_DIR"
# PYTORCH_ROOT=/home/sdp/repositories/pytorch/pytorch
PYTORCH_ROOT=/home/sdp/repositories/pytorch_experiment/pytorch
source "$DLE_ROOT/compiler/latest/env/vars.sh"
source "$DLE_ROOT/mkl/latest/env/vars.sh"
source "$DLE_ROOT/pti/latest/env/vars.sh"
source "$DLE_ROOT/umf/latest/env/vars.sh"
source "$DLE_ROOT/tcm/latest/env/vars.sh"
source "$DLE_ROOT/ccl/latest/env/vars.sh"
source "$DLE_ROOT/mpi/latest/env/vars.sh"

# Print Torch version
torch_version=$(python -c 'import torch; print(torch.__version__)')
echo "Torch version: $torch_version"
run_test() {
    local error_group="$1"
    local test_id="$2"
    local log_name
    log_name="$(printf '%s' "$test_id" | tr '/:' '_' | tr -s '_').log"

    printf '\n[%s] %s\nLog: %s/%s\n' "$error_group" "$test_id" "$LOG_DIR" "$log_name"
    timeout 180 python -m pytest -s -v "$test_id" >"$LOG_DIR/$log_name" 2>&1
    local status=$?
    printf 'Exit code: %d\n' "$status"
}

# Install dependencies
# pip install pytest-cov
# cd "$PYTORCH_ROOT"
# rm -rf /tmp/torchinductor_$(whoami)
# sed -i '/mkl/d' .ci/docker/requirements-ci.txt
# pip install -r .ci/docker/requirements-ci.txt
# pip install pytest-timeout
# pip install --upgrade typing-extensions
# pip install -r .ci/docker/ci_commit_pins/huggingface-requirements.txt
cd "$PYTORCH_ROOT/third_party/torch-xpu-ops/test/xpu"

export USE_CCL_V2=1
export BACKEND="xccl"
export WORLD_SIZE=4
export TEMP_DIR=/tmp
python -c 'import inspect,shutil,torch.testing._internal.distributed.distributed_test as m; shutil.copyfile("distributed/distributed_test.py", inspect.getsourcefile(m) or m.__file__)'
export PYTORCH_PRINT_REPRO_ON_FAILURE=0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTHONPATH="$PYTHONPATH:../../../../test/distributed/pipelining"
export ZE_AFFINITY_MASK="0,1,2,3"
# To avoid 'functorch.compile' error, we need to cd to the test directory
cd "$PYTORCH_ROOT/third_party/torch-xpu-ops/test"

# export CCL log level to debug level
# export CCL_LOG_LEVEL=DEBUG

# RuntimeError: An event cannot be enqueued for signaling or waiting behind a command which is not enqueued in the backend.
# run_test "runtime_error_an_event_cannot_be_enqueued_for_signaling_or_waiting_behind_a_command_which_is_not_enqueued_in_the_backend" \
# $PYTORCH_ROOT/test/distributed/_composable/fsdp/test_fully_shard_extensions.py::TestFullyShardAllGatherExtensionsMultiProcess::test_all_gather_extensions_train_parity

# FAILED with KeyError: 'duration_ms'
 timeout 60 python -m pytest -s -v $PYTORCH_ROOT/third_party/torch-xpu-ops/test/xpu/distributed/test_c10d_xccl_local.py::XCCLTraceTest::test_coalescing_manager_collective_timing_enabled_True
 timeout 60 python -m pytest -s -v $PYTORCH_ROOT/third_party/torch-xpu-ops/test/xpu/distributed/test_c10d_xccl_local.py::XCCLTraceTest::test_short_json_timing_enabled_True_include_collectives_True
 timeout 60 python -m pytest -s -v $PYTORCH_ROOT/third_party/torch-xpu-ops/test/xpu/distributed/test_c10d_xccl_local.py::XCCLTraceTest::test_short_pickle_timing_enabled_True_include_collectives_True
