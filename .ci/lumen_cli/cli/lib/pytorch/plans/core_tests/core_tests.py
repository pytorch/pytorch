"""
Core PyTorch tests — python test/run_test.py based.
Corresponds to jit_legacy / numpy_2 / default TEST_CONFIG in test.sh.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys

from cli.lib.pytorch.base import CoreTestPlan, is_xpu, run_test, TestStep


def _legacy_jit() -> None:
    run_test(
        "--include test_jit_legacy test_jit_fuser_legacy",
        "--verbose",
    )


def _setup_numpy_2() -> None:
    # test.sh: if [[ "${TEST_CONFIG}" == *numpy_2* ]]
    pandas_ver = subprocess.run(
        [sys.executable, "-c", "import pandas; print(pandas.__version__)"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    pkgs = ["--pre", "numpy==2.0.2", "scipy==1.13.1", "numba==0.60.0"]
    if pandas_ver:
        pkgs += [f"pandas=={pandas_ver}", "--force-reinstall"]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + pkgs)


def _numpy_2() -> None:
    run_test(
        "--include"
        " dynamo/test_functions.py"
        " dynamo/test_unspec.py"
        " test_binary_ufuncs.py"
        " test_fake_tensor.py"
        " test_linalg.py"
        " test_numpy_interop.py"
        " test_tensor_creation_ops.py"
        " test_torch.py"
        " torch_np/test_basic.py",
    )


def _python_shard(shard_id: int, num_shards: int, extra_args: list[str]) -> None:
    run_test(
        "--exclude-jit-executor",
        "--exclude-distributed-tests",
        "--exclude-quantization-tests",
        f"--shard {shard_id} {num_shards}",
        "--verbose",
        *extra_args,
    )


def _python_shard_steps(
    build_env: str, shard_id: int, num_shards: int, extra_args: list[str]
) -> list[TestStep]:
    return [
        TestStep(
            test_id="python_shard",
            fn=functools.partial(_python_shard, shard_id, num_shards, extra_args),
        )
    ]


CORE_TEST_PLANS: dict[str, CoreTestPlan] = {
    "pytorch_default_test": CoreTestPlan(
        group_id="pytorch_default_test",
        title="PyTorch Default Tests (test_python_shard)",
        # test.sh: default config → test_python_shard
        test_configs=["default"],
        # test.sh: if [[ "$TEST_CONFIG" == 'default' ]]; then export CUDA_VISIBLE_DEVICES=0; export HIP_VISIBLE_DEVICES=0
        env_vars={"CUDA_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0"},
        extra_args=[
            lambda env: "--xpu" if is_xpu(env) else None,
            lambda _: f"--include {os.environ['TESTS_TO_INCLUDE']}"
            if os.environ.get("TESTS_TO_INCLUDE")
            else None,
        ],
        get_steps_fn=_python_shard_steps,
    ),
    "pytorch_jit_legacy": CoreTestPlan(
        group_id="pytorch_jit_legacy",
        title="PyTorch JIT Legacy Tests",
        # test.sh: elif [[ "$TEST_CONFIG" == 'jit_legacy' ]]; then test_python_legacy_jit
        test_configs=["jit_legacy"],
        steps=[
            TestStep(test_id="jit_legacy", fn=_legacy_jit),
        ],
    ),
    "pytorch_numpy_2": CoreTestPlan(
        group_id="pytorch_numpy_2",
        title="PyTorch NumPy 2 Tests",
        # test.sh: if [[ "${TEST_CONFIG}" == *numpy_2* ]]
        test_configs=["numpy_2_x"],
        setup_fn=_setup_numpy_2,
        steps=[
            TestStep(test_id="numpy_2", fn=_numpy_2),
        ],
    ),
}
