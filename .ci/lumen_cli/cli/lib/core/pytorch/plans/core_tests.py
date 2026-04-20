"""
Core PyTorch tests — python test/run_test.py based.
Corresponds to default / cpuonly / dynamo TEST_CONFIG in test.sh.
"""

from __future__ import annotations

from cli.lib.core.pytorch.pytorch_test_library import (
    CoreTestPlan,
    is_cpu_only,
    is_gpu,
    TestStep,
)
from cli.lib.core.pytorch.run_test_helper import run_test


def _cpuonly() -> None:
    run_test(
        "--exclude-jit-executor",
        "--exclude-distributed-tests",
        "--exclude-quantization-tests",
        "--verbose",
    )


def _default_shard(shard_id: int, num_shards: int) -> None:
    run_test(
        "--exclude-jit-executor",
        "--exclude-distributed-tests",
        "--exclude-quantization-tests",
        f"--shard {shard_id} {num_shards}",
        "--verbose",
        "--upload-artifacts-while-running",
    )


def _dynamo_core() -> None:
    run_test(
        "--include-dynamo-core-tests",
        "--verbose",
        "--upload-artifacts-while-running",
    )


CORE_TEST_PLANS: dict[str, CoreTestPlan] = {
    "pytorch_cpuonly": CoreTestPlan(
        group_id="pytorch_cpuonly",
        title="PyTorch CPU-only Tests",
        # test.sh: elif [[ $TEST_CONFIG == 'nogpu_NO_AVX2' ]] || ...
        test_configs=["nogpu_NO_AVX2", "nogpu_AVX512"],
        run_on=[is_cpu_only],
        steps=[
            TestStep(test_id="cpuonly", fn=_cpuonly),
        ],
    ),
    "pytorch_default_shard": CoreTestPlan(
        group_id="pytorch_default_shard",
        title="PyTorch Default Tests (sharded)",
        # test.sh: elif [[ "${SHARD_NUMBER}" == 1 ... ]] / [[ "${SHARD_NUMBER}" == 2 ... ]]
        test_configs=["default"],
        run_on=[is_gpu],
        env_vars=lambda env: (
            {"CUDA_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0,1,2,3"}
            if "rocm" in env
            else {"CUDA_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0"}
        ),
        steps=[
            TestStep(test_id="default_shard", fn=lambda: _default_shard(1, 1)),
        ],
    ),
    "pytorch_dynamo_core": CoreTestPlan(
        group_id="pytorch_dynamo_core",
        title="PyTorch Dynamo Core Tests",
        # test.sh: elif [[ "${TEST_CONFIG}" == *dynamo_core* ]]
        test_configs=["dynamo_core"],
        run_on=["cuda", "rocm"],
        steps=[
            TestStep(test_id="dynamo_core", fn=_dynamo_core),
        ],
    ),
}
