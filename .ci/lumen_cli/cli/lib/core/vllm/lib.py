import logging
from typing import Any

from cli.lib.common.git_helper import clone_external_repo
from cli.lib.common.pip_helper import pip_install_packages
from cli.lib.common.utils import run_command, working_directory


logger = logging.getLogger(__name__)


def sample_vllm_test_library():
    """
    Simple sample to unblock the vllm ci development, which is mimic to
    https://github.com/vllm-project/vllm/blob/main/.buildkite/test-pipeline.yaml
    see run_test_plan for more details
    """
    # TODO(elainewy): Read from yaml file to handle the env and tests for vllm
    return {
        "vllm_basic_correctness_test": {
            "title": "Basic Correctness Test",
            "id": "vllm_basic_correctness_test",
            "steps": [
                "export VLLM_WORKER_MULTIPROC_METHOD=spawn",
                "pytest -v -s basic_correctness/test_cumem.py",
                "pytest -v -s basic_correctness/test_basic_correctness.py",
                "pytest -v -s basic_correctness/test_cpu_offload.py",
                "VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT=1 pytest -v -s basic_correctness/test_preemption.py",
            ],
        },
        "vllm_basic_models_test": {
            "title": "Basic models test",
            "id": "vllm_basic_models_test",
            "steps": [
                "pytest -v -s models/test_transformers.py",
                "pytest -v -s models/test_registry.py",
                "pytest -v -s models/test_utils.py",
                "pytest -v -s models/test_vision.py",
                "pytest -v -s models/test_initialization.py",
            ],
        },
        "vllm_entrypoints_test": {
            "title": "Entrypoints Test ",
            "id": "vllm_entrypoints_test",
            "steps": [
                "export VLLM_WORKER_MULTIPROC_METHOD=spawn",
                " ".join(
                    [
                        "pytest",
                        "-v",
                        "-s",
                        "entrypoints/llm",
                        "--ignore=entrypoints/llm/test_lazy_outlines.py",
                        "--ignore=entrypoints/llm/test_generate.py",
                        "--ignore=entrypoints/llm/test_generate_multiple_loras.py",
                        "--ignore=entrypoints/llm/test_collective_rpc.py",
                    ]
                ),
                "pytest -v -s entrypoints/llm/test_lazy_outlines.py",
                "pytest -v -s entrypoints/llm/test_generate.py ",
                "pytest -v -s entrypoints/llm/test_generate_multiple_loras.py",
                "VLLM_USE_V1=0 pytest -v -s entrypoints/offline_mode",
            ],
        },
        "vllm_regression_test": {
            "title": "Regression Test",
            "id": "vllm_regression_test",
            "package_install": ["modelscope"],
            "steps": [
                "pytest -v -s test_regression.py",
            ],
        },
        "vllm_lora_tp_test_distributed": {
            "title": "LoRA TP Test (Distributed)",
            "id": "vllm_lora_tp_test_distributed",
            "num_gpus": 4,
            "steps": [
                "export VLLM_WORKER_MULTIPROC_METHOD=spawn",
                "VLLM_WORKER_MULTIPROC_METHOD=spawn pytest -v -s -x lora/test_chatglm3_tp.py",
                "VLLM_WORKER_MULTIPROC_METHOD=spawn pytest -v -s -x lora/test_llama_tp.py",
                "VLLM_WORKER_MULTIPROC_METHOD=spawn pytest -v -s -x lora/test_multi_loras_with_tp.py",
            ],
        },
        "vllm_lora_280_failure_test": {
            "title": "LoRA 280 failure test",
            "id": "vllm_lora_280_failure_test",
            "steps": ["pytest -v lora/test_quant_model.py"],
        },
        "vllm_multi_model_processor_test": {
            "title": "Multi-Modal Processor Test",
            "id": "vllm_multi_model_processor_test",
            "package_install": ["git+https://github.com/TIGER-AI-Lab/Mantis.git"],
            "steps": [
                "pytest -v -s models/multimodal/processing --ignore models/multimodal/processing/test_tensor_schema.py",
            ],
        },
        "vllm_pytorch_compilation_unit_tests": {
            "title": "PyTorch Compilation Unit Tests",
            "id": "vllm_pytorch_compilation_unit_tests",
            "steps": [
                "pytest -v -s compile/test_pass_manager.py",
                "pytest -v -s compile/test_fusion.py",
                "pytest -v -s compile/test_fusion_attn.py",
                "pytest -v -s compile/test_silu_mul_quant_fusion.py",
                "pytest -v -s compile/test_sequence_parallelism.py",
                "pytest -v -s compile/test_async_tp.py",
                "pytest -v -s compile/test_fusion_all_reduce.py",
                "pytest -v -s compile/test_decorator.py",
            ],
        },
        "vllm_lora_test": {
            "title": "LoRA Test %N",
            "id": "lora_test",
            "parallelism": 4,
            "steps": [
                " ".join(
                    [
                        "pytest -v -s lora --shard-id=$$BUILDKITE_PARALLEL_JOB",
                        "--num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT",
                        "--ignore=lora/test_chatglm3_tp.py --ignore=lora/test_llama_tp.py",
                    ]
                ),
            ],
        },
    }


def check_parallelism(tests: Any, title: str, shard_id: int = 0, num_shards: int = 0):
    """
    a method to check if the test plan is parallelism or not.
    """
    parallelism = int(tests.get("parallelism", "0"))
    is_parallel = parallelism and parallelism > 1

    if not is_parallel:
        return False

    if shard_id > num_shards:
        raise RuntimeError(
            f"Test {title} expects {num_shards} shards, but invalid {shard_id} is provided"
        )

    if num_shards != parallelism:
        raise RuntimeError(
            f"Test {title} expects {parallelism} shards, but invalid {num_shards} is provided"
        )

    return True


def run_test_plan(
    test_plan: str,
    test_target: str,
    tests_map: dict[str, Any],
    shard_id: int = 0,
    num_shards: int = 0,
):
    """
    a method to run list of tests based on the test plan.
    """
    logger.info("run %s tests.....", test_target)
    if test_plan not in tests_map:
        raise RuntimeError(
            f"test {test_plan} not found, please add it to test plan pool"
        )
    tests = tests_map[test_plan]
    pkgs = tests.get("package_install", [])
    title = tests.get("title", "unknown test")

    is_parallel = check_parallelism(tests, title, shard_id, num_shards)
    if is_parallel:
        title = title.replace("%N", f"{shard_id}/{num_shards}")

    logger.info("Running tests: %s", title)
    if pkgs:
        logger.info("Installing packages: %s", pkgs)
        pip_install_packages(packages=pkgs, prefer_uv=True)
    with working_directory(tests.get("working_directory", "tests")):
        failures = []
        for step in tests["steps"]:
            if is_parallel:
                step = replace_buildkite_placeholders(step, shard_id, num_shards)
                logger.info("Running prallel step: %s", step)
            code = run_command(cmd=step, check=False, use_shell=True)
            if code != 0:
                failures.append(step)
        if failures:
            logger.error("Failed tests: %s", failures)
            raise RuntimeError(f"{len(failures)} pytest runs failed: {failures}")
        logger.info("Done. All tests passed")


def clone_vllm(dst: str = "vllm"):
    clone_external_repo(
        target="vllm",
        repo="https://github.com/vllm-project/vllm.git",
        dst=dst,
        update_submodules=True,
    )


def replace_buildkite_placeholders(step: str, shard_id: int, num_shards: int) -> str:
    mapping = {
        "$$BUILDKITE_PARALLEL_JOB_COUNT": str(num_shards),
        "$$BUILDKITE_PARALLEL_JOB": str(shard_id),
    }
    for k in sorted(mapping, key=len, reverse=True):
        step = step.replace(k, mapping[k])
    return step
