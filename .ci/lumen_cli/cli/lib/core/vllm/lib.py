import logging
import os
import textwrap
from typing import Any

from cli.lib.common.gh_summary import write_gh_step_summary
from cli.lib.common.git_helper import clone_external_repo
from cli.lib.common.pip_helper import pip_install_packages
from cli.lib.common.utils import run_command, temp_environ, working_directory
from jinja2 import Template


logger = logging.getLogger(__name__)

_TPL_VLLM_INFO = Template(
    textwrap.dedent("""\
    ##  Vllm against Pytorch CI Test Summary
    **Vllm Commit**: [{{ vllm_commit }}](https://github.com/vllm-project/vllm/commit/{{ vllm_commit }})
    {%- if torch_sha %}
    **Pytorch Commit**: [{{ torch_sha }}](https://github.com/pytorch/pytorch/commit/{{ torch_sha }})
    {%- endif %}
""")
)


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
            "env_vars": {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            },
            "steps": [
                "pytest -v -s basic_correctness/test_cumem.py",
                "pytest -v -s basic_correctness/test_basic_correctness.py",
                "pytest -v -s basic_correctness/test_cpu_offload.py",
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
            "env_vars": {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            },
            "steps": [
                " ".join(
                    [
                        "pytest",
                        "-v",
                        "-s",
                        "entrypoints/llm",
                        "--ignore=entrypoints/llm/test_generate.py",
                        "--ignore=entrypoints/llm/test_collective_rpc.py",
                    ]
                ),
                "pytest -v -s entrypoints/llm/test_generate.py",
                "pytest -v -s entrypoints/offline_mode",
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
            "env_vars": {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            },
            "num_gpus": 4,
            "steps": [
                "pytest -v -s -x lora/test_chatglm3_tp.py",
                "pytest -v -s -x lora/test_llama_tp.py",
                "pytest -v -s -x lora/test_llm_with_multi_loras.py",
            ],
        },
        "vllm_distributed_test_28_failure_test": {
            "title": "Distributed Tests (2 GPUs) pytorch 2.8 release failure",
            "id": "vllm_distributed_test_28_failure_test",
            "env_vars": {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            },
            "num_gpus": 4,
            "steps": [
                "pytest -v -s distributed/test_sequence_parallel.py",
            ],
        },
        "vllm_lora_28_failure_test": {
            "title": "LoRA pytorch 2.8 failure test",
            "id": "vllm_lora_28_failure_test",
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
        "vllm_multi_model_test_28_failure_test": {
            "title": "Multi-Model Test (Failed 2.8 release)",
            "id": "vllm_multi_model_test_28_failure_test",
            "package_install": ["git+https://github.com/TIGER-AI-Lab/Mantis.git"],
            "steps": [
                "pytest -v -s models/multimodal/generation/test_voxtral.py",
                "pytest -v -s models/multimodal/pooling",
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
        "vllm_languagde_model_test_extended_generation_28_failure_test": {
            "title": "Language Models Test (Extended Generation) 2.8 release failure",
            "id": "vllm_languagde_model_test_extended_generation_28_failure_test",
            "package_install": [
                "--no-build-isolation",
                "git+https://github.com/Dao-AILab/causal-conv1d@v1.5.0.post8",
            ],
            "steps": [
                "pytest -v -s models/language/generation/test_mistral.py",
            ],
        },
        "vllm_distributed_test_2_gpu_28_failure_test": {
            "title": "Distributed Tests (2 GPUs) pytorch 2.8 release failure",
            "id": "vllm_distributed_test_2_gpu_28_failure_test",
            "env_vars": {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            },
            "num_gpus": 4,
            "steps": [
                "pytest -v -s distributed/test_sequence_parallel.py",
            ],
        },
        # TODO(elainewy):need to add g6 with 4 gpus to run this test
        "vllm_lora_test": {
            "title": "LoRA Test %N",
            "id": "lora_test",
            "parallelism": 4,
            "steps": [
                "echo '[checking] list sharded lora tests:'",
                " ".join(
                    [
                        "pytest -q --collect-only lora",
                        "--shard-id=$$BUILDKITE_PARALLEL_JOB",
                        "--num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT",
                        "--ignore=lora/test_chatglm3_tp.py --ignore=lora/test_llama_tp.py",
                    ]
                ),
                "echo '[checking] Done. list lora tests'",
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
    with (
        working_directory(tests.get("working_directory", "tests")),
        temp_environ(tests.get("env_vars", {})),
    ):
        failures = []
        for step in tests["steps"]:
            logger.info("Running step: %s", step)
            if is_parallel:
                step = replace_buildkite_placeholders(step, shard_id, num_shards)
                logger.info("Running parallel step: %s", step)
            code = run_command(cmd=step, check=False, use_shell=True)
            if code != 0:
                failures.append(step)
            logger.info("Finish running step: %s", step)
        if failures:
            logger.error("Failed tests: %s", failures)
            raise RuntimeError(f"{len(failures)} pytest runs failed: {failures}")
        logger.info("Done. All tests passed")


def clone_vllm(dst: str = "vllm"):
    _, commit = clone_external_repo(
        target="vllm",
        repo="https://github.com/vllm-project/vllm.git",
        dst=dst,
        update_submodules=True,
    )
    return commit


def replace_buildkite_placeholders(step: str, shard_id: int, num_shards: int) -> str:
    mapping = {
        "$$BUILDKITE_PARALLEL_JOB_COUNT": str(num_shards),
        "$$BUILDKITE_PARALLEL_JOB": str(shard_id),
    }
    for k in sorted(mapping, key=len, reverse=True):
        step = step.replace(k, mapping[k])
    return step


def summarize_build_info(vllm_commit: str) -> bool:
    torch_sha = os.getenv("GITHUB_SHA")
    md = (
        _TPL_VLLM_INFO.render(vllm_commit=vllm_commit, torch_sha=torch_sha).strip()
        + "\n"
    )
    return write_gh_step_summary(md)
