#!/usr/bin/env python3

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Set

import jinja2
import json
import os
import sys
from typing_extensions import Literal

YamlShellBool = Literal["''", 1]
Arch = Literal["windows", "linux"]

DOCKER_REGISTRY = "308535385114.dkr.ecr.us-east-1.amazonaws.com"
GITHUB_DIR = Path(__file__).resolve().parent.parent

WINDOWS_CPU_TEST_RUNNER = "windows.4xlarge"
WINDOWS_CUDA_TEST_RUNNER = "windows.8xlarge.nvidia.gpu"
WINDOWS_RUNNERS = {
    WINDOWS_CPU_TEST_RUNNER,
    WINDOWS_CUDA_TEST_RUNNER,
}

LINUX_CPU_TEST_RUNNER = "linux.2xlarge"
LINUX_CUDA_TEST_RUNNER = "linux.8xlarge.nvidia.gpu"
LINUX_RUNNERS = {
    LINUX_CPU_TEST_RUNNER,
    LINUX_CUDA_TEST_RUNNER,
}

CUDA_RUNNERS = {
    WINDOWS_CUDA_TEST_RUNNER,
    LINUX_CUDA_TEST_RUNNER,
}
CPU_RUNNERS = {
    WINDOWS_CPU_TEST_RUNNER,
    LINUX_CPU_TEST_RUNNER,
}

LABEL_CIFLOW_ALL = "ciflow/all"
LABEL_CIFLOW_BAZEL = "ciflow/bazel"
LABEL_CIFLOW_COVERAGE = "ciflow/coverage"
LABEL_CIFLOW_CPU = "ciflow/cpu"
LABEL_CIFLOW_CUDA = "ciflow/cuda"
LABEL_CIFLOW_DEFAULT = "ciflow/default"
LABEL_CIFLOW_LIBTORCH = "ciflow/libtorch"
LABEL_CIFLOW_LINUX = "ciflow/linux"
LABEL_CIFLOW_SCHEDULED = "ciflow/scheduled"
LABEL_CIFLOW_SLOW = "ciflow/slow"
LABEL_CIFLOW_WIN = "ciflow/win"
LABEL_CIFLOW_XLA = "ciflow/xla"
LABEL_CIFLOW_NOARCH = "ciflow/noarch"


@dataclass
class CIFlowConfig:
    enabled: bool = False
    # For use to enable workflows to run on pytorch/pytorch-canary
    run_on_canary: bool = False
    labels: Set[str] = field(default_factory=set)
    trigger_action: str = 'unassigned'
    trigger_actor: str = 'pytorchbot'
    root_job_name: str = 'ciflow_should_run'
    root_job_condition: str = ''

    # trigger_action_only controls if we listen only on the trigger_action of a pull_request.
    # If it's False, we listen on all default pull_request actions, this is useful when
    # ciflow (via probot) is not automated yet.
    trigger_action_only: bool = False

    def gen_root_job_condition(self) -> None:
        # TODO: Make conditions strict
        # At the beginning of the rollout of ciflow, we keep everything the same as what we have
        # Once fully rollout, we can have strict constraints
        # e.g. ADD      env.GITHUB_ACTOR == '{self.trigger_actor}
        #      REMOVE   github.event.action !='{self.trigger_action}'
        label_conditions = [
            f"contains(github.event.pull_request.labels.*.name, '{label}')" for label in sorted(self.labels)]
        if self.run_on_canary:
            self.root_job_condition = "(github.repository_owner == 'pytorch') && "
        else:
            self.root_job_condition = "(github.repository == 'pytorch/pytorch') && "
        self.root_job_condition += f"((github.event_name != 'pull_request') || " \
            f"(github.event.action !='{self.trigger_action}') || " \
            f"({' || '.join(label_conditions)}))"

    def reset_root_job(self) -> None:
        self.root_job_name = ''
        self.root_job_condition = ''

    def __post_init__(self) -> None:
        if not self.enabled:
            self.reset_root_job()
            return
        self.labels.add(LABEL_CIFLOW_ALL)
        self.gen_root_job_condition()


@dataclass
class CIFlowRuleset:
    version = 'v1'
    output_file = f'{GITHUB_DIR}/generated-ciflow-ruleset.json'
    label_rules: Dict[str, Set[str]] = field(default_factory=dict)

    def add_label_rule(self, labels: Set[str], workflow_name: str) -> None:
        for label in labels:
            if label in self.label_rules:
                self.label_rules[label].add(workflow_name)
            else:
                self.label_rules[label] = {workflow_name}

    def generate_json(self) -> None:
        GENERATED = "generated"  # Note that please keep the variable GENERATED otherwise phabricator will hide the whole file
        output = {
            "__comment": f"@{GENERATED} DO NOT EDIT MANUALLY, Generation script: .github/scripts/generate_ci_workflows.py",
            "version": self.version,
            "label_rules": {
                label: sorted(list(workflows))
                for label, workflows in self.label_rules.items()
            }
        }
        with open(self.output_file, 'w') as outfile:
            json.dump(output, outfile, indent=2, sort_keys=True)
            outfile.write('\n')


@dataclass
class CIWorkflow:
    # Required fields
    arch: Arch
    build_environment: str
    test_runner_type: str

    # Optional fields
    ciflow_config: CIFlowConfig = field(default_factory=CIFlowConfig)
    cuda_version: str = ''
    docker_image_base: str = ''
    enable_doc_jobs: bool = False
    exclude_test: bool = False
    is_coverage: bool = False
    is_libtorch: bool = False
    is_scheduled: str = ''
    num_test_shards: int = 1
    on_pull_request: bool = False
    only_build_on_pull_request: bool = False
    only_run_smoke_tests_on_pull_request: bool = False
    num_test_shards_on_pull_request: int = -1
    distributed_test: bool = True

    # The following variables will be set as environment variables,
    # so it's easier for both shell and Python scripts to consume it if false is represented as the empty string.
    enable_jit_legacy_test: YamlShellBool = "''"
    enable_distributed_test: YamlShellBool = "''"
    enable_multigpu_test: YamlShellBool = "''"
    enable_nogpu_no_avx_test: YamlShellBool = "''"
    enable_nogpu_no_avx2_test: YamlShellBool = "''"
    enable_slow_test: YamlShellBool = "''"
    enable_docs_test: YamlShellBool = "''"
    enable_backwards_compat_test: YamlShellBool = "''"
    enable_xla_test: YamlShellBool = "''"
    enable_noarch_test: YamlShellBool = "''"

    def __post_init__(self) -> None:
        if self.is_libtorch:
            self.exclude_test = True

        if not self.on_pull_request:
            self.only_build_on_pull_request = False

        if self.distributed_test:
            self.enable_distributed_test = 1

        # If num_test_shards_on_pull_request is not user-defined, default to num_test_shards unless we are
        # only running smoke tests on the pull request.
        if self.num_test_shards_on_pull_request == -1:
            # Don't waste resources on runner spinup and cooldown for another shard if we are only running a few tests
            if self.only_run_smoke_tests_on_pull_request:
                self.num_test_shards_on_pull_request = 1
            else:
                self.num_test_shards_on_pull_request = self.num_test_shards
        self.assert_valid()

    def assert_valid(self) -> None:
        err_message = f"invalid test_runner_type for {self.arch}: {self.test_runner_type}"
        if self.arch == 'linux':
            assert self.test_runner_type in LINUX_RUNNERS, err_message
        if self.arch == 'windows':
            assert self.test_runner_type in WINDOWS_RUNNERS, err_message

        if self.ciflow_config.enabled:
            # make sure if LABEL_CIFLOW_DEFAULT is set, we then need to set trigger_action_only to False
            assert self.ciflow_config.trigger_action_only != (LABEL_CIFLOW_DEFAULT in self.ciflow_config.labels)
            assert self.on_pull_request
            assert LABEL_CIFLOW_ALL in self.ciflow_config.labels
            assert LABEL_CIFLOW_ALL in self.ciflow_config.root_job_condition
            if self.arch == 'linux':
                assert LABEL_CIFLOW_LINUX in self.ciflow_config.labels
            if self.arch == 'windows':
                assert LABEL_CIFLOW_WIN in self.ciflow_config.labels
            if self.test_runner_type in CUDA_RUNNERS:
                assert LABEL_CIFLOW_CUDA in self.ciflow_config.labels
            if self.test_runner_type in CPU_RUNNERS:
                assert LABEL_CIFLOW_CPU in self.ciflow_config.labels

    def generate_workflow_file(self, workflow_template: jinja2.Template) -> None:
        output_file_path = GITHUB_DIR / f"workflows/generated-{self.build_environment}.yml"
        with open(output_file_path, "w") as output_file:
            GENERATED = "generated"  # Note that please keep the variable GENERATED otherwise phabricator will hide the whole file
            output_file.writelines([f"# @{GENERATED} DO NOT EDIT MANUALLY\n"])
            try:
                content = workflow_template.render(asdict(self))
            except Exception as e:
                print(f"Failed on template: {workflow_template}", file=sys.stderr)
                raise e
            output_file.write(content)
            if content[-1] != "\n":
                output_file.write("\n")
        print(output_file_path)


WINDOWS_WORKFLOWS = [
    CIWorkflow(
        arch="windows",
        build_environment="win-vs2019-cpu-py3",
        cuda_version="cpu",
        test_runner_type=WINDOWS_CPU_TEST_RUNNER,
        on_pull_request=True,
        num_test_shards=2,
        ciflow_config=CIFlowConfig(
            enabled=True,
            run_on_canary=True,
            labels={LABEL_CIFLOW_DEFAULT, LABEL_CIFLOW_CPU, LABEL_CIFLOW_WIN}
        ),
    ),
    CIWorkflow(
        arch="windows",
        build_environment="win-vs2019-cuda10.2-py3",
        cuda_version="10.2",
        test_runner_type=WINDOWS_CUDA_TEST_RUNNER,
        on_pull_request=True,
        num_test_shards=2,
        ciflow_config=CIFlowConfig(
            enabled=True,
            trigger_action_only=True,
            labels={LABEL_CIFLOW_CUDA, LABEL_CIFLOW_WIN}
        ),
    ),
    CIWorkflow(
        arch="windows",
        build_environment="win-vs2019-cuda11.3-py3",
        cuda_version="11.3",
        test_runner_type=WINDOWS_CUDA_TEST_RUNNER,
        num_test_shards=2,
        on_pull_request=True,
        only_run_smoke_tests_on_pull_request=True,
        ciflow_config=CIFlowConfig(
            enabled=True,
            run_on_canary=True,
            labels={LABEL_CIFLOW_DEFAULT, LABEL_CIFLOW_CUDA, LABEL_CIFLOW_WIN}
        ),
    ),
    CIWorkflow(
        arch="windows",
        build_environment="periodic-win-vs2019-cuda11.1-py3",
        cuda_version="11.1",
        test_runner_type=WINDOWS_CUDA_TEST_RUNNER,
        num_test_shards=2,
        is_scheduled="45 0,4,8,12,16,20 * * *",
        on_pull_request=True,
        ciflow_config=CIFlowConfig(
            enabled=True,
            trigger_action_only=True,
            labels={LABEL_CIFLOW_SCHEDULED, LABEL_CIFLOW_WIN, LABEL_CIFLOW_CUDA}
        ),
    ),
]

LINUX_WORKFLOWS = [
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-py3.6-gcc5.4",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        on_pull_request=True,
        enable_jit_legacy_test=1,
        enable_doc_jobs=True,
        enable_docs_test=1,
        enable_backwards_compat_test=1,
        num_test_shards=2,
        ciflow_config=CIFlowConfig(
            enabled=True,
            run_on_canary=True,
            labels={LABEL_CIFLOW_DEFAULT, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CPU}
        ),
    ),
    # ParallelTBB does not have a maintainer and is currently flaky
    # CIWorkflow(
    #     arch="linux",
    #     build_environment="paralleltbb-linux-xenial-py3.6-gcc5.4",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    #     # This is a master only job despite on_pull_request is set to True
    #     on_pull_request=True,
    #     ciflow_config=CIFlowConfig(
    #         enabled=True,
    #         trigger_action_only=True,
    #         labels={LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CPU},
    #     ),
    # ),
    CIWorkflow(
        arch="linux",
        build_environment="parallelnative-linux-xenial-py3.6-gcc5.4",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        # This is a master only job despite on_pull_request is set to True
        on_pull_request=True,
        ciflow_config=CIFlowConfig(
            enabled=True,
            trigger_action_only=True,
            labels={LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CPU},
        ),
    ),
    # Build PyTorch with BUILD_CAFFE2=OFF
    CIWorkflow(
        arch="linux",
        build_environment="puretorch-linux-xenial-py3.6-gcc5.4",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc5.4",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        exclude_test=True,
        # This is a master only job despite on_pull_request is set to True
        on_pull_request=True,
        ciflow_config=CIFlowConfig(
            enabled=True,
            trigger_action_only=True,
            labels={LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CPU},
        ),
    ),
    # CIWorkflow(
    #     arch="linux",
    #     build_environment="linux-xenial-py3.6-gcc7",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3.6-gcc7",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # CIWorkflow(
    #     arch="linux",
    #     build_environment="linux-xenial-py3.6-clang5-asan",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-asan",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # CIWorkflow(
    #     arch="linux",
    #     build_environment="linux-xenial-py3.6-clang7-onnx",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang7-onnx",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-bionic-cuda10.2-py3.9-gcc7",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-cuda10.2-cudnn7-py3.9-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        num_test_shards=2,
        on_pull_request=True,
        ciflow_config=CIFlowConfig(
            enabled=True,
            run_on_canary=True,
            trigger_action_only=True,
            labels={LABEL_CIFLOW_SLOW, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CUDA}
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-cuda10.2-py3.6-gcc7",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda10.2-cudnn7-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        enable_jit_legacy_test=1,
        enable_multigpu_test=1,
        enable_nogpu_no_avx_test=1,
        enable_nogpu_no_avx2_test=1,
        enable_slow_test=1,
        num_test_shards=2,
        on_pull_request=True,
        ciflow_config=CIFlowConfig(
            enabled=True,
            trigger_action_only=True,
            labels=set([LABEL_CIFLOW_SLOW, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CUDA]),
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="libtorch-linux-xenial-cuda10.2-py3.6-gcc7",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda10.2-cudnn7-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        is_libtorch=True,
        on_pull_request=True,
        ciflow_config=CIFlowConfig(
            enabled=True,
            trigger_action_only=True,
            labels=set([LABEL_CIFLOW_LIBTORCH, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CUDA]),
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-cuda11.3-py3.6-gcc7",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.3-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        num_test_shards=2,
        on_pull_request=True,
        ciflow_config=CIFlowConfig(
            enabled=True,
            labels=set([LABEL_CIFLOW_DEFAULT, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CUDA]),
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="libtorch-linux-xenial-cuda11.3-py3.6-gcc7",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.3-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        is_libtorch=True,
        on_pull_request=True,
        ciflow_config=CIFlowConfig(
            enabled=True,
            trigger_action_only=True,
            labels=set([LABEL_CIFLOW_LIBTORCH, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CUDA]),
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="periodic-linux-xenial-cuda11.1-py3.6-gcc7",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.1-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        num_test_shards=2,
        is_scheduled="45 0,4,8,12,16,20 * * *",
        on_pull_request=True,
        ciflow_config=CIFlowConfig(
            enabled=True,
            trigger_action_only=True,
            labels={LABEL_CIFLOW_SCHEDULED, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CUDA}
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="periodic-libtorch-linux-xenial-cuda11.1-py3.6-gcc7",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-cuda11.1-cudnn8-py3-gcc7",
        test_runner_type=LINUX_CUDA_TEST_RUNNER,
        is_libtorch=True,
        is_scheduled="45 0,4,8,12,16,20 * * *",
        on_pull_request=True,
        ciflow_config=CIFlowConfig(
            enabled=True,
            trigger_action_only=True,
            labels={LABEL_CIFLOW_SCHEDULED, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_LIBTORCH, LABEL_CIFLOW_CUDA},
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-bionic-py3.8-gcc9-coverage",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-py3.8-gcc9",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        on_pull_request=True,
        is_coverage=True,
        num_test_shards=2,
        ciflow_config=CIFlowConfig(
            enabled=True,
            labels={LABEL_CIFLOW_DEFAULT, LABEL_CIFLOW_COVERAGE, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CPU},
        ),
    ),
    CIWorkflow(
        arch="linux",
        build_environment="linux-bionic-py3.6-clang9",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-py3.6-clang9",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        on_pull_request=True,
        num_test_shards=2,
        distributed_test=False,
        enable_noarch_test=1,
        ciflow_config=CIFlowConfig(
            enabled=True,
            labels={LABEL_CIFLOW_DEFAULT, LABEL_CIFLOW_LINUX, LABEL_CIFLOW_CPU, LABEL_CIFLOW_XLA, LABEL_CIFLOW_NOARCH},
        ),
    ),
    # CIWorkflow(
    #     arch="linux",
    #     build_environment="linux-bionic-rocm3.9-py3.6",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-rocm3.9-py3.6",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # CIWorkflow(
    #     arch="linux",
    #     build_environment="linux-xenial-py3.6-clang5-android-ndk-r19c-x86_32",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # CIWorkflow(
    #     arch="linux",
    #     build_environment="linux-xenial-py3.6-clang5-android-ndk-r19c-x86_64",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # CIWorkflow(
    #     arch="linux",
    #     build_environment="linux-xenial-py3.6-clang5-android-ndk-r19c-arm-v7a",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # CIWorkflow(
    #     arch="linux",
    #     build_environment="linux-xenial-py3.6-clang5-android-ndk-r19c-arm-v8a",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # CIWorkflow(
    #     arch="linux",
    #     build_environment="linux-xenial-py3.6-clang5-mobile",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-asan",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # CIWorkflow(
    #     arch="linux",
    #     build_environment="linux-xenial-py3.6-clang5-mobile-custom-dynamic",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # CIWorkflow(
    #     arch="linux",
    #     build_environment="linux-xenial-py3.6-clang5-mobile-custom-static",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
    # CIWorkflow(
    #     arch="linux",
    #     build_environment="linux-xenial-py3.6-clang5-mobile-code-analysis",
    #     docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c",
    #     test_runner_type=LINUX_CPU_TEST_RUNNER,
    # ),
]


BAZEL_WORKFLOWS = [
    CIWorkflow(
        arch="linux",
        build_environment="linux-xenial-py3.6-gcc7-bazel-test",
        docker_image_base=f"{DOCKER_REGISTRY}/pytorch/pytorch-linux-bionic-cuda10.2-cudnn7-py3.9-gcc7",
        test_runner_type=LINUX_CPU_TEST_RUNNER,
        on_pull_request=True,
        ciflow_config=CIFlowConfig(
            enabled=True,
            labels={LABEL_CIFLOW_DEFAULT, LABEL_CIFLOW_BAZEL, LABEL_CIFLOW_CPU, LABEL_CIFLOW_LINUX},
        ),
    ),
]

if __name__ == "__main__":
    jinja_env = jinja2.Environment(
        variable_start_string="!{{",
        loader=jinja2.FileSystemLoader(str(GITHUB_DIR.joinpath("templates"))),
        undefined=jinja2.StrictUndefined,
    )
    template_and_workflows = [
        (jinja_env.get_template("linux_ci_workflow.yml.j2"), LINUX_WORKFLOWS),
        (jinja_env.get_template("windows_ci_workflow.yml.j2"), WINDOWS_WORKFLOWS),
        (jinja_env.get_template("bazel_ci_workflow.yml.j2"), BAZEL_WORKFLOWS),
    ]
    # Delete the existing generated files first, this should align with .gitattributes file description.
    existing_workflows = GITHUB_DIR.glob("workflows/generated-*")
    for w in existing_workflows:
        try:
            os.remove(w)
        except Exception as e:
            print(f"Error occurred when deleting file {w}: {e}")

    ciflow_ruleset = CIFlowRuleset()
    for template, workflows in template_and_workflows:
        for workflow in workflows:
            workflow.generate_workflow_file(workflow_template=template)

            if workflow.ciflow_config.enabled:
                ciflow_ruleset.add_label_rule(workflow.ciflow_config.labels, workflow.build_environment)
            elif workflow.on_pull_request:
                # If ciflow is disabled but still on_pull_request, we can denote
                # it as a special label LABEL_CIFLOW_DEFAULT in the ruleset, which will be later
                # turned into an actual LABEL_CIFLOW_DEFAULT label in the workflow.
                # During the rollout phase, it has the same effect as LABEL_CIFLOW_DEFAULT
                ciflow_ruleset.add_label_rule({LABEL_CIFLOW_DEFAULT}, workflow.build_environment)
    ciflow_ruleset.generate_json()
