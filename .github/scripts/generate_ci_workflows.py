#!/usr/bin/env python3

import os
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal
from typing_extensions import TypedDict  # Python 3.11+

import generate_binary_build_matrix  # type: ignore[import]
import jinja2


Arch = Literal["windows", "linux", "macos"]

GITHUB_DIR = Path(__file__).resolve().parent.parent

LABEL_CIFLOW_TRUNK = "ciflow/trunk"
LABEL_CIFLOW_UNSTABLE = "ciflow/unstable"
LABEL_CIFLOW_BINARIES = "ciflow/binaries"
LABEL_CIFLOW_PERIODIC = "ciflow/periodic"
LABEL_CIFLOW_BINARIES_LIBTORCH = "ciflow/binaries_libtorch"
LABEL_CIFLOW_BINARIES_WHEEL = "ciflow/binaries_wheel"


@dataclass
class CIFlowConfig:
    # For use to enable workflows to run on pytorch/pytorch-canary
    run_on_canary: bool = False
    labels: set[str] = field(default_factory=set)
    # Certain jobs might not want to be part of the ciflow/[all,trunk] workflow
    isolated_workflow: bool = False
    unstable: bool = False

    def __post_init__(self) -> None:
        if not self.isolated_workflow:
            if LABEL_CIFLOW_PERIODIC not in self.labels:
                self.labels.add(
                    LABEL_CIFLOW_TRUNK if not self.unstable else LABEL_CIFLOW_UNSTABLE
                )


class Config(TypedDict):
    num_shards: int
    runner: str


@dataclass
class BinaryBuildWorkflow:
    os: str
    build_configs: list[dict[str, str]]
    package_type: str

    # Optional fields
    build_environment: str = ""
    ciflow_config: CIFlowConfig = field(default_factory=CIFlowConfig)
    is_scheduled: str = ""
    branches: str = "nightly"
    # Mainly for macos
    cross_compile_arm64: bool = False
    macos_runner: str = "macos-14-xlarge"
    use_split_build: bool = False
    # Mainly used for libtorch builds
    build_variant: str = ""

    def __post_init__(self) -> None:
        if self.build_environment == "":
            self.build_environment = "-".join(
                item
                for item in [self.os, "binary", self.package_type, self.build_variant]
                if item != ""
            )
        if self.use_split_build:
            # added to distinguish concurrency groups
            self.build_environment += "-split"

    def generate_workflow_file(self, workflow_template: jinja2.Template) -> None:
        output_file_path = (
            GITHUB_DIR
            / f"workflows/generated-{self.build_environment}-{self.branches}.yml"
        )
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


class OperatingSystem:
    LINUX = "linux"
    WINDOWS = "windows"
    WINDOWS_ARM64 = "windows-arm64"
    MACOS = "macos"
    MACOS_ARM64 = "macos-arm64"
    LINUX_AARCH64 = "linux-aarch64"
    LINUX_S390X = "linux-s390x"


LINUX_BINARY_BUILD_WORFKLOWS = [
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX,
        package_type="manywheel",
        build_configs=generate_binary_build_matrix.generate_wheels_matrix(
            OperatingSystem.LINUX
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_WHEEL},
            isolated_workflow=True,
        ),
    ),
    # See https://github.com/pytorch/pytorch/issues/138750
    #   BinaryBuildWorkflow(
    #     os=OperatingSystem.LINUX,
    #     package_type="manywheel",
    #     build_configs=generate_binary_build_matrix.generate_wheels_matrix(
    #         OperatingSystem.LINUX,
    #         use_split_build=True,
    #         arches=["11.8", "12.1", "12.4", "cpu"],
    #     ),
    #     ciflow_config=CIFlowConfig(
    #         labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_WHEEL},
    #         isolated_workflow=True,
    #     ),
    #     use_split_build=True,
    # ),
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX,
        package_type="libtorch",
        build_configs=generate_binary_build_matrix.generate_libtorch_matrix(
            OperatingSystem.LINUX,
            generate_binary_build_matrix.RELEASE,
            libtorch_variants=["shared-with-deps"],
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_LIBTORCH},
            isolated_workflow=True,
        ),
    ),
]

LINUX_BINARY_SMOKE_WORKFLOWS = [
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX,
        package_type="manywheel",
        build_configs=generate_binary_build_matrix.generate_wheels_matrix(
            OperatingSystem.LINUX,
            arches=["11.8", "12.6", "12.8"],
            python_versions=["3.9"],
        ),
        branches="main",
    ),
    # See https://github.com/pytorch/pytorch/issues/138750
    # BinaryBuildWorkflow(
    #     os=OperatingSystem.LINUX,
    #     package_type="manywheel",
    #     build_configs=generate_binary_build_matrix.generate_wheels_matrix(
    #         OperatingSystem.LINUX,
    #         arches=["11.8", "12.1", "12.4"],
    #         python_versions=["3.9"],
    #         use_split_build=True,
    #     ),
    #     ciflow_config=CIFlowConfig(
    #         labels={LABEL_CIFLOW_PERIODIC},
    #     ),
    #     branches="main",
    #     use_split_build=True,
    # ),
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX,
        package_type="libtorch",
        build_variant=generate_binary_build_matrix.RELEASE,
        build_configs=generate_binary_build_matrix.generate_libtorch_matrix(
            OperatingSystem.LINUX,
            generate_binary_build_matrix.RELEASE,
            arches=["cpu"],
            libtorch_variants=["shared-with-deps"],
        ),
        branches="main",
    ),
]

WINDOWS_BINARY_BUILD_WORKFLOWS = [
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS,
        package_type="wheel",
        build_configs=generate_binary_build_matrix.generate_wheels_matrix(
            OperatingSystem.WINDOWS
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_WHEEL},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS,
        package_type="libtorch",
        build_variant=generate_binary_build_matrix.RELEASE,
        build_configs=generate_binary_build_matrix.generate_libtorch_matrix(
            OperatingSystem.WINDOWS,
            generate_binary_build_matrix.RELEASE,
            libtorch_variants=["shared-with-deps"],
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_LIBTORCH},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS,
        package_type="libtorch",
        build_variant=generate_binary_build_matrix.DEBUG,
        build_configs=generate_binary_build_matrix.generate_libtorch_matrix(
            OperatingSystem.WINDOWS,
            generate_binary_build_matrix.DEBUG,
            libtorch_variants=["shared-with-deps"],
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_LIBTORCH},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS_ARM64,
        package_type="wheel",
        build_configs=generate_binary_build_matrix.generate_wheels_matrix(
            OperatingSystem.WINDOWS_ARM64,
            arches=["cpu"],
            python_versions=["3.11", "3.12", "3.13"],
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_WHEEL},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS_ARM64,
        package_type="libtorch",
        build_variant=generate_binary_build_matrix.RELEASE,
        build_configs=generate_binary_build_matrix.generate_libtorch_matrix(
            OperatingSystem.WINDOWS_ARM64,
            generate_binary_build_matrix.RELEASE,
            arches=["cpu"],
            libtorch_variants=["shared-with-deps"],
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_LIBTORCH},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS_ARM64,
        package_type="libtorch",
        build_variant=generate_binary_build_matrix.DEBUG,
        build_configs=generate_binary_build_matrix.generate_libtorch_matrix(
            OperatingSystem.WINDOWS_ARM64,
            generate_binary_build_matrix.DEBUG,
            arches=["cpu"],
            libtorch_variants=["shared-with-deps"],
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_LIBTORCH},
            isolated_workflow=True,
        ),
    ),
]

WINDOWS_BINARY_SMOKE_WORKFLOWS = [
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS,
        package_type="libtorch",
        build_variant=generate_binary_build_matrix.RELEASE,
        build_configs=generate_binary_build_matrix.generate_libtorch_matrix(
            OperatingSystem.WINDOWS,
            generate_binary_build_matrix.RELEASE,
            arches=["cpu"],
            libtorch_variants=["shared-with-deps"],
        ),
        branches="main",
        ciflow_config=CIFlowConfig(
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS,
        package_type="libtorch",
        build_variant=generate_binary_build_matrix.DEBUG,
        build_configs=generate_binary_build_matrix.generate_libtorch_matrix(
            OperatingSystem.WINDOWS,
            generate_binary_build_matrix.DEBUG,
            arches=["cpu"],
            libtorch_variants=["shared-with-deps"],
        ),
        branches="main",
        ciflow_config=CIFlowConfig(
            isolated_workflow=True,
        ),
    ),
]

MACOS_BINARY_BUILD_WORKFLOWS = [
    BinaryBuildWorkflow(
        os=OperatingSystem.MACOS_ARM64,
        package_type="libtorch",
        build_variant=generate_binary_build_matrix.RELEASE,
        build_configs=generate_binary_build_matrix.generate_libtorch_matrix(
            OperatingSystem.MACOS,
            generate_binary_build_matrix.RELEASE,
            libtorch_variants=["shared-with-deps"],
        ),
        cross_compile_arm64=False,
        macos_runner="macos-14-xlarge",
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_LIBTORCH},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.MACOS_ARM64,
        package_type="wheel",
        build_configs=generate_binary_build_matrix.generate_wheels_matrix(
            OperatingSystem.MACOS_ARM64
        ),
        cross_compile_arm64=False,
        macos_runner="macos-14-xlarge",
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_WHEEL},
            isolated_workflow=True,
        ),
    ),
]

AARCH64_BINARY_BUILD_WORKFLOWS = [
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX_AARCH64,
        package_type="manywheel",
        build_configs=generate_binary_build_matrix.generate_wheels_matrix(
            OperatingSystem.LINUX_AARCH64
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_WHEEL},
            isolated_workflow=True,
        ),
    ),
]

S390X_BINARY_BUILD_WORKFLOWS = [
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX_S390X,
        package_type="manywheel",
        build_configs=generate_binary_build_matrix.generate_wheels_matrix(
            OperatingSystem.LINUX_S390X
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_WHEEL},
            isolated_workflow=True,
        ),
    ),
]


def main() -> None:
    jinja_env = jinja2.Environment(
        variable_start_string="!{{",
        loader=jinja2.FileSystemLoader(str(GITHUB_DIR.joinpath("templates"))),
        undefined=jinja2.StrictUndefined,
    )

    # not ported yet
    template_and_workflows = [
        (
            jinja_env.get_template("linux_binary_build_workflow.yml.j2"),
            LINUX_BINARY_BUILD_WORFKLOWS,
        ),
        (
            jinja_env.get_template("linux_binary_build_workflow.yml.j2"),
            AARCH64_BINARY_BUILD_WORKFLOWS,
        ),
        (
            jinja_env.get_template("linux_binary_build_workflow.yml.j2"),
            S390X_BINARY_BUILD_WORKFLOWS,
        ),
        (
            jinja_env.get_template("linux_binary_build_workflow.yml.j2"),
            LINUX_BINARY_SMOKE_WORKFLOWS,
        ),
        (
            jinja_env.get_template("windows_binary_build_workflow.yml.j2"),
            WINDOWS_BINARY_BUILD_WORKFLOWS,
        ),
        (
            jinja_env.get_template("windows_binary_build_workflow.yml.j2"),
            WINDOWS_BINARY_SMOKE_WORKFLOWS,
        ),
        (
            jinja_env.get_template("macos_binary_build_workflow.yml.j2"),
            MACOS_BINARY_BUILD_WORKFLOWS,
        ),
    ]
    # Delete the existing generated files first, this should align with .gitattributes file description.
    existing_workflows = GITHUB_DIR.glob("workflows/generated-*")
    for w in existing_workflows:
        try:
            os.remove(w)
        except Exception as e:
            print(f"Error occurred when deleting file {w}: {e}")

    for template, workflows in template_and_workflows:
        # added Iterable check to appease the mypy gods
        if not isinstance(workflows, Iterable):
            raise Exception(  # noqa: TRY002
                f"How is workflows not iterable? {workflows}"
            )  # noqa: TRY002
        for workflow in workflows:
            workflow.generate_workflow_file(workflow_template=template)


if __name__ == "__main__":
    main()
