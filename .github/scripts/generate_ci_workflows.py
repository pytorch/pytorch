#!/usr/bin/env python3

import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Set

import generate_binary_build_matrix  # type: ignore[import]

import jinja2
from typing_extensions import TypedDict  # Python 3.11+

Arch = Literal["windows", "linux", "macos"]

GITHUB_DIR = Path(__file__).resolve().parent.parent

LABEL_CIFLOW_TRUNK = "ciflow/trunk"
LABEL_CIFLOW_UNSTABLE = "ciflow/unstable"
LABEL_CIFLOW_BINARIES = "ciflow/binaries"
LABEL_CIFLOW_PERIODIC = "ciflow/periodic"
LABEL_CIFLOW_BINARIES_LIBTORCH = "ciflow/binaries_libtorch"
LABEL_CIFLOW_BINARIES_CONDA = "ciflow/binaries_conda"
LABEL_CIFLOW_BINARIES_WHEEL = "ciflow/binaries_wheel"


@dataclass
class CIFlowConfig:
    # For use to enable workflows to run on pytorch/pytorch-canary
    run_on_canary: bool = False
    labels: Set[str] = field(default_factory=set)
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
    build_configs: List[Dict[str, str]]
    package_type: str

    # Optional fields
    build_environment: str = ""
    abi_version: str = ""
    ciflow_config: CIFlowConfig = field(default_factory=CIFlowConfig)
    is_scheduled: str = ""
    branches: str = "nightly"
    # Mainly for macos
    cross_compile_arm64: bool = False
    xcode_version: str = ""

    def __post_init__(self) -> None:
        if self.abi_version:
            self.build_environment = (
                f"{self.os}-binary-{self.package_type}-{self.abi_version}"
            )
        else:
            self.build_environment = f"{self.os}-binary-{self.package_type}"

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
    MACOS = "macos"
    MACOS_ARM64 = "macos-arm64"
    LINUX_AARCH64 = "linux-aarch64"


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
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX,
        package_type="conda",
        build_configs=generate_binary_build_matrix.generate_conda_matrix(
            OperatingSystem.LINUX
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_CONDA},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX,
        package_type="libtorch",
        abi_version=generate_binary_build_matrix.CXX11_ABI,
        build_configs=generate_binary_build_matrix.generate_libtorch_matrix(
            OperatingSystem.LINUX, generate_binary_build_matrix.CXX11_ABI
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_LIBTORCH},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX,
        package_type="libtorch",
        abi_version=generate_binary_build_matrix.PRE_CXX11_ABI,
        build_configs=generate_binary_build_matrix.generate_libtorch_matrix(
            OperatingSystem.LINUX, generate_binary_build_matrix.PRE_CXX11_ABI
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
            arches=["11.8", "12.1"],
            python_versions=["3.8"],
            gen_special_an_non_special_wheel=False,
        ),
        branches="main",
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX,
        package_type="libtorch",
        abi_version=generate_binary_build_matrix.CXX11_ABI,
        build_configs=generate_binary_build_matrix.generate_libtorch_matrix(
            OperatingSystem.LINUX,
            generate_binary_build_matrix.CXX11_ABI,
            arches=["cpu"],
            libtorch_variants=["shared-with-deps"],
        ),
        branches="main",
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX,
        package_type="libtorch",
        abi_version=generate_binary_build_matrix.PRE_CXX11_ABI,
        build_configs=generate_binary_build_matrix.generate_libtorch_matrix(
            OperatingSystem.LINUX,
            generate_binary_build_matrix.PRE_CXX11_ABI,
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
        package_type="conda",
        build_configs=generate_binary_build_matrix.generate_conda_matrix(
            OperatingSystem.WINDOWS
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_CONDA},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS,
        package_type="libtorch",
        abi_version=generate_binary_build_matrix.RELEASE,
        build_configs=generate_binary_build_matrix.generate_libtorch_matrix(
            OperatingSystem.WINDOWS, generate_binary_build_matrix.RELEASE
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_LIBTORCH},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS,
        package_type="libtorch",
        abi_version=generate_binary_build_matrix.DEBUG,
        build_configs=generate_binary_build_matrix.generate_libtorch_matrix(
            OperatingSystem.WINDOWS, generate_binary_build_matrix.DEBUG
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
        abi_version=generate_binary_build_matrix.RELEASE,
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
        abi_version=generate_binary_build_matrix.DEBUG,
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
        os=OperatingSystem.MACOS,
        package_type="wheel",
        build_configs=generate_binary_build_matrix.generate_wheels_matrix(
            OperatingSystem.MACOS
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_WHEEL},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.MACOS,
        package_type="conda",
        build_configs=generate_binary_build_matrix.generate_conda_matrix(
            OperatingSystem.MACOS
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_CONDA},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.MACOS,
        package_type="libtorch",
        abi_version=generate_binary_build_matrix.CXX11_ABI,
        build_configs=generate_binary_build_matrix.generate_libtorch_matrix(
            OperatingSystem.MACOS, generate_binary_build_matrix.CXX11_ABI
        ),
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
        cross_compile_arm64=True,
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_WHEEL},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.MACOS_ARM64,
        package_type="conda",
        cross_compile_arm64=True,
        build_configs=generate_binary_build_matrix.generate_conda_matrix(
            OperatingSystem.MACOS_ARM64
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_CONDA},
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
            raise Exception(f"How is workflows not iterable? {workflows}")
        for workflow in workflows:
            workflow.generate_workflow_file(workflow_template=template)


if __name__ == "__main__":
    main()
