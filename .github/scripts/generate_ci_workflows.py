#!/usr/bin/env python3

"""
Generate CI workflows. This script is based on common binary build matrix
and requires generate_binary_build_matrix.py file from test-infra repository.
To automatically pull this file and regenerate run it using  .githun/regenerate.sh
script.

"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Set, List, Iterable
from types import ModuleType
from enum import Enum
import json
import jinja2
import io
from contextlib import redirect_stdout

import os
import sys
from typing_extensions import Literal, TypedDict


Arch = Literal["windows", "linux", "macos"]

GITHUB_DIR = Path(__file__).resolve().parent.parent

LABEL_CIFLOW_TRUNK = "ciflow/trunk"
LABEL_CIFLOW_BINARIES = "ciflow/binaries"
LABEL_CIFLOW_PERIODIC = "ciflow/periodic"
LABEL_CIFLOW_BINARIES_LIBTORCH = "ciflow/binaries_libtorch"
LABEL_CIFLOW_BINARIES_CONDA = "ciflow/binaries_conda"
LABEL_CIFLOW_BINARIES_WHEEL = "ciflow/binaries_wheel"
CHANNEL = "nightly"
PRE_CXX11_ABI = "pre-cxx11"
CXX11_ABI = "cxx11-abi"
ENABLE = "enable"
DISABLE = "disable"
RELEASE = "release"
DEBUG = "debug"


@dataclass
class CIFlowConfig:
    # For use to enable workflows to run on pytorch/pytorch-canary
    run_on_canary: bool = False
    labels: Set[str] = field(default_factory=set)
    # Certain jobs might not want to be part of the ciflow/[all,trunk] workflow
    isolated_workflow: bool = False

    def __post_init__(self) -> None:
        if not self.isolated_workflow:
            if LABEL_CIFLOW_PERIODIC not in self.labels:
                self.labels.add(LABEL_CIFLOW_TRUNK)

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
    branches: str = CHANNEL
    # Mainly for macos
    cross_compile_arm64: bool = False
    xcode_version: str = ""

    def __post_init__(self) -> None:
        if self.abi_version:
            self.build_environment = f"{self.os}-binary-{self.package_type}-{self.abi_version}"
        else:
            self.build_environment = f"{self.os}-binary-{self.package_type}"

    def generate_workflow_file(self, workflow_template: jinja2.Template) -> None:
        output_file_path = GITHUB_DIR / f"workflows/generated-{self.build_environment}-{self.branches}.yml"
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

class PackageType(Enum):
    CONDA: str = "conda"
    WHEEL: str = "wheel"
    LIBTORCH: str = "libtorch"

class OperatingSystem(Enum):
    LINUX: str = "linux"
    WINDOWS: str = "windows"
    MACOS: str = "macos"
    MACOS_ARM64: str = "macos-arm64"

def import_module(fname: Path) -> ModuleType:
    import importlib.util
    spec = importlib.util.spec_from_file_location(fname.stem, fname)
    if spec is None:
        raise ImportError(f"Could not load spec for module {fname.stem} at: {fname}. \
        Please run the regenerate.sh from .github folder before running generate_ci_workflows")
    module = importlib.util.module_from_spec(spec)
    try:
        assert spec.loader is not None
        assert module is not None
        sys.modules[fname.stem] = module
        spec.loader.exec_module(module)
    except FileNotFoundError as e:
        raise ImportError(f"{e.strerror}: {fname}") from e
    return module

generate_binary_build_matrix = import_module(
    Path(__file__).parent / ".tools" / "generate_binary_build_matrix.py"
)

bin_bld_matrix : Dict[OperatingSystem, Dict[PackageType, List[Dict[str, str]]]] = {}
for osys in OperatingSystem:
    bin_bld_matrix[osys] = {}
    for package in PackageType:
        command = ["--channel", CHANNEL, "--operating-system", osys.value, "--package-type", package.value]
        if osys == OperatingSystem.MACOS_ARM64 and package == PackageType.LIBTORCH:
            continue
        elif osys == OperatingSystem.LINUX and package == PackageType.WHEEL:
            command += ["--with-py311", ENABLE, "--with-pypi-cudnn", ENABLE]
        f = io.StringIO()
        with redirect_stdout(f):
            generate_binary_build_matrix.main(command)
        bin_bld_matrix[osys][package] = json.loads(f.getvalue())["include"]


LINUX_BINARY_BUILD_WORFKLOWS = [
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX.value,
        package_type="manywheel",
        build_configs=bin_bld_matrix[OperatingSystem.LINUX][PackageType.WHEEL],
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_WHEEL},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX.value,
        package_type=PackageType.CONDA.value,
        build_configs=bin_bld_matrix[OperatingSystem.LINUX][PackageType.CONDA],
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_CONDA},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX.value,
        package_type=PackageType.LIBTORCH.value,
        abi_version=CXX11_ABI,
        build_configs=list(filter(
            lambda x: x["devtoolset"] == CXX11_ABI,
            bin_bld_matrix[OperatingSystem.LINUX][PackageType.LIBTORCH]
        )),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_LIBTORCH},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX.value,
        package_type=PackageType.LIBTORCH.value,
        abi_version=PRE_CXX11_ABI,
        build_configs=list(filter(
            lambda x: x["devtoolset"] == PRE_CXX11_ABI,
            bin_bld_matrix[OperatingSystem.LINUX][PackageType.LIBTORCH]
        )),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_LIBTORCH},
            isolated_workflow=True,
        ),
    ),
]

LINUX_BINARY_SMOKE_WORKFLOWS = [
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX.value,
        package_type="manywheel",
        build_configs=list(filter(
            lambda x:
            (x["gpu_arch_version"], x["python_version"]) == ("11.6", "3.7") and
            "pypi-cudnn" not in x["build_name"],
            bin_bld_matrix[OperatingSystem.LINUX][PackageType.WHEEL]
        )),
        branches="master",
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX.value,
        package_type=PackageType.LIBTORCH.value,
        abi_version=CXX11_ABI,
        build_configs=list(filter(
            lambda x:
            (x["devtoolset"], x["gpu_arch_type"], x["libtorch_variant"]) ==
            (CXX11_ABI, "cpu", "shared-with-deps"),
            bin_bld_matrix[OperatingSystem.LINUX][PackageType.LIBTORCH]
        )),
        branches="master",
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX.value,
        package_type=PackageType.LIBTORCH.value,
        abi_version=PRE_CXX11_ABI,
        build_configs=list(filter(
            lambda x:
            (x["devtoolset"], x["gpu_arch_type"], x["libtorch_variant"]) ==
            (PRE_CXX11_ABI, "cpu", "shared-with-deps"),
            bin_bld_matrix[OperatingSystem.LINUX][PackageType.LIBTORCH]
        )),
        branches="master",
    ),
]

WINDOWS_BINARY_BUILD_WORKFLOWS = [
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS.value,
        package_type=PackageType.WHEEL.value,
        build_configs=bin_bld_matrix[OperatingSystem.WINDOWS][PackageType.WHEEL],
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_WHEEL},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS.value,
        package_type=PackageType.CONDA.value,
        build_configs=bin_bld_matrix[OperatingSystem.WINDOWS][PackageType.CONDA],
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_CONDA},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS.value,
        package_type=PackageType.LIBTORCH.value,
        abi_version=RELEASE,
        build_configs=list(filter(
            lambda x:
            x["libtorch_config"] == RELEASE,
            bin_bld_matrix[OperatingSystem.WINDOWS][PackageType.LIBTORCH]
        )),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_LIBTORCH},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS.value,
        package_type=PackageType.LIBTORCH.value,
        abi_version=DEBUG,
        build_configs=list(filter(
            lambda x:
            x["libtorch_config"] == DEBUG,
            bin_bld_matrix[OperatingSystem.WINDOWS][PackageType.LIBTORCH]
        )),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_LIBTORCH},
            isolated_workflow=True,
        ),
    ),
]

WINDOWS_BINARY_SMOKE_WORKFLOWS = [
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS.value,
        package_type=PackageType.LIBTORCH.value,
        abi_version=RELEASE,
        build_configs=list(filter(
            lambda x:
            (x["libtorch_config"], x["gpu_arch_type"], x["libtorch_variant"]) ==
            (RELEASE, "cpu", "shared-with-deps"),
            bin_bld_matrix[OperatingSystem.WINDOWS][PackageType.LIBTORCH]
        )),
        branches="master",
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS.value,
        package_type=PackageType.LIBTORCH.value,
        abi_version=DEBUG,
        build_configs=list(filter(
            lambda x:
            (x["libtorch_config"], x["gpu_arch_type"], x["libtorch_variant"]) ==
            (DEBUG, "cpu", "shared-with-deps"),
            bin_bld_matrix[OperatingSystem.WINDOWS][PackageType.LIBTORCH]
        )),
        branches="master",
    ),
]

MACOS_BINARY_BUILD_WORKFLOWS = [
    BinaryBuildWorkflow(
        os=OperatingSystem.MACOS.value,
        package_type=PackageType.WHEEL.value,
        build_configs=bin_bld_matrix[OperatingSystem.MACOS][PackageType.WHEEL],
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_WHEEL},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.MACOS.value,
        package_type=PackageType.CONDA.value,
        build_configs=bin_bld_matrix[OperatingSystem.MACOS][PackageType.CONDA],
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_CONDA},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.MACOS.value,
        package_type=PackageType.LIBTORCH.value,
        abi_version=CXX11_ABI,
        build_configs=list(filter(
            lambda x:
            x["devtoolset"] == CXX11_ABI,
            bin_bld_matrix[OperatingSystem.MACOS][PackageType.LIBTORCH]
        )),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_LIBTORCH},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.MACOS.value,
        package_type=PackageType.LIBTORCH.value,
        abi_version=PRE_CXX11_ABI,
        build_configs=list(filter(
            lambda x:
            x["devtoolset"] == PRE_CXX11_ABI,
            bin_bld_matrix[OperatingSystem.MACOS][PackageType.LIBTORCH]
        )),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_LIBTORCH},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.MACOS_ARM64.value,
        package_type=PackageType.WHEEL.value,
        build_configs=bin_bld_matrix[OperatingSystem.MACOS_ARM64][PackageType.WHEEL],
        cross_compile_arm64=True,
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_WHEEL},
            isolated_workflow=True,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.MACOS_ARM64.value,
        package_type=PackageType.CONDA.value,
        cross_compile_arm64=True,
        build_configs=bin_bld_matrix[OperatingSystem.MACOS_ARM64][PackageType.CONDA],
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_CONDA},
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
        (jinja_env.get_template("linux_binary_build_workflow.yml.j2"), LINUX_BINARY_BUILD_WORFKLOWS),
        (jinja_env.get_template("linux_binary_build_workflow.yml.j2"), LINUX_BINARY_SMOKE_WORKFLOWS),
        (jinja_env.get_template("windows_binary_build_workflow.yml.j2"), WINDOWS_BINARY_BUILD_WORKFLOWS),
        (jinja_env.get_template("windows_binary_build_workflow.yml.j2"), WINDOWS_BINARY_SMOKE_WORKFLOWS),
        (jinja_env.get_template("macos_binary_build_workflow.yml.j2"), MACOS_BINARY_BUILD_WORKFLOWS),
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
