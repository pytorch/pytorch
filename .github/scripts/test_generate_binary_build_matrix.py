# ruff: noqa: S101

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("generate_binary_build_matrix.py")
SPEC = importlib.util.spec_from_file_location("generate_binary_build_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


def test_rocm_preview_is_a_named_rocm_lane() -> None:
    assert matrix.ROCM_PREVIEW.arch not in matrix.ROCM_ARCHES
    assert matrix.ROCM_PREVIEW_ARCHES == [matrix.ROCM_PREVIEW.arch]
    assert matrix.arch_type(matrix.ROCM_PREVIEW.arch) == "rocm"


def test_rocm_preview_routes_to_one_channel() -> None:
    configs = matrix.generate_wheels_matrix("linux")
    preview = [
        config
        for config in configs
        if config["gpu_arch_version"] == matrix.ROCM_PREVIEW.arch
    ]

    assert len(preview) == len(matrix.FULL_PYTHON_VERSIONS)
    assert {config["desired_cuda"] for config in preview} == {"rocmpreview"}
    assert {config["upload_subfolder"] for config in preview} == {
        matrix.ROCM_PREVIEW.channel
    }
    assert {config["container_image_tag_prefix"] for config in preview} == {
        matrix.ROCM_PREVIEW.channel
    }
    assert {config["pytorch_extra_install_requirements"] for config in preview} == {
        f"rocm[libraries,device-all]=={matrix.ROCM_PREVIEW.version}"
    }

    source = matrix.ROCM_NIGHTLY_SOURCE_MATRIX[matrix.ROCM_PREVIEW.channel]
    assert source["index_url"].endswith(f"/{matrix.ROCM_PREVIEW.channel}")


def test_stable_rocm_lanes_do_not_get_preview_routing() -> None:
    configs = matrix.generate_wheels_matrix("linux")
    stable = [
        config for config in configs if config["gpu_arch_version"] in matrix.ROCM_ARCHES
    ]

    assert stable
    assert all("upload_subfolder" not in config for config in stable)
