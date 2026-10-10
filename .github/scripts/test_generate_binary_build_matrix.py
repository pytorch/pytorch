import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("generate_binary_build_matrix.py")
SPEC = importlib.util.spec_from_file_location("generate_binary_build_matrix", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise AssertionError("unable to load generate_binary_build_matrix")
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_rocm_preview_is_a_named_rocm_lane() -> None:
    check(
        matrix.ROCM_PREVIEW.arch not in matrix.ROCM_ARCHES,
        "preview is numeric-independent",
    )
    check(
        matrix.ROCM_PREVIEW_ARCHES == [matrix.ROCM_PREVIEW.arch],
        "preview arches mismatch",
    )
    check(
        matrix.arch_type(matrix.ROCM_PREVIEW.arch) == "rocm", "preview lane is not ROCm"
    )


def test_rocm_preview_routes_to_one_channel() -> None:
    configs = matrix.generate_wheels_matrix("linux")
    preview = [
        config
        for config in configs
        if config["gpu_arch_version"] == matrix.ROCM_PREVIEW.arch
    ]

    check(
        len(preview) == len(matrix.FULL_PYTHON_VERSIONS),
        "preview Python matrix mismatch",
    )
    check(
        {config["desired_cuda"] for config in preview} == {"rocmpreview"},
        "preview channel mismatch",
    )
    check(
        {config["upload_subfolder"] for config in preview}
        == {matrix.ROCM_PREVIEW.channel},
        "preview upload route mismatch",
    )
    check(
        {config["container_image_tag_prefix"] for config in preview}
        == {matrix.ROCM_PREVIEW.channel},
        "preview image route mismatch",
    )
    check(
        {config["pytorch_extra_install_requirements"] for config in preview}
        == {f"rocm[libraries,device-all]=={matrix.ROCM_PREVIEW.version}"},
        "preview dependency mismatch",
    )

    source = matrix.ROCM_NIGHTLY_SOURCE_MATRIX[matrix.ROCM_PREVIEW.channel]
    check(
        source["index_url"].endswith(f"/{matrix.ROCM_PREVIEW.channel}"),
        "preview index mismatch",
    )


def test_stable_rocm_lanes_do_not_get_preview_routing() -> None:
    configs = matrix.generate_wheels_matrix("linux")
    stable = [
        config for config in configs if config["gpu_arch_version"] in matrix.ROCM_ARCHES
    ]

    check(stable, "stable ROCm lanes are missing")
    check(
        all("upload_subfolder" not in config for config in stable),
        "stable ROCm routes changed",
    )
