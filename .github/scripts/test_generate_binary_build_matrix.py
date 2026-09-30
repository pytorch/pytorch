from unittest import main, TestCase

import generate_binary_build_matrix as matrix


class TestRocmPreviewMatrix(TestCase):
    def preview_configs(self) -> list[dict[str, str]]:
        return [
            c
            for c in matrix.generate_wheels_matrix("linux")
            if c["gpu_arch_version"] == matrix.ROCM_PREVIEW.arch
        ]

    def test_preview_is_not_a_versioned_rocm_arch(self) -> None:
        self.assertNotIn(matrix.ROCM_PREVIEW.arch, matrix.ROCM_ARCHES)
        self.assertEqual(matrix.arch_type(matrix.ROCM_PREVIEW.arch), "rocm")

    def test_preview_uses_one_channel(self) -> None:
        configs = self.preview_configs()
        self.assertEqual(len(configs), len(matrix.FULL_PYTHON_VERSIONS))
        for config in configs:
            self.assertEqual(config["desired_cuda"], "rocmpreview")
            self.assertEqual(config["upload_subfolder"], "rocm-preview")
            self.assertEqual(config["container_image_tag_prefix"], "rocm-preview")
            self.assertEqual(
                config["pytorch_extra_install_requirements"],
                f"rocm[libraries,device-all]=={matrix.ROCM_PREVIEW.version}",
            )
        source = matrix.NIGHTLY_SOURCE_MATRIX["rocm-preview"]
        self.assertTrue(source["index_url"].endswith("/nightly/rocm-preview"))

    def test_non_preview_configs_have_no_subfolder_override(self) -> None:
        for platform in ("linux", "linux-aarch64", "windows"):
            for config in matrix.generate_wheels_matrix(platform):
                if config.get("gpu_arch_version") != matrix.ROCM_PREVIEW.arch:
                    self.assertNotIn("upload_subfolder", config)

    def test_versioned_rocm_names_are_unchanged(self) -> None:
        names = {
            c["build_name"]
            for c in matrix.generate_wheels_matrix("linux", python_versions=["3.11"])
            if c["gpu_arch_type"] == "rocm"
        }
        self.assertEqual(
            names,
            {
                "manywheel-py3_11-rocm7_14",
                "manywheel-py3_11-rocm10_0",
                "manywheel-py3_11-rocmpreview",
            },
        )


if __name__ == "__main__":
    main()
