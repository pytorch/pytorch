import os
import tempfile
import unittest

from tools.gen_vulkan_glsl import GLSLGenerator
from yaml.constructor import ConstructorError


class TestGLSLCodegen(unittest.TestCase):
    def test_assert_on_duplicate_key_yaml(self) -> None:
        yaml_with_duplicate_keys = """
conv2d_pw:
  parameter_names_with_default_values:
      TILE_SIZE_X: 1
      TILE_SIZE_Y: 1
  parameter_values:
    - TILE_SIZE_X: 2
      TILE_SIZE_Y: 2
    - TILE_SIZE_X: 2
      TILE_SIZE_Y: 4
    - TILE_SIZE_X: 4
      TILE_SIZE_Y: 2
    - TILE_SIZE_X: 4
      TILE_SIZE_Y: 4
conv2d_pw:
  parameter_names_with_default_values:
    - TILE_SIZE_X: 1
    - TILE_SIZE_Y: 1
  parameter_values:
    - TILE_SIZE_X: 2
      TILE_SIZE_Y: 2
    - TILE_SIZE_X: 2
      TILE_SIZE_Y: 4
    - TILE_SIZE_X: 4
      TILE_SIZE_Y: 2
    - TILE_SIZE_X: 4
      TILE_SIZE_Y: 4
"""

        generator = GLSLGenerator()
        with tempfile.NamedTemporaryFile(mode="w") as fp:
            fp.write(yaml_with_duplicate_keys)
            fp.flush()
            with self.assertRaisesRegex(
                ConstructorError, r"while constructing a mapping"
            ):
                generator.add_params_yaml(fp.name)

    def test_assert_keys_mismatch(self) -> None:
        yaml_with_key_mismatch = """
conv2d_pw:
  parameter_names_with_default_values:
      TILE_SIZE_X: 1
      TILE_SIZE_Y: 1
  parameter_values:
    - TILE_SIZE_X: 2
      TILE_SIZE_Z: 2
"""

        generator = GLSLGenerator()
        with tempfile.NamedTemporaryFile(mode="w") as fp:
            fp.write(yaml_with_key_mismatch)
            fp.flush()
            with self.assertRaisesRegex(KeyError, r"Invalid keys {'TILE_SIZE_Z'}"):
                generator.add_params_yaml(fp.name)

    def test_missing_key_default_val(self) -> None:
        yaml_with_key_mismatch = """
conv2d_pw:
  parameter_names_with_default_values:
      TILE_SIZE_X: 1
      TILE_SIZE_Y: 1
  parameter_values:
    - TILE_SIZE_X: 2
"""
        file_content = """
x = $TILE_SIZE_X + $TILE_SIZE_Y
"""

        generator = GLSLGenerator()
        with tempfile.NamedTemporaryFile(mode="w") as fp:
            fp.write(yaml_with_key_mismatch)
            fp.flush()
            generator.add_params_yaml(fp.name)
            with tempfile.TemporaryDirectory() as tmp_dir:
                template_file_name = os.path.join(tmp_dir, "conv2d_pw.glslt")
                with open(template_file_name, "w") as template_file:
                    template_file.write(file_content)
                    template_file.flush()
                    generator.generate(template_file.name, tmp_dir)
                    file_name_1 = os.path.join(tmp_dir, "conv2d_pw_1x1.glsl")
                    file_name_2 = os.path.join(tmp_dir, "conv2d_pw_2x1.glsl")
                    self.assertTrue(os.path.exists(file_name_1))
                    self.assertTrue(os.path.exists(file_name_2))
                    with open(file_name_1, "r") as f:
                        contents = f.read()
                        self.assertTrue("1 + 1" in contents)
                    with open(file_name_2, "r") as f:
                        contents = f.read()
                        self.assertTrue("2 + 1" in contents)
