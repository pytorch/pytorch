# Owner(s): ["module: unknown"]
import os
import pickle
from unittest.mock import patch


os.environ["ENV_TRUE"] = "1"
os.environ["ENV_FALSE"] = "0"

from typing import Optional

from torch.testing._internal import fake_config_module as config
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._config_module import _UNSET_SENTINEL


class TestConfigModule(TestCase):
    def test_base_value_loading(self):
        self.assertTrue(config.e_bool)
        self.assertTrue(config.nested.e_bool)
        self.assertTrue(config.e_optional)
        self.assertEqual(config.e_int, 1)
        self.assertEqual(config.e_float, 1.0)
        self.assertEqual(config.e_string, "string")
        self.assertEqual(config.e_list, [1])
        self.assertEqual(config.e_set, {1})
        self.assertEqual(config.e_tuple, (1,))
        self.assertEqual(config.e_dict, {1: 2})
        self.assertEqual(config.e_none, None)
        with self.assertRaises(
            AttributeError, msg="fake_config_module.does_not_exist does not exist"
        ):
            config.does_not_exist

    def test_type_loading(self):
        self.assertEqual(config.get_type("e_optional"), Optional[bool])
        self.assertEqual(config.get_type("e_none"), Optional[bool])

    def test_overrides(self):
        config.e_bool = False
        self.assertFalse(config.e_bool)
        config.nested.e_bool = False
        self.assertFalse(config.nested.e_bool)
        config.e_int = 2
        self.assertEqual(config.e_int, 2)
        config.e_float = 2.0
        self.assertEqual(config.e_float, 2.0)
        config.e_string = "string2"
        self.assertEqual(config.e_string, "string2")
        config.e_list = [2]
        self.assertEqual(config.e_list, [2])
        config.e_set = {2}
        self.assertEqual(config.e_set, {2})
        config.e_tuple = (2,)
        self.assertEqual(config.e_tuple, (2,))
        config.e_dict = {2: 3}
        self.assertEqual(config.e_dict, {2: 3})
        config.e_none = "not none"
        self.assertEqual(config.e_none, "not none")
        config.e_none = None
        self.assertEqual(config.e_none, None)
        config.e_optional = None
        self.assertEqual(config.e_optional, None)
        config.e_optional = False
        self.assertEqual(config.e_optional, False)
        with self.assertRaises(
            AttributeError, msg="fake_config_module.does_not_exist does not exist"
        ):
            config.does_not_exist = 0
        # Config changes get persisted between test cases
        for k in config._config:
            config._config[k].user_override = _UNSET_SENTINEL

    def test_none_override_semantics(self):
        config.e_bool = None
        self.assertIsNone(config.e_bool)
        for k in config._config:
            config._config[k].user_override = _UNSET_SENTINEL

    def test_reference_semantics(self):
        config.e_list.append(2)
        self.assertEqual(config.e_list, [1, 2])
        config.e_set.add(2)
        self.assertEqual(config.e_set, {1, 2})
        config.e_dict[2] = 3
        self.assertEqual(config.e_dict, {1: 2, 2: 3})
        for k in config._config:
            config._config[k].user_override = _UNSET_SENTINEL

    def test_env_name_semantics(self):
        self.assertTrue(config.e_env_default)
        self.assertFalse(config.e_env_default_FALSE)
        self.assertTrue(config.e_env_force)
        config.e_env_default = False
        self.assertFalse(config.e_env_default)
        config.e_env_force = False
        self.assertTrue(config.e_env_force)
        for k in config._config:
            config._config[k].user_override = _UNSET_SENTINEL

    def test_save_config(self):
        p = config.save_config()
        self.assertEqual(
            pickle.loads(p),
            {
                "_cache_config_ignore_prefix": ["magic_cache_config"],
                "e_bool": True,
                "e_dict": {1: 2},
                "e_float": 1.0,
                "e_int": 1,
                "e_list": [1],
                "e_none": None,
                "e_set": {1},
                "e_string": "string",
                "e_tuple": (1,),
                "nested.e_bool": True,
                "_e_ignored": True,
                "e_compile_ignored": True,
                "magic_cache_config_ignored": True,
                "_save_config_ignore": ["e_ignored"],
                "e_config": True,
                "e_jk": True,
                "e_jk_false": False,
                "e_env_default": True,
                "e_env_default_FALSE": False,
                "e_env_force": True,
                "e_optional": True,
            },
        )
        config.e_bool = False
        config.e_ignored = False
        config.load_config(p)
        self.assertTrue(config.e_bool)
        self.assertFalse(config.e_ignored)
        for k in config._config:
            config._config[k].user_override = _UNSET_SENTINEL

    def test_save_config_portable(self):
        p = config.save_config_portable()
        self.assertEqual(
            p,
            {
                "e_bool": True,
                "e_dict": {1: 2},
                "e_float": 1.0,
                "e_int": 1,
                "e_list": [1],
                "e_none": None,
                "e_set": {1},
                "e_string": "string",
                "e_tuple": (1,),
                "nested.e_bool": True,
                "e_ignored": True,
                "e_compile_ignored": True,
                "e_config": True,
                "e_jk": True,
                "e_jk_false": False,
                "e_env_default": True,
                "e_env_default_FALSE": False,
                "e_env_force": True,
                "e_optional": True,
            },
        )
        config.e_bool = False
        config._e_ignored = False
        config.load_config(p)
        self.assertTrue(config.e_bool)
        self.assertFalse(config._e_ignored)
        # Config changes get persisted between test cases
        for k in config._config:
            config._config[k].user_override = _UNSET_SENTINEL

    def test_codegen_config(self):
        config.e_bool = False
        config.e_ignored = False
        code = config.codegen_config()
        self.assertEqual(
            code,
            """torch.testing._internal.fake_config_module.e_bool = False
torch.testing._internal.fake_config_module.e_list = [1]
torch.testing._internal.fake_config_module.e_set = {1}
torch.testing._internal.fake_config_module.e_dict = {1: 2}
torch.testing._internal.fake_config_module._save_config_ignore = ['e_ignored']""",
        )
        # Config changes get persisted between test cases
        for k in config._config:
            config._config[k].user_override = _UNSET_SENTINEL

    def test_get_hash(self):
        self.assertEqual(config.get_hash(), b"\xf2C\xdbo\x99qq\x12\x11\xf7\xb4\xeewVpZ")
        # Test cached value
        self.assertEqual(config.get_hash(), b"\xf2C\xdbo\x99qq\x12\x11\xf7\xb4\xeewVpZ")
        self.assertEqual(config.get_hash(), b"\xf2C\xdbo\x99qq\x12\x11\xf7\xb4\xeewVpZ")
        config._hash_digest = "fake"
        self.assertEqual(config.get_hash(), "fake")

        config.e_bool = False
        self.assertNotEqual(
            config.get_hash(), b"\xf2C\xdbo\x99qq\x12\x11\xf7\xb4\xeewVpZ"
        )
        config.e_bool = True

        # Test ignored values
        config.e_compile_ignored = False
        self.assertEqual(config.get_hash(), b"\xf2C\xdbo\x99qq\x12\x11\xf7\xb4\xeewVpZ")
        for k in config._config:
            config._config[k].user_override = _UNSET_SENTINEL

    def test_dict_copy_semantics(self):
        p = config.shallow_copy_dict()
        self.assertDictEqual(
            p,
            {
                "e_bool": True,
                "e_dict": {1: 2},
                "e_float": 1.0,
                "e_int": 1,
                "e_list": [1],
                "e_none": None,
                "e_set": {1},
                "e_string": "string",
                "e_tuple": (1,),
                "nested.e_bool": True,
                "e_ignored": True,
                "_e_ignored": True,
                "e_compile_ignored": True,
                "_cache_config_ignore_prefix": ["magic_cache_config"],
                "_save_config_ignore": ["e_ignored"],
                "magic_cache_config_ignored": True,
                "e_config": True,
                "e_jk": True,
                "e_jk_false": False,
                "e_env_default": True,
                "e_env_default_FALSE": False,
                "e_env_force": True,
                "e_optional": True,
            },
        )
        p2 = config.to_dict()
        self.assertEqual(
            p2,
            {
                "e_bool": True,
                "e_dict": {1: 2},
                "e_float": 1.0,
                "e_int": 1,
                "e_list": [1],
                "e_none": None,
                "e_set": {1},
                "e_string": "string",
                "e_tuple": (1,),
                "nested.e_bool": True,
                "e_ignored": True,
                "_e_ignored": True,
                "e_compile_ignored": True,
                "_cache_config_ignore_prefix": ["magic_cache_config"],
                "_save_config_ignore": ["e_ignored"],
                "magic_cache_config_ignored": True,
                "e_config": True,
                "e_jk": True,
                "e_jk_false": False,
                "e_env_default": True,
                "e_env_default_FALSE": False,
                "e_env_force": True,
                "e_optional": True,
            },
        )
        p3 = config.get_config_copy()
        self.assertEqual(
            p3,
            {
                "e_bool": True,
                "e_dict": {1: 2},
                "e_float": 1.0,
                "e_int": 1,
                "e_list": [1],
                "e_none": None,
                "e_set": {1},
                "e_string": "string",
                "e_tuple": (1,),
                "nested.e_bool": True,
                "e_ignored": True,
                "_e_ignored": True,
                "e_compile_ignored": True,
                "_cache_config_ignore_prefix": ["magic_cache_config"],
                "_save_config_ignore": ["e_ignored"],
                "magic_cache_config_ignored": True,
                "e_config": True,
                "e_jk": True,
                "e_jk_false": False,
                "e_env_default": True,
                "e_env_default_FALSE": False,
                "e_env_force": True,
                "e_optional": True,
            },
        )

        # Shallow + deep copy semantics
        config.e_dict[2] = 3
        self.assertEqual(p["e_dict"], {1: 2})
        self.assertEqual(p2["e_dict"], {1: 2})
        self.assertEqual(p3["e_dict"], {1: 2})
        for k in config._config:
            config._config[k].user_override = _UNSET_SENTINEL

    def test_patch(self):
        self.assertTrue(config.e_bool)
        with config.patch("e_bool", False):
            self.assertFalse(config.e_bool)
        self.assertTrue(config.e_bool)
        with config.patch(e_bool=False):
            self.assertFalse(config.e_bool)
        self.assertTrue(config.e_bool)
        with self.assertRaises(AssertionError):
            with config.patch("does_not_exist"):
                pass

    def test_make_closur_patcher(self):
        revert = config._make_closure_patcher(e_bool=False)()
        self.assertFalse(config.e_bool)
        revert()
        self.assertTrue(config.e_bool)

    def test_unittest_patch(self):
        with patch("torch.testing._internal.fake_config_module.e_bool", False):
            with patch("torch.testing._internal.fake_config_module.e_bool", False):
                self.assertFalse(config.e_bool)
            # unittest.mock has some very weird semantics around deletion of attributes when undoing patches
            self.assertFalse(config.e_bool)
        self.assertTrue(config.e_bool)


if __name__ == "__main__":
    run_tests()
