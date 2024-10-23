# Owner(s): ["module: unknown"]
import pickle

from torch.testing._internal import fake_config_module as config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestConfigModule(TestCase):
    def test_base_value_loading(self):
        self.assertTrue(config.e_bool)
        self.assertTrue(config.nested.e_bool)
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
        with self.assertRaises(
            AttributeError, msg="fake_config_module.does_not_exist does not exist"
        ):
            config.does_not_exist = 0
        # Config changes get persisted between test cases
        del config.e_bool
        del config.nested.e_bool
        del config.e_int
        del config.e_float
        del config.e_string
        del config.e_list
        del config.e_set
        del config.e_tuple
        del config.e_dict
        del config.e_none

    def test_delete(self):
        self.assertTrue(config.e_bool)
        del config.e_bool
        self.assertTrue(config.e_bool)
        config.e_bool = False
        del config.e_bool
        self.assertTrue(config.e_bool)

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
            },
        )
        config.e_bool = False
        config.e_ignored = False
        config.load_config(p)
        self.assertTrue(config.e_bool)
        self.assertFalse(config.e_ignored)
        del config.e_ignored

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
            },
        )
        config.e_bool = False
        config._e_ignored = False
        config.load_config(p)
        self.assertTrue(config.e_bool)
        self.assertFalse(config._e_ignored)
        # Config changes get persisted between test cases
        del config._e_ignored

    def test_codegen_config(self):
        config.e_bool = False
        config.e_ignored = False
        code = config.codegen_config()
        self.assertEqual(
            code, "torch.testing._internal.fake_config_module.e_bool = False"
        )
        # Config changes get persisted between test cases
        del config.e_bool
        del config.e_ignored

    def test_get_hash(self):
        self.assertEqual(
            config.get_hash(), b"\xcd\x96\x93\xf5(\xf8(\xa5\x1c+O\n\xd3_\x0b\xa6"
        )
        # Test cached value
        self.assertEqual(
            config.get_hash(), b"\xcd\x96\x93\xf5(\xf8(\xa5\x1c+O\n\xd3_\x0b\xa6"
        )
        self.assertEqual(
            config._hash_digest, b"\xcd\x96\x93\xf5(\xf8(\xa5\x1c+O\n\xd3_\x0b\xa6"
        )
        config._hash_digest = "fake"
        self.assertEqual(config.get_hash(), "fake")

        # BUG
        config.e_bool = False
        self.assertNotEqual(
            config.get_hash(), b"\xcd\x96\x93\xf5(\xf8(\xa5\x1c+O\n\xd3_\x0b\xa6"
        )
        config.e_bool = True

        # Test ignored values
        config.e_compile_ignored = False
        self.assertEqual(
            config.get_hash(), b"\xcd\x96\x93\xf5(\xf8(\xa5\x1c+O\n\xd3_\x0b\xa6"
        )
        del config.e_compile_ignored

    def test_dict_copy_semantics(self):
        p = config.shallow_copy_dict()
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
                "_e_ignored": True,
                "e_compile_ignored": True,
                "_cache_config_ignore_prefix": ["magic_cache_config"],
                "_save_config_ignore": ["e_ignored"],
                "magic_cache_config_ignored": True,
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
            },
        )

        # Shallow + deep copy semantics
        config.e_dict[2] = 3
        self.assertEqual(p["e_dict"], {1: 2})
        self.assertEqual(p2["e_dict"], {1: 2})
        self.assertEqual(p3["e_dict"], {1: 2})
        del config.e_dict

    def test_patch(self):
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


if __name__ == "__main__":
    run_tests()
