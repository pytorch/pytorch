# Owner(s): ["module: unknown"]
import os
import pickle
import queue
import threading
import warnings
from unittest.mock import patch


os.environ["ENV_TRUE"] = "1"
os.environ["ENV_FALSE"] = "0"
os.environ["ENV_STR"] = "1234"
os.environ["ENV_STR_EMPTY"] = ""

from torch.testing._internal import (
    fake_config_module as config,
    fake_config_module2 as config2,
    fake_config_module3 as config3,
    fake_config_module_alias as alias_config,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._config_module import (
    _ConfigEntry,
    _UNSET_SENTINEL,
    alias_fields_from,
    Config,
)


# Module-level fixture for the alias_fields_from nested-subconfig-class
# rejection test. Top-level so its __qualname__ is a bare name
# (alias_fields_from separately rejects nested *parent* classes). Never
# decorated at module scope: applying alias_fields_from to it is expected to
# raise, so it is only ever passed to the decorator inside a test.
class _alias_parent_with_nested_class:
    e_bool = True

    class not_a_field:
        pass


class TestConfigModule(TestCase):
    def tearDown(self):
        # Config changes get persisted between test cases
        for k in config._config:
            config._config[k].user_override.set(_UNSET_SENTINEL)
            config._config[k].hide = False
        config._hash_cache_var.set(None)
        config._get_dict_dirty_keys_var.set(None)
        config._get_dict_cache_var.set(None)
        # Reset deprecation warning flags
        for k in config._config:
            config._config[k]._deprecation_warned = False

        # alias_config is a separate installed module (kept apart from config
        # so test_fuzzer.py's sweep doesn't trip over its list-typed alias --
        # see fake_config_module_alias.py), so its state needs the same reset.
        for k in alias_config._config:
            alias_config._config[k].user_override.set(_UNSET_SENTINEL)
            alias_config._config[k].hide = False
        alias_config._hash_cache_var.set(None)
        alias_config._get_dict_dirty_keys_var.set(None)
        alias_config._get_dict_cache_var.set(None)

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
        self.assertEqual(config.get_type("e_optional"), bool | None)
        self.assertEqual(config.get_type("e_none"), bool | None)

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

    def test_none_override_semantics(self):
        config.e_bool = None
        self.assertIsNone(config.e_bool)
        for k in config._config:
            config._config[k].user_override.set(_UNSET_SENTINEL)

    def test_reference_semantics(self):
        config.e_list.append(2)
        self.assertEqual(config.e_list, [1, 2])
        config.e_set.add(2)
        self.assertEqual(config.e_set, {1, 2})
        config.e_dict[2] = 3
        self.assertEqual(config.e_dict, {1: 2, 2: 3})

    def test_env_name_semantics(self):
        self.assertTrue(config.e_env_default)
        self.assertFalse(config.e_env_default_FALSE)
        self.assertTrue(config.e_env_force)
        config.e_env_default = False
        self.assertFalse(config.e_env_default)
        config.e_env_force = False
        self.assertTrue(config.e_env_force)

    def test_env_name_string_semantics(self):
        self.assertEqual(config.e_env_default_str, "1234")
        self.assertEqual(config.e_env_default_str_empty, "")
        config.e_env_default_str = "override"
        self.assertEqual(config.e_env_default_str, "override")

    def test_multi_env(self):
        self.assertTrue(config2.e_env_default_multi)
        self.assertTrue(config2.e_env_force_multi)

    def test_save_config(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            p = config.save_config()

        self.assertDictEqual(
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
                "e_env_default_str": "1234",
                "e_env_default_str_empty": "",
                "e_env_force": True,
                "e_optional": True,
                "e_deprecated": True,
                "e_not_deprecated": False,
            },
        )
        config.e_bool = False
        config.e_ignored = False
        config.load_config(p)
        self.assertTrue(config.e_bool)
        self.assertFalse(config.e_ignored)

    def test_save_config_with_patch(self):
        self.assertTrue(config.e_bool)
        with config.patch(e_bool=False):
            p = config.save_config()
            self.assertDictEqual(
                pickle.loads(p),
                {
                    "_cache_config_ignore_prefix": ["magic_cache_config"],
                    "e_bool": False,
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
                    "e_env_default_str": "1234",
                    "e_env_default_str_empty": "",
                    "e_env_force": True,
                    "e_optional": True,
                    "e_deprecated": True,
                    "e_not_deprecated": False,
                },
            )

    def test_save_config_portable(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            p = config.save_config_portable()

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
                "e_compile_ignored": True,
                "e_config": True,
                "e_jk": True,
                "e_jk_false": False,
                "e_env_default": True,
                "e_env_default_FALSE": False,
                "e_env_default_str": "1234",
                "e_env_default_str_empty": "",
                "e_env_force": True,
                "e_optional": True,
                "e_deprecated": True,
                "e_not_deprecated": False,
            },
        )
        config.e_bool = False
        config._e_ignored = False
        config.load_config(p)
        self.assertTrue(config.e_bool)
        self.assertFalse(config._e_ignored)

    def test_codegen_config(self):
        config.e_bool = False
        config.e_ignored = False
        code = config.codegen_config()
        self.assertEqual(
            code,
            """torch.testing._internal.fake_config_module.e_bool = False
torch.testing._internal.fake_config_module.e_env_default = True
torch.testing._internal.fake_config_module.e_env_default_FALSE = False
torch.testing._internal.fake_config_module.e_env_default_str = '1234'
torch.testing._internal.fake_config_module.e_env_default_str_empty = ''
torch.testing._internal.fake_config_module.e_env_force = True""",
        )

    def test_codegen_config_function(self):
        import logging
        import warnings

        config3.e_list = [print, warnings.warn, logging.warn]
        config3.e_set = {print}
        config3.e_func = warnings.warn
        code = config3.codegen_config()
        self.assertIn("import _warnings", code)
        self.assertIn("import logging", code)
        self.assertIn(
            """torch.testing._internal.fake_config_module3.e_list = ['print', '_warnings.warn', 'logging.warn']
torch.testing._internal.fake_config_module3.e_set = { print }
torch.testing._internal.fake_config_module3.e_func = _warnings.warn""",
            code,
        )

    def test_get_hash(self):
        hash_value = b"#d\x8b\xd3\xbc'\xf5\x0c\xcd\xb6\x87zDw6g"

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.assertEqual(
                config.get_hash(),
                hash_value,
            )
            # Test cached value
            self.assertEqual(
                config.get_hash(),
                hash_value,
            )
            self.assertEqual(
                config.get_hash(),
                hash_value,
            )

        config._hash_cache_var.set(b"fake")
        self.assertEqual(config.get_hash(), b"fake")

        config.e_bool = False
        self.assertNotEqual(
            config.get_hash(),
            hash_value,
        )
        config.e_bool = True

        # Test ignored values
        config.e_compile_ignored = False
        self.assertEqual(
            config.get_hash(),
            hash_value,
        )

    def test_dict_copy_semantics(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            p = config.shallow_copy_dict()
            p2 = config.to_dict()

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
                "e_env_default_str": "1234",
                "e_env_default_str_empty": "",
                "e_env_force": True,
                "e_optional": True,
                "e_deprecated": True,
                "e_not_deprecated": False,
            },
        )
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
                "e_env_default_str": "1234",
                "e_env_default_str_empty": "",
                "e_env_force": True,
                "e_optional": True,
                "e_deprecated": True,
                "e_not_deprecated": False,
            },
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
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
                "e_env_default_str": "1234",
                "e_env_default_str_empty": "",
                "e_env_force": True,
                "e_optional": True,
                "e_deprecated": True,
                "e_not_deprecated": False,
            },
        )

        # Shallow + deep copy semantics
        config.e_dict[2] = 3
        self.assertEqual(p["e_dict"], {1: 2})
        self.assertEqual(p2["e_dict"], {1: 2})
        self.assertEqual(p3["e_dict"], {1: 2})

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

    def _assert_patch_reentrant_across_threads(self, fn):
        barrier = threading.Barrier(2, timeout=5)
        errors: queue.SimpleQueue[str] = queue.SimpleQueue()

        def worker():
            try:
                fn(barrier)
            except Exception as e:
                errors.put(repr(e))

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        error_messages = []
        while True:
            try:
                error_messages.append(errors.get_nowait())
            except queue.Empty:
                break

        self.assertFalse(
            error_messages,
            lambda msg: f"{msg}\nconcurrent patch usage failed: {error_messages}",
        )
        self.assertTrue(config.e_bool)

    def test_patch_context_manager_is_reentrant_across_threads(self):
        patcher = config.patch(e_bool=False)

        def fn(barrier):
            with patcher:
                self.assertFalse(config.e_bool)
                barrier.wait()
                self.assertFalse(config.e_bool)

        self._assert_patch_reentrant_across_threads(fn)

    def test_patch_context_manager_is_reentrant_when_nested(self):
        patcher = config.patch(e_bool=False)

        with patcher:
            self.assertFalse(config.e_bool)
            with patcher:
                self.assertFalse(config.e_bool)
            self.assertFalse(config.e_bool)

        self.assertTrue(config.e_bool)

    def test_patch_decorator_is_reentrant_across_threads(self):
        @config.patch(e_bool=False)
        def fn(barrier):
            self.assertFalse(config.e_bool)
            barrier.wait()
            self.assertFalse(config.e_bool)

        self._assert_patch_reentrant_across_threads(fn)

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

    def test_bad_jk_type(self):
        with self.assertRaises(
            AssertionError,
            msg="AssertionError: justknobs only support booleans, thisisnotvalid is not a boolean",
        ):
            _ConfigEntry(Config(default="bad", justknob="fake_knob"), "test")

    def test_alias(self):
        self.assertFalse(config2.e_aliasing_bool)
        self.assertFalse(config.e_aliased_bool)
        with config2.patch(e_aliasing_bool=True):
            self.assertTrue(config2.e_aliasing_bool)
            self.assertTrue(config.e_aliased_bool)
        with config.patch(e_aliased_bool=True):
            self.assertTrue(config2.e_aliasing_bool)

    def test_alias_to_subconfig_field(self):
        # An alias whose target is a sub-config field resolves through to it,
        # both as a top-level alias and via an alias_fields_from subconfig.
        self.assertEqual(config.e_nested_alias_bool, config.nested.e_bool)
        self.assertEqual(config.nested_alias.e_bool, config.nested.e_bool)
        # value_type resolves through the alias so get_type is meaningful.
        self.assertEqual(config.get_type("e_nested_alias_bool"), bool)
        self.assertEqual(config.get_type("nested_alias.e_bool"), bool)
        # Both are recorded as aliases in _config.
        self.assertIsNotNone(config._config["e_nested_alias_bool"].alias)
        self.assertIsNotNone(config._config["nested_alias.e_bool"].alias)

    def test_alias_write_through_and_sibling(self):
        self.assertTrue(config.nested.e_bool)
        # Updating the target is reflected through every alias to it.
        with config.patch({"nested.e_bool": False}):
            self.assertFalse(config.e_nested_alias_bool)
            self.assertFalse(config.nested_alias.e_bool)
        # Writing through one alias mutates the shared target and is therefore
        # observed through the sibling alias too.
        with config.patch({"nested_alias.e_bool": False}):
            self.assertFalse(config.nested.e_bool)
            self.assertFalse(config.e_nested_alias_bool)

    def test_alias_mock_patch_teardown(self):
        # mock.patch does delattr+setattr on teardown; the aliased field must
        # remain readable afterwards. Regression for `hide` never being cleared
        # on the alias write path.
        orig = config.nested.e_bool
        with patch(
            "torch.testing._internal.fake_config_module.e_nested_alias_bool",
            not orig,
        ):
            self.assertEqual(config.e_nested_alias_bool, not orig)
        self.assertEqual(config.e_nested_alias_bool, orig)
        self.assertEqual(config.nested.e_bool, orig)

    def test_alias_delattr_marks_target_dirty(self):
        # delattr on an alias resets the target's user_override; the target
        # module's hash/get_dict must reflect that reset rather than the stale
        # pre-delete value. Regression for the target dirty-marking bypass.
        with config.patch({"nested.e_bool": False}):
            hash_before = config.get_hash()
            saved_before = config.save_config()
            delattr(config, "e_nested_alias_bool")
            # Target is reset back to its default value, not left at False.
            self.assertTrue(config.nested.e_bool)
            self.assertNotEqual(config.get_hash(), hash_before)
            self.assertNotEqual(config.save_config(), saved_before)

    def test_alias_fields_from_skips_methods(self):
        # alias_fields_from must only alias actual config fields. Functions of
        # the parent must not become config entries.
        self.assertNotIn("alias_child.method_not_a_field", alias_config._config)

    def test_alias_fields_from_rejects_nested_subconfig_class(self):
        # A nested class among the parent's fields cannot be meaningfully
        # aliased (an alias targets a single field, not a subtree). Unlike
        # inherit_fields_from, which could copy the class wholesale and let
        # install_config_module's visit() recurse into it, alias_fields_from
        # must raise here rather than silently dropping the whole subtree.
        with self.assertRaisesRegex(AssertionError, "not_a_field"):

            @alias_fields_from(_alias_parent_with_nested_class)
            class _unused_child:
                pass

    def test_alias_fields_from_value_type_resolution(self):
        # A parent field declared via Config(default=...) with no annotation
        # must resolve to the default's type, not the _Config class. An
        # annotated plain field resolves to its annotation. An annotated
        # Config(default=...) field must resolve to the annotation, not
        # type(default) (which would be NoneType for a str | None field).
        # Checked through the installed module's get_type(), since that (not
        # the decorator's raw output) is what real callers rely on.
        self.assertEqual(alias_config.get_type("alias_child.e_config_int"), int)
        self.assertEqual(alias_config.get_type("alias_child.e_annotated"), str)
        self.assertEqual(
            alias_config.get_type("alias_child.e_annotated_config"), str | None
        )

    def test_alias_fields_from_resolves_and_writes_through(self):
        # Aliased fields resolve to the parent's current value, and writes
        # through either name are observed via the other -- aliasing is not a
        # point-in-time copy.
        self.assertEqual(
            alias_config.alias_child.e_bool, alias_config.alias_parent.e_bool
        )
        with alias_config.patch({"alias_parent.e_config_int": 123}):
            self.assertEqual(alias_config.alias_child.e_config_int, 123)
        # alias_child_override only overrides e_bool, so e_config_int is a
        # sibling alias of alias_child's: writing through one is observed via
        # the parent and the other sibling too.
        with alias_config.patch({"alias_child.e_config_int": 456}):
            self.assertEqual(alias_config.alias_parent.e_config_int, 456)
            self.assertEqual(alias_config.alias_child_override.e_config_int, 456)

    def test_alias_fields_from_shares_mutable_default(self):
        # The docstring promises that for non-scalar defaults (lists, dicts)
        # the child and parent share the same object, so in-place mutation is
        # visible through either name. This matters because __getattr__ only
        # materializes the shared object (deepcopy-of-default into
        # user_override) on the *parent*'s entry -- an alias's own
        # user_override is never read.
        parent_list = alias_config.alias_parent.e_list
        child_list = alias_config.alias_child.e_list
        self.assertIs(parent_list, child_list)
        parent_list.append(99)
        self.assertEqual(alias_config.alias_child.e_list, [1, 2, 99])

    def test_alias_fields_from_child_override_independent(self):
        # A field the child explicitly defines must not be replaced by an
        # alias: it stays a genuinely independent config entry, distinct from
        # the parent's, and is still serialized on its own.
        self.assertIsNone(alias_config._config["alias_child_override.e_bool"].alias)
        with alias_config.patch(
            {"alias_parent.e_bool": False, "alias_child_override.e_bool": True}
        ):
            # Simultaneously different (both flipped from default) proves
            # independence -- if the child's "override" were actually an
            # alias, the second write would clobber the first.
            self.assertFalse(alias_config.alias_parent.e_bool)
            self.assertTrue(alias_config.alias_child_override.e_bool)
        saved = pickle.loads(alias_config.save_config())
        self.assertIn("alias_child_override.e_bool", saved)
        self.assertFalse(saved["alias_child_override.e_bool"])

        # Other (non-overridden) fields are still aliased and track the parent.
        self.assertIsNotNone(
            alias_config._config["alias_child_override.e_config_int"].alias
        )
        with alias_config.patch({"alias_parent.e_config_int": 42}):
            self.assertEqual(alias_config.alias_child_override.e_config_int, 42)

    def test_alias_unresolvable_raises(self):
        # An alias whose module prefix does not exist raises AttributeError.
        entry = _ConfigEntry(
            Config(alias="nonexistent_top_module_xyz.some_field"),
            name="bogus",
        )
        with self.assertRaisesRegex(AttributeError, "does not exist"):
            config._get_alias_module_and_name(entry)

    def test_alias_module_import_error_propagates(self):
        # A genuine ModuleNotFoundError raised from *inside* an existing
        # module's body (e.g. a missing third-party dependency) must
        # propagate rather than be swallowed as "config alias does not
        # exist" -- otherwise a clear "No module named 'x'" turns into a
        # confusing AttributeError far from the actual cause.
        entry = _ConfigEntry(
            Config(
                alias="torch.testing._internal.fake_config_module_missing_dep.e_field"
            ),
            name="bogus",
        )
        with self.assertRaises(ModuleNotFoundError):
            config._get_alias_module_and_name(entry)

    def test_alias_fields_from_qualname_assertion(self):
        # A nested (non-top-level) parent class must be rejected at decoration
        # time rather than producing an unresolvable alias.
        class Outer:
            class Inner:
                e_bool = True

        with self.assertRaisesRegex(AssertionError, "top-level config class"):
            alias_fields_from(Outer.Inner)

    def test_alias_excluded_from_save_and_hash(self):
        # Aliased subconfig fields must not appear in save_config()/
        # codegen_config() as independent keys; only the real target field is
        # serialized, and patching the target (not the alias) is what moves
        # the hash.
        saved = config.save_config()
        restored = pickle.loads(saved)
        self.assertNotIn("nested_alias.e_bool", restored)
        self.assertNotIn("e_nested_alias_bool", restored)

        hash_before = config.get_hash()
        with config.patch({"nested.e_bool": False}):
            self.assertNotEqual(config.get_hash(), hash_before)
            code = config.codegen_config()
            self.assertNotIn("nested_alias.e_bool", code)
            self.assertNotIn("e_nested_alias_bool", code)
            self.assertIn("nested.e_bool = False", code)

    def test_reference_is_default(self):
        t = config.e_dict
        self.assertTrue(config._is_default("e_dict"))
        t["a"] = "b"
        self.assertFalse(config._is_default("e_dict"))

    def test_invalid_config_int(self):
        with self.assertRaises(AssertionError):
            _ConfigEntry(
                Config(default=2, env_name_default="FAKE_DISABLE", value_type=int),
                "test",
            )

    def test_invalid_config_float(self):
        with self.assertRaises(AssertionError):
            _ConfigEntry(
                Config(default=2, env_name_force="FAKE_DISABLE", value_type=float),
                "test",
            )

    def test_deprecated_config(self):
        """Test that deprecated configs warn on access but still work normally."""
        config._config["e_deprecated"]._deprecation_warned = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Read access triggers warning
            self.assertTrue(config.e_deprecated)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, FutureWarning))
            msg = str(w[0].message)
            self.assertIn("e_deprecated", msg)
            self.assertIn("is no longer needed", msg)
            self.assertIn("will be removed in a future version of PyTorch", msg)

        # Write access also triggers warning
        config._config["e_deprecated"]._deprecation_warned = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config.e_deprecated = False
            self.assertEqual(len(w), 1)
            self.assertFalse(config.e_deprecated)

        # Functionality still works: patch, multiple accesses, etc.
        config.e_deprecated = True
        with config.patch(e_deprecated=False):
            self.assertFalse(config.e_deprecated)
        self.assertTrue(config.e_deprecated)

    def test_deprecated_config_warn_once(self):
        """Test that deprecation warning is only issued once per config."""
        config._config["e_deprecated"]._deprecation_warned = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # First access
            _ = config.e_deprecated
            # Second access
            _ = config.e_deprecated
            config.e_deprecated = True  # Write - no warn
            _ = config.e_deprecated  # Read again - no warn
            self.assertEqual(len(w), 1)

    def test_deprecated_alias_config(self):
        """Test that aliased deprecated configs issue their own warning."""
        config._config["e_deprecated_alias"]._deprecation_warned = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = config.e_deprecated_alias
            # Comes from e_not_deprecated = False
            self.assertFalse(value)
            self.assertEqual(len(w), 1)
            self.assertIn("e_deprecated_alias", str(w[0].message))
            self.assertIn("use something else instead", str(w[0].message))

    def test_no_warnings_for_normal_user_code(self):
        """Test that normal user operations don't produce deprecation warnings."""
        import torch

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Simple user code - compile and run a function
            @torch.compile(backend="eager")
            def fn(x):
                return x + 1

            _ = fn(torch.ones(3))

            # No deprecation warnings should be issued from config system
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, FutureWarning)
                and "deprecated" in str(warning.message).lower()
                and "config" in str(warning.message).lower()
            ]
            self.assertEqual(
                len(deprecation_warnings),
                0,
                lambda msg: f"{msg}\nUnexpected config deprecation warnings: {[str(x.message) for x in deprecation_warnings]}",
            )

    def test_patch_then_global(self):
        self.assertTrue(config.e_bool)
        with config.patch(e_bool=False):
            self.assertFalse(config.e_bool)

        config.e_bool = False
        self.assertFalse(config.e_bool)

    def test_is_default_patch(self):
        self.assertTrue(config.e_bool)
        with config.patch(e_bool=False):
            self.assertFalse(config._is_default("e_bool"))

    def test_dict_patch(self):
        self.assertTrue(config.e_bool)
        with config.patch(e_bool=False):
            d = config._get_dict()
            self.assertFalse(d["e_bool"])

    def test_set_in_patch(self):
        self.assertEqual(config.e_int, 1)
        with config.patch(e_int=2):
            self.assertEqual(config.e_int, 2)
            config.e_int = 3
            self.assertEqual(config.e_int, 3)
        # Exiting the patch resets e_int to 1, losing the fact that we explicitly set it to 4 - see ConfigModule._do_setattr
        self.assertEqual(config.e_int, 1)

    def test_nested_patch(self):
        self.assertEqual(config.e_int, 1)
        self.assertEqual(config.e_string, "string")
        with config.patch(e_int=2, e_string="inner"):
            self.assertEqual(config.e_int, 2)
            self.assertEqual(config.e_string, "inner")
            with config.patch(e_int=3):
                self.assertEqual(config.e_int, 3)
                self.assertEqual(config.e_string, "inner")
                config.e_int = 4
                self.assertEqual(config.e_int, 4)
            self.assertEqual(config.e_int, 2)
        self.assertEqual(config.e_int, 1)
        self.assertEqual(config.e_string, "string")

    def test_get_dict_readonly_values_caching(self):
        """readonly_values=True should return cached results on repeated calls."""
        d1 = config._get_dict(readonly_values=True)
        d2 = config._get_dict(readonly_values=True)
        self.assertEqual(d1, d2)

    def test_get_dict_readonly_values_cache_invalidation(self):
        """Mutating a config key should be reflected in subsequent cached calls."""
        d1 = config._get_dict(readonly_values=True)
        self.assertEqual(d1["e_int"], 1)
        config.e_int = 42
        d2 = config._get_dict(readonly_values=True)
        self.assertEqual(d2["e_int"], 42)

    def test_get_dict_readonly_values_does_not_mutate_prior_result(self):
        """Mutating config after a cached call should not affect the prior result."""
        d1 = config._get_dict(readonly_values=True)
        self.assertEqual(d1["e_int"], 1)
        config.e_int = 42
        d2 = config._get_dict(readonly_values=True)
        self.assertEqual(d1["e_int"], 1)
        self.assertEqual(d2["e_int"], 42)

    def test_get_dict_readonly_values_multiple_cache_keys(self):
        """Different filter arguments should produce independent cached results."""
        d_all = config._get_dict(readonly_values=True)
        d_filtered = config._get_dict(ignored_keys=["e_int"], readonly_values=True)
        self.assertIn("e_int", d_all)
        self.assertNotIn("e_int", d_filtered)

        # Mutate and verify both cache entries update correctly
        config.e_string = "changed"
        d_all2 = config._get_dict(readonly_values=True)
        d_filtered2 = config._get_dict(ignored_keys=["e_int"], readonly_values=True)
        self.assertEqual(d_all2["e_string"], "changed")
        self.assertEqual(d_filtered2["e_string"], "changed")

    def test_get_dict_readonly_values_full_dirty_clears_cache(self):
        """Exceeding the dirty keys cap should trigger a full cache rebuild."""
        config._get_dict(readonly_values=True)
        # Mutate more distinct keys than the cap to force fully dirty state.
        all_keys = [k for k in config._config if config._config[k].alias is None]
        self.assertGreater(len(all_keys), config._GET_DICT_DIRTY_KEYS_CAP)
        for key in all_keys[: config._GET_DICT_DIRTY_KEYS_CAP + 1]:
            setattr(config, key, config._config[key].default)
        self.assertIsNone(config._get_dict_dirty_keys_var.get())
        d = config._get_dict(readonly_values=True)
        self.assertIsNotNone(d)

    def test_get_dict_non_readonly_ignores_cache(self):
        """Non-readonly calls should not use or pollute the cache."""
        d1 = config._get_dict(readonly_values=False)
        self.assertIsNone(config._get_dict_cache_var.get())
        config.e_int = 42
        d2 = config._get_dict(readonly_values=False)
        self.assertEqual(d2["e_int"], 42)
        self.assertEqual(d1["e_int"], 1)

    def test_get_dict_cache_not_poisoned_by_caller_mutation(self):
        """Mutating a returned dict must not affect subsequent cached calls."""
        d1 = config._get_dict(readonly_values=True)
        d1["e_int"] = 999
        d2 = config._get_dict(readonly_values=True)
        self.assertEqual(d2["e_int"], 1)

    def test_get_dict_cache_invalidated_by_delattr(self):
        """Deleting a config key should invalidate the cache."""
        config.e_int = 42
        d1 = config._get_dict(readonly_values=True)
        self.assertEqual(d1["e_int"], 42)
        del config.e_int
        d2 = config._get_dict(readonly_values=True)
        # __delattr__ resets user_override, so the default is returned
        self.assertEqual(d2["e_int"], 1)

    def test_get_dict_cache_invalidated_by_patch(self):
        """Config changes via patch() should be reflected in cached results."""
        d1 = config._get_dict(readonly_values=True)
        self.assertEqual(d1["e_int"], 1)
        with config.patch(e_int=42):
            d2 = config._get_dict(readonly_values=True)
            self.assertEqual(d2["e_int"], 42)
        d3 = config._get_dict(readonly_values=True)
        self.assertEqual(d3["e_int"], 1)

    def test_get_dict_cache_invalidated_by_closure_patcher(self):
        """Config changes via _make_closure_patcher should be reflected in cached results."""
        d1 = config._get_dict(readonly_values=True)
        self.assertEqual(d1["e_int"], 1)
        change_fn = config._make_closure_patcher(e_int=42)
        revert = change_fn()
        try:
            d2 = config._get_dict(readonly_values=True)
            self.assertEqual(d2["e_int"], 42)
        finally:
            revert()
        d3 = config._get_dict(readonly_values=True)
        self.assertEqual(d3["e_int"], 1)

    def test_get_hash_per_thread_isolation(self):
        """Each thread should compute its own hash based on its own config values."""
        import threading

        base_hash = config.get_hash()

        thread_hash = [None]
        error = [None]

        def thread_fn():
            try:
                config.e_int = 999
                thread_hash[0] = config.get_hash()
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=thread_fn)
        t.start()
        t.join()
        if error[0] is not None:
            raise error[0]

        # The main thread's hash should be unchanged (e_int still at default)
        self.assertEqual(config.get_hash(), base_hash)
        # The child thread should have gotten a different hash
        self.assertNotEqual(thread_hash[0], base_hash)

    def test_hash_dirty_on_setattr(self):
        """Direct config assignment should mark the hash dirty."""
        config.get_hash()
        self.assertFalse(config._hash_dirty_var.get())
        config.e_int = 42
        self.assertTrue(config._hash_dirty_var.get())

    def test_hash_dirty_on_delattr(self):
        """Deleting a config key should mark the hash dirty."""
        config.get_hash()
        self.assertFalse(config._hash_dirty_var.get())
        del config.e_int
        self.assertTrue(config._hash_dirty_var.get())

    def test_hash_dirty_on_patch(self):
        """Entering and exiting a patch should mark the hash dirty."""
        config.get_hash()
        self.assertFalse(config._hash_dirty_var.get())
        with config.patch(e_int=42):
            self.assertTrue(config._hash_dirty_var.get())
            config.get_hash()
            self.assertFalse(config._hash_dirty_var.get())
        # Exiting the patch restores old values via __setattr__, which marks dirty
        self.assertTrue(config._hash_dirty_var.get())

    def test_hash_dirty_on_closure_patcher(self):
        """_make_closure_patcher should mark the hash dirty on apply and revert."""
        config.get_hash()
        self.assertFalse(config._hash_dirty_var.get())
        change_fn = config._make_closure_patcher(e_int=42)
        revert = change_fn()
        self.assertTrue(config._hash_dirty_var.get())
        config.get_hash()
        self.assertFalse(config._hash_dirty_var.get())
        revert()
        self.assertTrue(config._hash_dirty_var.get())

    def test_hash_dirty_on_load_config(self):
        """Loading a saved config should mark the hash dirty."""
        saved = config.save_config()
        config.get_hash()
        self.assertFalse(config._hash_dirty_var.get())
        config.load_config(saved)
        self.assertTrue(config._hash_dirty_var.get())


if __name__ == "__main__":
    run_tests()
