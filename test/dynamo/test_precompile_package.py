# Owner(s): ["module: dynamo"]

import importlib.machinery
import math
import os
import sys
import sysconfig
import traceback
import types
import xml.parsers.expat  # noqa: F401
from unittest import mock

import numpy

import torch
import torch._dynamo.precompile_package as dynamo_package_lint
import torch._inductor.test_case
import torch.nn.functional as F
from torch._dynamo.guards import CheckFunctionManager, GuardBuilder, strip_local_scope
from torch._dynamo.source import (
    AttrSource,
    DictGetItemSource,
    GetItemSource,
    GlobalSource,
    LocalSource,
)
from torch._dynamo.types import GuardFilterEntry
from torch._guards import Guard
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


def _user_op(x):
    return x + 1


_OWN = GlobalSource(__name__)
_BUILTINS_DICT = GlobalSource("__builtins_dict___0")
_HERE = traceback.StackSummary.from_list([(__file__, 1, "forward", "")])
_ELSEWHERE = traceback.StackSummary.from_list([(F.__file__, 1, "forward", "")])


def _entry(
    source, value, guard_type="ID_MATCH", derived=(), has_value=True, user_stack=None
):
    guard = Guard(source, getattr(GuardBuilder, guard_type))
    guard.user_stack = user_stack
    return GuardFilterEntry(
        name=strip_local_scope(source.name),
        has_value=has_value,
        value=value,
        guard_type=guard_type,
        derived_guard_types=tuple(derived),
        is_global=isinstance(source, GlobalSource),
        orig_guard=guard,
    )


# Names torch or the stdlib own, including the shapes with no file of their own:
# torch._C._nn owns F.gelu, torch.ops carries a relative __file__, pyexpat.errors
# is a stdlib submodule with no location evidence at all.
_LIBRARY_NAMES = (
    "torch",
    "torch._C",
    "torch._C._nn",
    "torch.ops",
    "os.path",
    "collections.abc",
    "sys",
    "zipimport",
    "pyexpat.errors",
    "xml.parsers.expat.model",
)

# Rows: risky?, source, value, _entry keywords. The trusted namespaces are the
# torch, stdlib and own-module globals the test builds through _module_namespaces;
# G['impl'] is an aliased user module and G['config'] a config module, neither
# trusted. Every risky row was once a silent wrong answer on a serving machine.
_RISKY_DROP_CASES = {
    "torch_namespace_read": (False, AttrSource(GlobalSource("F"), "gelu"), F.gelu, {}),
    "stdlib_namespace_read": (False, AttrSource(GlobalSource("math"), "sqrt"), math.sqrt, {}),
    "dynamo_import_alias_read": (False, AttrSource(GlobalSource("__import_torch"), "relu"), torch.relu, {}),
    "trusted_module_itself": (False, GlobalSource("F"), F, {}),
    "own_module_def_read_as_namespace": (False, AttrSource(_OWN, "_user_op"), _user_op, {}),
    "own_module_def_under_another_name": (True, AttrSource(_OWN, "act"), _user_op, {}),
    "aliased_user_module": (True, AttrSource(GlobalSource("impl"), "op"), _user_op, {}),
    "config_module_attribute": (True, AttrSource(GlobalSource("config"), "attn_impl"), _user_op, {}),
    "module_in_attribute": (True, AttrSource(AttrSource(LocalSource("self"), "ns"), "gelu"), F.gelu, {}),
    "instance_attribute": (True, AttrSource(LocalSource("self"), "act"), F.gelu, {}),
    "builtin_in_a_slot": (True, AttrSource(LocalSource("self"), "act"), abs, {}),
    "builtin_read_ordinary": (False, DictGetItemSource(_BUILTINS_DICT, "len"), len, {}),
    "user_code_injected_into_builtins": (True, DictGetItemSource(_BUILTINS_DICT, "op"), _user_op, {}),
    "registry_keyed_by_builtin_name": (True, DictGetItemSource(GlobalSource("_OPS"), "len"), len, {}),
    "dict_lookup": (True, DictGetItemSource(GlobalSource("DISPATCH"), "act"), _user_op, {}),
    "global_bound_to_own_def": (False, GlobalSource("_user_op"), _user_op, {"user_stack": _HERE}),
    "global_bound_to_torch_def": (False, GlobalSource("silu"), F.silu, {}),
    "global_alias_of_a_def": (True, GlobalSource("act"), _user_op, {"user_stack": _HERE}),
    "cross_module_from_import": (True, GlobalSource("_user_op"), _user_op, {"user_stack": _ELSEWHERE}),
    "closure_cell": (True, LocalSource("fn"), _user_op, {}),
    "value_unreadable": (True, AttrSource(_OWN, "x"), None, {"has_value": False}),
    "nested_resume_plumbing": (False, GetItemSource(LocalSource("__nested_frame_values"), 0), _user_op, {}),
}  # fmt: skip


class TestPrecompilePackage(torch._inductor.test_case.TestCase):
    def test_default_guard_filter_drops_the_unserializable_types(self):
        unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
        identity = [_entry(GlobalSource("g"), None, guard_type=t) for t in unsupported]
        self.assertEqual(
            dynamo_package_lint.default_guard_filter_fn(identity),
            [False] * len(unsupported),
        )
        entries = [
            _entry(GlobalSource("g"), None, "TENSOR_MATCH"),
            _entry(GlobalSource("g"), None, "TYPE_MATCH", derived=("NN_MODULE",)),
            # Stricter than serialize_guards, which lets a BUILTIN_MATCH through
            # despite its derived ID_MATCH; the drop is deliberate.
            _entry(GlobalSource("g"), None, "BUILTIN_MATCH", derived=("ID_MATCH",)),
        ]
        self.assertEqual(
            dynamo_package_lint.default_guard_filter_fn(entries), [True, False, False]
        )
        compose = dynamo_package_lint._compose_with_default
        drop_first = compose(lambda es: [False] + [True] * (len(es) - 1))
        self.assertEqual(drop_first(entries), [False, False, False])
        with self.assertRaisesRegex(ValueError, "returned 1 decisions for 3 guards"):
            compose(lambda es: [True])(entries)

    def test_roots_tell_the_stdlib_install_and_torch_dirs_apart(self):
        stdlib = dynamo_package_lint._stdlib_roots()
        install = dynamo_package_lint._install_roots()
        torch_roots = dynamo_package_lint._torch_roots()
        self.assertTrue(stdlib and install and torch_roots)
        # purelib nests inside a stdlib root (conda) or platstdlib (venv), so
        # the two sets must stay distinguishable for the exclusion to work.
        self.assertEqual(set(stdlib) & set(install), set())
        norm, within = dynamo_package_lint._norm, dynamo_package_lint._within
        self.assertTrue(within(norm(os.__file__), stdlib))
        self.assertTrue(within(norm(numpy.__file__), install))
        self.assertIn(norm(os.path.dirname(torch.__file__)), torch_roots)
        root = os.path.join(os.sep, "a", "b")
        self.assertTrue(within(root, (root,)))
        self.assertTrue(within(os.path.join(root, "c"), (root,)))
        self.assertFalse(within(root + "c", (root,)))

    def test_library_module_requires_the_name_to_resolve_to_the_stdlib(self):
        # The risky-drop waiver keys on the OWNER's module name, and a name is
        # not an identity: graphlib, queue, code and distutils are all stdlib
        # names a third party ships, and purelib NESTS inside stdlib (conda) or
        # platstdlib (venv), so a __file__ prefix check waived every shadow.
        self.addCleanup(dynamo_package_lint._classify_file.cache_clear)
        stdlib_root = sysconfig.get_paths()["stdlib"]

        def fake(name, **attrs):
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            return module

        def is_library(name, module):
            with mock.patch.dict(sys.modules, {name: module}):
                return dynamo_package_lint._is_library_module(name)

        site_packages = os.path.join(stdlib_root, "site-packages")
        shadows = {
            "under an install dir": fake("graphlib", __file__=os.path.join(site_packages, "graphlib", "__init__.py")),
            "no __file__, no __spec__": fake("graphlib"),
            "relative __file__": fake("graphlib", __file__="graphlib.py"),
            "namespace package": fake("graphlib", __spec__=importlib.machinery.ModuleSpec("graphlib", None, is_package=True)),
        }  # fmt: skip
        for label, module in shadows.items():
            self.assertFalse(is_library("graphlib", module), label)
        # An install root nested inside the stdlib root with no site-packages
        # component in the path: only the _install_roots exclusion catches it.
        vendored = os.path.join(stdlib_root, "vendored")
        nested = fake(
            "graphlib", __file__=os.path.join(vendored, "graphlib", "__init__.py")
        )
        for install_roots, expected in (
            ((dynamo_package_lint._norm(vendored),), False),
            ((), True),
        ):
            dynamo_package_lint._classify_file.cache_clear()
            with mock.patch.object(
                dynamo_package_lint, "_install_roots", return_value=install_roots
            ):
                self.assertEqual(
                    is_library("graphlib", nested), expected, install_roots
                )
        # A located parent does not vouch for a descendant located elsewhere.
        shadowed_sub = fake(
            "collections.abc", __file__=os.path.join(site_packages, "abc.py")
        )
        self.assertFalse(is_library("collections.abc", shadowed_sub))
        self.assertFalse(dynamo_package_lint._is_library_module("not_a_stdlib_name"))
        self.assertFalse(dynamo_package_lint._is_library_module(None))

    @parametrize("name", _LIBRARY_NAMES)
    def test_library_module_keeps_the_waiver_for_the_real_library(self, name):
        self.assertTrue(
            dynamo_package_lint._is_library_module(name), f"{name} lost its waiver"
        )

    def test_unrepointable_binding_predicates(self):
        reads_a_builtin = dynamo_package_lint._reads_a_builtin
        self.assertTrue(reads_a_builtin(DictGetItemSource(_BUILTINS_DICT, "len"), len))
        # A builtin parked in a slot, a user table keyed by a builtin's name and
        # user code injected into builtins are all reads from a slot.
        self.assertFalse(reads_a_builtin(AttrSource(LocalSource("self"), "act"), abs))
        self.assertFalse(
            reads_a_builtin(DictGetItemSource(GlobalSource("_OPS"), "len"), len)
        )
        self.assertFalse(
            reads_a_builtin(DictGetItemSource(_BUILTINS_DICT, "op"), _user_op)
        )

        synthesized = dynamo_package_lint._is_dynamo_synthesized
        self.assertTrue(
            synthesized(GetItemSource(LocalSource("__nested_frame_values"), 0))
        )
        self.assertTrue(synthesized(LocalSource("__nested_resume_fns")))
        # A global spelled like one is a user binding.
        self.assertFalse(synthesized(GlobalSource("__nested_frame_values")))
        self.assertFalse(synthesized(LocalSource("x")))

        alias_module = dynamo_package_lint._dynamo_alias_module
        self.assertIs(alias_module("__import_torch_dot_nn_dot_functional"), F)
        self.assertIsNone(alias_module("F"))
        owning_module = dynamo_package_lint._owning_module
        self.assertEqual(owning_module(F), "torch.nn.functional")
        self.assertEqual(owning_module(F.gelu), "torch._C._nn")
        self.assertIsNone(owning_module(3))

        defined_where_read = dynamo_package_lint._defined_where_read
        self.assertTrue(defined_where_read(_user_op, _HERE))
        # Paths are compared normalized, so another spelling of the file matches.
        unnormalized = os.path.join(
            os.path.dirname(__file__), os.curdir, os.path.basename(__file__)
        )
        stack = traceback.StackSummary.from_list([(unnormalized, 1, "forward", "")])
        self.assertTrue(defined_where_read(_user_op, stack))
        self.assertFalse(defined_where_read(_user_op, _ELSEWHERE))
        self.assertFalse(defined_where_read(_user_op, None))
        self.assertFalse(defined_where_read(F.silu, _HERE))

    def test_minted_global_names_match_dynamo(self):
        # The predicates restate names Dynamo mints inline, in
        # install_builtins_dict_in_fglobals and import_source; a rename there
        # must fail here rather than silently turn the lint off.
        seen = []

        def record(entries):
            seen.extend(entries)
            return [True] * len(entries)

        lin = torch.nn.Linear(2, 2)

        def fn(x):
            return lin(x) + len(x.shape)

        compiled = torch.compile(
            fn, backend="eager", options={"guard_filter_fn": record}
        )
        compiled(torch.ones(2))
        reads_a_builtin = dynamo_package_lint._reads_a_builtin
        self.assertTrue(
            any(reads_a_builtin(e.orig_guard.originating_source, e.value) for e in seen)
        )
        roots = {
            dynamo_package_lint._source_root(e.orig_guard.originating_source)
            for e in seen
        }
        aliases = {
            r.global_name
            for r in roots
            if isinstance(r, GlobalSource) and r.global_name.startswith("__import_")
        }
        alias = "__import_torch_dot_nn_dot_modules_dot_linear"
        self.assertIn(alias, aliases)
        self.assertIs(
            dynamo_package_lint._dynamo_alias_module(alias), torch.nn.modules.linear
        )

    def test_module_namespaces_trust_only_bindings_config_cannot_repoint(self):
        mypkg = types.ModuleType("mypkg")
        layers = types.ModuleType("mypkg.layers")
        impl_b = types.ModuleType("mypkg.impl_b")
        entries = [
            _entry(GlobalSource("mypkg"), mypkg),  # import mypkg
            _entry(AttrSource(GlobalSource("mypkg"), "layers"), layers),  # import mypkg.layers
            _entry(AttrSource(GlobalSource("mypkg"), "impl"), impl_b),  # from . import impl_b as impl
            _entry(GlobalSource("impl"), impl_b),  # import mypkg.impl_b as impl
            _entry(AttrSource(GlobalSource("other"), "sub"), layers),  # parent never guarded
            _entry(GlobalSource("F"), F),  # import torch.nn.functional as F
            _entry(AttrSource(GlobalSource("torch"), "_dynamo"), torch._dynamo),  # library, parent or not
            _entry(GlobalSource("__import_mypkg_dot_layers"), layers),  # Dynamo's alias for an inlined function's globals
            _entry(AttrSource(GlobalSource("__import_torch"), "Tensor"), torch.Tensor),  # alias without a module-valued guard
            _entry(GlobalSource("config"), torch._dynamo.config),
        ]  # fmt: skip
        with mock.patch.dict(sys.modules, {"mypkg.layers": layers}):
            namespaces = dynamo_package_lint._module_namespaces(entries)
        self.assertEqual(
            set(namespaces),
            {
                "G['mypkg']",
                "G['mypkg'].layers",
                "G['F']",
                "G['torch']._dynamo",
                "G['__import_mypkg_dot_layers']",
                "G['__import_torch']",
            },
        )
        self.assertIs(namespaces["G['__import_torch']"], torch)
        self.assertIs(namespaces["G['mypkg'].layers"], layers)

    @parametrize("shape", sorted(_RISKY_DROP_CASES))
    def test_risky_drop_decision_table(self, shape):
        risky, source, value, kw = _RISKY_DROP_CASES[shape]
        entry = _entry(source, value, **kw)
        modules = [
            (GlobalSource("F"), F),
            (GlobalSource("math"), math),
            (_OWN, sys.modules[__name__]),
            (GlobalSource("impl"), types.ModuleType("mypkg.impl_b")),
            (GlobalSource("config"), torch._dynamo.config),
        ]
        entries = [_entry(s, m) for s, m in modules] + [entry]
        namespaces = dynamo_package_lint._module_namespaces(entries)
        self.assertEqual(dynamo_package_lint._is_risky_drop(entry, namespaces), risky)


instantiate_parametrized_tests(TestPrecompilePackage)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
