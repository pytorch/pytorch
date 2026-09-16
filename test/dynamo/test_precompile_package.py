# Owner(s): ["module: dynamo"]

import importlib.machinery
import itertools
import math
import os
import subprocess
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

    def test_guard_policy_classification_is_total(self):
        # A guard type in no set is KEPT, so a drop policy can only ever
        # drop what _INVARIANT_DROPPABLE_GUARD_TYPES names. This test is
        # what makes the never-drop claim enforceable:
        # a guard type added to GuardBuilder fails here until someone triages
        # it into exactly one of the four sets.
        from torch._dynamo.guards import GuardBuilder
        from torch._dynamo.precompile_package import (
            _IDENTITY_GUARD_TYPES,
            _INVARIANT_DROPPABLE_GUARD_TYPES,
            _NOOP_GUARD_TYPES,
            _SHAPE_BEARING_GUARD_TYPES,
            _UNMODELLED_GUARD_TYPES,
        )

        # dir() rather than vars(): a guard method added on GuardBuilderBase or
        # a future mixin is a GuardBuilder guard type too.
        guard_types = {
            name
            for name in dir(GuardBuilder)
            if name.isupper() and callable(getattr(GuardBuilder, name))
        }
        sets = {
            "_SHAPE_BEARING_GUARD_TYPES": _SHAPE_BEARING_GUARD_TYPES,
            "_UNMODELLED_GUARD_TYPES": _UNMODELLED_GUARD_TYPES,
            "_INVARIANT_DROPPABLE_GUARD_TYPES": _INVARIANT_DROPPABLE_GUARD_TYPES,
            "_NOOP_GUARD_TYPES": _NOOP_GUARD_TYPES,
        }
        classified: frozenset[str] = frozenset().union(*sets.values())
        self.assertEqual(
            sorted(guard_types - classified),
            [],
            "unclassified GuardBuilder guard type(s): add each to exactly one "
            "policy set in torch/_dynamo/precompile_package.py (KEPT until then)",
        )
        self.assertEqual(
            sorted(classified - guard_types),
            [],
            "phantom entries: no GuardBuilder method by these names",
        )
        for (a_name, a), (b_name, b) in itertools.combinations(sets.items(), 2):
            self.assertEqual(sorted(a & b), [], f"{a_name} overlaps {b_name}")
        # The identity guards the default filter drops are droppable by
        # construction; a literal rewrite of the set must not lose that.
        self.assertTrue(_IDENTITY_GUARD_TYPES <= _INVARIANT_DROPPABLE_GUARD_TYPES)

    def test_noop_guard_type_follows_the_hook_guard_config(self):
        # EMPTY_NN_MODULE_HOOKS_DICT emits nothing under the default config and
        # a SEQUENCE_LENGTH on the hook dicts otherwise, so whether a report may
        # treat it as a marker depends on the config the frame compiled under.
        from torch._dynamo.precompile_package import _is_noop_guard_type

        self.assertTrue(_is_noop_guard_type("GRAD_MODE"))
        self.assertFalse(_is_noop_guard_type("TENSOR_MATCH"))
        self.assertTrue(_is_noop_guard_type("EMPTY_NN_MODULE_HOOKS_DICT"))
        with torch._dynamo.config.patch(skip_nnmodule_hook_guards=False):
            self.assertFalse(_is_noop_guard_type("EMPTY_NN_MODULE_HOOKS_DICT"))
            self.assertTrue(_is_noop_guard_type("GRAD_MODE"))

    def test_normalize_scrubs_addresses_and_counters_but_not_user_constants(self):
        # Both directions matter: anything run-varying that survives makes the
        # committed report churn, and anything meaningful that is erased makes
        # two variants guarding different values render one fact.
        from torch._dynamo.precompile_package import _normalize

        cases = {
            "___check_obj_id(G['fn'], 140311678493200), type=<class 'function'>": "___check_obj_id(G['fn'], <id>), type=<class 'function'>",
            "G['__builtins_dict___6']['len']": "G['__builtins_dict___<n>']['len']",
            "G['__import_mod_140311678493200_c1']": "G['__import_mod_<id>_c<n>']",
            "G['___unnamed_scope_140311678493200_c1']": "G['___unnamed_scope_<id>_c<n>']",
            "G['_140311678493200_c3'] is not None": "G['_<id>_c<n>'] is not None",
            "top_saved_tensors_hooks ids == (139, 140)": "top_saved_tensors_hooks ids == (<ids>)",
            # User constants and identifiers are not addresses.
            "L['dims'][0] == 140311678493200": "L['dims'][0] == 140311678493200",
            "L['w_1_c2'] == 3": "L['w_1_c2'] == 3",
            "len(L['xs']) == 6": "len(L['xs']) == 6",
        }
        for text, expected in cases.items():
            self.assertEqual(_normalize(text), expected, text)

    def test_code_fingerprint_recurses_into_container_and_nested_consts(self):
        # _code_fingerprint names a callable by its body so an ACT2FN-style table
        # can be told apart. Two lambdas can differ ONLY inside a constant the
        # outer co_code does not distinguish: a tuple, a frozenset, or a nested
        # code object. Filtering those out whole -- rather than recursing -- gives
        # both the same digest, _object_identity names them identically, and the
        # guard that split the two compilations is reported as an invariant of
        # each.
        from torch._dynamo.precompile_package import _code_fingerprint, _stable_consts

        pairs = {
            "tuple const": (lambda x: x * (1, 2), lambda x: x * (1, 3)),
            "frozenset const": (lambda x: x in {1, 2}, lambda x: x in {1, 3}),
            # Not called: what matters is the nested code object in co_consts.
            "nested code": (lambda x: (lambda y: y + 1), lambda x: (lambda y: y + 2)),
            # A subscript with Ellipsis folds to ONE const tuple at one index, so
            # a const type outside the stable set must keep its slot.
            "ellipsis const": (lambda x: x[..., 0], lambda x: x[0, ...]),
        }
        for label, (left, right) in pairs.items():
            self.assertEqual(
                left.__code__.co_code,
                right.__code__.co_code,
                f"{label}: the pair must differ only in co_consts",
            )
            self.assertNotEqual(
                _code_fingerprint(left.__code__),
                _code_fingerprint(right.__code__),
                f"{label}: two different bodies share a fingerprint",
            )
        # An unrenderable const keeps its position as a type marker.
        self.assertEqual(_stable_consts((object(), 1)), ("<object>", 1))
        # And the digest is a function of the body, not of the code object:
        # the same source compiled twice must agree.
        src = "lambda x: (x * 2, 'a', (lambda y: y + 1))"
        self.assertEqual(
            _code_fingerprint(compile(src, "<a>", "eval")),
            _code_fingerprint(compile(src, "<b>", "eval")),
        )

    def test_code_fingerprint_is_stable_across_processes(self):
        # The digest goes into a file meant to be committed and diffed, so it
        # has to agree in a fresh interpreter; within one process any two calls
        # trivially agree, which is why the positive case above cannot catch a
        # regression that puts an address back (repr of an arbitrary const).
        from torch._dynamo.precompile_package import _code_fingerprint

        src = "lambda x: (x * 2, 'a', (lambda y: y + 1), x in {1, 2})"
        probe = (
            "from torch._dynamo.precompile_package import _code_fingerprint;"
            f"print(_code_fingerprint(compile({src!r}, '<p>', 'eval')))"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        self.assertEqual(
            out.stdout.strip(), _code_fingerprint(compile(src, "<p>", "eval"))
        )

    def test_object_identity_puts_the_digest_before_the_truncation_point(self):
        from torch._dynamo.precompile_package import _code_fingerprint, _object_identity

        self.assertEqual(
            _object_identity(torch.nn.functional), "is module torch.nn.functional"
        )
        self.assertEqual(_object_identity(object()), "is a builtins.object")

        def fn():
            pass

        # A qualname that alone exceeds the 160-character bound: the site and
        # digest must survive the cut and the qualname tail is what goes.
        fn.__qualname__ = "Outer." * 40 + "fn"
        rendered = _object_identity(fn)
        code = fn.__code__
        prefix = f"is @{os.path.basename(code.co_filename)}:{code.co_firstlineno}#{_code_fingerprint(code)} "
        self.assertEqual(len(rendered), 160)
        self.assertTrue(rendered.startswith(prefix), rendered)
        self.assertNotIn("#", rendered[len(prefix) :])

    def test_value_fingerprint_dispatches_on_the_guard_type(self):
        from torch.compiler._precompile_types import GuardFact

        fingerprint = dynamo_package_lint._value_fingerprint
        src = LocalSource("x")
        x = torch.zeros(2, 3)
        with torch.inference_mode():
            inference = torch.zeros(2, 3)
        variants = (x, x.double(), x[:, :2], torch.nn.Parameter(x), inference)
        rendered = [fingerprint(_entry(src, v, "TENSOR_MATCH")) for v in variants]
        self.assertEqual(len(set(rendered)), len(variants), rendered)
        for line in rendered:
            self.assertTrue(line.startswith("check_tensor(<value>, "), line)
        # The guard type decides, not the value's type: NOT_NONE_MATCH is what
        # Dynamo installs on an optimizer's .grad, and it checks only presence.
        for guard_type in ("NOT_NONE_MATCH", "TYPE_MATCH", "COW_TENSOR_MATCH"):
            self.assertEqual(fingerprint(_entry(src, x, guard_type)), "")
        self.assertEqual(fingerprint(_entry(LocalSource("n"), 1, "TYPE_MATCH")), "")
        self.assertEqual(
            fingerprint(_entry(_OWN, _user_op, "ID_MATCH")),
            dynamo_package_lint._object_identity(_user_op),
        )
        grad_mode = _entry(src, None, "GRAD_MODE", has_value=False)
        with torch.no_grad():
            self.assertEqual(fingerprint(grad_mode), "grad_enabled=False")

        class Opaque(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                raise RuntimeError("no attribute reads")

        opaque = torch.zeros(2).as_subclass(Opaque)
        unrenderable = fingerprint(_entry(src, opaque, "TENSOR_MATCH"))
        self.assertEqual(unrenderable, "type=Opaque, <unrenderable>")
        # Once the boilerplate parts are filtered a TENSOR_MATCH renders no code,
        # so the value is what keeps two shape specializations in a fixed order.
        facts = [GuardFact("TENSOR_MATCH", "L['x']", (), v, True) for v in rendered]
        ordered = sorted(facts, key=dynamo_package_lint._fact_order)
        self.assertEqual([f.value for f in ordered], sorted(rendered))

    def test_saved_hooks_fingerprint_mirrors_what_the_guard_stores(self):
        fingerprint = dynamo_package_lint._saved_hooks_fingerprint
        self.assertEqual(fingerprint(), "hooks=None")
        # The guard stores None for hooks it cannot inline, so plain-Python
        # hooks are one value to it and must be one value here.
        with torch.autograd.graph.saved_tensors_hooks(_user_op, _user_op):
            self.assertEqual(fingerprint(), "hooks=None")

        def identity(x):
            return x

        pack = torch.fx.symbolic_trace(identity)
        unpack = torch.fx.symbolic_trace(identity)
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            rendered = fingerprint()
        # Named by rendered graph, never by address: two GraphModules with one
        # code read the same here although the guard compares their ids.
        digest = dynamo_package_lint._hash_text(pack.code)
        self.assertEqual(rendered, f"hooks=({digest}, {digest})")

    def test_wont_generalize_cancels_pins_only_within_a_frame(self):
        from torch._dynamo.precompile_package import (
            _pins_a_value,
            _SHAPE_BEARING_GUARD_TYPES,
            _VALUE_EQUALITY_GUARD_TYPES,
            _wont_generalize,
        )
        from torch.compiler._precompile_types import GuardFact

        # A value pin is never policy-droppable, so a new value-pinning guard
        # type has to be triaged into the shape-bearing set to land here.
        self.assertTrue(_VALUE_EQUALITY_GUARD_TYPES <= _SHAPE_BEARING_GUARD_TYPES)
        self.assertTrue(_pins_a_value("EQUALS_MATCH", "scale"))
        self.assertTrue(_pins_a_value("CONSTANT_MATCH", "___stack0"))
        self.assertTrue(_pins_a_value("CONSTANT_SUBCLASS_MATCH", "n"))
        # Reached THROUGH an argument, or a container element: not counted.
        self.assertFalse(_pins_a_value("CONSTANT_MATCH", "self.eps"))
        self.assertFalse(_pins_a_value("EQUALS_MATCH", "dims[0]"))
        self.assertFalse(_pins_a_value("EQUALS_MATCH", "G['CFG'].width"))
        self.assertFalse(_pins_a_value("TENSOR_MATCH", "x"))
        self.assertFalse(_pins_a_value("SEQUENCE_LENGTH", "xs"))

        def fact(guard_type, source):
            return GuardFact(guard_type, source, (), "", True)

        entry = ("step", "m.py", 1)
        resume_a = ("torch_dynamo_resume_in_step_at_7", "m.py", 7)
        resume_b = ("torch_dynamo_resume_in_step_at_9", "m.py", 9)
        kept = {
            ("EQUALS_MATCH", "scale"),
            ("EQUALS_MATCH", "mode"),
            ("CONSTANT_MATCH", "___stack0"),
            ("TENSOR_MATCH", "x"),
        }
        pinned_scale = fact("EQUALS_MATCH", "scale")
        pinned_mode = fact("EQUALS_MATCH", "mode")
        generic_scale = fact("TYPE_MATCH", "scale")
        x = fact("TENSOR_MATCH", "x")
        guard_sets = {
            # Two variants of the entry: one pins scale and mode, the other
            # serves scale generically -- the ordinary shape once two examples
            # are captured -- so only mode stays pinned.
            entry: [
                frozenset({pinned_scale, pinned_mode, x}),
                frozenset({generic_scale, pinned_mode, x}),
            ],
            # ___stack0 is a tensor in this resume frame ...
            resume_a: [frozenset({fact("TENSOR_MATCH", "___stack0")})],
            # ... and the .item() int in this one. The tensor elsewhere is a
            # different local under the same bare name and must not cancel it.
            resume_b: [frozenset({fact("CONSTANT_MATCH", "___stack0")})],
        }
        self.assertEqual(_wont_generalize(kept, guard_sets), ("___stack0", "mode"))
        # Nothing pinned: nothing to report, whatever the frames say.
        self.assertEqual(_wont_generalize({("TENSOR_MATCH", "x")}, guard_sets), ())


instantiate_parametrized_tests(TestPrecompilePackage)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
