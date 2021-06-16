import torch._C as C
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._python_dispatcher import PythonDispatcher

from collections import namedtuple
import itertools
import re

# TODO: Expand the dispatcher API to be a generic API for interfacing with
# the dispatcher from Python!
#
# These are exhaustive tests for commutativity of dispatch behavior.  If you're
# looking for more usage-info style tests, check op_registration_test.cpp
#
# Things not tested here:
#   - Listeners
#   - Top level namespace registrations
#   - Fallback
#   - Exotic overloads of CppFunction/schema
#
# Things not directly tested here:
#   - Internal state of Dispatcher makes sense.  This is indirectly
#     tested by the invariant testing

Result = namedtuple('Result', 'state table provenance')

dispatch_keys_to_check = (
    'Undefined',
    'CPU',
    'CUDA',
    'XLA',
    'AutogradOther',
    'AutogradCPU',
    'AutogradCUDA',
    'AutogradXLA')

def extract_dispatch_table_with_keys(table, dispatch_keys):
    extracted = ''
    table_entries = table.split('\n')
    regex = re.compile(r"registered at .*FallbackKernel\.cpp.*(\[)")
    for k in dispatch_keys:
        for t in table_entries:
            if t.startswith(k):
                # mask out file:line info for in-tree backend fallback
                entry = regex.sub('registered in pytorch framework [', t)
                extracted += (entry + '\n')
    return extracted

class TestDispatch(TestCase):
    namespace_index = 0

    def test_all_invariants(self):
        # Check that the regular stuff is OK!
        C._dispatch_check_all_invariants()

    # You probably don't want to call this directly; if your constructors
    # don't commute, you can still run commute with a fixed ctor_order
    # so that you can test that the destructors still commute
    def run_ops(self, name, ops, ctor_order=None, dtor_order=None,
                results=None, expect_raises=False):
        """
        Given a list of operator registrations, run the registrations in the
        order specified by ctor_order, and then run the deregistrations in
        dtor_order.

        If results is specified, intermediate results are checked for consistency
        with results stored in results (and stored in results if this is the
        first time we've seen them).  Results are expected to be equivalent
        modulo commutativity and inverses (thus, results is keyed on a frozenset
        of in effect registrations from ops).  Results stores namedtuple
        Result[state, table, provenance], where state is a string that contains
        non-derived kernel registered or error message if it doesn't pass;
        table is a string that contains computed dispatch table entries;
        provenance is a string that describes how exactly we got this string.

        If expect_raises is True, it is not an error to raise an exception.  Instead,
        we'll store the exception string (instead of the dispatcher state)
        in results.  In principle we should flag these differently, but it's
        very obvious when you get an error in one case but not another.
        """
        # By allocating every test into a fresh namespace, this makes it less
        # likely that a bug in the testing framework will result in tests
        # interfering with each other
        self.__class__.namespace_index += 1
        if results is None:
            results = {}
        if ctor_order is None:
            ctor_order = list(range(len(ops)))
        if dtor_order is None:
            dtor_order = list(reversed(ctor_order))
        # Refs which retain the c10::Module object so we can explicitly control
        # when each deregistration happens (deregistration occurs when the
        # object gets deallocated).
        refs = [None] * len(ops)
        # Keep track of the set "in effect" registrations
        active_ops = set()

        # double underscore to make it less likely we conflict with something
        # else
        test_namespace = "__test{}__".format(self.namespace_index)

        def check_invariants(actual_provenance):
            C._dispatch_check_invariants(name)
            # Normalize the test namespace so that expected outputs are stable
            actual_state = C._dispatch_dump(
                "{}::{}".format(test_namespace, name)).replace(test_namespace, "test")
            actual_table = C._dispatch_dump_table(
                "{}::{}".format(test_namespace, name)).replace(test_namespace, "test")
            expected_state, expected_table, expected_provenance = results.setdefault(
                frozenset(active_ops),
                Result(actual_state, actual_table, actual_provenance)
            )
            self.assertMultiLineEqual(
                expected_state, actual_state,
                "expected from {}; actual from {}"
                .format(expected_provenance, actual_provenance)
            )
            self.assertMultiLineEqual(
                expected_table, actual_table,
                "expected from {}; actual from {}"
                .format(expected_provenance, actual_provenance)
            )

        results.setdefault(frozenset(), Result("", "", "hardcoded initial state"))
        check_invariants("initial state")
        # In the order specified by ctor_order, run registrations
        set_to_report = frozenset(range(len(ops)))
        for i, op_ix in enumerate(ctor_order):
            # It would be better to DEF here, but because we manage
            # lifetime of multiple registrations with multiple Library
            # references (refs), we can't deal with the strict checking
            # from DEF.
            refs[op_ix] = C._dispatch_library("FRAGMENT", test_namespace, "")
            active_ops.add(op_ix)
            try:
                ops[op_ix](refs[op_ix])
                check_invariants("running ctors {}".format(ctor_order[:i + 1]))
            except RuntimeError as e:
                if not expect_raises:
                    raise
                actual = str(e).replace(test_namespace, "test")
                actual = actual.split("\nException raised from ")[0]
                expected, _, expected_provenance = results.setdefault(
                    frozenset(active_ops),
                    Result(actual, "", "error after running ctors {}".format(ctor_order[:i + 1]))
                )
                self.assertMultiLineEqual(expected, actual, expected_provenance)
                set_to_report = frozenset(active_ops)
                active_ops.remove(op_ix)
                # NB: this finally test asserts that if a registrations fails,
                # the dispatcher is left in the same state *that it was before*!
                check_invariants(
                    "running ctors {} and then failing to run ctor {} "
                    "(did this failure leave the dispatcher in a wedged state? "
                    "it shouldn't!)"
                    .format(ctor_order[:i], op_ix))
                break
        last_ctor = i
        if expect_raises and len(active_ops) == len(ops):
            # Destroy references first, as some test frameworks (like pytest)
            # will retain references in the exception raised by assertTrue! EW!
            refs = None
            self.assertTrue(
                False,
                "expected exception to be raised, but nothing was raised "
                "(after running ctors {})".format(ctor_order))
        # In the order specified by dtor_order, run deregistrations
        for i, op_ix in enumerate(dtor_order):
            # Trigger a destruction
            refs[op_ix] = None
            # discard not remove, since we may not have actually deregistered
            # anything if there was an error raised
            if expect_raises:
                active_ops.discard(op_ix)
            else:
                active_ops.remove(op_ix)
            check_invariants(
                "running ctors {}, then running dtors {}"
                .format(ctor_order[:last_ctor + 1], dtor_order[:i + 1])
            )
        return results[set_to_report][0]

    # Operator registrations are commutative (as static initializers can
    # run in any order) and invertible (by deregistration).  (Subject
    # to some caveats: some legacy behavior in the system are not commutative--
    # we want to get rid of these!)
    #
    # So while in principle we could simply test a set of operations
    # by just running them one by one in the order specified by the user,
    # we can get more assurance about these extra properties by doing
    # more work:
    #
    # 1. Don't run the registrations once in a fixed order: run every possible
    #    permutation.  Similarly, run every permutation of deregistration order.
    #
    # 2. Don't just check the end state of the dispatcher: for every
    #    subset of operator registrations, ensure that the computed
    #    intermediate state is path independent.  One thing to note:
    #    in this function, we assume each operation is unique.  In general,
    #    there may be duplicated registrations, but these are usually
    #    idempotent or legacy.  We test for behavior here separately.
    #
    # NB: checking all permutations means this function is exponential in
    # the length of ops!  So don't pass too many ops to this function!
    def commute(self, name, ops, ctor_order=None, expect_raises=False):
        results = {}

        def go(ctor_order):
            for dtor_order in itertools.permutations(range(len(ops))):
                self.run_ops(
                    name, ops, ctor_order, dtor_order,
                    results=results, expect_raises=expect_raises)

        if ctor_order is not None:
            go(ctor_order)
        else:
            for ctor_order in itertools.permutations(range(len(ops))):
                go(ctor_order)

        # Return the "full" Result namedtuple after all operations are run.
        # If this KeyErrors, that means that there did not exist any
        # ordering of ctors which got us to the "end".  That's an
        # error in test construction: it means you could have
        # factored the test into two smaller ones.
        return results[frozenset(range(len(ops)))]

    def test_def(self):
        state = self.commute("foo", [
            # m.def("foo(Tensor x) -> Tensor")
            lambda m: m.def_("foo(Tensor x) -> Tensor"),
            # m.impl("test_def", [](const Tensor& x) { return x })
            lambda m: m.impl_t_t("foo"),
            # m.impl("test_def", kCPU, [](const Tensor& x) { return x })
            lambda m: m.impl_t_t("foo", dispatch="CPU"),
            # m.impl("test_def", kAutograd, [](const Tensor& x) { return x })
            lambda m: m.impl_t_t("foo", dispatch="Autograd"),
            # m.impl("test_def", kAutogradCPU, [](const Tensor& x) { return x })
            lambda m: m.impl_t_t("foo", dispatch="AutogradCPU")
        ]).state
        self.assertExpectedInline(state, '''\
name: test::foo
schema: test::foo(Tensor x) -> (Tensor)
debug: registered at /dev/null:0
alias analysis kind: FROM_SCHEMA
CPU: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
AutogradCPU: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
Autograd[alias]: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
CompositeImplicitAutograd[alias]: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
''')

    def test_def_impl_schema_mismatch(self):
        # NB: an impl-impl mismatch is not reported eagerly; you'll find out
        # about it because one of them won't match with def
        state = self.commute("foo", [
            # m.def("foo(Tensor x, Tensor y) -> Tensor")
            lambda m: m.def_("foo(Tensor x, Tensor y) -> Tensor"),
            # m.impl("foo", [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo"),
        ], expect_raises=True).state
        self.assertExpectedInline(state, '''\
Inferred operator schema for a C++ kernel function doesn't match the expected function schema.
  operator: test::foo
  expected schema: test::foo(Tensor x, Tensor y) -> (Tensor)
    registered at /dev/null:0
  inferred schema: (Tensor _0) -> (Tensor _0)
    impl_t_t
  reason: The number of arguments is different. 2 vs 1.''')

    def test_def_with_inference(self):
        state = self.commute("foo", [
            # m.def("foo", [](const Tensor & x) { return x })
            lambda m: m.def_name_t_t("foo"),
            # m.impl("foo", torch::kCPU, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "CPU"),
            # m.impl("foo", torch::kAutograd, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "Autograd"),
            # m.impl("foo", torch::kAutogradCPU, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "AutogradCPU")
        ]).state
        self.assertExpectedInline(state, '''\
name: test::foo
schema: test::foo(Tensor _0) -> (Tensor _0)
debug: registered at /dev/null:0
alias analysis kind: CONSERVATIVE
CPU: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
AutogradCPU: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
Autograd[alias]: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
CompositeImplicitAutograd[alias]: default_def_name_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
''')

    def test_def_only(self):
        state = self.commute("foo", [
            # m.def("foo(Tensor x, Tensor y) -> Tensor")
            lambda m: m.def_("foo(Tensor x, Tensor y) -> Tensor"),
        ]).state
        self.assertExpectedInline(state, '''\
name: test::foo
schema: test::foo(Tensor x, Tensor y) -> (Tensor)
debug: registered at /dev/null:0
alias analysis kind: FROM_SCHEMA
''')

    def test_impl_only(self):
        state = self.commute("foo", [
            # m.impl("foo", [](const Tensor& x) { return x })
            lambda m: m.impl_t_t("foo"),
            # m.impl("foo", torch::kCPU, [](const Tensor& x) { return x })
            lambda m: m.impl_t_t("foo", "CPU"),
            # m.impl("foo", torch::kAutograd, [](const Tensor& x) { return x })
            lambda m: m.impl_t_t("foo", "Autograd"),
            # m.impl("foo", torch::kAutogradCPU, [](const Tensor& x) { return x })
            lambda m: m.impl_t_t("foo", "AutogradCPU")
        ]).state
        self.assertExpectedInline(state, '''\
name: test::foo
schema: (none)
CPU: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
AutogradCPU: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
Autograd[alias]: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
CompositeImplicitAutograd[alias]: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
''')

    def test_computed_table(self):
        result = self.commute("foo", [
            # m.def("foo", [](const Tensor & x) { return x })
            lambda m: m.def_name_t_t("foo"),
            # m.impl("foo", torch::kCPU, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "CPU", debug="fn_cpu"),
            # m.impl("foo", torch::kCUDA, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "XLA", debug="fn_xla"),
            # m.impl("foo", torch::kAutograd, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "Autograd", debug="fn_autograd"),
            # m.impl("foo", torch::kAutogradCPU, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "AutogradCPU", debug="fn_autogradcpu")
        ])
        state, table = result.state, result.table
        self.assertExpectedInline(state, '''\
name: test::foo
schema: test::foo(Tensor _0) -> (Tensor _0)
debug: registered at /dev/null:0
alias analysis kind: CONSERVATIVE
CPU: fn_cpu :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
XLA: fn_xla :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
AutogradCPU: fn_autogradcpu :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
Autograd[alias]: fn_autograd :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
CompositeImplicitAutograd[alias]: default_def_name_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
''')

        # computed dispatch table is too big, so we only check on a few entries we're interested in.
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check)

        self.assertExpectedInline(extracted_table, '''\
Undefined: default_def_name_t_t [math kernel]
CPU: fn_cpu [kernel]
CUDA: default_def_name_t_t [math kernel]
XLA: fn_xla [kernel]
AutogradOther: default_def_name_t_t [math kernel]
AutogradCPU: fn_autogradcpu [kernel]
AutogradCUDA: default_def_name_t_t [math kernel]
AutogradXLA: fn_autograd [autograd kernel]
''')

    def test_computed_table_with_cpu_math_autogradcpu_fallthrough(self):
        global_m = C._dispatch_library("IMPL", "_", "AutogradCPU")
        result = self.commute("foo", [
            # m.def("foo", [](const Tensor & x) { return x })
            lambda m: m.def_name_t_t("foo"),
            # m.impl("foo", torch::kCPU, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "CPU"),
        ])
        state, table = result.state, result.table
        self.assertExpectedInline(state, '''\
name: test::foo
schema: test::foo(Tensor _0) -> (Tensor _0)
debug: registered at /dev/null:0
alias analysis kind: CONSERVATIVE
CPU: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
CompositeImplicitAutograd[alias]: default_def_name_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
''')

        # computed dispatch table is too big, so we only check on a few entries we're interested in.
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check)

        self.assertExpectedInline(extracted_table, '''\
Undefined: default_def_name_t_t [math kernel]
CPU: impl_t_t [kernel]
CUDA: default_def_name_t_t [math kernel]
XLA: default_def_name_t_t [math kernel]
AutogradOther: default_def_name_t_t [math kernel]
AutogradCPU: fallthrough registered in pytorch framework [backend fallback]
AutogradCUDA: default_def_name_t_t [math kernel]
AutogradXLA: default_def_name_t_t [math kernel]
''')

    def test_computed_table_with_math(self):
        global_m = C._dispatch_library("IMPL", "_", "AutogradCPU")
        result = self.commute("foo", [
            # m.def("foo(Tensor x) -> Tensor")
            lambda m: m.def_("foo(Tensor x) -> Tensor"),
            # m.impl("foo", torch::kCompositeImplicitAutograd, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "CompositeImplicitAutograd"),
        ])
        state, table = result.state, result.table
        self.assertExpectedInline(state, '''\
name: test::foo
schema: test::foo(Tensor x) -> (Tensor)
debug: registered at /dev/null:0
alias analysis kind: FROM_SCHEMA
CompositeImplicitAutograd[alias]: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
''')

        # computed dispatch table is too big, so we only check on a few entries we're interested in.
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check)

        self.assertExpectedInline(extracted_table, '''\
Undefined: impl_t_t [math kernel]
CPU: impl_t_t [math kernel]
CUDA: impl_t_t [math kernel]
XLA: impl_t_t [math kernel]
AutogradOther: impl_t_t [math kernel]
AutogradCPU: impl_t_t [math kernel]
AutogradCUDA: impl_t_t [math kernel]
AutogradXLA: impl_t_t [math kernel]
''')

    def test_computed_table_with_cpu_math(self):
        global_m = C._dispatch_library("IMPL", "_", "AutogradCPU")
        result = self.commute("foo", [
            # m.def("foo(Tensor x) -> Tensor")
            lambda m: m.def_("foo(Tensor x) -> Tensor"),
            # m.impl("foo", torch::kCPU, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "CPU", debug="fn_cpu"),
            # m.impl("foo", torch::kCompositeImplicitAutograd, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "CompositeImplicitAutograd", debug="fn_math"),
        ])
        state, table = result.state, result.table
        self.assertExpectedInline(state, '''\
name: test::foo
schema: test::foo(Tensor x) -> (Tensor)
debug: registered at /dev/null:0
alias analysis kind: FROM_SCHEMA
CPU: fn_cpu :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
CompositeImplicitAutograd[alias]: fn_math :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
''')

        # computed dispatch table is too big, so we only check on a few entries we're interested in.
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check)

        self.assertExpectedInline(extracted_table, '''\
Undefined: fn_math [math kernel]
CPU: fn_cpu [kernel]
CUDA: fn_math [math kernel]
XLA: fn_math [math kernel]
AutogradOther: fn_math [math kernel]
AutogradCPU: fallthrough registered in pytorch framework [backend fallback]
AutogradCUDA: fn_math [math kernel]
AutogradXLA: fn_math [math kernel]
''')

    def test_computed_table_with_autograd(self):
        global_m = C._dispatch_library("IMPL", "_", "AutogradCPU")
        result = self.commute("foo", [
            # m.def("foo(Tensor x) -> Tensor")
            lambda m: m.def_("foo(Tensor x) -> Tensor"),
            # m.impl("foo", torch::kAutograd, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "Autograd"),
        ])
        state, table = result.state, result.table
        self.assertExpectedInline(state, '''\
name: test::foo
schema: test::foo(Tensor x) -> (Tensor)
debug: registered at /dev/null:0
alias analysis kind: FROM_SCHEMA
Autograd[alias]: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
''')

        # computed dispatch table is too big, so we only check on a few entries we're interested in.
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check)

        self.assertExpectedInline(extracted_table, '''\
AutogradOther: impl_t_t [autograd kernel]
AutogradCPU: impl_t_t [autograd kernel]
AutogradCUDA: impl_t_t [autograd kernel]
AutogradXLA: impl_t_t [autograd kernel]
''')

    # Now that catchAll maps to CompositeImplicitAutograd, registering to both
    # catchAll and CompositeImplicitAutograd breaks commutativity.
    def test_computed_table_with_cpu_autograd_math(self):
        result = self.commute("foo", [
            # m.def("foo(Tensor x) -> Tensor")
            lambda m: m.def_("foo(Tensor x) -> Tensor"),
            # m.impl("foo", torch::kCPU, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "CPU", debug="fn_cpu"),
            # m.impl("foo", torch::kAutograd, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "Autograd", debug="fn_autograd"),
            # m.impl("foo", torch::kCompositeImplicitAutograd, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "CompositeImplicitAutograd", debug="fn_math"),
        ])
        state, table = result.state, result.table
        self.assertExpectedInline(state, '''\
name: test::foo
schema: test::foo(Tensor x) -> (Tensor)
debug: registered at /dev/null:0
alias analysis kind: FROM_SCHEMA
CPU: fn_cpu :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
Autograd[alias]: fn_autograd :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
CompositeImplicitAutograd[alias]: fn_math :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
''')

        # computed dispatch table is too big, so we only check on a few entries we're interested in.
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check)

        self.assertExpectedInline(extracted_table, '''\
Undefined: fn_math [math kernel]
CPU: fn_cpu [kernel]
CUDA: fn_math [math kernel]
XLA: fn_math [math kernel]
AutogradOther: fn_math [math kernel]
AutogradCPU: fn_autograd [autograd kernel]
AutogradCUDA: fn_math [math kernel]
AutogradXLA: fn_math [math kernel]
''')

    def test_computed_table_with_ambiguous_autogradother(self):
        result = self.commute("foo", [
            # m.def("foo(Tensor x) -> Tensor")
            lambda m: m.def_("foo(Tensor x) -> Tensor"),
            # m.impl("foo", torch::kCompositeImplicitAutograd, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "CompositeImplicitAutograd", debug="fn_math"),
            # m.impl("foo", torch::kQuantizedCPU, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "QuantizedCPU", debug="fn_quantizedcpu"),
        ])
        state, table = result.state, result.table
        self.assertExpectedInline(state, '''\
name: test::foo
schema: test::foo(Tensor x) -> (Tensor)
debug: registered at /dev/null:0
alias analysis kind: FROM_SCHEMA
QuantizedCPU: fn_quantizedcpu :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
CompositeImplicitAutograd[alias]: fn_math :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
''')

        # computed dispatch table is too big, so we only check on a few entries we're interested in.
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check + ('QuantizedCPU',))

        self.assertExpectedInline(extracted_table, '''\
Undefined: fn_math [math kernel]
CPU: fn_math [math kernel]
CUDA: fn_math [math kernel]
XLA: fn_math [math kernel]
AutogradOther: ambiguous_autogradother [ambiguous autogradother]
AutogradCPU: fn_math [math kernel]
AutogradCUDA: fn_math [math kernel]
AutogradXLA: fn_math [math kernel]
QuantizedCPU: fn_quantizedcpu [kernel]
''')

    def test_computed_table_with_cpu_defaultbackend(self):
        result = self.commute("foo", [
            # m.def("foo(Tensor x) -> Tensor")
            lambda m: m.def_("foo(Tensor x) -> Tensor"),
            # m.impl("foo", torch::kCPU, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "CPU", debug="fn_cpu"),
            # m.impl("foo", torch::kCompositeExplicitAutograd, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "CompositeExplicitAutograd", debug="fn_defaultbackend"),
        ])
        state, table = result.state, result.table
        self.assertExpectedInline(state, '''\
name: test::foo
schema: test::foo(Tensor x) -> (Tensor)
debug: registered at /dev/null:0
alias analysis kind: FROM_SCHEMA
CPU: fn_cpu :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
CompositeExplicitAutograd[alias]: fn_defaultbackend :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
''')

        # computed dispatch table is too big, so we only check on a few entries we're interested in.
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check)

        self.assertExpectedInline(extracted_table, '''\
Undefined: fn_defaultbackend [default backend kernel]
CPU: fn_cpu [kernel]
CUDA: fn_defaultbackend [default backend kernel]
XLA: fn_defaultbackend [default backend kernel]
AutogradOther: fallthrough registered in pytorch framework [backend fallback]
AutogradCPU: fallthrough registered in pytorch framework [backend fallback]
AutogradCUDA: fallthrough registered in pytorch framework [backend fallback]
AutogradXLA: fallthrough registered in pytorch framework [backend fallback]
''')

    def test_computed_table_with_cpu_autograd_defaultbackend(self):
        result = self.commute("foo", [
            # m.def("foo(Tensor x) -> Tensor")
            lambda m: m.def_("foo(Tensor x) -> Tensor"),
            # m.impl("foo", torch::kCPU, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "CPU", debug="fn_cpu"),
            # m.impl("foo", torch::kAutograd, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "Autograd", debug="fn_autograd"),
            # m.impl("foo", torch::kCompositeExplicitAutograd, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "CompositeExplicitAutograd", debug="fn_defaultbackend"),
        ])
        state, table = result.state, result.table
        self.assertExpectedInline(state, '''\
name: test::foo
schema: test::foo(Tensor x) -> (Tensor)
debug: registered at /dev/null:0
alias analysis kind: FROM_SCHEMA
CPU: fn_cpu :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
Autograd[alias]: fn_autograd :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
CompositeExplicitAutograd[alias]: fn_defaultbackend :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
''')

        # computed dispatch table is too big, so we only check on a few entries we're interested in.
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check + ('QuantizedCPU',))

        self.assertExpectedInline(extracted_table, '''\
Undefined: fn_defaultbackend [default backend kernel]
CPU: fn_cpu [kernel]
CUDA: fn_defaultbackend [default backend kernel]
XLA: fn_defaultbackend [default backend kernel]
AutogradOther: fn_autograd [autograd kernel]
AutogradCPU: fn_autograd [autograd kernel]
AutogradCUDA: fn_autograd [autograd kernel]
AutogradXLA: fn_autograd [autograd kernel]
QuantizedCPU: fn_defaultbackend [default backend kernel]
''')

    def test_computed_table_with_cpu_autograd_math_defaultbackend(self):
        result = self.commute("foo", [
            # m.def("foo(Tensor x) -> Tensor")
            lambda m: m.def_("foo(Tensor x) -> Tensor"),
            # m.impl("foo", torch::kCPU, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "CPU", debug="fn_cpu"),
            # m.impl("foo", torch::kAutograd, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "Autograd", debug="fn_autograd"),
            # m.impl("foo", torch::kCompositeImplicitAutograd, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "CompositeImplicitAutograd", debug="fn_math"),
            # m.impl("foo", torch::kCompositeExplicitAutograd, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "CompositeExplicitAutograd", debug="fn_defaultbackend"),
        ])
        state, table = result.state, result.table
        self.assertExpectedInline(state, '''\
name: test::foo
schema: test::foo(Tensor x) -> (Tensor)
debug: registered at /dev/null:0
alias analysis kind: FROM_SCHEMA
CPU: fn_cpu :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
Autograd[alias]: fn_autograd :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
CompositeImplicitAutograd[alias]: fn_math :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
CompositeExplicitAutograd[alias]: fn_defaultbackend :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
''')

        # computed dispatch table is too big, so we only check on a few entries we're interested in.
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check)

        self.assertExpectedInline(extracted_table, '''\
Undefined: fn_defaultbackend [default backend kernel]
CPU: fn_cpu [kernel]
CUDA: fn_defaultbackend [default backend kernel]
XLA: fn_defaultbackend [default backend kernel]
AutogradOther: fn_autograd [autograd kernel]
AutogradCPU: fn_autograd [autograd kernel]
AutogradCUDA: fn_autograd [autograd kernel]
AutogradXLA: fn_autograd [autograd kernel]
''')

    def test_multiple_def_error(self):
        ops = [
            # m.def("foo(Tensor x, Tensor y) -> Tensor")
            lambda m: m.def_("foo(Tensor x, Tensor y) -> Tensor"),
            # m.def("foo(Tensor x, Tensor y) -> Tensor")
            lambda m: m.def_("foo(Tensor x, Tensor y) -> Tensor"),
        ]
        self.assertExpectedInline(
            self.commute("foo", ops, expect_raises=True).state,
            '''Tried to register an operator (test::foo(Tensor x, Tensor y) -> (Tensor)) with the same name and overload '''
            '''name multiple times. Each overload's schema should only be registered with a single call to def(). '''
            '''Duplicate registration: registered at /dev/null:0. Original registration: registered at /dev/null:0'''
        )

    def test_def_with_explicit_alias(self):
        state = self.commute("foo", [
            # m.def(torch::schema(
            #   "foo(Tensor x, Tensor y) -> Tensor",
            #   AliasAnalysisKind::PURE))
            lambda m: m.def_("foo(Tensor x, Tensor y) -> Tensor",
                             alias="PURE_FUNCTION")
        ]).state
        self.assertExpectedInline(state, '''\
name: test::foo
schema: test::foo(Tensor x, Tensor y) -> (Tensor)
debug: registered at /dev/null:0
alias analysis kind: PURE_FUNCTION
''')

    def test_multiple_def_alias_defaulting(self):
        ops = [
            # m.def(torch::schema("foo(Tensor x) -> Tensor",
            #                     c10::AliasAnalysisKind::PURE_FUNCTION))
            lambda m: m.def_("foo(Tensor x) -> Tensor", alias="PURE_FUNCTION"),
            # RegisterOperators().op("foo(Tensor x) -> Tensor")
            lambda m: m.def_legacy("foo(Tensor x) -> Tensor"),
        ]
        self.assertExpectedInline(
            self.commute("foo", ops, expect_raises=True).state,
            '''Tried to register an operator (test::foo(Tensor x) -> (Tensor)) with the same name and overload '''
            '''name multiple times. Each overload's schema should only be registered with a single call to def(). '''
            '''Duplicate registration: registered at /dev/null:0. Original registration: registered at /dev/null:0'''
        )

    def test_multiple_def_alias_mismatch(self):
        ops = [
            # m.def(torch::schema("foo(Tensor x) -> Tensor",
            #                     c10::AliasAnalysisKind::PURE_FUNCTION))
            lambda m: m.def_("foo(Tensor x) -> Tensor", alias="PURE_FUNCTION"),
            # m.def(torch::schema("foo(Tensor x) -> Tensor",
            #                     c10::AliasAnalysisKind::CONSERVATIVE))
            lambda m: m.def_("foo(Tensor x) -> Tensor", alias="CONSERVATIVE"),
        ]
        self.assertExpectedInline(
            self.commute("foo", ops, expect_raises=True).state,
            '''Tried to register an operator (test::foo(Tensor x) -> (Tensor)) with the same name and overload '''
            '''name multiple times. Each overload's schema should only be registered with a single call to def(). '''
            '''Duplicate registration: registered at /dev/null:0. Original registration: registered at /dev/null:0'''
        )

    def test_multiple_fallback(self):
        global_m = C._dispatch_library("IMPL", "_", "XLA")
        global_m.fallback_fallthrough(),
        try:
            global_m.fallback_fallthrough(),
        except RuntimeError as e:
            self.assertExpectedInline(
                str(e),
                '''Tried to register multiple backend fallbacks for the same dispatch key XLA; previous registration '''
                '''registered at /dev/null:0, new registration registered at /dev/null:0'''
            )
        else:
            self.assertTrue(False)

    def test_overwrite_math(self):
        ops = [
            lambda m: m.impl_t_t("foo", debug="fn1"),
            lambda m: m.impl_t_t("foo", debug="fn2"),
        ]
        # Not commutative
        self.assertExpectedInline(
            self.commute("foo", ops, ctor_order=(0, 1)).state,
            '''\
name: test::foo
schema: (none)
CompositeImplicitAutograd[alias]: fn2 :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
CompositeImplicitAutograd[alias] (inactive): fn1 :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
'''
        )

class TestPythonDispatcher(TestCase):
    def test_basic(self):
        dispatcher = PythonDispatcher()
        dispatcher.register(["CPU", "XLA", "CompositeImplicitAutograd"])
        self.assertExpectedInline(
            dispatcher.dispatchTable(),
            '''\

Computed Dispatch Table
key             kernel
---------------------------
CPU             fn_CPU [kernel]
XLA             fn_XLA [kernel]
QuantizedCPU    fn_CompositeImplicitAutograd [math kernel]
AutogradOther   fn_CompositeImplicitAutograd [math kernel]
AutogradCPU     fallthrough [backend fallback]
AutogradXLA     fallthrough [backend fallback]
'''
        )

    def test_math_autogradcpu(self):
        dispatcher = PythonDispatcher()
        dispatcher.register(["CPU", "XLA", "CompositeImplicitAutograd", "AutogradCPU"])
        self.assertExpectedInline(
            dispatcher.dispatchTable(),
            '''\

Computed Dispatch Table
key             kernel
---------------------------
CPU             fn_CPU [kernel]
XLA             fn_XLA [kernel]
QuantizedCPU    fn_CompositeImplicitAutograd [math kernel]
AutogradOther   fn_CompositeImplicitAutograd [math kernel]
AutogradCPU     fn_AutogradCPU [kernel]
AutogradXLA     fallthrough [backend fallback]
'''
        )
        self.assertExpectedInline(
            dispatcher.registrations(),
            '''\

Registered Kernels
key             kernel
---------------------------
CPU             fn_CPU
XLA             fn_XLA
AutogradCPU     fn_AutogradCPU
CompositeImplicitAutograd[alias] fn_CompositeImplicitAutograd
'''
        )

    def test_defaultbackend_autogradcpu(self):
        dispatcher = PythonDispatcher()
        dispatcher.register(["CPU", "XLA", "CompositeExplicitAutograd", "AutogradCPU"])
        self.assertExpectedInline(
            dispatcher.dispatchTable(),
            '''\

Computed Dispatch Table
key             kernel
---------------------------
CPU             fn_CPU [kernel]
XLA             fn_XLA [kernel]
QuantizedCPU    fn_CompositeExplicitAutograd [default backend kernel]
AutogradOther   fallthrough [backend fallback]
AutogradCPU     fn_AutogradCPU [kernel]
AutogradXLA     fallthrough [backend fallback]
'''
        )

        self.assertExpectedInline(
            dispatcher.registrations(),
            '''\

Registered Kernels
key             kernel
---------------------------
CPU             fn_CPU
XLA             fn_XLA
AutogradCPU     fn_AutogradCPU
CompositeExplicitAutograd[alias] fn_CompositeExplicitAutograd
'''
        )

    def test_autogradother(self):
        dispatcher = PythonDispatcher()
        dispatcher.register(["CPU", "QuantizedCPU", "CompositeImplicitAutograd"])
        self.assertExpectedInline(
            dispatcher.dispatchTable(),
            '''\

Computed Dispatch Table
key             kernel
---------------------------
CPU             fn_CPU [kernel]
XLA             fn_CompositeImplicitAutograd [math kernel]
QuantizedCPU    fn_QuantizedCPU [kernel]
AutogradOther   ambiguous_autogradother [ambiguous autogradother]
AutogradCPU     fallthrough [backend fallback]
AutogradXLA     fn_CompositeImplicitAutograd [math kernel]
'''
        )

        self.assertExpectedInline(
            dispatcher.registrations(),
            '''\

Registered Kernels
key             kernel
---------------------------
CPU             fn_CPU
QuantizedCPU    fn_QuantizedCPU
CompositeImplicitAutograd[alias] fn_CompositeImplicitAutograd
'''
        )

    def test_duplicate_registrations(self):
        dispatcher = PythonDispatcher()

        with self.assertRaisesRegex(RuntimeError, r"Overriden is not allowed"):
            dispatcher.register(["CPU", "CPU"])

    def test_defaultbackend_math(self):
        dispatcher = PythonDispatcher()

        with self.assertRaisesRegex(
                RuntimeError,
                r"Registration to both CompositeImplicitAutograd and CompositeExplicitAutograd is not allowed"):
            dispatcher.register(["CompositeExplicitAutograd", "CompositeImplicitAutograd"])


if __name__ == '__main__':
    run_tests()
