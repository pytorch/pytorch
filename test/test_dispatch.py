import torch._C as C
from torch.testing._internal.common_utils import TestCase, run_tests

import itertools
import unittest

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
        of in effect registrations from ops).  Results stores Tuple[str, provenance],
        where provenance is a string that describes how exactly we got this
        string.

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
            actual = C._dispatch_dump(
                "{}::{}".format(test_namespace, name)).replace(test_namespace, "test")
            expected, expected_provenance = results.setdefault(
                frozenset(active_ops),
                (actual, actual_provenance)
            )
            self.assertMultiLineEqual(
                expected, actual,
                "expected from {}; actual from {}"
                .format(expected_provenance, actual_provenance)
            )

        results.setdefault(frozenset(), ("", "hardcoded initial state"))
        check_invariants("initial state")
        # In the order specified by ctor_order, run registrations
        set_to_report = frozenset(range(len(ops)))
        for i, op_ix in enumerate(ctor_order):
            refs[op_ix] = C._dispatch_import(test_namespace)
            active_ops.add(op_ix)
            try:
                ops[op_ix](refs[op_ix])
                check_invariants("running ctors {}".format(ctor_order[:i + 1]))
            except RuntimeError as e:
                if not expect_raises:
                    raise
                actual = str(e).replace(test_namespace, "test")
                expected, expected_provenance = results.setdefault(
                    frozenset(active_ops),
                    (actual, "error after running ctors {}".format(ctor_order[:i + 1]))
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

        # Return the "full" state after all operations are run.
        # If this KeyErrors, that means that there did not exist any
        # ordering of ctors which got us to the "end".  That's an
        # error in test construction: it means you could have
        # factored the test into two smaller ones.
        return results[frozenset(range(len(ops)))][0]

    def test_def(self):
        r = self.commute("foo", [
            # m.def("foo(Tensor x) -> Tensor")
            lambda m: m.def_("foo(Tensor x) -> Tensor"),
            # m.impl("test_def", [](const Tensor& x) { return x })
            lambda m: m.impl_t_t("foo"),
            # m.impl("test_def", kAutograd, [](const Tensor& x) { return x })
            lambda m: m.impl_t_t("foo", dispatch="autograd")
        ])
        self.assertExpectedInline(r, '''\
name: test::foo
schema: test::foo(Tensor x) -> (Tensor)
alias analysis kind: FROM_SCHEMA
Autograd: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
catchall: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
''')

    def test_def_impl_schema_mismatch(self):
        # NB: an impl-impl mismatch is not reported eagerly; you'll find out
        # about it because one of them won't match with def
        r = self.commute("foo", [
            # m.def("foo(Tensor x, Tensor y) -> Tensor")
            lambda m: m.def_("foo(Tensor x, Tensor y) -> Tensor"),
            # m.impl("foo", [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo"),
        ], expect_raises=True)
        self.assertExpectedInline(r, '''In registration for test::foo: expected schema of operator to be "test::foo(Tensor x, Tensor y) -> (Tensor)", but got inferred schema "(Tensor _0) -> (Tensor _0)". The number of arguments is different. 2 vs 1.''')  # noqa

    def test_def_with_inference(self):
        r = self.commute("foo", [
            # m.def("foo", [](const Tensor & x) { return x })
            lambda m: m.def_name_t_t("foo"),
            # m.impl("foo", torch::kAutograd, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "autograd")
        ])
        self.assertExpectedInline(r, '''\
name: test::foo
schema: test::foo(Tensor _0) -> (Tensor _0)
alias analysis kind: CONSERVATIVE
Autograd: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
catchall: default_def_name_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
''')

    def test_def_only(self):
        r = self.commute("foo", [
            # m.def("foo(Tensor x, Tensor y) -> Tensor")
            lambda m: m.def_("foo(Tensor x, Tensor y) -> Tensor"),
        ])
        self.assertExpectedInline(r, '''\
name: test::foo
schema: test::foo(Tensor x, Tensor y) -> (Tensor)
alias analysis kind: FROM_SCHEMA
''')

    def test_impl_only(self):
        r = self.commute("foo", [
            # m.impl("foo", [](const Tensor& x) { return x })
            lambda m: m.impl_t_t("foo"),
            # m.impl("foo", torch::kAutograd, [](const Tensor& x) { return x })
            lambda m: m.impl_t_t("foo", "autograd")
        ])
        self.assertExpectedInline(r, '''\
name: test::foo
schema: (none)
Autograd: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
catchall: impl_t_t :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
''')

    # Can't do this yet for BC reasons
    @unittest.expectedFailure
    def test_multiple_def_error(self):
        r = self.commute("foo", [
            # m.def("foo(Tensor x, Tensor y) -> Tensor")
            lambda m: m.def_("foo(Tensor x, Tensor y) -> Tensor"),
            # m.def("foo(Tensor x, Tensor y) -> Tensor")
            lambda m: m.def_("foo(Tensor x, Tensor y) -> Tensor"),
        ], expect_raises=True)
        # TODO: fill in the error message here
        # self.assertExpectedInline(r, '''''')

    def test_def_with_explicit_alias(self):
        r = self.commute("foo", [
            # m.def(torch::schema(
            #   "foo(Tensor x, Tensor y) -> Tensor",
            #   AliasAnalysisKind::PURE))
            lambda m: m.def_("foo(Tensor x, Tensor y) -> Tensor",
                             alias="PURE_FUNCTION")
        ])
        self.assertExpectedInline(r, '''\
name: test::foo
schema: test::foo(Tensor x, Tensor y) -> (Tensor)
alias analysis kind: PURE_FUNCTION
''')

    # TODO: get rid of this test when multiple defs are wrong
    def test_multiple_def_schema_mismatch(self):
        # error message is order dependent
        ops = [
            # m.def("foo(Tensor x, Tensor y) -> Tensor")
            lambda m: m.def_("foo(Tensor x, Tensor y) -> Tensor"),
            # m.def("foo(Tensor x) -> Tensor")
            lambda m: m.def_("foo(Tensor x) -> Tensor"),
        ]
        self.assertExpectedInline(
            self.commute("foo", ops, ctor_order=(0, 1), expect_raises=True),
            '''Tried to register multiple operators with the same name and the same overload name but different schemas: test::foo(Tensor x) -> (Tensor) vs test::foo(Tensor x, Tensor y) -> (Tensor)'''  # noqa
        )
        self.assertExpectedInline(
            self.commute("foo", ops, ctor_order=(1, 0), expect_raises=True),
            '''Tried to register multiple operators with the same name and the same overload name but different schemas: test::foo(Tensor x, Tensor y) -> (Tensor) vs test::foo(Tensor x) -> (Tensor)'''  # noqa
        )

    def test_multiple_def_alias_defaulting(self):
        # TODO: should be an error in both directions soon
        ops = [
            # m.def(torch::schema("foo(Tensor x) -> Tensor",
            #                     c10::AliasAnalysisKind::PURE_FUNCTION))
            lambda m: m.def_("foo(Tensor x) -> Tensor", alias="PURE_FUNCTION"),
            # RegisterOperators().op("foo(Tensor x) -> Tensor")
            lambda m: m.def_legacy("foo(Tensor x) -> Tensor"),
        ]
        self.assertExpectedInline(
            self.commute("foo", ops, ctor_order=(0, 1)),
            '''\
name: test::foo
schema: test::foo(Tensor x) -> (Tensor)
alias analysis kind: PURE_FUNCTION
'''
        )
        # NB: When run with ctor order (1, 0), the destructors are NOT
        # COMMUTATIVE.  THIS IS A BUG, however we are purposely leaving the bug
        # in as it is very benign (only leaves us in a bad state during
        # destruction, when no useful work is being done), will be fixed when we
        # make alias defaulting a hard error, and is very nontrivial to fix
        # prior to that.

    def test_multiple_def_alias_mismatch(self):
        # error message is order dependent
        ops = [
            # m.def(torch::schema("foo(Tensor x) -> Tensor",
            #                     c10::AliasAnalysisKind::PURE_FUNCTION))
            lambda m: m.def_("foo(Tensor x) -> Tensor", alias="PURE_FUNCTION"),
            # m.def(torch::schema("foo(Tensor x) -> Tensor",
            #                     c10::AliasAnalysisKind::CONSERVATIVE))
            lambda m: m.def_("foo(Tensor x) -> Tensor", alias="CONSERVATIVE"),
        ]
        self.assertExpectedInline(
            self.commute("foo", ops, ctor_order=(0, 1), expect_raises=True),
            '''Tried to define the schema for test::foo with different alias analysis kinds: PURE_FUNCTION vs CONSERVATIVE'''  # noqa
        )
        self.assertExpectedInline(
            self.commute("foo", ops, ctor_order=(1, 0), expect_raises=True),
            '''Tried to define the schema for test::foo with different alias analysis kinds: CONSERVATIVE vs PURE_FUNCTION'''  # noqa
        )

    def test_overwrite_catchall(self):
        ops = [
            lambda m: m.impl_t_t("foo", debug="fn1"),
            lambda m: m.impl_t_t("foo", debug="fn2"),
        ]
        # Not commutative
        self.assertExpectedInline(
            self.commute("foo", ops, ctor_order=(0, 1)),
            '''\
name: test::foo
schema: (none)
catchall: fn2 :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
catchall (inactive): fn1 :: (Tensor _0) -> (Tensor _0) [ boxed unboxed ]
'''
        )

if __name__ == '__main__':
    run_tests()
