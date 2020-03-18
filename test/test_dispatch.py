# import torch
import torch._C as C
from torch.testing._internal.common_utils import TestCase, run_tests

import itertools
import collections
import unittest

# Tests for C++ framework should be run in C++; users will expect tests in C++;
# tests should be close to the location they're testing
#   Greg: This is simply not true about the codebase today
#   Comments make this clearer
# Non-C++ tests don't make usage patterns clear
#   Plan on record: Basic tests in C++, but exhaustive testing should happen here
#   Alternative: use C++ extensions?
#     C++ extension tests are slower
#     Pasting C++ strings together is more complicated and less clear
# Would it be cool if we could register things to dispatcher in Python?
#   Yes! But not in scope right now
# Test-only Python API, versus full API

class TestDispatch(TestCase):
    def test_all_invariants(self):
        C._dispatch_check_all_invariants()

    def run_permutation(self, name, ops, ctor_order, dtor_order,
                        results=None, raises=False):
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

        If raises is True, it is not an error to raise an exception.  Instead,
        we'll store the exception string (instead of the dispatcher state)
        in results.  In principle we should flag these differently, but it's
        very obvious when you get an error in one case but not another.
        """
        if results is None:
            results = {}
        # Refs which retain the c10::Module object so we can explicitly control
        # when each deregistration happens (deregistration occurs when the
        # object gets deallocated).
        refs = [None] * len(ops)
        # Keep track of the set "in effect" registrations
        active_ops = set()
        def check_invariants(actual_provenance):
            C._dispatch_check_invariants(name)
            actual = C._dispatch_dump(name)
            expected, expected_provenance = results.setdefault(
                frozenset(active_ops),
                (actual, actual_provenance)
            )
            self.assertMultiLineEqual(
                expected, actual,
                "expected from {}; actual from {}"
                .format(expected_provenance, actual_provenance)
            )
        check_invariants("initial state")
        # In the order specified by ctor_order, run registrations
        for i, op_ix in enumerate(ctor_order):
            refs[op_ix] = C._dispatch_import()
            active_ops.add(op_ix)
            try:
                ops[op_ix](refs[op_ix])
                check_invariants("running ctors {}".format(ctor_order[:i+1]))
            except RuntimeError as e:
                if not raises:
                    raise
                actual = str(e)
                expected, expected_provenance = results.setdefault(
                    frozenset(active_ops),
                    (actual, "error after running ctors {}".format(ctor_order[:i+1]))
                )
                self.assertMultiLineEqual(expected, actual, expected_provenance)
                active_ops.remove(op_ix)
                # NB: this finally test asserts that if a registrations fails,
                # the dispatcher is left in the same state *that it was before*!
                check_invariants("running ctors {} and then failing to run ctor {} "
                    "(did this failure leave the dispatcher in a wedged state? "
                    "it shouldn't!)"
                    .format(ctor_order[:i], op_ix))
                break
        last_ctor = i
        if raises and len(active_ops) == len(ops):
            self.assertTrue(False,
                "expected exception to be raised, but nothing was raised "
                "(after running ctors {})".format(ctor_order))
        # In the order specified by dtor_order, run deregistrations
        for i, op_ix in enumerate(dtor_order):
            # Trigger a destruction
            refs[op_ix] = None
            # discard not remove, since we may not have actually deregistered
            # anything if there was an error raised
            if raises:
                active_ops.discard(op_ix)
            else:
                active_ops.remove(op_ix)
            check_invariants(
                "running ctors {}, then running dtors {}"
                .format(ctor_order[:last_ctor+1], dtor_order[:i+1])
            )
        return results


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
    def commute(self, name, ops, raises=False):
        results = {}
        for ctor_order in itertools.permutations(range(len(ops))):
            for dtor_order in itertools.permutations(range(len(ops))):
                self.run_permutation(
                    name, ops, ctor_order, dtor_order,
                    results=results, raises=raises)
        # Return the "full" state after all operations are run.
        # If this KeyErrors, that means that there did not exist any
        # ordering of ctors which got us to the "end".
        return results[frozenset(range(len(ops)))][0]

    def test_def(self):
        r = self.commute("test::foo", [
            # m.def("test::foo(Tensor x) -> Tensor")
            lambda m: m.def_("test::foo(Tensor x) -> Tensor"),
            # m.impl("test::foo", [](const Tensor& x) { return x })
            lambda m: m.impl_t_t("test::foo"),
            # m.impl("test::foo",
            #        torch::dispatch_autograd([](const Tensor& x) { return x }))
            lambda m: m.impl_t_t("test::foo", "autograd")
        ])
        self.assertExpectedInline(r, '''\
name: test::foo
schema: test::foo(Tensor x) -> (Tensor)
VariableTensorId: boxed unboxed :: (Tensor _0) -> (Tensor _0)
catchall: boxed unboxed :: (Tensor _0) -> (Tensor _0)
''')

    def test_def_impl_schema_mismatch(self):
        # NB: an impl-impl mismatch is not reported eagerly; you'll find out
        # about it because one of them won't match with def
        r = self.commute("test::foo", [
            # m.def("test::foo(Tensor x, Tensor y) -> Tensor")
            lambda m: m.def_("test::foo(Tensor x, Tensor y) -> Tensor"),
            # m.impl("test::foo", [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("test::foo"),
        ], raises=True)
        self.assertExpectedInline(r, '''In registration for test::foo: expected schema of operator to be "test::foo(Tensor x, Tensor y) -> (Tensor)", but got inferred schema "(Tensor _0) -> (Tensor _0)". The number of arguments is different. 2 vs 1.''')  # noqa

    def test_def_with_inference(self):
        r = self.commute("test::foo", [
            # m.def("test::foo", [](const Tensor & x) { return x })
            lambda m: m.def_t_t("test::foo"),
            # m.impl("test::foo", torch::dispatch_autograd([](const Tensor & x) { return x }))
            lambda m: m.impl_t_t("test::foo", "autograd")
        ])
        self.assertExpectedInline(r, '''\
name: test::foo
schema: test::foo(Tensor _0) -> (Tensor _0)
VariableTensorId: boxed unboxed :: (Tensor _0) -> (Tensor _0)
catchall: boxed unboxed :: (Tensor _0) -> (Tensor _0)
''')

    def test_def_only(self):
        r = self.commute("test::foo", [
            # m.def("test::foo(Tensor x, Tensor y) -> Tensor")
            lambda m: m.def_("test::foo(Tensor x, Tensor y) -> Tensor"),
        ])
        self.assertExpectedInline(r, '''\
name: test::foo
schema: test::foo(Tensor x, Tensor y) -> (Tensor)
''')

    def test_impl_only(self):
        r = self.commute("test::foo", [
            # m.impl("test::foo", [](const Tensor& x) { return x })
            lambda m: m.impl_t_t("test::foo"),
            # m.impl("test::foo",
            #        torch::dispatch_autograd([](const Tensor& x) { return x }))
            lambda m: m.impl_t_t("test::foo", "autograd")
        ])
        self.assertExpectedInline(r, '''\
name: test::foo
schema: (none)
VariableTensorId: boxed unboxed :: (Tensor _0) -> (Tensor _0)
catchall: boxed unboxed :: (Tensor _0) -> (Tensor _0)
''')

    # Can't do this yet for BC reasons
    @unittest.expectedFailure
    def test_multiple_def_error(self):
        r = self.commute("test::foo", [
            # m.def("test::foo(Tensor x, Tensor y) -> Tensor")
            lambda m: m.def_("test::foo(Tensor x, Tensor y) -> Tensor"),
            # m.def("test::foo(Tensor x, Tensor y) -> Tensor")
            lambda m: m.def_("test::foo(Tensor x, Tensor y) -> Tensor"),
        ], raises=True)
        self.assertExpectedInline(r, '''''')

    # Alias analysis kind
    # Overwriting
    #   catchall / dispatch key
    #
    # What if I def a boxed functions with no schema inference?
    #
    # fallback tests
    # xla preautograd tests

if __name__ == '__main__':
    run_tests()
