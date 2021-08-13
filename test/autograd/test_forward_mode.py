# Do not add this to test/run_test.py as this is run with
# test/test_autograd.py.
#
# If you add a TestCase here, import it in test/test_autograd.py.

import torch

from torch.testing._internal.common_utils import TestCase, run_tests
import torch.autograd.forward_ad as fwAD


class TestAutogradForwardMode(TestCase):
    def tearDown(self):
        # Ensure that a failing test won't make others fail
        while fwAD._current_level >= 0:
            fwAD.exit_dual_level()

        super().tearDown()

    def test_forward_level_cleanup(self):
        def get_tensor_and_weak_ref():
            # Create a new Tensor and weak reference
            t = torch.rand(2, requires_grad=True)
            return t, torch._C._WeakTensorRef(t)

        # Sanity check that the helper function works as expected
        t, t_ref = get_tensor_and_weak_ref()
        self.assertFalse(t_ref.expired())

        del t
        self.assertTrue(t_ref.expired())

        # Main test code
        foo = torch.rand(2)

        with fwAD.dual_level():
            tangent, tangent_ref = get_tensor_and_weak_ref()
            self.assertFalse(tangent_ref.expired())

            dual = fwAD.make_dual(foo, tangent)
            self.assertFalse(tangent_ref.expired())

            # Make sure that the tangent we provided has been re-used as is
            self.assertTrue(fwAD.unpack_dual(dual)[1] is tangent)

            # Make sure that dual is keeping the tangent alive
            del tangent
            self.assertFalse(tangent_ref.expired())

            # Make sure that the dual level does not keep the c++
            # version of the tangent alive
            del dual
            self.assertTrue(tangent_ref.expired())

    def test_size_check(self):
        foo = torch.rand(2)
        tangent = torch.rand(3)

        with fwAD.dual_level():
            with self.assertRaisesRegex(RuntimeError, "Trying to set a forward gradient that has a different size"):
                dual = fwAD.make_dual(foo, tangent)

            dual = fwAD.make_dual(foo, tangent[1:])

    # The following test functions want to ensure all the following behaviors:
    #   - Ensure that default level system in the python binding works
    #   - Ensure that only level 0 exists and nesting is properly disabled
    #   - Ensure that printing works fine
    #   - Ensure that basic packing/unpacking works
    #   - Ensure that advanced packing/unpacking works
    #     - For memory / version counter share
    #     - For backward AD (regular ops)
    #   - Ensure that view + inplace for both modes work fine
    #   - Ensure we do proper cleanup on exit of a level

    def test_default_level(self):
        foo = torch.rand(2)
        bar = torch.rand(2)

        with fwAD.dual_level():
            baz = fwAD.make_dual(foo, bar)
            baz_primal, baz_tangent = fwAD.unpack_dual(baz)
        self.assertEqual(baz_primal, foo)
        # We don't actually need to enforce that these two are the exact same python
        # object, feel free to relax in the future
        self.assertIs(baz_tangent, bar)

        baz_primal, baz_tangent = fwAD.unpack_dual(baz)
        self.assertEqual(baz_primal, foo)
        self.assertEqual(baz_tangent, None)

    def test_nested_level(self):
        with fwAD.dual_level() as level:
            # For now only level 0 exists
            self.assertEqual(level, 0)

        with fwAD.dual_level():
            with self.assertRaisesRegex(RuntimeError, "Nested forward mode AD is not supported at the moment"):
                nest_level = fwAD.enter_dual_level()

    def test_print(self):
        with fwAD.dual_level() as level:
            a = torch.rand(3)
            self.assertFalse("tangent=" in str(a))

            b = fwAD.make_dual(a, torch.rand(3))
            self.assertFalse("tangent=" in str(a))
            self.assertTrue("tangent=" in str(b))

            b_primal, b_tangent = fwAD.unpack_dual(b)
            self.assertFalse("tangent=" in str(b_primal))
            self.assertFalse("tangent=" in str(b_tangent))

    def test_basic_packing_unpacking(self):
        foo = torch.rand(2)
        bar = torch.rand(2)

        with fwAD.dual_level():
            baz = fwAD.make_dual(foo, bar)
            baz_primal, baz_tangent = fwAD.unpack_dual(baz)
            self.assertEqual(baz_primal, foo)
            self.assertIs(baz_tangent, bar)

            # Check that packing/unpacking did not change the input
            foo_primal, foo_tangent = fwAD.unpack_dual(foo)
            self.assertEqual(foo_primal, foo)
            self.assertIsNone(foo_tangent)

    def test_advanced_packing_unpacking(self):
        foo = torch.rand(2)
        bar = torch.ones(2)

        # Memory and version counter check
        with fwAD.dual_level():
            dual = fwAD.make_dual(foo, bar)

            # Ensure that they are sharing memory and version counter
            self.assertEqual(dual.storage().data_ptr(), foo.storage().data_ptr())

            # Ensure we properly share the version counter
            self.assertEqual(foo._version, dual._version)
            foo.add_(1)
            self.assertEqual(foo._version, dual._version)

            # Unpacking should only create aliases as well
            dual_primal, dual_tangent = fwAD.unpack_dual(dual)
            self.assertEqual(dual_primal.storage().data_ptr(), foo.storage().data_ptr())
            self.assertEqual(dual_tangent.storage().data_ptr(), bar.storage().data_ptr())
            # And the tangent is actually re-used as-is so it is still the same Tensor
            self.assertIs(dual_tangent, bar)

            # Ensure we properly share the version counter
            self.assertEqual(foo._version, dual_primal._version)
            foo.add_(1)
            self.assertEqual(foo._version, dual_primal._version)
            self.assertEqual(bar._version, dual_tangent._version)
            bar.add_(1)
            self.assertEqual(bar._version, dual_tangent._version)

        # backward mode check
        with fwAD.dual_level():
            foo.requires_grad_()
            bar.requires_grad_()

            # Check that backward gradients properly propagates through packing/unpacking
            dual = fwAD.make_dual(foo, bar)
            p, t = fwAD.unpack_dual(dual)

            gfoo, gbar = torch.autograd.grad(p.sum(), (foo, bar), retain_graph=True, allow_unused=True)
            self.assertEqual(gfoo, torch.ones_like(foo))
            self.assertIsNone(gbar)

            gfoo, gbar = torch.autograd.grad(t.sum(), (foo, bar), retain_graph=True, allow_unused=True)
            self.assertIsNone(gfoo)
            self.assertEqual(gbar, torch.ones_like(bar))

            # Check that forward gradients are impacted by detach()
            detached_dual = dual.detach()
            out = detached_dual * 2
            p, t = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertEqual(p, foo * 2)
            self.assertIsNone(t)

            # Check that forward gradients are not impacted by no_grad
            with torch.no_grad():
                out = dual * 3
            p, t = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertFalse(t.requires_grad)
            self.assertEqual(p, foo * 3)
            self.assertEqual(t, bar * 3)

            # Check that forward gradients are not impacted by inplace detach
            dual = dual.clone()
            dual.detach_()
            out = dual * 2
            p, t = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertEqual(p, foo * 2)
            self.assertIsNone(t)

    def test_view_inplace_non_differentiable_views(self):
        original_foo = torch.rand(2, dtype=torch.double)
        original_bar = torch.ones(2, dtype=torch.double)

        # Do clones to be able to compare the values updated inplace
        # with the original content of these Tensors
        foo = original_foo.clone()
        bar = original_bar.clone()

        with fwAD.dual_level():
            # Note that in this test, we use "update" to mean computing the right tangent for the dual
            # All the inplace operations here are expected to update the primal value of the Tensors but
            # not always their tangents.
            # Also all mentions of "non differentiable view" here means non forward differentiable view
            # unless specified otherwise.
            # See note [Forward Grad View/inplace] for more details on how these views work.

            # Check that inplace ops do not update non-differentiable views
            # Non differentiable view
            dual = fwAD.make_dual(foo, bar)
            dual *= 2
            # Check that non differentiable view's tangent was not updated
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            # Check that the computed result is correct
            self.assertEqual(bar, original_bar * 2)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 2)
            self.assertEqual(foo, original_foo * 2)
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 2)
            # Other non differentiable view
            dual_primal, dual_tangent = fwAD.unpack_dual(dual)
            self.assertIsNone(fwAD.unpack_dual(dual_primal)[1])
            self.assertIsNone(fwAD.unpack_dual(dual_tangent)[1])
            dual_primal *= 2
            # Ensure dual's tangent did not change
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 4)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 2)
            dual_tangent *= 2
            # Ensure dual's primal did not change
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 4)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 4)


    def test_view_inplace_differentiable_views(self):
        original_foo = torch.rand(2)
        original_bar = torch.ones(2)

        # Do clones to be able to compare the values updated inplace
        # with the original content of these Tensors
        foo = original_foo.clone()
        bar = original_bar.clone()

        with fwAD.dual_level():
            # Check that inplace ops do update differentiable view but stop at non differentiable ones
            # A non differentiable view
            dual = fwAD.make_dual(foo, bar)
            # A differentiable view
            view = dual.narrow(0, 0, 1)
            view *= 2
            # Check that non differentiable view was not updated
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            # Check that differentiable view was updated
            self.assertEqual(fwAD.unpack_dual(dual)[1], torch.tensor([2., 1.]))
            self.assertEqual(fwAD.unpack_dual(view)[1], torch.tensor([2.]))

            # Check that we track differentiable view even for Tensors that are not dual
            baz = torch.rand(2)
            baz += dual
            self.assertEqual(fwAD.unpack_dual(baz)[1], fwAD.unpack_dual(dual)[1])
            # Updates on view should as well
            baz = torch.rand(2)
            baz[0] = dual[0]
            self.assertEqual(fwAD.unpack_dual(baz)[1][0], fwAD.unpack_dual(dual)[1][0])
            # Unused values get a gradient of 0
            self.assertEqual(fwAD.unpack_dual(baz)[1][1], 0.)

            # Check that forward non-differentiable views do prevent gradient update
            baz = torch.rand(2)
            view = baz.detach()
            view += dual
            self.assertIsNone(fwAD.unpack_dual(baz)[1])

    def test_grad_cleanup(self):
        foo = torch.rand(2)
        bar = torch.rand(2)
        baz = torch.rand(2)

        with fwAD.dual_level():
            dual = fwAD.make_dual(foo, bar)
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            self.assertIs(fwAD.unpack_dual(dual)[1], bar)

        self.assertIsNone(fwAD.unpack_dual(dual)[1])

        with fwAD.dual_level():
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            new_dual = fwAD.make_dual(foo, baz)

            dual_primal, dual_tangent = fwAD.unpack_dual(dual)
            new_dual_primal, new_dual_tangent = fwAD.unpack_dual(new_dual)
            self.assertEqual(dual_primal, new_dual_primal)
            self.assertIsNone(dual_tangent)
            self.assertEqual(new_dual_tangent, baz)

    def test_detach_view_tracking(self):
        # Default detach is both forward and backward non-differentiable
        foo = torch.rand(2)
        foo_weak = torch._C._WeakTensorRef(foo)

        out = foo.detach()

        del foo
        self.assertTrue(foo_weak.expired())

    def test_out_variant(self):

        with fwAD.dual_level():
            foo = fwAD.make_dual(torch.rand(2), torch.rand(2))
            bar = torch.rand(2)

            with self.assertRaisesRegex(RuntimeError, "out= function"):
                torch.add(bar, bar, out=foo)

            with self.assertRaisesRegex(RuntimeError, "out= function"):
                torch.add(foo, bar, out=bar)


if __name__ == '__main__':
    run_tests()
