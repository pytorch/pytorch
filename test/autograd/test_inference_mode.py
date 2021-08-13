# Do not add this to test/run_test.py as this is run with
# test/test_autograd.py.
#
# If you add a TestCase here, import it in test/test_autograd.py.

import torch

from torch.testing._internal.common_utils import TestCase, run_tests


class TestAutogradInferenceMode(TestCase):
    def _is_inference_tensor(self, tensor):
        try:
            err_msg = "Inference tensors do not track version counter"
            with self.assertRaisesRegex(RuntimeError, err_msg):
                tensor._version
            return True
        except AssertionError as e:
            return False

    def test_inference_mode_context_manager(self):
        self.assertFalse(torch.is_inference_mode_enabled())
        with torch.inference_mode():
            self.assertTrue(torch.is_inference_mode_enabled())
            with torch.inference_mode(False):
                self.assertFalse(torch.is_inference_mode_enabled())
            self.assertTrue(torch.is_inference_mode_enabled())
        self.assertFalse(torch.is_inference_mode_enabled())

    def test_inference_mode_decorator(self):
        @torch.inference_mode()
        def func(x):
            self.assertTrue(torch.is_inference_mode_enabled())
            return x * x

        for requires_grad in (True, False):
            c = torch.ones(1, 2, 3, requires_grad=requires_grad)
            d = func(c)
            self.assertTrue(torch.is_inference(d))
            self.assertFalse(d.requires_grad)

    def test_inference_mode_tensor_creation(self):
        with torch.inference_mode():
            # new tensors created through constructors are inference tensors
            c = torch.ones(1, 2, 3)
            self.assertFalse(c.requires_grad)
            self.assertTrue(torch.is_inference(c))

            # requires_grad doesn't change inference tensor behavior in InferenceMode
            tmp = torch.ones(1, 2, 3, requires_grad=True)
            self.assertTrue(tmp.requires_grad)
            self.assertTrue(torch.is_inference(tmp))

            tmp = torch.ones(1, 2, 3).requires_grad_(False)
            self.assertFalse(tmp.requires_grad)
            self.assertTrue(torch.is_inference(tmp))

    def test_inference_mode_existing_autograd_session(self):
        s = torch.ones(1, 2, 3, requires_grad=True)
        a = s.clone()

        # `a` gets saved outside of inference mode
        out = a * a
        with torch.inference_mode():
            a.add_(2)

        self.assertFalse(torch.is_inference(a))
        # tensors created outside of inference mode aren't
        # inference tensors, so they will still have their
        # version counters tracked
        err_msg = ("one of the variables needed for gradient computation has been "
                   "modified by an inplace operation")
        with self.assertRaisesRegex(RuntimeError, err_msg):
            out.backward(torch.ones_like(out))

    def test_inference_mode_inf_tensor_in_inf_mode_functional_op(self):
        def functional_op(x):
            return x * x

        with torch.inference_mode():
            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                # performing a non-view operation produces a inference tensor
                # that does not require grad
                func_out = functional_op(c)
                self.assertTrue(torch.is_inference(func_out))
                self.assertFalse(func_out.requires_grad)

    def test_inference_mode_inf_tensor_in_inf_mode_inplace_op(self):
        @torch.inference_mode()
        def run_test(fn):
            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                # after performing inplace operation, tensor is still
                # an inference tensor
                fn(c)
                self.assertTrue(torch.is_inference(c))
                self.assertEqual(c.requires_grad, requires_grad)
        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))

    def test_inference_mode_inf_tensor_in_inf_mode_view_op(self):
        with torch.inference_mode():
            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                # perform view operation produces inference tensor
                # that does not require grad
                view_out = c.view(-1)
                self.assertTrue(torch.is_inference(view_out))
                self.assertFalse(view_out.requires_grad)

    def test_inference_mode_inf_tensor_in_normal_mode_functional_op(self):
        def functional_op(x):
            return x * x

        for requires_grad in (True, False):
            with torch.inference_mode():
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

        func_out = functional_op(c)
        self.assertFalse(torch.is_inference(func_out))
        self.assertFalse(func_out.requires_grad)
        self.assertTrue(func_out.is_leaf)

    def test_inference_mode_inf_tensor_in_normal_mode_inplace_op(self):
        def run_test(fn):
            for requires_grad in (False, True):
                with torch.inference_mode():
                    c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                if requires_grad:
                    # leaf variable that requires grad is being used in an inplace
                    # operation when requires_grad=True
                    pass
                else:
                    err_msg = "Inplace update to inference tensor outside InferenceMode"
                    with self.assertRaisesRegex(RuntimeError, err_msg):
                        fn(c)
        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))

    def test_inference_mode_inf_tensor_in_normal_mode_view_op(self):
        for requires_grad in (True, False):
            with torch.inference_mode():
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

            out = c.view(-1)
            self.assertTrue(torch.is_inference(out))
            self.assertFalse(out.requires_grad)
            self.assertFalse(out._is_view())
            self.assertTrue(out.is_leaf)

    def test_normal_tensor_inplace_output_in_inference_mode(self):
        def run_test(fn):
            for requires_grad in (True, False):
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                a = s.clone()

                with torch.inference_mode():
                    fn(a)
                    self.assertFalse(torch.is_inference(a))
                    self.assertEqual(a.requires_grad, requires_grad)

                    # inplace -> inplace
                    fn(a)
                    self.assertFalse(torch.is_inference(a))
                    self.assertEqual(a.requires_grad, requires_grad)

                    # inplace -> inplace -> view
                    view_out = a.view(-1)
                    self.assertFalse(torch.is_inference(view_out))
                    self.assertEqual(view_out.requires_grad, requires_grad)
        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))

    def test_normal_tensor_inplace_output_in_normal_mode(self):
        def run_test(fn):
            for requires_grad in (True, False):
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                a = s.clone()

                with torch.inference_mode():
                    fn(a)
                    self.assertFalse(torch.is_inference(a))
                    self.assertEqual(a.requires_grad, requires_grad)

                fn(a)
                self.assertFalse(torch.is_inference(a))
                self.assertEqual(a.requires_grad, requires_grad)

                # inplace -> inplace
                fn(a)
                self.assertFalse(torch.is_inference(a))
                self.assertEqual(a.requires_grad, requires_grad)

                # inplace -> inplace -> view
                view_out = a.view(-1)
                self.assertFalse(torch.is_inference(view_out))
                self.assertEqual(view_out.requires_grad, requires_grad)
            run_test(lambda x: x.add_(2))
            run_test(lambda x: x.transpose_(0, 1))

    def test_normal_tensor_view_output_in_inference_mode(self):
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()

            with torch.inference_mode():
                out = a.view(-1)
                self.assertFalse(torch.is_inference(out))
                self.assertEqual(out.requires_grad, requires_grad)
                self.assertTrue(out._is_view())

                # view -> view
                tmp = out.view(-1)
                self.assertFalse(torch.is_inference(tmp))
                self.assertEqual(tmp.requires_grad, requires_grad)
                self.assertTrue(tmp._is_view())
                self.assertTrue(tmp.is_leaf)

                # view -> view -> inplace
                self.assertTrue(torch.is_inference_mode_enabled())
                tmp.add_(2)
                self.assertFalse(torch.is_inference(tmp))
                self.assertEqual(tmp.requires_grad, requires_grad)
                # Accessing is_leaf in python tries to update grad_fn and raises:
                # A view was created in inference mode and its base or
                # another view of its base has been modified inplace in normal mode
                # tmp.is_leaf
                self.assertEqual(a._version, tmp._version)

    def test_normal_tensor_view_output_in_normal_mode(self):
        def functional_op(x):
            return x * x

        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()

            with torch.inference_mode():
                out = a.view(-1)
                self.assertFalse(torch.is_inference(out))
                self.assertEqual(out.requires_grad, requires_grad)
                self.assertTrue(out._is_view())
                self.assertTrue(out.is_leaf)

            tmp = functional_op(out)
            self.assertFalse(torch.is_inference(tmp))
            self.assertEqual(tmp.requires_grad, requires_grad)

            if requires_grad:
                err_msg = "A view was created in inference mode and is being modified inplace"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    out.add_(2)
                pass
            else:
                out.add_(2)

            tmp = out.view(2, 3)
            self.assertFalse(torch.is_inference(tmp))
            self.assertEqual(tmp.requires_grad, requires_grad)

    def test_mix_inference_and_normal_tensor_functional_op(self):
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)

            with torch.inference_mode():
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

            # add is safe since it doesn't save any variable for backward
            out = c.add(s)
            self.assertFalse(torch.is_inference(out))
            self.assertEqual(out.requires_grad, requires_grad)
            if requires_grad:
                # leaf inference tensor with requires_grad=True can still have gradient
                out.backward(torch.ones_like(out))
                self.assertEqual(c.grad, torch.ones_like(c))

            if requires_grad:
                err_msg = "Inference tensors cannot be saved for backward"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    c * s

                # inference tensor in TensorList input
                inputs = [s, c]
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    torch.stack(inputs)


    def test_mix_inference_and_normal_tensor_inplace_op(self):
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()

            with torch.inference_mode():
                c = torch.ones(1, 2, 3)

            self.assertTrue(torch.is_inference(c))
            if requires_grad:
                err_msg = "Inference tensors cannot be saved for backward"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    a.mul_(c)

                # inference tensor in TensorList input
                err_msg = ("out=... arguments don't support automatic differentiation, "
                           "but one of the arguments requires grad")
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    torch.mul(s, s, out=c)
            else:
                a.mul_(c)
                err_msg = "Inplace update to inference tensor outside InferenceMode is not allowed"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    torch.mul(s, s, out=c)

    def test_mix_inference_and_normal_tensor_view_op(self):
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)

            with torch.inference_mode():
                c = torch.ones(1, 2, 3)

            # view_as is a composite op which calls view with only one
            # tensor argument. So there isn't a mixed inference and normal
            # tensor inputs for view ops
            tmp1 = c.view_as(s)
            self.assertTrue(torch.is_inference(tmp1))
            self.assertFalse(tmp1.requires_grad)

            # this is fine since its equivalent as s.view(c.sizes()) which
            # isn't a mixed input scenario
            tmp2 = s.view_as(c)
            self.assertFalse(torch.is_inference(tmp2))
            self.assertEqual(tmp2.requires_grad, requires_grad)

    def test_inference_mode_handle_direct_view_on_rebase(self):
        def run_test(fn):
            for requires_grad in (True, False):
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                a = s.clone()

                with torch.inference_mode():
                    view_out = a.view_as(a)

                if requires_grad:
                    err_msg = "A view was created in inference mode and is being modified inplace"
                    with self.assertRaisesRegex(RuntimeError, err_msg):
                        fn(view_out)
                    pass
                else:
                    fn(view_out)
        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))

    def test_inference_mode_handle_indirect_view_on_rebase(self):
        def run_test(fn):
            for requires_grad in (True, False):
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                a = s.clone()

                with torch.inference_mode():
                    view_out = a.view(-1)

                fn(a)
                if requires_grad:
                    err_msg = "A view was created in inference mode and its base or another view "
                    with self.assertRaisesRegex(RuntimeError, err_msg):
                        view_out.grad_fn
                    pass
                else:
                    view_out.grad_fn
        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))


if __name__ == '__main__':
    run_tests()
