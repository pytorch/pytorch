import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.logging_tensor import LoggingTensor, capture_logs, log_input

def maybe_log_functional_input(name, handle, is_logging):
    if is_logging:
        assert torch._is_functional_tensor(handle)
        log_input(name, torch._from_functional_tensor(handle))

class TestFunctionalization(TestCase):

    def test_functionalization(self):

        def f1(x, *, logging: bool):
            # simple test: 1 view op, 1 inplace op
            with capture_logs() as logs:
                maybe_log_functional_input("x", x, logging)
                tmp = torch.ones(4, 2)
                y = x.view(4, 2)
                y.add_(tmp)
                z = x * x
            if logging:
                self.assertExpectedInline('\n'.join(logs), """\
$0 = input('x')
$1 = torch._ops.aten.view($0, [4, 2])
$2 = torch._ops.aten.add($1, tensor([[1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]]))
$3 = torch._ops.aten.view($2, [4, 2])
$4 = torch._ops.aten.mul($3, $3)""")
            else:
                return y

        def f2(x, *, logging: bool):
            # test: view ops that take a subset of the original tensor (select/diagonal)
            with capture_logs() as logs:
                maybe_log_functional_input("x", x, logging)
                tmp = torch.ones(2)
                y = x.diagonal()
                y.add_(tmp)
                z = x * x
            if logging:
                self.assertExpectedInline('\n'.join(logs), """\
$0 = input('x')
$1 = torch._ops.aten.diagonal($0)
$2 = torch._ops.aten.add($1, tensor([1., 1.]))
$3 = torch._ops.aten.diagonal_scatter($0, $2)
$4 = torch._ops.aten.mul($3, $3)""")
            else:
                return z

        def f3(x, *, logging: bool):
            # test: view ops that return multiple tensors (split)
            with capture_logs() as logs:
                maybe_log_functional_input("x", x, logging)
                tmp = torch.ones(2)
                y1, y2 = x.split(2)
                y3 = y2.diagonal()
                y3.add_(tmp)
                z = x * x
            if logging:
                self.assertExpectedInline('\n'.join(logs), """\
$0 = input('x')
$1, $2 = torch._ops.aten.split($0, 2)
$3 = torch._ops.aten.diagonal($2)
$4 = torch._ops.aten.add($3, tensor([1., 1.]))
$5, $6 = torch._ops.aten.split($0, 2)
$7 = torch._ops.aten.diagonal_scatter($6, $4)
$8 = torch._ops.aten.slice_scatter($0, $7, 0, 2, 4)
$9 = torch._ops.aten.mul($8, $8)""")
            else:
                return y3

        def f4(x, *, logging: bool):
            # test: view + inplace op (transpose_)
            with capture_logs() as logs:
                maybe_log_functional_input("x", x, logging)
                tmp = torch.ones(4)
                x.transpose_(1, 0)
                y = x[0]
                y.add_(tmp)
            if logging:
                self.assertExpectedInline('\n'.join(logs), """\
$0 = input('x')
$1 = torch._ops.aten.transpose($0, 1, 0)
$2 = torch._ops.aten.select($1, 0, 0)
$3 = torch._ops.aten.add($2, tensor([1., 1., 1., 1.]))""")
            else:
                return y

        def f5(x, *, logging: bool):
            # test: everything
            with capture_logs() as logs:
                maybe_log_functional_input("x", x, logging)
                tmp = torch.ones(2, 2)
                y = x.view(8)
                z0 = y.reshape(2, 4)
                z1 = z0.transpose(1, 0)
                z1.unsqueeze_(0)
                z1.squeeze_()
                z2, z3 = z1.split(2)
                z2.add_(tmp)
                z4 = z0[0] + z2.reshape(4)
            if logging:
                self.assertExpectedInline('\n'.join(logs), """\
$0 = input('x')
$1 = torch._ops.aten.view($0, [8])
$2 = torch._ops.aten._reshape_alias($1, [2, 4], [4, 1])
$3 = torch._ops.aten.transpose($2, 1, 0)
$4 = torch._ops.aten.view($0, [8])
$5 = torch._ops.aten._reshape_alias($4, [2, 4], [4, 1])
$6 = torch._ops.aten.transpose($5, 1, 0)
$7 = torch._ops.aten.unsqueeze($6, 0)
$8 = torch._ops.aten.view($0, [8])
$9 = torch._ops.aten._reshape_alias($8, [2, 4], [4, 1])
$10 = torch._ops.aten.transpose($9, 1, 0)
$11 = torch._ops.aten.unsqueeze($10, 0)
$12 = torch._ops.aten.squeeze($11)
$13, $14 = torch._ops.aten.split($12, 2)
$15 = torch._ops.aten.add($13, tensor([[1., 1.],
        [1., 1.]]))
$16 = torch._ops.aten.select($2, 0, 0)
$17 = torch._ops.aten.view($0, [8])
$18 = torch._ops.aten._reshape_alias($17, [2, 4], [4, 1])
$19 = torch._ops.aten.transpose($18, 1, 0)
$20 = torch._ops.aten.unsqueeze($19, 0)
$21 = torch._ops.aten.squeeze($20)
$22 = torch._ops.aten.slice_scatter($21, $15, 0, 0, 2)
$23 = torch._ops.aten.unsqueeze($22, 0)
$24 = torch._ops.aten.squeeze($23, 0)
$25 = torch._ops.aten.transpose($24, 1, 0)
$26 = torch._ops.aten._reshape_alias($25, [8], [1])
$27 = torch._ops.aten.view($26, [4, 2])
$28 = torch._ops.aten.view($27, [8])
$29 = torch._ops.aten._reshape_alias($28, [2, 4], [4, 1])
$30 = torch._ops.aten.transpose($29, 1, 0)
$31 = torch._ops.aten.unsqueeze($30, 0)
$32 = torch._ops.aten.squeeze($31)
$33, $34 = torch._ops.aten.split($32, 2)
$35 = torch._ops.aten.clone($33, memory_format=0)
$36 = torch._ops.aten._unsafe_view($35, [4])
$37 = torch._ops.aten.view($27, [8])
$38 = torch._ops.aten._reshape_alias($37, [2, 4], [4, 1])
$39 = torch._ops.aten.select($38, 0, 0)
$40 = torch._ops.aten.add($39, $36)""")
            else:
                return z2

        def assert_functionalization(func, inpt):
            input_clone = inpt.clone()

            input_clone2 = inpt.clone()
            input_clone_logging = LoggingTensor(inpt.clone())

            input_functional = torch._to_functional_tensor(input_clone2)
            input_functional_logging = torch._to_functional_tensor(input_clone_logging)

            # Runs expecttests for LoggingTensor output.
            torch._enable_functionalization()
            func(input_functional_logging, logging=True)
            torch._disable_functionalization()

            # Compare outputs (and mutated inputs), with and without functionalization.
            out_ref = func(inpt, logging=False)

            torch._enable_functionalization()
            out_functional = func(input_functional, logging=False)
            torch._disable_functionalization()

            # We need to sync the input tensors first, in case there are any queued mutations left.
            torch._sync(input_functional)
            torch._sync(out_functional)
            self.assertEqual(out_ref, torch._from_functional_tensor(out_functional))
            self.assertEqual(inpt, torch._from_functional_tensor(input_functional))  # input mutations should still occur

        assert_functionalization(f1, torch.ones(4, 2))
        assert_functionalization(f2, torch.ones(2, 2))
        assert_functionalization(f3, torch.ones(4, 2))
        assert_functionalization(f4, torch.ones(4, 2))
        assert_functionalization(f5, torch.ones(4, 2))

if __name__ == '__main__':
    run_tests()
