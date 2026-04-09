# Owner(s): ["module: dynamo"]
# flake8: noqa: F403,F405

try:
    from ._higher_order_ops_test_utils import *
except ImportError:
    from _higher_order_ops_test_utils import *


class FuncTorchHigherOrderOpTests(
    torch._dynamo.test_case.TestCaseWithNestedGraphBreaks
):
    def tearDown(self):
        # Ensure that in the case of a test failure, the next test won't fail
        # because of a previous call to _vmap_increment_nesting that wasn't undone
        # i.e. test_vmap_free_tensor fails when PYTORCH_TEST_WITH_DYNAMO=1
        # and the call to increment nesting is not undone
        try:
            if TEST_WITH_TORCHDYNAMO:
                warn = False
                while ci := torch._C._functorch.peek_interpreter_stack():
                    if ci.key() == torch._C._functorch.TransformType.Vmap:
                        warn = True
                        torch._C._functorch._vmap_decrement_nesting()
                    else:
                        break

                if warn:
                    msg = (
                        "Interpreter stack is not empty. Test should have called "
                        "'torch._C._functorch._vmap_decrement_nesting()'"
                    )
                    warnings.warn(msg)
        finally:
            super().tearDown()

    def test_teardown_resets_nested_graph_breaks(self):
        expected_nested_state = getattr(
            self, "prev_nested_graph_breaks", torch._dynamo.config.nested_graph_breaks
        )

        def _check_flag():
            self.assertEqual(
                torch._dynamo.config.nested_graph_breaks, expected_nested_state
            )

        self.addCleanup(_check_flag)
        # Sanity check: these tests always run with nested graph breaks enabled.
        self.assertTrue(torch._dynamo.config.nested_graph_breaks)

    def _compile_check(self, fn, inputs, fullgraph=True, graph_idx=0):
        backend = EagerAndRecordGraphs()
        actual = fn(*inputs)
        expected = torch.compile(fn, backend=backend, fullgraph=fullgraph)(*inputs)

        self.assertEqual(actual, expected)

        wrapped_gm = backend.graphs[graph_idx]
        return wrapped_gm

    def test_hessian(self):
        counters.clear()

        def wrapper_fn(x):
            return torch.func.hessian(torch.sin)(x)

        x = torch.randn(4, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]"):
        l_x_ = L_x_

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = l_x_.new_zeros(12, 12)

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        child: "f32[12, 4, 3]" = chunk.view(12, 4, 3);  chunk = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        child_1: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(child, 0, 1);  child = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        child_2: "f32[4, 3]" = torch._make_dual(l_x_, child_1, level = 0);  child_1 = None

        _wrap_for_grad: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = _wrap_for_grad = None

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_primals: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(child_2, 3);  child_2 = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[4, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_primals);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        primals_out: "f32[4, 3]" = torch.sin(diff_primals)

        results: "f32[4, 3]" = torch._C._functorch._unwrap_for_grad(primals_out, 3)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        tensor_1: "i64[1]" = torch.tensor((12,))
        cumsum_1: "i64[1]" = tensor_1.cumsum(dim = 0);  tensor_1 = None
        getitem_1: "i64[0]" = cumsum_1[slice(None, -1, None)];  cumsum_1 = None
        neg_1: "i64[0]" = getitem_1.neg();  getitem_1 = None
        unbind_1 = neg_1.unbind();  neg_1 = unbind_1 = None

        chunk_1: "f32[12, 12]" = results.new_zeros(12, 12);  results = None

        diagonal_1: "f32[12]" = chunk_1.diagonal(0)
        fill__1: "f32[12]" = diagonal_1.fill_(1);  diagonal_1 = fill__1 = None

        basis: "f32[12, 4, 3]" = chunk_1.view(12, 4, 3);  chunk_1 = None

        lazy_load_decompositions_1 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_1 = None

        _vmap_increment_nesting_1 = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting_1 = None

        _add_batch_dim_1: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(basis, 0, 3);  basis = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([primals_out], [diff_primals], [_add_batch_dim_1], retain_graph = True, create_graph = True);  primals_out = diff_primals = _add_batch_dim_1 = None
        batched_outputs: "f32[4, 3]" = _autograd_grad[0];  _autograd_grad = None

        chunked_result: "f32[12, 4, 3]" = torch._functorch.predispatch._remove_batch_dim(batched_outputs, 3, 12, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        split = chunked_result.split((12,), dim = 0);  chunked_result = None
        split_1: "f32[12, 4, 3]" = split[0];  split = None

        output_input: "f32[4, 3, 4, 3]" = split_1.view((4, 3, 4, 3));  split_1 = None

        _unpack_dual = torch._unpack_dual(output_input, level = 0);  output_input = None
        primal: "f32[4, 3, 4, 3]" = _unpack_dual[0]
        dual: "f32[4, 3, 4, 3]" = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[4, 3, 4, 3]" = torch._C._functorch._unwrap_for_grad(primal, 2);  primal = primals_out_unflatten = None
        tangents_out_unflatten: "f32[4, 3, 4, 3]" = torch._C._functorch._unwrap_for_grad(dual, 2);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        results_1: "f32[12, 4, 3, 4, 3]" = torch._functorch.predispatch._remove_batch_dim(tangents_out_unflatten, 1, 12, 0);  tangents_out_unflatten = None

        _vmap_decrement_nesting_1 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None

        movedim: "f32[4, 3, 4, 3, 12]" = results_1.movedim(0, -1);  results_1 = None
        split_2 = movedim.split((12,), dim = -1);  movedim = None
        jac_out_in: "f32[4, 3, 4, 3, 12]" = split_2[0];  split_2 = None

        unflatten: "f32[4, 3, 4, 3, 4, 3]" = jac_out_in.unflatten(-1, (4, 3));  jac_out_in = None
        return (unflatten,)
""",
        )

    def test_hessian_argnums(self):
        counters.clear()

        def fn(x, y):
            return x.sin()

        def wrapper_fn(x, y):
            return torch.func.hessian(fn, argnums=(1,))(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            "\n".join(actual.split("\n")[:-2]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = l_y_.new_zeros(12, 12)

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        child: "f32[12, 3, 4]" = chunk.view(12, 3, 4);  chunk = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        child_1: "f32[3, 4]" = torch._functorch.predispatch._add_batch_dim(child, 0, 1);  child = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        child_3: "f32[3, 4]" = torch._make_dual(l_y_, child_1, level = 0);  child_1 = None

        child_2: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = None
        _wrap_for_grad_1: "f32[3, 4]" = torch._C._functorch._wrap_for_grad(l_y_, 2);  l_y_ = _wrap_for_grad_1 = None

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad_2: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(child_2, 3);  child_2 = None
        child_4: "f32[3, 4]" = torch._C._functorch._wrap_for_grad(child_3, 3);  child_3 = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 4]" = torch._functorch.eager_transforms._set_tensor_requires_grad(child_4);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        primals_out: "f32[4, 3]" = _wrap_for_grad_2.sin();  _wrap_for_grad_2 = None

        results: "f32[4, 3]" = torch._C._functorch._unwrap_for_grad(primals_out, 3)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        tensor_1: "i64[1]" = torch.tensor((12,))
        cumsum_1: "i64[1]" = tensor_1.cumsum(dim = 0);  tensor_1 = None
        getitem_1: "i64[0]" = cumsum_1[slice(None, -1, None)];  cumsum_1 = None
        neg_1: "i64[0]" = getitem_1.neg();  getitem_1 = None
        unbind_1 = neg_1.unbind();  neg_1 = unbind_1 = None

        chunk_1: "f32[12, 12]" = results.new_zeros(12, 12);  results = None

        diagonal_1: "f32[12]" = chunk_1.diagonal(0)
        fill__1: "f32[12]" = diagonal_1.fill_(1);  diagonal_1 = fill__1 = None

        basis: "f32[12, 4, 3]" = chunk_1.view(12, 4, 3);  chunk_1 = None

        lazy_load_decompositions_1 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_1 = None

        _vmap_increment_nesting_1 = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting_1 = None

        _add_batch_dim_1: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(basis, 0, 3);  basis = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([primals_out], [child_4], [_add_batch_dim_1], retain_graph = True, create_graph = True);  primals_out = child_4 = _add_batch_dim_1 = None
        child_5: "f32[3, 4]" = _autograd_grad[0];  _autograd_grad = None

        child_6: "f32[12, 3, 4]" = torch._functorch.predispatch._remove_batch_dim(child_5, 3, 12, 0);  child_5 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        split = child_6.split((12,), dim = 0);  child_6 = None
        split_1: "f32[12, 3, 4]" = split[0];  split = None

        child_7: "f32[4, 3, 3, 4]" = split_1.view((4, 3, 3, 4));  split_1 = None

        _unpack_dual = torch._unpack_dual(child_7, level = 0);  child_7 = None
        primal: "f32[4, 3, 3, 4]" = _unpack_dual[0];  _unpack_dual = None

        tangent: "f32[4, 3, 3, 4]" = torch.zeros_like(primal)

        child_8: "f32[4, 3, 3, 4]" = torch._C._functorch._unwrap_for_grad(primal, 2);  primal = child_8 = None
        child_9: "f32[4, 3, 3, 4]" = torch._C._functorch._unwrap_for_grad(tangent, 2);  tangent = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        child_10: "f32[12, 4, 3, 3, 4]" = torch._functorch.predispatch._remove_batch_dim(child_9, 1, 12, 0);  child_9 = None

        _vmap_decrement_nesting_1 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None

        movedim: "f32[4, 3, 3, 4, 12]" = child_10.movedim(0, -1);  child_10 = None
        split_2 = movedim.split((12,), dim = -1);  movedim = None
        jac_out_in: "f32[4, 3, 3, 4, 12]" = split_2[0];  split_2 = None

        unflatten: "f32[4, 3, 3, 4, 3, 4]" = jac_out_in.unflatten(-1, (3, 4));  jac_out_in = None""",
        )

        self.assertExpectedInline(
            actual.split("\n")[-2],
            """        return (unflatten,)""",
        )

    def test_jacrev(self):
        counters.clear()

        def wrapper_fn(x):
            return torch.func.jacrev(torch.sin)(x)

        x = torch.randn(4, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_primals: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[4, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_primals);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        primals_out: "f32[4, 3]" = torch.sin(diff_primals)

        results: "f32[4, 3]" = torch._C._functorch._unwrap_for_grad(primals_out, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = results.new_zeros(12, 12);  results = None

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        basis: "f32[12, 4, 3]" = chunk.view(12, 4, 3);  chunk = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(basis, 0, 1);  basis = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([primals_out], [diff_primals], [_add_batch_dim], retain_graph = True, create_graph = True);  primals_out = diff_primals = _add_batch_dim = None
        batched_outputs: "f32[4, 3]" = _autograd_grad[0];  _autograd_grad = None

        chunked_result: "f32[12, 4, 3]" = torch._functorch.predispatch._remove_batch_dim(batched_outputs, 1, 12, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        split = chunked_result.split((12,), dim = 0);  chunked_result = None
        split_1: "f32[12, 4, 3]" = split[0];  split = None

        output_input: "f32[4, 3, 4, 3]" = split_1.view((4, 3, 4, 3));  split_1 = None
        return (output_input,)
""",
        )

    def test_jacrev_two_tensors_argnums(self):
        counters.clear()

        def fn(x, y):
            return y.sin()

        def wrapper_fn(x, y):
            return torch.func.jacrev(fn, argnums=1)(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = _wrap_for_grad = None
        diff_primals: "f32[3, 4]" = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 4]" = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_primals);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        primals_out: "f32[3, 4]" = diff_primals.sin()

        results: "f32[3, 4]" = torch._C._functorch._unwrap_for_grad(primals_out, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = results.new_zeros(12, 12);  results = None

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        basis: "f32[12, 3, 4]" = chunk.view(12, 3, 4);  chunk = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 4]" = torch._functorch.predispatch._add_batch_dim(basis, 0, 1);  basis = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([primals_out], [diff_primals], [_add_batch_dim], retain_graph = True, create_graph = True);  primals_out = diff_primals = _add_batch_dim = None
        batched_outputs: "f32[3, 4]" = _autograd_grad[0];  _autograd_grad = None

        chunked_result: "f32[12, 3, 4]" = torch._functorch.predispatch._remove_batch_dim(batched_outputs, 1, 12, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        split = chunked_result.split((12,), dim = 0);  chunked_result = None
        split_1: "f32[12, 3, 4]" = split[0];  split = None

        output_input: "f32[3, 4, 3, 4]" = split_1.view((3, 4, 3, 4));  split_1 = None
        return (output_input,)
""",
        )

    def test_jacrev_has_aux(self):
        counters.clear()

        def fn(x, y):
            return y.sin(), x

        def wrapper_fn(x, y):
            return torch.func.jacrev(fn, argnums=1, has_aux=True)(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        aux: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        diff_primals: "f32[3, 4]" = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 4]" = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_primals);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        primals_out: "f32[3, 4]" = diff_primals.sin()

        aux_1: "f32[4, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None
        results: "f32[3, 4]" = torch._C._functorch._unwrap_for_grad(primals_out, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = results.new_zeros(12, 12);  results = None

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        basis: "f32[12, 3, 4]" = chunk.view(12, 3, 4);  chunk = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 4]" = torch._functorch.predispatch._add_batch_dim(basis, 0, 1);  basis = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([primals_out], [diff_primals], [_add_batch_dim], retain_graph = True, create_graph = True);  primals_out = diff_primals = _add_batch_dim = None
        batched_outputs: "f32[3, 4]" = _autograd_grad[0];  _autograd_grad = None

        chunked_result: "f32[12, 3, 4]" = torch._functorch.predispatch._remove_batch_dim(batched_outputs, 1, 12, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        split = chunked_result.split((12,), dim = 0);  chunked_result = None
        split_1: "f32[12, 3, 4]" = split[0];  split = None

        output_input: "f32[3, 4, 3, 4]" = split_1.view((3, 4, 3, 4));  split_1 = None
        return (output_input, aux_1)
""",
        )

    def test_vjp(self):
        counters.clear()

        def fn(x):
            return x.sin().sum()

        def wrapper_fn(x, v):
            (out, vjpfunc) = torch.func.vjp(fn, x)
            return out

        x = torch.randn([5])
        v = torch.randn(5)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[5]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[5]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        child: "f32[5]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad);  child = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[5]" = _wrap_for_grad.sin();  _wrap_for_grad = None
        primals_out: "f32[]" = sin.sum();  sin = None

        results: "f32[]" = torch._C._functorch._unwrap_for_grad(primals_out, 1);  primals_out = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (results,)
""",
        )

    def test_vjp_multiple_outputs(self):
        counters.clear()

        def wrapper_fn(x, v):
            fn = lambda x: (x.sin(), x.cos())  # noqa: E731
            (out, vjpfunc) = torch.func.vjp(fn, x)
            vjps = vjpfunc((v, v))
            return out, vjps

        x = torch.randn([5])
        v = torch.randn(5)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[5]", L_v_: "f32[5]"):
        l_x_ = L_x_
        l_v_ = L_v_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[5]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        child_2: "f32[5]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        child: "f32[5]" = _wrap_for_grad.sin()
        child_1: "f32[5]" = _wrap_for_grad.cos();  _wrap_for_grad = None

        _unwrap_for_grad: "f32[5]" = torch._C._functorch._unwrap_for_grad(child, 1)
        _unwrap_for_grad_1: "f32[5]" = torch._C._functorch._unwrap_for_grad(child_1, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([child, child_1], [child_2], [l_v_, l_v_], retain_graph = True, create_graph = True);  child = child_1 = child_2 = l_v_ = None
        getitem: "f32[5]" = _autograd_grad[0];  _autograd_grad = None
        return (_unwrap_for_grad, _unwrap_for_grad_1, getitem)
""",
        )

    def test_vjp_multiple_outputs_python_struct(self):
        counters.clear()

        def wrapper_fn(x, v):
            fn = lambda x: {"first": x.sin(), "second": x.cos()}  # noqa: E731
            (out, vjpfunc) = torch.func.vjp(fn, x)
            vjps = vjpfunc({"first": v, "second": v.sin()})
            return out, vjps

        x = torch.randn([5])
        v = torch.randn(5)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[5]", L_v_: "f32[5]"):
        l_x_ = L_x_
        l_v_ = L_v_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[5]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        child_2: "f32[5]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        child: "f32[5]" = _wrap_for_grad.sin()
        child_1: "f32[5]" = _wrap_for_grad.cos();  _wrap_for_grad = None

        _unwrap_for_grad: "f32[5]" = torch._C._functorch._unwrap_for_grad(child, 1)
        _unwrap_for_grad_1: "f32[5]" = torch._C._functorch._unwrap_for_grad(child_1, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        child_3: "f32[5]" = l_v_.sin()

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([child, child_1], [child_2], [l_v_, child_3], retain_graph = True, create_graph = True);  child = child_1 = child_2 = l_v_ = child_3 = None
        getitem: "f32[5]" = _autograd_grad[0];  _autograd_grad = None
        return (_unwrap_for_grad, _unwrap_for_grad_1, getitem)
""",
        )

    def test_vjp_has_aux(self):
        counters.clear()

        def fn(x):
            return x.sin().sum(), x

        def wrapper_fn(x, v):
            (out, vjpfunc, _) = torch.func.vjp(fn, x, has_aux=True)
            return out

        x = torch.randn([5])
        v = torch.randn(5)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[5]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        aux: "f32[5]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        child: "f32[5]" = torch._functorch.eager_transforms._set_tensor_requires_grad(aux);  child = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[5]" = aux.sin()
        primals_out: "f32[]" = sin.sum();  sin = None

        aux_1: "f32[5]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = aux_1 = None
        results: "f32[]" = torch._C._functorch._unwrap_for_grad(primals_out, 1);  primals_out = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (results,)
""",
        )

    def test_functional_call(self):
        def wrapper_fn(model, params, inputs, targets):
            prediction = torch.func.functional_call(model, params, (inputs,))
            return torch.nn.functional.mse_loss(prediction, targets)

        model = torch.nn.Linear(3, 3)
        params = dict(model.named_parameters())
        inputs = torch.randn(64, 3)
        targets = torch.randn(64, 3)

        wrapped_gm = self._compile_check(wrapper_fn, (model, params, inputs, targets))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_model_parameters_weight_: "f32[3, 3]", L_model_parameters_bias_: "f32[3]", L_inputs_: "f32[64, 3]", L_targets_: "f32[64, 3]"):
        l_model_parameters_weight_ = L_model_parameters_weight_
        l_model_parameters_bias_ = L_model_parameters_bias_
        l_inputs_ = L_inputs_
        l_targets_ = L_targets_

        prediction: "f32[64, 3]" = torch._C._nn.linear(l_inputs_, l_model_parameters_weight_, l_model_parameters_bias_);  l_inputs_ = l_model_parameters_weight_ = l_model_parameters_bias_ = None

        mse_loss: "f32[]" = torch.nn.functional.mse_loss(prediction, l_targets_);  prediction = l_targets_ = None
        return (mse_loss,)
""",
        )

    def test_functional_call_sequential_params_and_buffers(self):
        # copied from test/test_stateless.py
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l1 = torch.nn.Linear(1, 1)
                self.register_buffer("buffer", torch.ones(1))
                self.foo = 0.0

            def forward(self, x):
                return self.l1(x) + self.buffer

        def wrapper_fn(model, params, buffers, inputs):
            # two separate dictionaries
            return torch.func.functional_call(model, (params, buffers), inputs)

        model = MockModule()
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        inputs = torch.tensor([[1.5]])

        wrapped_gm = self._compile_check(
            wrapper_fn, (model, params, buffers, inputs), fullgraph=False
        )
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_inputs_: "f32[1, 1]", L_model_modules_l1_parameters_weight_: "f32[1, 1]", L_model_modules_l1_parameters_bias_: "f32[1]", L_model_buffers_buffer_: "f32[1]"):
        l_inputs_ = L_inputs_
        l_model_modules_l1_parameters_weight_ = L_model_modules_l1_parameters_weight_
        l_model_modules_l1_parameters_bias_ = L_model_modules_l1_parameters_bias_
        l_model_buffers_buffer_ = L_model_buffers_buffer_
        linear: "f32[1, 1]" = torch._C._nn.linear(l_inputs_, l_model_modules_l1_parameters_weight_, l_model_modules_l1_parameters_bias_);  l_inputs_ = l_model_modules_l1_parameters_weight_ = l_model_modules_l1_parameters_bias_ = None
        add: "f32[1, 1]" = linear + l_model_buffers_buffer_;  linear = l_model_buffers_buffer_ = None
        return (add,)
"""
        # We found Windows/Linux have some empty line difference, empty_line_normalizer will help fix it.
        self.assertExpectedInline(
            empty_line_normalizer(actual),
            empty_line_normalizer(normalize_gm(expected)),
        )

    def test_grad(self):
        counters.clear()

        def fn(x):
            return x.sin().sum()

        def wrapper_fn(x):
            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_args: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[3, 3, 3]" = diff_args.sin()
        output: "f32[]" = sin.sum();  sin = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input: "f32[3, 3, 3]" = _autograd_grad[0];  _autograd_grad = None

        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None
        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (grad_input_1,)
""",
        )

    def test_grad_freevar_tensor(self):
        counters.clear()
        y = torch.randn(3, 3)

        def fn(x):
            return (x.sin() + y).sum()

        def wrapper_fn(x):
            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)
        expected = wrapper_fn(x)
        actual = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(actual, expected)

    def test_grad_freevar_python_scalar(self):
        counters.clear()
        y = 3

        def fn(x):
            return (x.sin() + y).sum()

        def wrapper_fn(x):
            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_args: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[3, 3, 3]" = diff_args.sin()
        add: "f32[3, 3, 3]" = sin + 3;  sin = None
        output: "f32[]" = add.sum();  add = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input: "f32[3, 3, 3]" = _autograd_grad[0];  _autograd_grad = None

        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None
        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (grad_input_1,)
""",
        )

    def test_grad_capture_tensor(self):
        counters.clear()

        def wrapper_fn(x):
            y = torch.randn(3)

            def fn(x):
                return (x.sin() + y).sum()

            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)

        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        y: "f32[3]" = torch.randn(3)

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_args: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[3, 3, 3]" = diff_args.sin()
        add: "f32[3, 3, 3]" = sin + y;  sin = y = None
        output: "f32[]" = add.sum();  add = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input: "f32[3, 3, 3]" = _autograd_grad[0];  _autograd_grad = None

        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None
        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (grad_input_1,)
""",
        )

    def test_grad_closure_scalar(self):
        counters.clear()

        def wrapper_fn(x):
            y = 3.14

            def fn(x):
                return (x.sin() + y).sum()

            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)

        # Graph break because dynamo is unable to get source `fn` and
        # functools.wraps in `grad` leads to graph-break
        wrapped_gm = self._compile_check(wrapper_fn, (x,), fullgraph=False)

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_args: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[3, 3, 3]" = diff_args.sin()
        add: "f32[3, 3, 3]" = sin + 3.14;  sin = None
        output: "f32[]" = add.sum();  add = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input: "f32[3, 3, 3]" = _autograd_grad[0];  _autograd_grad = None

        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None
        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (grad_input_1,)
""",
        )

    def test_grad_has_aux(self):
        counters.clear()

        y = 3.14

        def fn(x):
            return ((x.sin() + y).sum(), x.cos())

        def wrapper_fn(x):
            return torch.func.grad(fn, has_aux=True)(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_args: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[3, 3, 3]" = diff_args.sin()
        add: "f32[3, 3, 3]" = sin + 3.14;  sin = None
        output: "f32[]" = add.sum();  add = None
        aux: "f32[3, 3, 3]" = diff_args.cos()

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input: "f32[3, 3, 3]" = _autograd_grad[0];  _autograd_grad = None

        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None
        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None
        aux_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (grad_input_1, aux_1)
""",
        )

    def test_grad_two_tensor_has_aux(self):
        counters.clear()

        def fn(x, y):
            return ((x.sin() + y).sum(), x.cos())

        def wrapper_fn(x, y):
            return torch.func.grad(fn, has_aux=True)(x, y)

        y = torch.randn(3, 3, 3)
        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_args: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        _wrap_for_grad_1: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[3, 3, 3]" = diff_args.sin()
        add: "f32[3, 3, 3]" = sin + _wrap_for_grad_1;  sin = _wrap_for_grad_1 = None
        output: "f32[]" = add.sum();  add = None
        aux: "f32[3, 3, 3]" = diff_args.cos()

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input: "f32[3, 3, 3]" = _autograd_grad[0];  _autograd_grad = None

        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None
        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None
        aux_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (grad_input_1, aux_1)
""",
        )

    def test_grad_two_tensor_all_grad_has_aux(self):
        counters.clear()

        nums = (0, 1)

        def fn(x, y):
            return ((x.sin() + y).sum(), x.cos())

        def wrapper_fn_const_var(x, y):
            return torch.func.grad(fn, argnums=(0, 1), has_aux=True)(x, y)

        def wrapper_fn_tuple_var(x, y):
            return torch.func.grad(fn, argnums=nums, has_aux=True)(x, y)

        y = torch.randn(3, 3, 3)
        x = torch.randn(3, 3, 3)
        wrapped_gm_const_var = self._compile_check(wrapper_fn_const_var, (x, y))
        wrapped_gm_tuple_var = self._compile_check(wrapper_fn_tuple_var, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual_const_var = normalize_gm(
            wrapped_gm_const_var.print_readable(print_output=False)
        )
        actual_tuple_var = normalize_gm(
            wrapped_gm_tuple_var.print_readable(print_output=False)
        )
        self.assertExpectedInline(
            actual_const_var,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        child: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        child_1: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(child);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None
        set_inplace_requires_grad_allowed_2 = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed_2 = None

        _set_tensor_requires_grad_1: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(child_1);  _set_tensor_requires_grad_1 = None

        set_inplace_requires_grad_allowed_3 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_3 = None

        sin: "f32[3, 3, 3]" = child.sin()
        add: "f32[3, 3, 3]" = sin + child_1;  sin = None
        output: "f32[]" = add.sum();  add = None
        aux: "f32[3, 3, 3]" = child.cos()

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [child, child_1], create_graph = True);  child = child_1 = None
        getitem: "f32[3, 3, 3]" = _autograd_grad[0]
        getitem_1: "f32[3, 3, 3]" = _autograd_grad[1];  _autograd_grad = None

        _unwrap_for_grad: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(getitem, 1);  getitem = None
        _unwrap_for_grad_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(getitem_1, 1);  getitem_1 = None
        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None
        aux_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (_unwrap_for_grad, _unwrap_for_grad_1, aux_1)
""",
        )
        self.assertExpectedInline(
            actual_tuple_var,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        child: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        child_1: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(child);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None
        set_inplace_requires_grad_allowed_2 = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed_2 = None

        _set_tensor_requires_grad_1: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(child_1);  _set_tensor_requires_grad_1 = None

        set_inplace_requires_grad_allowed_3 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_3 = None

        sin: "f32[3, 3, 3]" = child.sin()
        add: "f32[3, 3, 3]" = sin + child_1;  sin = None
        output: "f32[]" = add.sum();  add = None
        aux: "f32[3, 3, 3]" = child.cos()

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [child, child_1], create_graph = True);  child = child_1 = None
        getitem: "f32[3, 3, 3]" = _autograd_grad[0]
        getitem_1: "f32[3, 3, 3]" = _autograd_grad[1];  _autograd_grad = None

        _unwrap_for_grad: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(getitem, 1);  getitem = None
        _unwrap_for_grad_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(getitem_1, 1);  getitem_1 = None
        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None
        aux_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (_unwrap_for_grad, _unwrap_for_grad_1, aux_1)
""",
        )

    def test_grad_over_grad(self):
        counters.clear()

        def fn(x):
            return x.sin().sum()

        def wrapper_fn(x):
            return torch.func.grad(torch.func.grad(fn))(x)

        x = torch.randn(())
        wrapped_gm = self._compile_check(wrapper_fn, (x,), fullgraph=False)

        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_args: "f32[]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[]" = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None
        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable_1 = None
        _grad_increment_nesting_1 = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting_1 = None

        diff_args_1: "f32[]" = torch._C._functorch._wrap_for_grad(diff_args, 2)

        set_inplace_requires_grad_allowed_2 = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed_2 = None

        _set_tensor_requires_grad_1: "f32[]" = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args_1);  _set_tensor_requires_grad_1 = None

        set_inplace_requires_grad_allowed_3 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_3 = None

        sin: "f32[]" = diff_args_1.sin()
        output: "f32[]" = sin.sum();  sin = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args_1], create_graph = True);  diff_args_1 = None
        grad_input: "f32[]" = _autograd_grad[0];  _autograd_grad = None

        grad_input_1: "f32[]" = torch._C._functorch._unwrap_for_grad(grad_input, 2);  grad_input = None
        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 2);  output = output_1 = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_disable_2 = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable_2 = None

        _autograd_grad_1 = torch._functorch.eager_transforms._autograd_grad((grad_input_1,), [diff_args], create_graph = True);  diff_args = None
        grad_input_2: "f32[]" = _autograd_grad_1[0];  _autograd_grad_1 = None

        grad_input_3: "f32[]" = torch._C._functorch._unwrap_for_grad(grad_input_2, 1);  grad_input_2 = None
        output_2: "f32[]" = torch._C._functorch._unwrap_for_grad(grad_input_1, 1);  grad_input_1 = output_2 = None

        _grad_decrement_nesting_1 = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting_1 = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (grad_input_3,)
""",
        )

    def test_grad_with_graph_break(self):
        counters.clear()

        def fn(x):
            torch._dynamo.graph_break()
            return x.sin().sum()

        def wrapper_fn(x):
            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)
        actual = wrapper_fn(x)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x)
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(actual, expected)

    def test_grad_with_side_effect(self):
        counters.clear()

        foo = [1, 2]

        def fn(x):
            foo.append(3)
            return x.sin().sum()

        def wrapper_fn(x):
            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)
        actual = wrapper_fn(x)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_grad_pytree(self):
        counters.clear()

        def fn(x):
            x1, x2 = x
            return x1.sin().sum() + x2

        def wrapper_fn(x):
            return torch.func.grad(fn)(x)

        x1 = torch.randn(3, 3, 3)
        x2 = torch.randn(())
        actual = wrapper_fn((x1, x2))
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
            (x1, x2)
        )
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_grad_non_tensor_input(self):
        counters.clear()

        def fn(x, y):
            return x.sin().sum() + y

        def wrapper_fn(x, y):
            return torch.func.grad(fn)(x, y)

        x = torch.randn(3, 3, 3)
        y = 3.0
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_args: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[3, 3, 3]" = diff_args.sin()
        sum_1: "f32[]" = sin.sum();  sin = None
        output: "f32[]" = sum_1 + 3.0;  sum_1 = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input: "f32[3, 3, 3]" = _autograd_grad[0];  _autograd_grad = None

        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None
        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (grad_input_1,)
""",
        )

    def test_grad_fn_with_kwargs(self):
        def fn(x, y):
            return (x + y).sum()

        def wrapper_fn(x, y):
            return torch.func.grad(fn)(x, y=y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        actual = wrapper_fn(x, y)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x, y)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_jacfwd(self):
        counters.clear()

        def wrapper_fn(x):
            return torch.func.jacfwd(torch.sin)(x)

        x = torch.randn(4, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]"):
        l_x_ = L_x_

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = l_x_.new_zeros(12, 12)

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        child: "f32[12, 4, 3]" = chunk.view(12, 4, 3);  chunk = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        child_1: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(child, 0, 1);  child = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[4, 3]" = torch._make_dual(l_x_, child_1, level = 0);  child_1 = None

        _wrap_for_grad: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = _wrap_for_grad = None

        result_duals: "f32[4, 3]" = torch.sin(_make_dual);  _make_dual = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal: "f32[4, 3]" = _unpack_dual[0]
        dual: "f32[4, 3]" = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[4, 3]" = torch._C._functorch._unwrap_for_grad(primal, 2);  primal = primals_out_unflatten = None
        tangents_out_unflatten: "f32[4, 3]" = torch._C._functorch._unwrap_for_grad(dual, 2);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        results: "f32[12, 4, 3]" = torch._functorch.predispatch._remove_batch_dim(tangents_out_unflatten, 1, 12, 0);  tangents_out_unflatten = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        movedim: "f32[4, 3, 12]" = results.movedim(0, -1);  results = None
        split = movedim.split((12,), dim = -1);  movedim = None
        jac_out_in: "f32[4, 3, 12]" = split[0];  split = None

        unflatten: "f32[4, 3, 4, 3]" = jac_out_in.unflatten(-1, (4, 3));  jac_out_in = None
        return (unflatten,)
""",
        )

    def test_jacfwd_two_tensors_argnums(self):
        counters.clear()

        def fn(x, y):
            return y.sin()

        def wrapper_fn(x, y):
            return torch.func.jacfwd(fn, argnums=1)(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = l_y_.new_zeros(12, 12)

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        child: "f32[12, 3, 4]" = chunk.view(12, 3, 4);  chunk = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        child_1: "f32[3, 4]" = torch._functorch.predispatch._add_batch_dim(child, 0, 1);  child = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[3, 4]" = torch._make_dual(l_y_, child_1, level = 0);  child_1 = None

        _wrap_for_grad: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = _wrap_for_grad = None
        _wrap_for_grad_1: "f32[3, 4]" = torch._C._functorch._wrap_for_grad(l_y_, 2);  l_y_ = _wrap_for_grad_1 = None

        result_duals: "f32[3, 4]" = _make_dual.sin();  _make_dual = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal: "f32[3, 4]" = _unpack_dual[0]
        dual: "f32[3, 4]" = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[3, 4]" = torch._C._functorch._unwrap_for_grad(primal, 2);  primal = primals_out_unflatten = None
        tangents_out_unflatten: "f32[3, 4]" = torch._C._functorch._unwrap_for_grad(dual, 2);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        results: "f32[12, 3, 4]" = torch._functorch.predispatch._remove_batch_dim(tangents_out_unflatten, 1, 12, 0);  tangents_out_unflatten = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        movedim: "f32[3, 4, 12]" = results.movedim(0, -1);  results = None
        split = movedim.split((12,), dim = -1);  movedim = None
        jac_out_in: "f32[3, 4, 12]" = split[0];  split = None

        unflatten: "f32[3, 4, 3, 4]" = jac_out_in.unflatten(-1, (3, 4));  jac_out_in = None
        return (unflatten,)
""",
        )

    def test_jacfwd_has_aux(self):
        counters.clear()

        def fn(x, y):
            return y.sin(), x

        def wrapper_fn(x, y):
            return torch.func.jacfwd(fn, argnums=1, has_aux=True)(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = l_y_.new_zeros(12, 12)

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        child: "f32[12, 3, 4]" = chunk.view(12, 3, 4);  chunk = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        child_1: "f32[3, 4]" = torch._functorch.predispatch._add_batch_dim(child, 0, 1);  child = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[3, 4]" = torch._make_dual(l_y_, child_1, level = 0);  child_1 = None

        aux: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = None
        _wrap_for_grad_1: "f32[3, 4]" = torch._C._functorch._wrap_for_grad(l_y_, 2);  l_y_ = _wrap_for_grad_1 = None

        result_duals: "f32[3, 4]" = _make_dual.sin();  _make_dual = None

        aux_1: "f32[4, 3]" = torch._C._functorch._unwrap_for_grad(aux, 2);  aux = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal: "f32[3, 4]" = _unpack_dual[0]
        dual: "f32[3, 4]" = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[3, 4]" = torch._C._functorch._unwrap_for_grad(primal, 2);  primal = primals_out_unflatten = None
        tangents_out_unflatten: "f32[3, 4]" = torch._C._functorch._unwrap_for_grad(dual, 2);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        results: "f32[12, 3, 4]" = torch._functorch.predispatch._remove_batch_dim(tangents_out_unflatten, 1, 12, 0);  tangents_out_unflatten = None
        aux_2: "f32[12, 4, 3]" = torch._functorch.predispatch._remove_batch_dim(aux_1, 1, 12, 0);  aux_1 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        aux_3: "f32[4, 3]" = aux_2[0];  aux_2 = None

        movedim: "f32[3, 4, 12]" = results.movedim(0, -1);  results = None
        split = movedim.split((12,), dim = -1);  movedim = None
        jac_out_in: "f32[3, 4, 12]" = split[0];  split = None

        unflatten: "f32[3, 4, 3, 4]" = jac_out_in.unflatten(-1, (3, 4));  jac_out_in = None
        return (unflatten, aux_3)
""",
        )

    def test_jacfwd_randomness(self):
        counters.clear()

        def fn(x, y):
            return y.sin(), x

        def wrapper_fn(x, y):
            return torch.func.jacfwd(fn, randomness="same")(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = l_x_.new_zeros(12, 12)

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        child: "f32[12, 4, 3]" = chunk.view(12, 4, 3);  chunk = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'same');  _vmap_increment_nesting = None

        child_1: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(child, 0, 1);  child = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        child_3: "f32[4, 3]" = torch._make_dual(l_x_, child_1, level = 0);  child_1 = None

        _wrap_for_grad: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = _wrap_for_grad = None
        _wrap_for_grad_1: "f32[3, 4]" = torch._C._functorch._wrap_for_grad(l_y_, 2);  l_y_ = None

        child_2: "f32[3, 4]" = _wrap_for_grad_1.sin();  _wrap_for_grad_1 = None

        _unpack_dual = torch._unpack_dual(child_2, level = 0);  child_2 = None
        primal: "f32[3, 4]" = _unpack_dual[0];  _unpack_dual = None

        tangent: "f32[3, 4]" = torch.zeros_like(primal)

        _unpack_dual_1 = torch._unpack_dual(child_3, level = 0);  child_3 = None
        primal_1: "f32[4, 3]" = _unpack_dual_1[0]
        dual: "f32[4, 3]" = _unpack_dual_1[1];  _unpack_dual_1 = None

        child_4: "f32[3, 4]" = torch._C._functorch._unwrap_for_grad(primal, 2);  primal = child_4 = None
        child_5: "f32[4, 3]" = torch._C._functorch._unwrap_for_grad(primal_1, 2);  primal_1 = child_5 = None
        child_6: "f32[3, 4]" = torch._C._functorch._unwrap_for_grad(tangent, 2);  tangent = None
        child_7: "f32[4, 3]" = torch._C._functorch._unwrap_for_grad(dual, 2);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        child_8: "f32[12, 3, 4]" = torch._functorch.predispatch._remove_batch_dim(child_6, 1, 12, 0);  child_6 = None
        child_9: "f32[12, 4, 3]" = torch._functorch.predispatch._remove_batch_dim(child_7, 1, 12, 0);  child_7 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        movedim: "f32[3, 4, 12]" = child_8.movedim(0, -1);  child_8 = None
        split = movedim.split((12,), dim = -1);  movedim = None
        jac_out_in: "f32[3, 4, 12]" = split[0];  split = None

        unflatten: "f32[3, 4, 4, 3]" = jac_out_in.unflatten(-1, (4, 3));  jac_out_in = None

        movedim_1: "f32[4, 3, 12]" = child_9.movedim(0, -1);  child_9 = None
        split_1 = movedim_1.split((12,), dim = -1);  movedim_1 = None
        jac_out_in_1: "f32[4, 3, 12]" = split_1[0];  split_1 = None

        unflatten_1: "f32[4, 3, 4, 3]" = jac_out_in_1.unflatten(-1, (4, 3));  jac_out_in_1 = None
        return (unflatten, unflatten_1)
""",
        )

    def test_jvp_simple(self):
        counters.clear()

        def fn(x):
            return x.sin().sum()

        def wrapper_fn(x, v):
            return torch.func.jvp(fn, (x,), (v,))

        x = torch.randn(3, 3)
        v = torch.randn(3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_v_: "f32[3, 3]"):
        l_x_ = L_x_
        l_v_ = L_v_

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[3, 3]" = torch._make_dual(l_x_, l_v_, level = 0);  l_x_ = l_v_ = None

        sin: "f32[3, 3]" = _make_dual.sin();  _make_dual = None
        result_duals: "f32[]" = sin.sum();  sin = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal: "f32[]" = _unpack_dual[0]
        dual: "f32[]" = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(primal, 1);  primal = None
        tangents_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(dual, 1);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None
        return (primals_out_unflatten, tangents_out_unflatten)
""",
        )

    def test_jvp_has_aux(self):
        counters.clear()

        def fn(x):
            return x.sin().sum(), x

        def wrapper_fn(x, v):
            return torch.func.jvp(fn, (x,), (v,), has_aux=True)

        x = torch.randn(3, 3)
        v = torch.randn(3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_v_: "f32[3, 3]"):
        l_x_ = L_x_
        l_v_ = L_v_

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        aux: "f32[3, 3]" = torch._make_dual(l_x_, l_v_, level = 0);  l_x_ = l_v_ = None

        sin: "f32[3, 3]" = aux.sin()
        result_duals: "f32[]" = sin.sum();  sin = None

        aux_1: "f32[3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal: "f32[]" = _unpack_dual[0]
        dual: "f32[]" = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(primal, 1);  primal = None
        tangents_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(dual, 1);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None
        return (primals_out_unflatten, tangents_out_unflatten, aux_1)
""",
        )

    def test_jvp_two_tensors_has_aux(self):
        counters.clear()

        def fn(x, y):
            return (x.sin().sum() + y.cos()), x

        def wrapper_fn(x, y, v):
            return torch.func.jvp(fn, (x, y), (v, v), has_aux=True)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        v = torch.randn(3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_y_: "f32[3, 3]", L_v_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_
        l_v_ = L_v_

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        aux: "f32[3, 3]" = torch._make_dual(l_x_, l_v_, level = 0);  l_x_ = None

        _maybe_load_decompositions_1 = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions_1 = None

        _make_dual_1: "f32[3, 3]" = torch._make_dual(l_y_, l_v_, level = 0);  l_y_ = l_v_ = None

        sin: "f32[3, 3]" = aux.sin()
        sum_1: "f32[]" = sin.sum();  sin = None
        cos: "f32[3, 3]" = _make_dual_1.cos();  _make_dual_1 = None
        result_duals: "f32[3, 3]" = sum_1 + cos;  sum_1 = cos = None

        aux_1: "f32[3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal: "f32[3, 3]" = _unpack_dual[0]
        dual: "f32[3, 3]" = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[3, 3]" = torch._C._functorch._unwrap_for_grad(primal, 1);  primal = None
        tangents_out_unflatten: "f32[3, 3]" = torch._C._functorch._unwrap_for_grad(dual, 1);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None
        return (primals_out_unflatten, tangents_out_unflatten, aux_1)
""",
        )

    def test_jvp_two_tensors_disable_grad(self):
        counters.clear()

        def fn(x):
            return x.sin().sum()

        def wrapper_fn(x, v):
            with torch.autograd.forward_ad._set_fwd_grad_enabled(False):
                return torch.func.jvp(fn, (x,), (v,))

        x = torch.randn(3, 3)
        v = torch.randn(3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_v_: "f32[3, 3]"):
        l_x_ = L_x_
        l_v_ = L_v_

        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled = None
        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[3, 3]" = torch._make_dual(l_x_, l_v_, level = 0);  l_x_ = l_v_ = None

        sin: "f32[3, 3]" = _make_dual.sin();  _make_dual = None
        result_duals: "f32[]" = sin.sum();  sin = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal: "f32[]" = _unpack_dual[0]
        dual: "f32[]" = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(primal, 1);  primal = None
        tangents_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(dual, 1);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_2 = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled_2 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None
        _set_fwd_grad_enabled_3 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_3 = None
        return (primals_out_unflatten, tangents_out_unflatten)
""",
        )

    def test_jvp_two_tensors_disable_enable_disable_grad(self):
        counters.clear()

        def fn(x):
            return x.sin().sum()

        def wrapper_fn(x, v):
            with torch.autograd.forward_ad._set_fwd_grad_enabled(False):  # (1)
                with torch.autograd.forward_ad._set_fwd_grad_enabled(True):  # (2)
                    with torch.autograd.forward_ad._set_fwd_grad_enabled(False):  # (3)
                        return torch.func.jvp(fn, (x,), (v,))  # (4)

            # Start True
            # False      (1)
            #   True     (2)
            #     False  (3)
            #       True (4)
            #     True   (undo 3)
            #   False    (undo 2)
            # True       (undo 1)

        x = torch.randn(3, 3)
        v = torch.randn(3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_v_: "f32[3, 3]"):
        l_x_ = L_x_
        l_v_ = L_v_

        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _set_fwd_grad_enabled_2 = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled_2 = None
        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled_3 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_3 = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[3, 3]" = torch._make_dual(l_x_, l_v_, level = 0);  l_x_ = l_v_ = None

        sin: "f32[3, 3]" = _make_dual.sin();  _make_dual = None
        result_duals: "f32[]" = sin.sum();  sin = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal: "f32[]" = _unpack_dual[0]
        dual: "f32[]" = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(primal, 1);  primal = None
        tangents_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(dual, 1);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_4 = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled_4 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None
        _set_fwd_grad_enabled_5 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_5 = None
        _set_fwd_grad_enabled_6 = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled_6 = None
        _set_fwd_grad_enabled_7 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_7 = None
        return (primals_out_unflatten, tangents_out_unflatten)
""",
        )

    def test_jvp_freevar_tensor(self):
        counters.clear()
        y = torch.randn(3, 3)

        def fn(x):
            return (x.sin() + y).sum()

        def wrapper_fn(x):
            return torch.func.jvp(fn, (x,), (x,))

        x = torch.randn(3, 3)
        expected = wrapper_fn(x)
        actual = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(actual, expected)

    def test_jvp_jvp(self):
        counters.clear()

        if check_dynamic_shape_capture():
            self.skipTest("test fails with dynamic shapes")

        def fn(x):
            return torch.func.jvp(torch.sin, (x,), (x,))

        def wrapper_fn(x):
            return torch.func.jvp(fn, (x,), (x,))

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        child: "f32[3, 3, 3]" = torch._make_dual(l_x_, l_x_, level = 0);  l_x_ = None

        _jvp_increment_nesting_1 = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting_1 = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None

        _maybe_load_decompositions_1 = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions_1 = None

        _make_dual_1: "f32[3, 3, 3]" = torch._make_dual(child, child, level = 0);  child = None

        result_duals: "f32[3, 3, 3]" = torch.sin(_make_dual_1);  _make_dual_1 = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal: "f32[3, 3, 3]" = _unpack_dual[0]
        dual: "f32[3, 3, 3]" = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(primal, 2);  primal = None
        tangents_out_unflatten: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(dual, 2);  dual = None

        _set_fwd_grad_enabled_2 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_2 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        _unpack_dual_1 = torch._unpack_dual(primals_out_unflatten, level = 0);  primals_out_unflatten = None
        primal_1: "f32[3, 3, 3]" = _unpack_dual_1[0]
        dual_1: "f32[3, 3, 3]" = _unpack_dual_1[1];  _unpack_dual_1 = None
        _unpack_dual_2 = torch._unpack_dual(tangents_out_unflatten, level = 0);  tangents_out_unflatten = None
        primal_2: "f32[3, 3, 3]" = _unpack_dual_2[0]
        dual_2: "f32[3, 3, 3]" = _unpack_dual_2[1];  _unpack_dual_2 = None

        _unwrap_for_grad_2: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(primal_1, 1);  primal_1 = None
        _unwrap_for_grad_3: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(primal_2, 1);  primal_2 = None
        _unwrap_for_grad_4: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(dual_1, 1);  dual_1 = None
        _unwrap_for_grad_5: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(dual_2, 1);  dual_2 = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_3 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_3 = None
        _jvp_decrement_nesting_1 = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting_1 = None
        return (_unwrap_for_grad_2, _unwrap_for_grad_3, _unwrap_for_grad_4, _unwrap_for_grad_5)
""",
        )

    def test_jvp_freevar_python_scalar(self):
        counters.clear()
        y = 3

        def fn(x):
            return (x.sin() + y).sum()

        def wrapper_fn(x):
            return torch.func.jvp(fn, (x,), (x,))

        x = torch.randn(3, 3, 3)
        expected = wrapper_fn(x)
        actual = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(actual, expected)

    def test_linearize_jvp_fn(self):
        counters.clear()

        def wrapper_fn(x):
            output, jvp_fn = torch.func.linearize(torch.sin, x)
            return output, jvp_fn(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,), fullgraph=False, graph_idx=0)

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_self_buffers_tensor_constant0_: "f32[3, 3, 3]"):
        l_self_buffers_tensor_constant0_ = L_self_buffers_tensor_constant0_

        alias_default: "f32[3, 3, 3]" = torch.ops.aten.alias.default(l_self_buffers_tensor_constant0_);  l_self_buffers_tensor_constant0_ = None

        sin_default: "f32[3, 3, 3]" = torch.ops.aten.sin.default(alias_default)

        alias_default_1: "f32[3, 3, 3]" = torch.ops.aten.alias.default(alias_default)

        cos_default: "f32[3, 3, 3]" = torch.ops.aten.cos.default(alias_default_1);  alias_default_1 = None

        alias_default_2: "f32[3, 3, 3]" = torch.ops.aten.alias.default(sin_default);  alias_default_2 = None
        return (alias_default, cos_default, sin_default)
""",
        )

        wrapped_gm = self._compile_check(wrapper_fn, (x,), fullgraph=False, graph_idx=1)
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_self_modules_FX_CONST_FOLDED_ATTRS_parameters_0_: "f32[3, 3, 3]", L_self_modules_FX_CONST_FOLDED_ATTRS_parameters_1_: "f32[3, 3, 3]", L_flat_tangents_1_: "f32[3, 3, 3]"):
        l_self_modules_fx_const_folded_attrs_parameters_0_ = L_self_modules_FX_CONST_FOLDED_ATTRS_parameters_0_
        l_self_modules_fx_const_folded_attrs_parameters_1_ = L_self_modules_FX_CONST_FOLDED_ATTRS_parameters_1_
        l_flat_tangents_1_ = L_flat_tangents_1_

        _new_zeros_with_same_feature_meta_default: "f32[3, 3, 3]" = torch.ops.aten._new_zeros_with_same_feature_meta.default(l_flat_tangents_1_, l_self_modules_fx_const_folded_attrs_parameters_0_);  l_self_modules_fx_const_folded_attrs_parameters_0_ = None

        copy__default: "f32[3, 3, 3]" = torch.ops.aten.copy_.default(_new_zeros_with_same_feature_meta_default, l_flat_tangents_1_);  _new_zeros_with_same_feature_meta_default = l_flat_tangents_1_ = None

        mul_tensor: "f32[3, 3, 3]" = torch.ops.aten.mul.Tensor(copy__default, l_self_modules_fx_const_folded_attrs_parameters_1_);  copy__default = l_self_modules_fx_const_folded_attrs_parameters_1_ = None
        return (mul_tensor,)
""",
        )

    @config.patch(error_on_recompile=True)
    def test_vmap_recompile(self):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        torch.vmap(fn)(x)
        # should not recompile on second call. See Pytorch issue #118493
        torch.vmap(fn)(x)

    @xfailIfTorchDynamo
    @config.patch(error_on_recompile=True)
    def test_vmap_recompile_different_config(self):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        torch.vmap(fn)(x)
        with self.assertRaises(torch._dynamo.exc.RecompileError):
            fn(x)

    @config.patch(error_on_recompile=True)
    def test_vmap_recompile_same_config(self):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        torch.vmap(torch.vmap(fn, randomness="same"), randomness="same")(x)
        with self.assertRaises(torch._dynamo.exc.RecompileError):
            torch.vmap(torch.vmap(fn, randomness="same"), randomness="error")(x)

    @config.patch(error_on_recompile=True)
    def test_vmap_recompile_with_randomness(self):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        torch.vmap(fn, randomness="same")(x)
        with self.assertRaises(torch._dynamo.exc.RecompileError):
            torch.vmap(fn, randomness="different")(x)

    def test_vmap_call_torch_compile_fn(self):
        def wrapped_fn(x):
            return x.sin()

        x = torch.randn(3, 4)
        fn = torch.compile(backend="aot_eager", fullgraph=True)(wrapped_fn)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Calling torch.func.vmap\\(compiled_fn\\) function from eager mode is not supported",
        ):
            torch.func.vmap(fn)(x)

    def test_vmap_call_compiled_backward_fn(self):
        # See PyTorch issue #138422
        @torch.compile(backend="aot_eager")
        def f(x):
            return x**2

        x = torch.randn(2, requires_grad=True)
        y = f(x)

        def get_vjp(v):
            return torch.autograd.grad(y, x, v)

        with self.assertRaisesRegex(
            RuntimeError,
            "It looks like you're trying to call a compiled backward function within vmap/grad/vjp, which isn't supported",
        ):
            torch.func.vjp(get_vjp, x)

    def test_vjp_call_compiled_backward_fn(self):
        # See PyTorch issue #138422
        @torch.compile(backend="aot_eager")
        def f(x):
            return x**2

        x = torch.randn(2, requires_grad=True)
        y = f(x)

        def get_vjp(v):
            return torch.autograd.grad(y, x, v)

        with self.assertRaisesRegex(
            RuntimeError,
            "It looks like you're trying to call a compiled backward function within vmap/grad/vjp, which isn't supported",
        ):
            torch.func.vjp(get_vjp, x)

    def test_grad_call_compiled_backward_fn(self):
        # See PyTorch issue #138422
        @torch.compile(backend="aot_eager")
        def f(x):
            return x**2

        x = torch.randn(2, requires_grad=True)
        y = f(x)

        def get_vjp(v):
            return torch.autograd.grad(y, x, v)

        with self.assertRaisesRegex(
            RuntimeError,
            "It looks like you're trying to call a compiled backward function within vmap/grad/vjp, which isn't supported",
        ):
            torch.func.grad(get_vjp)(x)

    def test_grad_call_torch_compile_fn(self):
        def wrapped_fn(x):
            return x.sin().sum()

        x = torch.randn(3, 4)
        fn = torch.compile(backend="aot_eager", fullgraph=True)(wrapped_fn)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Calling torch.func.grad\\(compiled_fn\\) function from eager mode is not supported",
        ):
            torch.func.grad(fn)(x)

    def test_jvp_call_torch_compile_fn(self):
        def wrapped_fn(x):
            return x.sin().sum()

        x = torch.randn(3, 4)
        fn = torch.compile(backend="aot_eager", fullgraph=True)(wrapped_fn)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Calling torch.func.jvp\\(compiled_fn\\) function from eager mode is not supported",
        ):
            torch.func.jvp(fn, (x,), (x,))

    @config.patch(error_on_recompile=True)
    def test_grad_recompile(self):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.func.grad(torch.sin)(x)

        x = torch.randn([])
        torch.func.grad(fn)(x)
        # should not recompile on second call
        torch.func.grad(fn)(x)

    def test_vmap_get_wrapped(self):
        counters.clear()

        def g(x):
            return x.sin()

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn():
            return torch.vmap(g)

        x = torch.randn(3, 4)
        expected = torch.vmap(g)(x)
        wrapper = fn()
        got = wrapper(x)
        self.assertEqual(expected, got)

    def test_vmap_with_conditional_graph_break(self):
        def g(x):
            if len(x.shape) < 2:
                torch._dynamo.graph_break()
                return x.sin()
            else:
                return x.cos()

        @torch.compile(backend="aot_eager")
        def fn(x):
            return torch.vmap(g)(x)

        counters.clear()
        x = torch.randn(2, 3)
        expected = x.sin()
        got = fn(x)
        self.assertEqual(expected, got)
        self.assertEqual(len(counters["graph_break"]), 1)

        counters.clear()
        y = torch.randn(2, 3, 4)
        expected = y.cos()
        got = fn(y)
        self.assertEqual(expected, got)
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_vmap_with_graph_break(self):
        counters.clear()

        def g(x):
            y = x.cos()
            print("hi")
            return y.sin()

        def fn(x):
            return torch.vmap(g)(x)

        x = torch.randn(3, 4)
        opt = torch.compile(fn, backend="aot_eager", fullgraph=False)
        expected = fn(x)
        got = opt(x)
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(expected, got)

    def test_vmap_with_graph_break_2(self):
        counters.clear()

        def cos(x):
            print("cos")
            return x.cos()

        def sin(x):
            print("sin")
            return x.sin()

        def g(x):
            y = cos(x)
            return sin(y)

        def fn(x):
            return torch.vmap(g, randomness="same")(x)

        x = torch.randn(3, 4)
        opt = torch.compile(fn, backend="aot_eager", fullgraph=False)
        expected = fn(x)
        got = opt(x)
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(expected, got)

    def test_vmap_with_graph_break_lambda(self):
        counters.clear()

        def sin(x):
            print("sin")
            return x.sin()

        def fn(x):
            return torch.vmap(lambda x: sin(x))(x)

        x = torch.randn(3, 4)
        opt = torch.compile(fn, backend="aot_eager", fullgraph=False)
        expected = fn(x)
        got = opt(x)
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(expected, got)

    def test_vmap_scalar_tensor_indexing(self):
        data = torch.arange(20).reshape(2, 10)
        b_indices = torch.arange(2)
        n_indices = torch.arange(10)

        def vmap_index_fn(data_in, b_indices, n_indices):
            def index_fn(b, n):
                return data_in[b, n]

            return torch.func.vmap(index_fn, in_dims=(None, 0))(b_indices, n_indices)

        eager_result = vmap_index_fn(data, b_indices, n_indices)

        compiled_result = torch.compile(vmap_index_fn, backend="eager", fullgraph=True)(
            data, b_indices, n_indices
        )

        self.assertEqual(eager_result, compiled_result)

    def test_vmap(self):
        def fn(x):
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1))(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        sum_1: "f32[3]" = _add_batch_dim.sum(0)
        sum_2: "f32[3]" = _add_batch_dim.sum(1);  _add_batch_dim = None
        batched_outputs: "f32[3]" = sum_1 + sum_2;  sum_1 = sum_2 = None

        _remove_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._remove_batch_dim(batched_outputs, 1, 3, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim,)
""",
        )

    def test_vmap_free_const(self):
        y = 3

        def fn(x):
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1) + y)(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        sum_1: "f32[3]" = _add_batch_dim.sum(0)
        sum_2: "f32[3]" = _add_batch_dim.sum(1);  _add_batch_dim = None
        add: "f32[3]" = sum_1 + sum_2;  sum_1 = sum_2 = None
        batched_outputs: "f32[3]" = add + 3;  add = None

        _remove_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._remove_batch_dim(batched_outputs, 1, 3, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim,)
""",
        )

    def test_vmap_free_tensor(self):
        y = torch.randn(3, 3)

        def fn(x):
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1) + y)(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_y_: "f32[3, 3]", L_x_: "f32[3, 3, 3]"):
        l_y_ = L_y_
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        sum_1: "f32[3]" = _add_batch_dim.sum(0)
        sum_2: "f32[3]" = _add_batch_dim.sum(1);  _add_batch_dim = None
        add: "f32[3]" = sum_1 + sum_2;  sum_1 = sum_2 = None
        batched_outputs: "f32[3, 3]" = add + l_y_;  add = l_y_ = None

        _remove_batch_dim: "f32[3, 3, 3]" = torch._functorch.predispatch._remove_batch_dim(batched_outputs, 1, 3, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim,)
""",
        )

    def test_vmap_two_inputs(self):
        def fn(x, y):
            return torch.func.vmap(
                lambda x, y: x.sum(0) + x.sum(1) + y, in_dims=(0, 1)
            )(x, y)

        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3)
        wrapped_gm = self._compile_check(fn, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None
        _add_batch_dim_1: "f32[3]" = torch._functorch.predispatch._add_batch_dim(l_y_, 1, 1);  l_y_ = None

        sum_1: "f32[3]" = _add_batch_dim.sum(0)
        sum_2: "f32[3]" = _add_batch_dim.sum(1);  _add_batch_dim = None
        add: "f32[3]" = sum_1 + sum_2;  sum_1 = sum_2 = None
        batched_outputs: "f32[3]" = add + _add_batch_dim_1;  add = _add_batch_dim_1 = None

        _remove_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._remove_batch_dim(batched_outputs, 1, 3, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim,)
""",
        )

    def test_vmap_two_inputs_tuple_in_dims(self):
        in_dims = (0, 1)

        def fn(x, y):
            return torch.func.vmap(
                lambda x, y: x.sum(0) + x.sum(1) + y, in_dims=in_dims
            )(x, y)

        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3)
        wrapped_gm = self._compile_check(fn, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None
        _add_batch_dim_1: "f32[3]" = torch._functorch.predispatch._add_batch_dim(l_y_, 1, 1);  l_y_ = None

        sum_1: "f32[3]" = _add_batch_dim.sum(0)
        sum_2: "f32[3]" = _add_batch_dim.sum(1);  _add_batch_dim = None
        add: "f32[3]" = sum_1 + sum_2;  sum_1 = sum_2 = None
        batched_outputs: "f32[3]" = add + _add_batch_dim_1;  add = _add_batch_dim_1 = None

        _remove_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._remove_batch_dim(batched_outputs, 1, 3, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim,)
""",
        )

    def test_vmap_over_vmap_two_inputs(self):
        def fn(x, y):
            return torch.func.vmap(torch.func.vmap(lambda x, y: x + y, in_dims=1))(x, y)

        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        child: "f32[3, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None
        child_1: "f32[3, 3]" = torch._functorch.predispatch._add_batch_dim(l_y_, 0, 1);  l_y_ = None

        lazy_load_decompositions_1 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_1 = None

        _vmap_increment_nesting_1 = torch._functorch.predispatch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting_1 = None

        _add_batch_dim_2: "f32[3]" = torch._functorch.predispatch._add_batch_dim(child, 1, 2);  child = None
        _add_batch_dim_3: "f32[3]" = torch._functorch.predispatch._add_batch_dim(child_1, 1, 2);  child_1 = None

        batched_outputs: "f32[3]" = _add_batch_dim_2 + _add_batch_dim_3;  _add_batch_dim_2 = _add_batch_dim_3 = None

        batched_outputs_1: "f32[3, 3]" = torch._functorch.predispatch._remove_batch_dim(batched_outputs, 2, 3, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        _remove_batch_dim_1: "f32[3, 3, 3]" = torch._functorch.predispatch._remove_batch_dim(batched_outputs_1, 1, 3, 0);  batched_outputs_1 = None

        _vmap_decrement_nesting_1 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None
        return (_remove_batch_dim_1,)
""",
        )

    def test_vmap_over_vmap_captured(self):
        x = torch.ones(2, 3)
        y = torch.ones(5, 3)

        def fn(x):
            return torch.func.vmap(torch.func.vmap(lambda y: x * y))(y)

        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 3]", L_y_: "f32[5, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(5, 'error');  _vmap_increment_nesting = None

        child: "f32[3]" = torch._functorch.predispatch._add_batch_dim(l_y_, 0, 1);  l_y_ = None

        lazy_load_decompositions_1 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_1 = None

        _vmap_increment_nesting_1 = torch._functorch.predispatch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting_1 = None

        _add_batch_dim_1: "f32[]" = torch._functorch.predispatch._add_batch_dim(child, 0, 2);  child = None

        batched_outputs: "f32[2, 3]" = l_x_ * _add_batch_dim_1;  l_x_ = _add_batch_dim_1 = None

        batched_outputs_1: "f32[3, 2, 3]" = torch._functorch.predispatch._remove_batch_dim(batched_outputs, 2, 3, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        _remove_batch_dim_1: "f32[5, 3, 2, 3]" = torch._functorch.predispatch._remove_batch_dim(batched_outputs_1, 1, 5, 0);  batched_outputs_1 = None

        _vmap_decrement_nesting_1 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None
        return (_remove_batch_dim_1,)
""",
        )

    def test_vmap_multiple_outputs(self):
        x = torch.ones(2, 4, 3)

        def fn(x):
            return torch.vmap(lambda x: (x.sum(0), x.sum(1)))(x)

        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 4, 3]"):
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(2, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        child: "f32[3]" = _add_batch_dim.sum(0)
        child_1: "f32[4]" = _add_batch_dim.sum(1);  _add_batch_dim = None

        _remove_batch_dim: "f32[2, 3]" = torch._functorch.predispatch._remove_batch_dim(child, 1, 2, 0);  child = None
        _remove_batch_dim_1: "f32[2, 4]" = torch._functorch.predispatch._remove_batch_dim(child_1, 1, 2, 0);  child_1 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim, _remove_batch_dim_1)
""",
        )

    def test_vmap_multiple_outputs_diff_dims(self):
        x = torch.ones(2, 4, 3)

        def fn(x):
            return torch.vmap(lambda x: (x.sum(0), x.sum(1)), out_dims=(1, 0))(x)

        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 4, 3]"):
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(2, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        child: "f32[3]" = _add_batch_dim.sum(0)
        child_1: "f32[4]" = _add_batch_dim.sum(1);  _add_batch_dim = None

        _remove_batch_dim: "f32[3, 2]" = torch._functorch.predispatch._remove_batch_dim(child, 1, 2, 1);  child = None
        _remove_batch_dim_1: "f32[2, 4]" = torch._functorch.predispatch._remove_batch_dim(child_1, 1, 2, 0);  child_1 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim, _remove_batch_dim_1)
""",
        )

    def test_vmap_multiple_outputs_out_dims_tuple(self):
        x = torch.ones(2, 4, 3)
        out_dims = (1, 0)

        def fn(x):
            return torch.vmap(lambda x: (x.sum(0), x.sum(1)), out_dims=out_dims)(x)

        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 4, 3]"):
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(2, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        child: "f32[3]" = _add_batch_dim.sum(0)
        child_1: "f32[4]" = _add_batch_dim.sum(1);  _add_batch_dim = None

        _remove_batch_dim: "f32[3, 2]" = torch._functorch.predispatch._remove_batch_dim(child, 1, 2, 1);  child = None
        _remove_batch_dim_1: "f32[2, 4]" = torch._functorch.predispatch._remove_batch_dim(child_1, 1, 2, 0);  child_1 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim, _remove_batch_dim_1)
""",
        )

    def test_vmap_kwargs(self):
        counters.clear()
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        def fn(x, y):
            return torch.func.vmap(lambda x, y: x + y)(x, y=y)

        actual = fn(x, y)
        expected = torch.compile(fn, backend="aot_eager", fullgraph=False)(x, y)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_vmap_pytree_inputs(self):
        counters.clear()
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        def vmap_fn(inps):
            x = inps["x"]
            y = inps["y"]
            return x + y

        def fn(x, y):
            return torch.func.vmap(vmap_fn)({"x": x, "y": y})

        actual = fn(x, y)
        expected = torch.compile(fn, backend="aot_eager", fullgraph=False)(x, y)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_vmap_side_effects(self):
        counters.clear()
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        some_list = []

        def f(x, y):
            some_list.append(1)
            return x + y

        def wrapper_fn(x, y):
            return torch.func.vmap(f)(x, y)

        actual = wrapper_fn(x, y)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x, y)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)
        self.assertEqual(some_list, [1, 1])

    @unittest.expectedFailure
    def test_vmap_side_effects_append_input(self):
        counters.clear()
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        some_list = []

        def f(x, y):
            some_list.append(x)
            return x + y

        def wrapper_fn(x, y):
            return torch.func.vmap(f)(x, y)

        actual = wrapper_fn(x, y)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x, y)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_vmap_previous_illegal_op_no_graph_break(self):
        counters.clear()

        # calling .stride() would previously graph break
        def bad_fn(x):
            y = x.view((4, 3))
            y.stride()
            return y

        def wrapper_fn(x):
            return torch.func.vmap(bad_fn)(x)

        x = torch.randn(2, 3, 4)
        actual = wrapper_fn(x)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_vmap_multiple_invocation_in_dims(self):
        counters.clear()

        def wrapper_fn(x, in_dims):
            return torch.func.vmap(torch.sum, in_dims)(x)

        x = torch.randn(3, 3, 3, 3)
        cnt = CompileCounter()
        opt = torch.compile(wrapper_fn, backend=cnt, fullgraph=False, dynamic=True)
        expected = wrapper_fn(x, 0), wrapper_fn(x, 1), wrapper_fn(x, 2)
        # Third invocation of `opt` makes `in_dims` as SymInt.
        actual = opt(x, 0), opt(x, 1), opt(x, 2)
        self.assertEqual(expected, actual)
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 18)

    def test_vmap_multiple_invocation_out_dims(self):
        counters.clear()

        def wrapper_fn(x, out_dims):
            return torch.func.vmap(lambda x: torch.sum(x, 0), out_dims=out_dims)(x)

        x = torch.randn(3, 3, 3, 3)
        cnt = CompileCounter()
        opt = torch.compile(wrapper_fn, backend=cnt, fullgraph=False, dynamic=True)
        expected = wrapper_fn(x, 0), wrapper_fn(x, 1), wrapper_fn(x, 2)
        # Third invocation of `opt` makes `in_dims` as SymInt.
        actual = opt(x, 0), opt(x, 1), opt(x, 2)
        self.assertEqual(expected, actual)
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 18)

    def test_vmap_out_dims_None(self):
        # issue https://github.com/pytorch/pytorch/issues/149509
        def fn(x, y):
            return x, y * 2

        def wrapper_fn(x, y):
            return torch.func.vmap(fn, in_dims=(None, 0), out_dims=(None, 0))(x, y)

        x, y = torch.randn(4), torch.randn(3, 4)
        expected = wrapper_fn(x, y)
        got = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)(x, y)
        self.assertEqual(expected, got)

    def test_vmap_new_tensor_in_body(self):
        def fn(x):
            return x + torch.ones(3)

        def wrapper_fn(x):
            return torch.func.vmap(fn)(x)

        x = torch.randn(
            3,
        )
        opt = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)
        expected = wrapper_fn(x)
        actual = opt(x)
        self.assertEqual(expected, actual)

    def test_vmap_new_tensor_unused_in_body(self):
        def fn(x):
            return torch.tensor(0.5)

        def wrapper_fn(x):
            return torch.func.vmap(fn)(x)

        x = torch.randn(3)
        opt = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)
        expected = wrapper_fn(x)
        actual = opt(x)
        self.assertEqual(expected, actual)

    def test_vmap_new_tensor_implicit_via_op(self):
        def wrapper_fn(x):
            return torch.func.vmap(lambda t: torch.add(t, 0.5))(x)

        x = torch.randn(3)
        opt = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)
        expected = wrapper_fn(x)
        actual = opt(x)
        self.assertEqual(expected, actual)
