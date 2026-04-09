try:
    from ._repros_common import (
        _GLOBAL_CPU_TENSOR,
        Any,
        collections,
        CompileCounter,
        dataclasses,
        deepcopy,
        EagerAndRecordGraphs,
        Enum,
        fresh_cache,
        functools,
        gc,
        global_fn,  # noqa: F401
        IncByOne,
        IncByTwo,
        IntEnum,
        itertools,
        maybe,
        mock,
        namedtuple,
        nn,
        np,
        parametrize,
        pytree,
        requires_cuda,
        same,
        serialTest,
        skipIfWindows,
        torch,
        TwoTensor,
        typing,
        unittest,
        warnings,
        weakref,
        XSoftmax,
    )
except ImportError:
    from _repros_common import (
        _GLOBAL_CPU_TENSOR,
        Any,
        collections,
        CompileCounter,
        dataclasses,
        deepcopy,
        EagerAndRecordGraphs,
        Enum,
        fresh_cache,
        functools,
        gc,
        global_fn,  # noqa: F401
        IncByOne,
        IncByTwo,
        IntEnum,
        itertools,
        maybe,
        mock,
        namedtuple,
        nn,
        np,
        parametrize,
        pytree,
        requires_cuda,
        same,
        serialTest,
        skipIfWindows,
        torch,
        TwoTensor,
        typing,
        unittest,
        warnings,
        weakref,
        XSoftmax,
    )


_GLOBAL_FN_SENTINEL = global_fn


class ReproTestsMixin2:
    def test_batchnorm_e2e(self):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(
                    64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
                )
                self.conv1 = torch.nn.Conv2d(
                    64,
                    64,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    bias=False,
                )

            def forward(self, x):
                x1 = self.bn(x)
                x2 = self.conv1(x1)
                out = torch.nn.functional.relu(x2)
                return (out,)

        torch.manual_seed(1337)

        m_ref = Repro()
        m_test = deepcopy(m_ref)

        @torch.compile(backend="aot_eager_decomp_partition")
        def compiled_fn(x):
            return m_test(x)

        x_ref = torch.randn(2, 64, 32, 32, requires_grad=True)
        x_test = x_ref.clone()

        # Loop multiple times: each iteration the running_mean/var on batchnorm will update,
        # which changes the output of the next iteration
        for _ in range(3):
            ref = m_ref(x_ref)
            res = compiled_fn(x_test)

            self.assertTrue(same(ref, res))

            for r in ref:
                if r.requires_grad:
                    r.sum().backward()
            for r in res:
                if r.requires_grad:
                    r.sum().backward()

            for param_ref, param_test in zip(m_ref.parameters(), m_test.parameters()):
                self.assertTrue(same(param_ref, param_test))
            # Assert running_mean/var
            for buffer_ref, buffer_test in zip(m_ref.buffers(), m_test.buffers()):
                self.assertTrue(same(buffer_ref, buffer_test))

    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_dynamic_shapes_right_side(self):
        def f(x):
            return torch.ones(5 * x.shape[0])

        inp = torch.randn(6, 5)

        gm, _ = torch._dynamo.export(f, aten_graph=True)(torch.randn(4, 5))
        self.assertEqual(gm(inp).shape, f(inp).shape)

    @torch._dynamo.config.patch("specialize_int", False)
    def test_maybe_multiply_symint(self):
        # https://github.com/pytorch/pytorch/issues/97346
        from torch._functorch.aot_autograd import aot_module_simplified

        def my_aot_compiler(gm, example_inputs):
            def my_compiler(gm, example_inputs):
                return gm.forward

            # Invoke AOTAutograd
            return aot_module_simplified(gm, example_inputs, fw_compiler=my_compiler)

        def my_example(t1, t2, d):
            out = torch.add(t1, t2, alpha=d)
            return out

        compiled_fn = torch.compile(backend=my_aot_compiler, dynamic=True)(my_example)

        t1 = torch.arange(3, dtype=torch.float32).requires_grad_(True)
        t2 = torch.arange(3, dtype=torch.float32).requires_grad_(True)

        ra = compiled_fn(t1, t2, 5)
        self.assertEqual(ra, torch.tensor([0.0, 6.0, 12.0]))

        ra = compiled_fn(t1, t2, 6)
        self.assertEqual(ra, torch.tensor([0.0, 7.0, 14.0]))

    def test_build_map_unpack_with_call(self):
        def forward_with_cond_scale(x, t, cond_scale, self_cond, other1, other2):
            return x.sin() + t + cond_scale + self_cond + other1 + other2

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            d1 = dict(other1=5)
            d2 = dict(other2=4)
            text_cond = {**d1, **d2}
            return forward_with_cond_scale(x, 1, cond_scale=2, self_cond=3, **text_cond)

        self.assertTrue(same(fn(torch.ones(4)), torch.ones(4).sin() + 15))

    @torch._dynamo.config.patch(verbose=True)
    def test_graph_break_unsupported_fake(self):
        counter = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=counter)
        def f(x):
            return torch.ops.test_sample.foo(x + 1) + 1

        f(torch.randn(3, device="cpu"))

        self.assertEqual(counter.op_count, 2)
        self.assertEqual(counter.frame_count, 2)

    def test_delattr(self):
        class MyObj:
            def __init__(self, a, b):
                self.a = a
                self.b = b

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, obj):
            del obj.a
            obj.c = x + 1
            del obj.c
            tmp = MyObj(x + 2, x + 3)
            del tmp.b
            if hasattr(obj, "a"):
                return x + 1
            return tmp

        x = torch.zeros([])
        obj1 = MyObj(x, x)
        obj2 = fn(x, obj1)
        self.assertFalse(hasattr(obj1, "a"))
        self.assertFalse(hasattr(obj1, "c"))
        self.assertFalse(hasattr(obj2, "b"))
        self.assertEqual(obj1.b.item(), 0)
        self.assertEqual(obj2.a.item(), 2)

    def test_delattr_return(self):
        class MyObject:
            def __init__(self, val):
                self.val = val
                self.deletion_attempted = False

            def __delattr__(self, attr):
                if attr == "val":
                    self.deletion_attempted = True
                else:
                    super().__delattr__(attr)

        @torch.compile(fullgraph=True, backend="eager")
        def test_delattr(input_tensor):
            instance_a = MyObject(1)
            instance_b = MyObject(2)
            del instance_a.val
            del instance_b.val
            exists_a = hasattr(instance_a, "val")
            exists_b = hasattr(instance_b, "val")
            deletion_attempted_a = instance_a.deletion_attempted
            deletion_attempted_b = instance_b.deletion_attempted
            return (
                input_tensor + 1,
                exists_a,
                exists_b,
                deletion_attempted_a,
                deletion_attempted_b,
            )

        result = test_delattr(torch.ones(1))
        self.assertEqual(result[0], torch.tensor([2.0]))
        self.assertEqual(result[1:], (True, True, True, True))

    def test_delattr_raises(self):
        class MyObj:
            def __init__(self, a, b):
                self.a = a
                self.b = b

        @torch.compile(backend="eager")
        def fn(x, obj):
            del obj.a
            x = x + 1
            obj.a  # will raise
            return x

        x = torch.zeros([])
        obj1 = MyObj(x, x)
        self.assertRaises(AttributeError, lambda: fn(x, obj1))

    def test_delsubscr(self):
        @torch.compile(backend="eager")
        def fn(x):
            del x["a"]
            y = x["b"] + 1
            return y

        x = {"a": torch.tensor([1]), "b": torch.tensor([1])}
        result = fn(x)
        self.assertFalse(hasattr(x, "a"))
        self.assertEqual(result.item(), 2)

    def test_delsubscr_raises(self):
        @torch.compile(backend="eager")
        def fn(x):
            del x["a"]
            y = x["a"] + 1  # should raise KeyError
            return y

        x = {"a": torch.tensor([1]), "b": torch.tensor([1])}
        self.assertRaises(KeyError, lambda: fn(x))

    def test_attached_attribute_in_dir(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear(x))

        mod = torch.compile(MyModule(), backend="eager")
        mod.is_compiled = True
        self.assertTrue("is_compiled" in dir(mod))

    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    def test_dynamic_shapes_implicit_guard(self):
        def f(x):
            y = x * x.size(x.shape[0])
            torch.sum(y, [y.shape[0]])
            return y

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(f, backend=cnt, fullgraph=True)
        opt_fn(torch.randn(3, 1, 1, 1, 1))
        self.assertEqual(cnt.frame_count, 1)

    def test_dalle2_maybe(self):
        def normalize(x):
            return x.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, normalize_img):
            lowres_cond_img = x.sin()
            lowres_cond_img = maybe(normalize_img)(lowres_cond_img)
            return lowres_cond_img

        self.assertEqual(fn(torch.ones([]), normalize), torch.ones([]).sin().cos())

    def test_functools_wraps(self):
        def cool_name(x):
            return x.sin()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = x.cos()

            @functools.wraps(cool_name)
            def uncool_name():
                return cool_name(y)

            return uncool_name

        result = fn(torch.ones([]))
        self.assertEqual(result.__name__, "cool_name")
        self.assertEqual(result(), torch.ones([]).cos().sin())

    def test_dynamic_shapes_float_guard(self):
        def f(x):
            return torch.nn.functional.dropout(x, x.shape[0] / 6)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(f, backend=cnt, fullgraph=True)
        opt_fn(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tensor_item(self):
        def f(x, y):
            val = y.item()
            return x.sum() + val

        gm, _ = torch._dynamo.export(
            f,
            aten_graph=True,
        )(
            torch.zeros(6, 4),
            torch.tensor(1),
        )
        self.assertEqual(
            f(torch.zeros(6, 4), torch.tensor(1)),
            gm(torch.zeros(6, 4), torch.tensor(1)),
        )
        self.assertEqual(
            f(torch.zeros(6, 4), torch.tensor(2)),
            gm(torch.zeros(6, 4), torch.tensor(2)),
        )

    def test_dataclass_init_with_default_factory_with_inputs(self):
        @dataclasses.dataclass
        class DClass:
            sharding_contexts: Any = dataclasses.field(default_factory=list)
            a: int = 1

        def fn(x, inp_list):
            d = DClass(inp_list)
            d.sharding_contexts.append(x.sin() + d.a)
            return d

        x = torch.randn(4)
        inp_list1 = [1, 2, 3]
        inp_list2 = [2, 3, 4]
        inp_list3 = [1, 2]
        ref1 = fn(x, inp_list1)
        ref2 = fn(x, inp_list2)
        ref3 = fn(x, inp_list3)

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")

        opt_ret1 = opt_fn(x, inp_list1)
        opt_ret2 = opt_fn(x, inp_list2)
        opt_ret3 = opt_fn(x, inp_list3)
        self.assertEqual(ref1.sharding_contexts, opt_ret1.sharding_contexts)
        self.assertEqual(ref2.sharding_contexts, opt_ret2.sharding_contexts)
        self.assertEqual(ref3.sharding_contexts, opt_ret3.sharding_contexts)

    def test_list_index(self):
        for i, list_type in enumerate(
            (
                list,
                tuple,
                torch.Size,
                collections.deque,
                namedtuple("FourElems", "one two three four", defaults=[0, 0, 0, 0]),
            )
        ):
            torch._dynamo.reset()
            for index in ([], [2], [0, 3]):

                def f(t):
                    if i == 4:  # namedtuple
                        xs = list_type(1, 2, 3, 4)
                    else:
                        xs = list_type([1, 2, 3, 4])
                    res = xs.index(3, *index)
                    return t + res

                res = torch.compile(f, backend="eager", fullgraph=True)(torch.zeros(1))

                self.assertEqual(res, torch.tensor([2.0]))

    def test_list_index_not_found(self):
        def f(t):
            xs = ["bar", "foo", "baz", "buzz"]
            res = xs.index("non-existent")
            return t + res

        # Raising ValueError from item not found is unsupported
        with self.assertRaises(
            torch._dynamo.exc.Unsupported,
        ):
            torch.compile(f, backend="eager", fullgraph=True)(torch.zeros(1))

    def test_list_index_tensor_unsupported(self):
        for index in ([], [2], [0, 3]):

            def f(t):
                xs = [torch.tensor([i]) for i in range(4)]
                res = xs.index(torch.tensor([2]), *index)
                return t + res

            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "Data-dependent branching",
            ):
                torch.compile(f, backend="eager", fullgraph=True)(torch.zeros(1))

    def test_hf_xsoftmax_inference(self):
        def fn(input, mask):
            return XSoftmax.apply(input + 1, mask, 1) + 2

        fn_opt = torch.compile(fn, backend="eager", fullgraph=True)

        inputs = [
            torch.randn(4, 10),
            torch.randn(4, 10) < 0,
        ]
        expected = fn(*inputs)
        actual = fn_opt(*inputs)
        self.assertTrue(same(actual, expected))

    @mock.patch("torch._dynamo.config.guard_nn_modules", True)
    def test_hf_xsoftmax_training(self):
        from torch._dynamo.utils import counters

        counters.clear()

        def fn(input, mask):
            return XSoftmax.apply(input, mask, 1)

        cnt = torch._dynamo.testing.CompileCounter()
        fn_opt = torch.compile(fn, backend=cnt, fullgraph=False)

        torch.manual_seed(1234)
        inputs1 = [
            torch.randn(4, 10, requires_grad=True),
            torch.randn(4, 10) < 0,
        ]
        torch.manual_seed(1234)
        inputs2 = [
            torch.randn(4, 10, requires_grad=True),
            torch.randn(4, 10) < 0,
        ]

        expected = fn(*inputs1)
        actual = fn_opt(*inputs2)
        self.assertTrue(same(actual, expected))
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(cnt.frame_count, 1)
        cnt.clear()
        counters.clear()

        expected.sum().backward()
        actual.sum().backward()
        self.assertTrue(same(inputs1[0].grad, inputs2[0].grad))

        # currently we don't capture the backwards frame
        self.assertEqual(cnt.frame_count, 0)
        self.assertEqual(cnt.op_count, 0)
        self.assertEqual(dict(counters["frames"]), {})
        self.assertEqual(dict(counters["graph_break"]), {})

    def test_autograd_function_graph_break(self):
        class MySin(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                torch._dynamo.graph_break()
                ctx.save_for_backward(x)
                return x.sin()

            @staticmethod
            def backward(ctx, gx):
                (x,) = ctx.saved_tensors
                return gx * x.cos()

        x = torch.randn([], requires_grad=True)

        @torch.compile(backend="eager")
        def fn(x):
            return MySin.apply(x)

        y = fn(x)
        self.assertEqual(y, x.sin())

        (gx,) = torch.autograd.grad(y, x)
        self.assertEqual(gx, x.cos())

    def test_jit_trace_errors(self):
        @torch.compile(backend="eager", dynamic=True)
        def f(x):
            return x + 1

        with self.assertRaises(RuntimeError):
            torch.jit.trace(f, torch.randn(3))

    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_tensor_split(self):
        def f(x):
            return torch.split(x, x.shape[0] // 2, dim=0)[0]

        gm, _ = torch._dynamo.export(
            f,
            aten_graph=True,
        )(
            torch.zeros(6, 4),
        )

        self.assertEqual(f(torch.ones(8, 4)), gm(torch.ones(8, 4)))

    @skipIfWindows(
        msg="TODO: (xuhancn) fix, AssertionError: tensor([[0.1000, 0.1000, 0.1000,  ..., 0.1000, 0.1000, 0.1000],"
    )
    def test_optim_state_references_cleared(self):
        model = torch.nn.Linear(2048, 2048, bias=False)
        x = torch.ones(2048)
        state_ref = 0

        optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)

        def opt_step():
            optimizer.step()

        compiled_opt_step = torch.compile(opt_step, backend="eager")

        def compiled_model_step(x):
            optimizer.zero_grad()
            y = model(x)
            torch.sum(y).backward()
            compiled_opt_step()

        compiled_model_step(x)

        # Picked "square_avg" arbitrarily to check that
        # optimizer state tensors are deallocated
        state_ref = weakref.ref(
            optimizer.state[optimizer.param_groups[0]["params"][0]]["square_avg"]
        )
        optimizer = None

        self.assertIsNone(state_ref())

    def test_grad_references_cleared(self):
        model = torch.nn.Linear(2048, 2048, bias=False)
        x = torch.ones(2048)
        optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)

        def opt_step():
            optimizer.step()

        compiled_opt_step = torch.compile(opt_step, backend="eager")

        def compiled_model_step(x):
            optimizer.zero_grad(True)
            y = model(x)
            torch.sum(y).backward()
            compiled_opt_step()

        compiled_model_step(x)
        param_grad_ref = weakref.ref(next(iter(model.parameters())).grad)
        optimizer.zero_grad(True)
        self.assertIsNone(param_grad_ref())

    def test_batch_encoding_clone_inputs(self):
        class BatchEncoding(dict):
            """
            Copied from test_tokenization
            """

            def __init__(
                self,
                data,
            ):
                super().__init__(data)

            def __getattr__(self, item: str):
                try:
                    return self.data[item]
                except KeyError as e:
                    raise AttributeError from e

        encoding = BatchEncoding({"key": torch.rand((1, 4))})
        cloned_encoding = torch._dynamo.utils.clone_inputs(encoding)
        self.assertTrue(type(cloned_encoding) is not dict)

    def test_iadd_graph_break(self):
        def fn(x):
            a = ()
            x = torch.sin(x)
            a += (x,)
            return a

        x = torch.randn(4)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_odict_get_item_index_name(self):
        d = {float: torch.float32, np.float16: torch.float16}

        @torch.compile(backend="eager")
        def f(x, y1, y2):
            return torch.zeros(5, dtype=d[y1]), torch.zeros(5, dtype=d[y2])

        f(torch.zeros(4), float, np.float16)

    def test_dedup_global(self):
        @torch.compile(backend="eager")
        def f():
            return _GLOBAL_CPU_TENSOR + _GLOBAL_CPU_TENSOR

        self.assertEqual(f(), _GLOBAL_CPU_TENSOR + _GLOBAL_CPU_TENSOR)

    def test_randint_out_dynamic(self):
        def randint_fn(high, size, out):
            return torch.randint(high, size, out=out)

        opt_model = torch.compile(randint_fn, backend="eager")

        out1 = torch.empty(10, dtype=torch.int32)
        opt_model(17, (10,), out1)

        out2 = torch.empty(12, dtype=torch.int32)
        opt_model(17, (12,), out2)

    @requires_cuda
    @serialTest()
    def test_mem_leak_guards(self):
        def gn(x0, x):
            return x0 * x

        class MyMod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            @torch._dynamo.disable(recursive=False)
            def forward(self, running_x):
                # This line creates an temp tensor, which should not be leaked
                running_x = torch.sin(running_x)
                x = running_x
                # This creates a TENSOR_ALIASING guard
                x = gn(running_x, running_x)
                # This creates a NO_TENSOR_ALIASING guard which was leaking memory
                x = gn(running_x, x)
                return x

        mod = MyMod().cuda()

        fn = torch.compile(mod, backend="eager")
        x = torch.randn(10, 10, device="cuda")
        torch.cuda.reset_peak_memory_stats()

        fn(x)
        peak_mem1 = torch.cuda.max_memory_allocated()

        for _ in range(1000):
            fn(x)
        peak_mem2 = torch.cuda.max_memory_allocated()
        self.assertTrue(peak_mem1 == peak_mem2)

    @requires_cuda
    def test_guard_default_device(self):
        try:
            torch.set_default_device("cuda")

            counter = torch._dynamo.testing.CompileCounter()

            @torch.compile(backend=counter)
            def f():
                x = torch.randn(3)
                return x * 2

            self.assertEqual(f().device.type, "cuda")
            self.assertEqual(counter.frame_count, 1)

            torch.set_default_device("cpu")

            self.assertEqual(f().device.type, "cpu")
            self.assertEqual(counter.frame_count, 2)

        finally:
            torch.set_default_device(None)

    def test_list_self_reference(self):
        # Issue - https://github.com/pytorch/pytorch/issues/100150
        root = []
        root[:] = [root, root, None, None]

        @torch.compile(fullgraph=False, backend="eager")
        def test_bug():
            return root[0]

        test_bug()

    def test_hf_bigbird_unsqueeze(self):
        def torch_bmm_nd(inp_1, inp_2, ndim=None):
            torch._dynamo.graph_break()
            return torch.bmm(inp1, inp2)

        def fn(inp1, inp2, inp3, inp4, c):
            a = torch_bmm_nd(inp1, inp2, 4)
            a.unsqueeze_(2)
            a = a * 2

            b = torch_bmm_nd(inp3, inp4, 4)
            b.unsqueeze_(2)
            l = a + b

            out = torch.cat([a, b, c], dim=2)
            return out, l

        inp1 = torch.rand(1, 64, 448)
        inp2 = torch.rand(1, 448, 64)
        inp3 = torch.rand(1, 64, 448)
        inp4 = torch.rand(1, 448, 64)
        c = torch.rand(1, 64, 1, 64)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        opt_fn(inp1, inp2, inp3, inp4, c)
        self.assertEqual(cnt.frame_count, 3)

    def test_torch_variable_type(self):
        # from torchvision
        def check_type(obj, types_or_checks):
            for type_or_check in types_or_checks:
                if (
                    isinstance(obj, type_or_check)
                    if isinstance(type_or_check, type)
                    else type_or_check(obj)
                ):
                    return True
            return False

        opt_check_type = torch.compile(check_type, backend="eager")
        ref = check_type(torch.randn(4), [torch.Tensor])
        res = opt_check_type(torch.randn(4), [torch.Tensor])
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_inference_mode_dynamic_shapes(self):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, param):
                z = torch.matmul(param, param)
                return z

        model = Repro()
        # Need a 3d tensor to actually cause the error:
        # we go down a path of the C++ matmul decomp that calls sizes().
        inp = torch.randn(4, 4, 4, requires_grad=True)
        model = torch.compile(model, backend="aot_eager", dynamic=True)
        with torch.inference_mode():
            model(inp)

    def test_kwargs_out_list_variable(self):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, param):
                z = torch.frexp(**param)
                return z

        model = Repro()
        params = {"input": torch.tensor([[0.0, 1, 2, 4]])}
        params["out"] = [
            torch.empty(0, dtype=torch.float32),  # mantissa
            torch.empty(0, dtype=torch.int32),  # exponent
        ]

        model = torch.compile(model, backend="eager")
        mantissa, exponent = model(params)
        ref_mantissa = torch.tensor([[0.0000, 0.5000, 0.5000, 0.5000]])
        ref_exponent = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
        self.assertEqual(ref_mantissa, mantissa)
        self.assertEqual(ref_exponent, exponent)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_split_with_sizes_aot_autograd(self):
        def fn(result, split_sizes):
            rs = torch.ops.aten.split_with_sizes(result, split_sizes.tolist())
            return rs

        example_inputs = (
            torch.randn(32, requires_grad=True),
            torch.tensor((7, 16, 9)),
        )
        actual = torch.compile(fn, fullgraph=True, backend="aot_eager")(*example_inputs)
        expected = fn(*example_inputs)
        self.assertEqual(actual, expected)

    def test_unspecialized_nn_module_with_torch_variable_attribute(self):
        """
        In this case self.fn = something that should be a TorchVariable.
        When it's not a TorchVariable, dynamo tries to trace through and fails.
        This makes sure that the self.fn is handled as a TorchVariable.
        """

        class UserModule(torch.nn.Module):
            torchdynamo_force_dynamic = True  # forced to be a UnspecializedNNModule

            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, **inp):
                return self.fn(**inp)

        inputs = {
            "input": torch.randn([2, 9]).uniform_(0, 1),
            "target": torch.randn([2, 9]).uniform_(0, 1),
            "reduction": "mean",
        }

        mod = UserModule(torch.nn.functional.binary_cross_entropy)
        ref = mod(**inputs)
        res = torch.compile(mod, backend="eager", fullgraph=True)(**inputs)
        self.assertEqual(ref, res)

    def test_string_format(self):
        s = "temp{i}"

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            if s.format(i=4) == "temp4":
                return torch.sin(x)
            return torch.cos(x)

        x = torch.randn(4)
        self.assertEqual(fn(x), torch.sin(x))

    @unittest.skip("Fails with incorrect result with fullgraph constraints")
    def test_int_format(self):
        def fn(num: int):
            return format(num, "b")

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=False)
        self.assertEqual(fn(10), opt_fn(10))

    def test_empty_list_contains_with_jump(self):
        def fn(x, l):
            if x in l:
                return x.cos()
            return x.sin()

        counter = CompileCounter()
        torch.compile(fn, backend=counter)(torch.randn([2, 2]), [])
        self.assertEqual(counter.frame_count, 1)

    def test_get_type_hints(self):
        class Foo:
            pass

        def fn(x):
            typing.get_type_hints(Foo, include_extras=True)
            return torch.sin(x)

        x = torch.randn(4)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_graph_break_on_jit_isinstance(self):
        @torch.compile(backend="eager")
        def fn(x):
            if torch.jit.isinstance(x, typing.List[str]):  # noqa: UP006
                return x * 2
            return x

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.rand(4)
        self.assertTrue(same(fn(x), opt_fn(x)))

    def test_graph_break_on_jit_isinstance_pep585(self):
        @torch.compile(backend="eager")
        def fn(x):
            if torch.jit.isinstance(x, list[str]):
                return x * 2
            return x

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.rand(4)
        self.assertTrue(same(fn(x), opt_fn(x)))

    def test_add_sub_alpha_out(self):
        test_cases = (
            (torch.randn(2, 3, 4), 1, 2, torch.zeros(2, 3, 4)),
            (2, 1.1, 0.4, torch.tensor(0.0)),
        )
        for op in [torch.add, torch.sub]:
            for inp, other, alpha, out in test_cases:
                compiled_fn = torch.compile(op, dynamic=True, backend="eager")
                eager_out = out.clone()
                compiled_out = out.clone()
                op(inp, other, alpha=alpha, out=eager_out)
                compiled_fn(inp, other, alpha=alpha, out=compiled_out)
                self.assertTrue(same(eager_out, compiled_out))

    def test_negative_shape_guard(self):
        def fn(x):
            if x.size() != (5, 1, 2, 3):
                return x.cos()
            return x.sin()

        counter = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, dynamic=True)

        x = torch.ones(5, 1, 3, 4)
        x2 = torch.ones(5, 1, 2, 3)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x2), opt_fn(x2))
        self.assertEqual(counter.frame_count, 2)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_deferred_runtime_asserts(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            y = x.item()
            torch._check(y >= 0)
            if y >= 0:
                return x * 2
            else:
                return x * 3

        f(torch.tensor([3]))
        self.assertRaises(RuntimeError, lambda: f(torch.tensor([-2])))

    def test_addr_alpha_beta_out(self):
        inp = torch.randn(2, 3)
        vec1 = torch.randn(2)
        vec2 = torch.randn(3)
        alpha = 2
        beta = 5

        out = torch.zeros(2, 3)
        compile_out = torch.zeros(2, 3)

        torch.addr(inp, vec1, vec2, alpha=alpha, beta=beta, out=out)
        compiled_fn = torch.compile(torch.addr, dynamic=True, backend="eager")
        compiled_fn(inp, vec1, vec2, alpha=alpha, beta=beta, out=compile_out)
        self.assertTrue(same(out, compile_out))

    def test_setattr_requires_grad_graph_breaks(self):
        def fn(x):
            z = x + 4
            x.requires_grad = True
            y = x * z
            return y

        for backend in ["count", "eager", "aot_eager"]:
            if backend == "count":
                backend = CompileCounter()
            opt_fn = torch.compile(fn, backend=backend)

            eager = torch.zeros(5)
            compiled = eager.clone()

            out_eager = fn(eager)
            out_opt = opt_fn(compiled)

            self.assertEqual(out_eager, out_opt)

            out_eager.sum().backward()
            out_opt.sum().backward()

            self.assertEqual(eager, compiled)
            if isinstance(backend, CompileCounter):
                self.assertEqual(backend.frame_count, 2)  # graph breaks

    def test_dynamic_shapes_double_not_equal(self):
        # https://github.com/pytorch/pytorch/issues/113393
        def fn(x):
            if x.size() != (5, 1, 2, 3):
                return x.cos()
            return x.sin()

        opt_fn = torch.compile(fn, backend="eager")

        x = torch.ones(5, 1, 2, 3)
        x2 = torch.ones(5, 1, 3, 4)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x2), opt_fn(x2))

    def test_user_defined_object_callable(self):
        # https://github.com/pytorch/pytorch/issues/114019
        class MyCallable:
            def __call__(self, x):
                return x + 1

        def fn(x):
            # Create in graph - will not have source
            return MyCallable()(x)

        fn_opt = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn_opt(torch.zeros(1)), fn(torch.zeros(1)))

    @torch._dynamo.config.patch(log_compilation_metrics=True)
    def test_many_views_with_mutation(self):
        # When symbolic storage offsets were added in #113734, tensors_definitely_do_not_overlap
        # began adding shape guards - a quadratic amount relative to the number of inputs.
        # Test this configuration, and test that a reasonable number of guards are added.
        # Note, when dynamic shapes are turned on, this test fails and we still get quadratic guards.
        def fn(x):
            x[0].relu_()
            return torch.cat(x).sum()

        AMT = 32
        src = torch.rand(16 * (AMT + 1))

        x = [src.as_strided((4, 4), (4, 1), 3 + 16 * i) for i in range(AMT)]

        torch._dynamo.reset()
        torch._dynamo.utils.clear_compilation_metrics()

        torch.compile(fn, backend="aot_eager")(x)

        all_metrics = torch._dynamo.utils.get_compilation_metrics()

        total_guards = sum(metric.guard_count for metric in all_metrics)
        self.assertLess(total_guards, AMT * 8)

        total_shape_env_guards = sum(
            metric.shape_env_guard_count for metric in all_metrics
        )
        self.assertLess(total_shape_env_guards, AMT * 8)

    def test_subclass_graph_output_repro(self):
        @torch._dynamo.allow_in_graph
        def to_subclass(x):
            return TwoTensor(x.clone(), x.clone())

        def f(x):
            tmp_subclass = to_subclass(x)
            return tmp_subclass.view(-1)

        x = torch.ones(2)
        out_ref = f(x)
        out_test = torch.compile(f, backend="aot_eager")(x)
        self.assertEqual(out_ref, out_test)

    def test_numpy_tobytes_no_error(self):
        def fn(x):
            x += 1
            z = x.tobytes()
            x += 1
            return z

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        opt_arg, arg = np.array([1, 2]), np.array([1, 2])
        self.assertEqual(opt_fn(opt_arg), fn(arg))
        self.assertEqual(cnt.frame_count, 2)

    def test_numpy_not_ndarray_recompiles(self):
        import torch

        def fn(x=None):
            if x is None:
                x = np.ones(3)
            elif isinstance(x, int):
                x = np.ones(6)
            elif isinstance(x, str):
                x = np.ones(9)
            return x**2

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)

        x = np.zeros((2, 2))

        self.assertEqual(opt_fn(x), fn(x))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(opt_fn(), fn())
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(opt_fn(10), fn(10))
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(opt_fn("10"), fn("10"))
        self.assertEqual(cnt.frame_count, 4)

    @parametrize(
        "backend",
        ["eager", "aot_eager", "inductor"],
    )
    @parametrize(
        "func_name",
        ["func1", "func2", "func3"],
    )
    def test_tensor_set_data(self, backend, func_name):
        # https://github.com/pytorch/pytorch/issues/113030
        def func1(x, y):
            x.data = y
            x.add_(1)
            return x

        def func2(x, y):
            x.data = y
            y.data = torch.zeros([0])
            return x

        def func3(x, y):
            z = x
            x.data = y
            y.data = torch.zeros([0])
            return torch.tensor(x is z)

        funcs = {"func1": func1, "func2": func2, "func3": func3}
        func = funcs[func_name]

        if backend != "eager" and func is func1:
            # add_ not working w/ aot_autograd?
            return

        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

        compiled_fn = torch.compile(func, backend=cnt, fullgraph=True)
        requires_grad = func is not func1
        for _ in range(5):
            # Inputs
            eager_a = torch.ones([6], requires_grad=requires_grad)
            compiled_a = torch.ones([6], requires_grad=requires_grad)

            eager_b = torch.ones([6], requires_grad=requires_grad)
            compiled_b = torch.ones([6], requires_grad=requires_grad)

            # Eager
            out_eager = func(eager_a, eager_b)
            # Compiled
            out_compiled = compiled_fn(compiled_a, compiled_b)
            self.assertEqual(eager_a, compiled_a)
            self.assertEqual(eager_b, compiled_b)
            self.assertTrue(torch.equal(out_eager, out_compiled))

            # func1 hits a leaf Variable that requires grad is being used in an in-place operation
            if requires_grad:
                bwd_inp_eager = torch.randn([6])
                bwd_inp_compiled = torch.clone(bwd_inp_eager)
                eager_a.backward(bwd_inp_eager)
                compiled_a.backward(bwd_inp_compiled)
                self.assertEqual(eager_a.grad, compiled_a.grad)

        # Prove guarding works - we run the compiled_fn 5 times
        # frame_count should stay at 1.
        self.assertEqual(cnt.frame_count, 1)

    def test_tensor_set_data_mismatched_dtype(self):
        def func(x, y):
            x.data = y.to(dtype=torch.bfloat16)

        x1 = torch.tensor([], dtype=torch.float32)
        x2 = torch.tensor([], dtype=torch.float32)
        y1 = torch.tensor([1, 2, 3], dtype=torch.float32)
        y2 = torch.tensor([1, 2, 3], dtype=torch.float32)
        func(x1, y1)
        torch.compile(func, backend="eager")(x2, y2)
        self.assertEqual(x1, x2)
        self.assertEqual(x1.data, x2.data)
        self.assertEqual(y1, y2)

    def test_user_ctor_ctx_manager(self):
        class UserCtxManager:
            def __enter__(self):
                return 1

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        def fn(x, y):
            ucm = UserCtxManager()  # noqa: F841
            return x * x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        x = torch.rand([2, 2])
        opt_fn(x, x)
        self.assertExpectedInline(cnt.frame_count, """1""")

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_arange_in_bounds(self):
        # see https://github.com/pytorch/pytorch/issues/113002
        class PaddingNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, lengths):
                max_seq_len = lengths.max().item()
                row_vector = torch.arange(0, max_seq_len, 1)
                matrix = torch.unsqueeze(lengths, dim=-1)
                mask = row_vector < matrix
                mask = mask.type(torch.float32)
                mask_3d_btd = mask[:, :, None]
                return mask_3d_btd

        model = PaddingNet()
        lengths = torch.tensor([5, 4, 4, 4], dtype=torch.int32)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(model, backend=cnt, fullgraph=True)
        opt_fn(lengths)
        self.assertEqual(cnt.frame_count, 1)

    def test_overlapping_inputs_with_dynamic_shapes_error(self):
        @torch.compile(backend="aot_eager")
        def fn(a, b, c, d, e, f):
            a.mul_(2)
            b.mul_(2)
            c.mul_(2)
            d.mul_(2)
            e.mul_(2)
            f.mul_(2)

            base = torch.ones(2, 20)
            a = base[:, 0:2]
            b = base[:, 2:4]
            c = base[:, 4:6]
            d = base[:, 6:8]
            e = base[:, 8:10]
            f = base[:, 10:12]
            f2 = base[:, 10:14]
            fn(a, b, c, d, e, f)
            with self.assertRaisesRegex(
                AssertionError, "is being compiled with dynamic shapes"
            ):
                fn(a, b, c, d, e, f2)

    def test_user_ctor_ctx_manager_custom_init(self):
        class UserCtxManager:
            def __init__(self, x):
                x[0] = 10

            def __enter__(self):
                return 1

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        def fn(x, y):
            ucm = UserCtxManager(y)  # noqa: F841
            return x * y[0]

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        x = torch.rand([2, 2])
        self.assertEqual(opt_fn(x, [5]), fn(x, [5]))
        self.assertExpectedInline(cnt.frame_count, """1""")

    def test_user_ctor_ctx_manager_custom_init_graph_break(self):
        counter = [0]

        class UserCtxManager:
            def __init__(self, k):
                k[0] += 1

            def __enter__(self):
                return 1

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        def fn(x, counter):
            x = x * x
            ucm = UserCtxManager(counter)  # noqa: F841
            return x * x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.rand([2, 2])
        self.assertEqual(opt_fn(x, counter), fn(x, counter))
        self.assertEqual(counter[0], 2)
        for _ in range(10):
            opt_fn(x, counter)
        self.assertEqual(counter[0], 12)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """2""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")

    def test_many_overlapping_inputs_does_not_explode_guards(self):
        from torch._dynamo.backends.common import aot_autograd

        # Before, this was (9702, 0)
        num_shape_guards = None
        num_aot_guards = None
        num_compiles = 0

        def guard_count_backend(gm, *args):
            nonlocal num_shape_guards
            nonlocal num_aot_guards
            nonlocal num_compiles
            num_shape_guards = len(
                torch._guards.TracingContext.try_get().fake_mode.shape_env.guards
            )
            num_aot_guards = len(
                torch._guards.TracingContext.try_get().guards_context.aotautograd_guards
            )
            num_compiles += 1
            return gm

        aot_guard_counter = aot_autograd(fw_compiler=guard_count_backend)

        @torch.compile(backend=aot_guard_counter, dynamic=True)
        def f(*args):
            for a in args:
                a.add_(1)

        x = torch.ones(1000, requires_grad=True)
        args = x.split(10)

        with torch.no_grad():
            f(*args)
        # In this example, there were 4950 guards (roughly (# tensors) ^ 2 // 2),
        # because every pair of aliased inputs needs a guard.
        self.assertTrue(num_aot_guards < 5000)
        # But there are no dynamic shape guards.
        self.assertEqual(num_shape_guards, 0)
        # don't recompile
        with torch.no_grad():
            f(*args)
        self.assertEqual(num_compiles, 1)

    def test_issue134451(self):
        class BoundingBox2DIndex(IntEnum):
            _X = 0
            _Y = 1
            _HEADING = 2
            _LENGTH = 3
            _WIDTH = 4

            @classmethod
            def size(cls):
                return 5

            @classmethod
            @property
            def X(cls):
                return cls._X

            @classmethod
            @property
            def Y(cls):
                return cls._Y

            @classmethod
            @property
            def HEADING(cls):
                return cls._HEADING

            @classmethod
            @property
            def LENGTH(cls):
                return cls._LENGTH

            @classmethod
            @property
            def WIDTH(cls):
                return cls._WIDTH

            @classmethod
            @property
            def POINT(cls):
                # assumes X, Y have subsequent indices
                return slice(cls._X, cls._Y + 1)

            @classmethod
            @property
            def STATE_SE2(cls):
                # assumes X, Y, HEADING have subsequent indices
                return slice(cls._X, cls._HEADING + 1)

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._mlp_states = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, BoundingBox2DIndex.size()),
                )

            def forward(self, x):
                agent_states = self._mlp_states(x)
                agent_states[..., BoundingBox2DIndex.POINT] = (
                    agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
                )
                agent_states[..., BoundingBox2DIndex.HEADING] = (
                    agent_states[..., BoundingBox2DIndex.HEADING].tanh() * torch.pi
                )
                return agent_states

        model = SimpleModel().eval()
        input_tensor = torch.randn(1, 10, dtype=torch.float32)
        opt = torch.compile(model.eval(), backend="eager", fullgraph=True)
        try:
            expected = model(input_tensor)
        except Exception as e:
            raise unittest.SkipTest(
                "eager failed, requires Python between 3.9 and 3.12"
            ) from e
        actual = opt(input_tensor)
        self.assertEqual(actual, expected)

    def test_invalid_seq_unpack(self):
        def myfn(arg):
            (a, b) = arg  # noqa: F841

        def fn():
            return myfn((1, 2, 3))

        try:
            torch.compile(fn, backend="eager")()
        except ValueError:
            pass
        else:
            self.fail("expected exception")

    def test_udf_classes_reconstruction(self):
        def fn(x):
            o = T(5)
            return o.x + x

        opt_fn = torch.compile(fn, backend="eager")
        T = IncByOne

        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

        # This should recompile
        T = IncByTwo
        self.assertEqual(fn(x), opt_fn(x))

    def test_contains_range_constprop(self):
        def fn(x):
            # dynamo should const prop to False
            if 3 in range(10):
                return x + 1
            else:
                return x + 2

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.zeros(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_as_strided_on_base_with_mutation_works(self):
        def foo(a):
            f = a.as_strided((2,), (1,), 0)
            f.add_(1.0)
            return a

        a = torch.randn(2, 4)
        a_ref = a.clone()
        out_ref = foo(a_ref)
        f_compiled = torch.compile(foo, backend="aot_eager")
        out = f_compiled(a)
        self.assertEqual(out_ref, out)
        self.assertEqual(a_ref, a)

    def test_as_strided_on_existing_view_banned(self):
        def foo(a):
            e = a.diagonal()
            f = e.as_strided((2,), (1,), 0)
            f.add_(1.0)
            return a

        a = torch.randn(2, 4)
        a_ref = a.clone()
        foo(a_ref)
        f_compiled = torch.compile(foo, backend="aot_eager")
        with self.assertRaisesRegex(
            RuntimeError,
            "encountered a mutation on a view chain of length 2, where view 1 was an as_strided",
        ):
            f_compiled(a)

    def test_preserve_stride_with_clone(self) -> None:
        A = torch.rand(5, 5, device="cuda" if torch.cuda.is_available() else "cpu")
        B = torch.rand(5, 5, device="cuda" if torch.cuda.is_available() else "cpu")

        def fn(
            src: torch.Tensor, count: torch.Tensor
        ) -> tuple[tuple[int, ...], tuple[int, ...]]:
            Q, R = torch.linalg.qr(src)
            rhs = torch.ones(Q.shape[0], 1, device=src.device)
            a = torch.linalg.solve_triangular(R, Q.T @ rhs, upper=True)
            cloned = a.clone(memory_format=torch.preserve_format)
            return a.stride(), cloned.stride()

        a_stride, cloned_stride = fn(A, torch.zeros(1))
        self.assertEqual(
            a_stride,
            cloned_stride,
            f"Strides should match in eager: {a_stride} against {cloned_stride}",
        )

        compiled_a_stride, compiled_cloned_stride = torch.compile(fn, backend="eager")(
            B, torch.zeros(1)
        )
        self.assertEqual(
            compiled_a_stride,
            compiled_cloned_stride,
            f"Strides should match in eager: {compiled_a_stride} against {compiled_cloned_stride}",
        )

    def test_clone_not_memory_dense(self):
        def foo() -> torch.Tensor:
            x = torch.randn(10, 8).t()[::2, ::2]
            y = x.clone()
            return y

        y = foo()
        self.assertEqual(
            y.stride(),
            (1, 4),
            "Reference eager implementation should have stride (1, 4)",
        )
        y = torch.compile(foo, backend="eager")()
        self.assertEqual(
            y.stride(), (1, 4), "Compile with eager backend should have stride (1, 4)"
        )
        y = torch.compile(foo, backend="aot_eager")()
        self.assertEqual(
            y.stride(),
            (1, 4),
            "Compile with aot_eager backend should have stride (1, 4)",
        )
        y = torch.compile(foo, backend="inductor")()
        self.assertEqual(
            y.stride(),
            (1, 4),
            "Compile with inductor backend should have stride (1, 4)",
        )

    @unittest.expectedFailure
    def test_lru_cache_tracing(self):
        from functools import lru_cache

        counter = 0

        @lru_cache
        def cached_fn(x):
            nonlocal counter
            counter += 1
            return x + 1

        compiled_fn = torch.compile(cached_fn, backend="eager")

        t = torch.randn(2, 2)
        result1 = compiled_fn(t)
        self.assertEqual(counter, 1)

        result2 = compiled_fn(t)
        self.assertEqual(counter, 1)
        self.assertEqual(result1, result2)

    def test_dont_aggressively_write_assert(self):
        record_graph = torch._dynamo.testing.EagerAndRecordGraphs()

        @torch.compile(dynamic=True, backend=record_graph)
        def f(x):
            assert x.shape[0] > 3  # noqa: S101
            assert x[0].sum() > 0  # noqa: S101
            assert 1 % (x.shape[0] // 2) != 0  # noqa: S101
            assert 32 * (x.shape[0] // 2) ** 2 - 16 * (x.shape[0] // 2) != 0  # noqa: S101
            return x.cos()

        f(torch.ones(6, 4))
        graph = record_graph.graphs[0]
        # It is bit annoying that we generate useless statements for
        # shape guards, but DCE should be able to remove them since t
        # there is no backed assert on them. The reason this is ok is
        # because dynamo will only skip the assert statement, but not
        # the instructions before it.

        # The code generation can non-deterministically use either form
        generated_code = str(graph.code).strip().replace(".gt(0)", " > 0")
        self.assertExpectedInline(
            generated_code,
            """\
def forward(self, s77 : torch.SymInt, s27 : torch.SymInt, L_x_ : torch.Tensor):
    l_x_ = L_x_
    getitem_2 = l_x_[0]
    sum_1 = getitem_2.sum();  getitem_2 = None
    gt_1 = sum_1 > 0;  sum_1 = None
    _assert_async = torch._assert_async(gt_1, 'assertion error');  gt_1 = _assert_async = None
    cos = l_x_.cos();  l_x_ = None
    return (cos,)""",
        )
        for node in graph.graph.nodes:
            if "example_value" in node.meta and isinstance(
                node.meta["example_value"], torch._subclasses.fake_tensor.FakeTensor
            ):
                shape_env = node.meta["example_value"].fake_mode.shape_env
                lower_ranges = [val.lower for val in shape_env.var_to_range.values()]
                self.assertTrue(lower_ranges == [4, 2])

        @torch.compile(dynamic=True, backend=record_graph)
        def f_fail(x):
            assert x.shape[0] < 3  # noqa: S101

        # We graph-break here, so the failure should be eager
        with self.assertRaisesRegex(AssertionError, ""):
            f_fail(torch.ones(6, 4))

    def test_detectron2_instances_cat(self):
        class Instances:
            def __init__(self, image_size: tuple[int, int], **kwargs: Any):
                self._image_size = image_size
                self._fields: dict[str, Any] = {}
                for k, v in kwargs.items():
                    self.set(k, v)

            @property
            def image_size(self) -> tuple[int, int]:
                return self._image_size

            def __setattr__(self, name: str, val: Any) -> None:
                if name.startswith("_"):
                    super().__setattr__(name, val)
                else:
                    self.set(name, val)

            def __getattr__(self, name: str) -> Any:
                if name == "_fields" or name not in self._fields:
                    raise AttributeError(
                        f"Cannot find field '{name}' in the given Instances!"
                    )
                return self._fields[name]

            def __len__(self) -> int:
                for v in self._fields.values():
                    # use __len__ because len() has to be int and is not friendly to tracing
                    return v.__len__()
                raise NotImplementedError("Empty Instances does not support __len__!")

            def set(self, name: str, value: Any) -> None:
                with warnings.catch_warnings(record=True):
                    data_len = len(value)
                if len(self._fields):
                    assert len(self) == data_len, (  # noqa: S101
                        f"Adding a field of length {data_len} to a Instances of length {len(self)}"
                    )
                self._fields[name] = value

            def get(self, name: str) -> Any:
                return self._fields[name]

            @staticmethod
            def cat(instance_lists: list["Instances"]) -> "Instances":
                assert all(isinstance(i, Instances) for i in instance_lists)  # noqa: S101
                assert len(instance_lists) > 0  # noqa: S101
                if len(instance_lists) == 1:
                    return instance_lists[0]

                image_size = instance_lists[0].image_size
                if not isinstance(
                    image_size, torch.Tensor
                ):  # could be a tensor in tracing
                    for i in instance_lists[1:]:
                        assert i.image_size == image_size  # noqa: S101
                ret = Instances(image_size)
                for k in instance_lists[0]._fields:
                    values = [i.get(k) for i in instance_lists]
                    v0 = values[0]
                    if isinstance(v0, torch.Tensor):
                        values = torch.cat(values, dim=0)
                    elif isinstance(v0, list):
                        values = list(itertools.chain(*values))
                    elif hasattr(type(v0), "cat"):
                        values = type(v0).cat(values)
                    else:
                        raise ValueError(
                            f"Unsupported type {type(v0)} for concatenation"
                        )
                    ret.set(k, values)
                return ret

        instances = [
            Instances((16, 16), a=torch.randn(16, 16), b=torch.randn(16, 16))
            for _ in range(3)
        ]

        @torch.compile(backend="eager", fullgraph=True)
        def fn(instances):
            return instances[0].cat(instances)

        actual = fn(instances)
        expected = instances[0].cat(instances)
        self.assertEqual(type(actual), type(expected))
        self.assertEqual(actual.__dict__, expected.__dict__)

    def test_weakref_construction(self):
        def fn(x, y):
            x_weak = weakref.ref(x)
            return x_weak() * y

        x = torch.randn(4)
        y = torch.randn(4)

        ref = fn(x, y)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_weakref(self):
        def fn(x_weak, weight, y):
            if x_weak is not None and x_weak() is not weight:
                return torch.sin(y)
            return torch.cos(y)

        weight = torch.randn(4)
        y = torch.randn(4)
        x_weak = weakref.ref(weight)

        ref = fn(x_weak, weight, y)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x_weak, weight, y)
        self.assertEqual(ref, res)

    def test_weakref_proxy(self):
        class DummyTrainer:
            def __init__(self, x):
                self.foo = x

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.trainer = None

            def foo(self):
                return self.trainer.foo

        x = torch.randn(4)
        model = DummyModel()
        trainer = DummyTrainer(x)
        model.trainer = weakref.proxy(trainer)
        compiled_foo = torch.compile(model.foo, backend="eager", fullgraph=True)
        self.assertEqual(compiled_foo(), x)

    def test_weakref_reconstruct(self):
        def fn(x_weak, weight, y):
            y = torch.sin(y)
            referent = x_weak()
            torch._dynamo.graph_break()
            if referent is not weight:
                return torch.sin(y)
            return torch.cos(y)

        weight = torch.randn(4)
        y = torch.randn(4)
        x_weak = weakref.ref(weight)

        ref = fn(x_weak, weight, y)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        res = opt_fn(x_weak, weight, y)
        self.assertEqual(ref, res)
        self.assertEqual(cnt.frame_count, 2)

    def test_return_weakref(self):
        def f(t):
            t = t * 2
            wr = weakref.ref(t)
            return wr, t

        ref_t = torch.randn(2, 2, requires_grad=True)
        ref_y = f(ref_t)

        t = ref_t.detach().clone().requires_grad_()
        y = torch.compile(f, backend="eager", fullgraph=True)(t)
        self.assertEqual(ref_y[0](), y[0]())

    def test_weakref_del(self):
        def fn(x_weak, y):
            x = x_weak()
            if x is not None:
                return torch.sin(y)
            return torch.cos(y)

        weight = torch.randn(4)
        x_weak = weakref.ref(weight)
        y = torch.randn(4)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        ref = fn(x_weak, y)
        res = opt_fn(x_weak, y)
        self.assertEqual(ref, res)

        del weight
        gc.collect()
        ref = fn(x_weak, y)
        res = opt_fn(x_weak, y)
        self.assertEqual(ref, res)

    @skipIfWindows(msg="TODO: (xuhancn) fix, AssertionError: False is not true")
    def test_weakref_callback(self):
        called1 = False

        def callback1(ref):
            nonlocal called1
            called1 = True
            if not torch.compiler.is_compiling():
                raise RuntimeError("callback1 expected to be compiled")

        # weakref callbacks that should be called in the compiled region will be compiled.
        # But the exact place in the compiled code that the callback is made is undefined.
        @torch.compile(backend="eager")
        def fn(x):
            y = x + 1
            ref = weakref.ref(y, callback1)
            torch._dynamo.graph_break()
            return ref

        fn(torch.ones(3))
        self.assertTrue(called1)

        called2 = False

        def callback2(ref):
            nonlocal called2
            called2 = True
            if torch.compiler.is_compiling():
                raise RuntimeError("callback2 expected to not be compiled")

        # weakref callbacks that fire outside the compiled region work
        @torch.compile(backend="eager")
        def gn(x):
            y = x + 1
            ref = weakref.ref(y, callback2)
            torch._dynamo.graph_break()
            return y, ref

        y, _ = gn(torch.ones(3))
        del y
        self.assertTrue(called2)

        def callback3(ref):
            raise RuntimeError("callback3 should not be called")

        # The callback will NOT be called if both the weakref and the referrent are
        # deleted in the same compiled region (graph breaks act like a "memory sync"
        # and thus make things tricky - the callback is actually expected to be called).
        # This test does NOT mean that this behavior is part of the (weak)ref programming
        # model, but rather reminds us that this is an intentionally allowed weakref-Dynamo behavior.
        @torch.compile(backend="eager")
        def hn(x):
            y = x + 1
            _ = weakref.ref(y, callback3)

        hn(torch.ones(3))

    def test_super_in_staticmethod(self):
        class A:
            @staticmethod
            def foo():
                return super().__init__()

        def fn(obj):
            return obj.foo()

        obj = A()

        try:
            fn(obj)
        except Exception as e:
            orig_str = str(e)
        self.assertIn("no arguments", orig_str)

        try:
            torch.compile(backend="eager")(fn)(obj)
        except Exception as e:
            compiled_str = str(e)
        self.assertEqual(orig_str, compiled_str)

    def test_super_staticmethod(self):
        class Parent:
            @staticmethod
            def greet():
                return 5

        class Child(Parent):
            @staticmethod
            def greet(x):
                return x * super(Child, Child).greet()

        child = Child()

        def fn(x):
            return child.greet(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_super_classmethod(self):
        class Parent:
            @classmethod
            def greet(cls):
                if cls == Parent:
                    return 4
                if cls == Child:
                    return 3
                if cls == GrandChild:
                    return 5
                return 2

        class Child(Parent):
            def greet(self, x):
                return x * super().greet()

        class GrandChild(Child):
            pass

        grand_child = GrandChild()

        def fn(x):
            return grand_child.greet(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_super_classmethod_inheritance(self):
        class GrandParent:
            @classmethod
            def greet(cls, x):
                return cls.A * x

        class Parent(GrandParent):
            @classmethod
            def greet(cls, x):
                return super().greet(x)

        class Child(Parent):
            A = 5

            @classmethod
            def greet(cls, x):
                return super().greet(x)

        child = Child()

        def fn(x):
            return child.greet(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_super_diamond(self):
        class A:
            def __init__(self):
                super().__init__()
                self.a = 5

        class Nothing:
            pass

        class B(Nothing, A):
            def __init__(self):
                super().__init__()
                self.b = 10

            def run(self, x):
                return self.a * self.b * x

        def fn(x):
            b = B()
            return b.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_vc_bumped_in_inference_graph(self):
        @torch.compile(backend="eager")
        def f(x):
            return x.mul_(2)

        x = torch.randn(4)
        vc_before = x._version
        f(x)
        vc_after = x._version
        self.assertTrue(vc_after > vc_before)

    def test_nn_module_callable(self):
        class M(nn.Module):
            def forward(self, x):
                return x.sin()

        def f(m):
            return callable(m)

        res = torch.compile(f, fullgraph=True, backend="eager")(M())
        self.assertTrue(res)

    def test_stk_sdd_is_transposed(self):
        def _is_transposed(x):
            return (
                not x.is_contiguous()
                and x.stride()[0] == 1
                and x.stride()[1] == x.size()[0]
            )

        class SDD(torch.autograd.Function):
            @staticmethod
            def forward(ctx, lhs, rhs):
                ctx.save_for_backward(lhs, rhs)
                out = torch.full_like(lhs, 1.0, dtype=lhs.dtype, device=lhs.device)
                return out

            @staticmethod
            def backward(ctx, dy):
                saved_tensors = ctx.saved_tensors
                lhs, rhs = saved_tensors[:2]
                trans_a = _is_transposed(lhs)
                trans_b = _is_transposed(rhs)
                dlhs = None
                if ctx.needs_input_grad[0]:
                    dlhs = torch.full_like(lhs, 1.0 if trans_a else 2.0)
                drhs = None
                if ctx.needs_input_grad[1]:
                    drhs = torch.full_like(rhs, 1.0 if trans_b else 2.0)
                return dlhs, drhs, None, None

        x1 = torch.randn((8, 8), requires_grad=True)
        y1 = torch.randn((8, 8)).transpose(0, 1).requires_grad_(True)
        x2 = torch.randn((8, 8), requires_grad=True)
        y2 = torch.randn((8, 8)).transpose(0, 1).requires_grad_(True)

        SDD.apply(x1, y1).sum().backward()

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return SDD.apply(x2, y2)

        fn().sum().backward()

        self.assertEqual(x1.grad, x2.grad)
        self.assertEqual(y1.grad, y2.grad)

    def test_partially_initialized_module_property(self):
        class Matrix(torch.nn.Module):
            def __init__(self, data):
                super().__init__()
                self._data = data
                self.foo = 10 * self.blocking

            @property
            def data(self):
                return self._data

            @property
            def blocking(self):
                return self.data.shape[1]

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return Matrix(torch.randn(10, 20))

        v = fn()
        self.assertEqual(v.foo, 200)
        self.assertEqual(v.data.shape, (10, 20))
        self.assertEqual(type(v), Matrix)

    def test_classmethod_with_slots(self):
        class Mock:
            __slots__ = ("_a",)

            def __init__(self):
                self._a = 2

            @classmethod
            def _m(cls):
                return 3

            def run(self, x):
                return torch.sin(x) * self._a * self._m()

        def fn(x):
            mock = Mock()
            return mock.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_nn_parametrize(self):
        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(10, 10))

            def forward(self, x):
                return self.param @ x

        class Parametrization(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x)

        m = Module()
        torch.nn.utils.parametrize.register_parametrization(
            m, "param", Parametrization()
        )

        sin_found = False

        def backend(gm, _):
            nonlocal sin_found
            for node in gm.graph.nodes:
                if node.target is torch.sin:
                    sin_found = True
            return gm

        opt_m = torch.compile(m, backend=backend, fullgraph=True)
        inp = torch.randn(10, 10)
        self.assertEqual(m(inp), opt_m(inp))
        self.assertTrue(sin_found)

        torch.nn.utils.parametrize.remove_parametrizations(m, "param")
        sin_found = False
        self.assertEqual(m(inp), opt_m(inp))
        self.assertFalse(sin_found)

    def test_nn_module_property_closure(self):
        x = torch.randn(10, 10)

        class Mod(torch.nn.Module):
            @property
            def y(self):
                return torch.ones(10, 10) + x

            def forward(self, x):
                return x @ self.y

        mod = Mod()

        def fn(x):
            return mod(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        inp = torch.randn(10, 10)
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_global_fn_mutation(self):
        def foo(x, y):
            return global_fn(x) + y

        x = torch.ones(1)
        y = torch.ones(1)

        opt = torch.compile(foo, fullgraph=True, backend="eager")
        self.assertEqual(opt(x, y), foo(x, y))

        # Change global_fn
        global global_fn

        def new_fn(x):
            return torch.cos(x)

        global_fn = new_fn
        self.assertEqual(opt(x, y), foo(x, y))

    def test_list_reverse(self):
        def ladder(x):
            trail = x.size(-1)
            assert trail > 2  # noqa: S101
            weights = []
            for s in [trail, trail - 1, trail - 2]:
                weights.append(torch.ones(s, s - 1))

            for w in weights:
                x = x @ w

            weights.reverse()

            for w in weights:
                x = x @ w.t()

            return x

        data = torch.randn(3, 4)
        opt_ladder = torch.compile(ladder, fullgraph=True, backend="eager")
        self.assertEqual(opt_ladder(data), ladder(data))

    def test_trace_functional_tensor_with(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        def f(a, tmp):
            a_view = a.view(-1)
            with torch.no_grad():
                a.set_(tmp)
                a_view.mul_(2)
            return a + tmp

        fake_mode = FakeTensorMode()
        with FunctionalTensorMode():
            inp = torch.ones(3, 3, requires_grad=True)
            inp = fake_mode.from_tensor(inp, static_shapes=True)
            inp = FunctionalTensor.to_functional(inp)

            tmp = torch.ones(3, 3, requires_grad=True)
            tmp = fake_mode.from_tensor(tmp, static_shapes=True)
            tmp = FunctionalTensor.to_functional(tmp)

            opt_f = torch.compile(f, backend="eager")
            with self.assertRaisesRegex(
                RuntimeError, "cannot mutate tensors with frozen storage"
            ):
                opt_f(inp, tmp)

    def test_const_dict_keyerror(self):
        d = {}

        def fn(x):
            try:
                y = d[0]
            except KeyError:
                y = 1
            return x + y

        opt_fn = torch.compile(fn, backend="eager")
        inp = torch.randn(3, 3)
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_nonconst_issubclass(self):
        def fn(x):
            if issubclass(x.__class__, np.ndarray):
                return 1
            return 0

        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(np.ones([3, 3]))

    def test_issue126128(self):
        def fn():
            x = torch.randn(1, 10)
            y = torch.randn(10, 1)
            return torch.mm(x, y).sum()

        def fn2():
            x = torch.randn(10, 100)
            y = torch.randn(100, 10)
            return torch.mm(x, y).sum()

        with fresh_cache():
            torch.compile(fn, backend="eager")()

        torch.compile(fn2, backend="eager")()

    def test_jit_script_defaults(self):
        @torch.jit.script
        def fast_cos(x, c: float = 2.0):
            return torch.cos(x) * c

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fast_cos = fast_cos

            def forward(self, x):
                return self.fast_cos(x)

        mod = Mod()
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        x = torch.randn(4)

        self.assertEqual(mod(x), opt_mod(x))

    def test_enum(self):
        class ExplicitEnum(str, Enum):  # noqa: SLOT000
            @classmethod
            def _missing_(cls, value):
                raise ValueError(
                    f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
                )

        class PaddingStrategy(ExplicitEnum):
            LONGEST = "longest"
            MAX_LENGTH = "max_length"
            DO_NOT_PAD = "do_not_pad"

        def fn(x):
            a = PaddingStrategy("longest")
            if a == PaddingStrategy.LONGEST:
                return torch.sin(x)
            return torch.cos(x)

        x = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_hasattr_builtin(self):
        class MyClass:
            foo: int = 1

        def func(x, m):
            if getattr(type(m), "foo", 0):
                return x + MyClass.foo
            return x

        opt_func = torch.compile(func, backend="eager", fullgraph=True)
        m = MyClass()
        x = torch.zeros(())
        self.assertEqual(func(x, m), opt_func(x, m))
        self.assertEqual(func(x, 0), opt_func(x, 0))

    def test_grad(self):
        # Write to `grad` or `_grad` should reflective in reading from the other,
        # and should be codegen-ed.
        def fn(x, y):
            x._grad = y + 1
            y.grad = x + 2
            return x.grad.data, y._grad.data

        x0 = torch.randn(4, requires_grad=True)
        y0 = torch.randn(4, requires_grad=True)
        x1 = x0.clone()
        y1 = y0.clone()
        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x0, y0), opt_fn(x1, y1))
        self.assertEqual(x0.grad, x1.grad)
        self.assertEqual(y0.grad, y1.grad)

    def test_nn_module_stack_bc(self):
        from torch._dynamo.mutation_guard import GenerationTracker

        def compiler(gm, *args):
            module_stacks = [
                node.meta.get("nn_module_stack", None) for node in gm.graph.nodes
            ]
            module_stacks, _ = pytree.tree_flatten(module_stacks)
            module_stacks = [x for x in module_stacks if isinstance(x, str)]
            for stack in module_stacks:
                self.assertTrue("_module" not in stack)
            return gm.forward

        class SubMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.submod1 = SubMod()
                self.submod2 = SubMod()

            def forward(self, x):
                return self.submod1(x) + self.submod2(x)

        mod = Mod()
        opt_mod = torch.compile(mod, backend=compiler)
        opt_mod(torch.randn(2, 2))

        mod = Mod()
        opt_mod = torch.compile(mod, backend=compiler)
        opt_mod(torch.randn(2, 2))

        # an example similar to Pippy usecase
        mod = Mod()
        GenerationTracker.tag(mod.submod1)
        GenerationTracker.mark_class_dynamic(type(mod.submod1))
        mod = Mod()
        opt_mod = torch.compile(mod, backend=compiler)
        opt_mod(torch.randn(2, 2))

    def test_is_make_fx_tracing(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            torch.nn.modules.activation._is_make_fx_tracing()
            return torch.sin(x)

        fn(torch.rand(4))

    def test_export_vs_dynamo_for_multiheadattention(self):
        # More details at https://github.com/pytorch/pytorch/issues/164062

        # Ensure that both dynamo and export do not take the fast path.
        with torch.no_grad():
            inp = torch.randn(1, 2, 64)
            mha = nn.MultiheadAttention(64, 2, dropout=0.1, batch_first=True)
            mha.eval()

            backend = EagerAndRecordGraphs()
            mha_compile = torch.compile(mha, backend=backend, fullgraph=True)
            mha_compile(inp, inp, inp)
            torch.compiler.reset()

            mha_export = torch._dynamo.export(mha)(inp, inp, inp)

            compile_nodes = backend.graphs[0].graph.find_nodes(
                op="call_function", target=torch._native_multi_head_attention
            )
            export_nodes = mha_export.graph_module.graph.find_nodes(
                op="call_function", target=torch._native_multi_head_attention
            )
            self.assertEqual(len(compile_nodes), 0)
            self.assertEqual(len(export_nodes), 0)

    def test_negative_floor_div_solve(self):
        class CompiledClass(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nums = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                self.t = 5

            def forward(self):
                self.num = self.nums[self.t // 12]
                self.t += 1
                return self.num

        m = CompiledClass()
        m = torch.compile(m, backend="eager")

        # the first call works
        m()
        # the second call causes a failure
        m()

    def test_tensor_random(self):
        def random_op(tensor, args, kwargs):
            res = tensor.random_(*args, **kwargs)
            return res

        random_op = torch.compile(random_op, backend="eager")
        tensor = torch.randn([2, 3])
        random_op(tensor, [], {"from": -10, "to": 10})
        random_op(tensor, [-10], {"to": 10})
        random_op(tensor, [-10, 10], {})

    def test_tensor_uniform(self):
        def uniform_op(tensor, args, kwargs):
            res = tensor.uniform_(*args, **kwargs)
            return res

        uniform_op = torch.compile(uniform_op, backend="eager")
        tensor = torch.randn([2, 3])
        uniform_op(tensor, [], {"from": -10, "to": 10})
        uniform_op(tensor, [-10], {"to": 10})
        uniform_op(tensor, [-10, 10], {})

    def test_data_attr_mutation_after_saved_for_bw(self):
        def f(x):
            out = x.sin()
            x.data.mul_(2)
            return out

        x = torch.randn(4, requires_grad=True)
        x_test = x.detach().clone().requires_grad_(True)

        out = f(x)
        out_test = torch.compile(f, backend="aot_eager")(x_test)
        self.assertEqual(out, out_test)

        out.sum().backward()
        out_test.sum().backward()
        self.assertEqual(x.grad, x_test.grad)

    def test_map_with_multiple_args(self):
        def f(a, b):
            return a[0] * b[0] + a[1] * b[1]

        def gen_inps(len_x, len_y):
            x = [torch.randn(5) for _ in range(len_x)]
            y = [torch.randn(5) for _ in range(len_y)]
            return x, y

        def g(x, y):
            return map(f, x, y)

        opt_g = torch.compile(g, fullgraph=True, backend="eager")

        inps = gen_inps(3, 3)
        self.assertEqual(type(g(*inps)), type(opt_g(*inps)))
        self.assertEqual(tuple(g(*inps)), tuple(opt_g(*inps)))

        inps = gen_inps(3, 5)
        self.assertEqual(type(g(*inps)), type(opt_g(*inps)))
        self.assertEqual(tuple(g(*inps)), tuple(opt_g(*inps)))
