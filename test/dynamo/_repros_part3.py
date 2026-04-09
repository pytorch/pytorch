try:
    from ._repros_common import (
        aot_graph_capture_backend,
        collections,
        CompileCounter,
        CompileCounterWithBackend,
        contextlib,
        copy,
        create_block_mask,
        dataclasses,
        dist,
        EagerAndRecordGraphs,
        expectedFailureDynamic,
        ExplainWithBackend,
        F,
        flex_attention,
        functools,
        fw_graph,
        HAS_MSGSPEC,
        HAS_OMEGACONG,
        Literal,
        mock,
        msgspec,
        nn,
        OmegaConf,
        os,
        parametrize,
        profile,
        ProfilerActivity,
        requires_cuda,
        serialTest,
        skipIfNotPy312,
        skipIfPy312,
        skipIfWindows,
        SM70OrLater,
        sys,
        TEST_CUDA,
        torch,
        TypedDict,
        typing,
        unittest,
        weakref,
        xfailIfS390X,
    )
except ImportError:
    from _repros_common import (
        aot_graph_capture_backend,
        collections,
        CompileCounter,
        CompileCounterWithBackend,
        contextlib,
        copy,
        create_block_mask,
        dataclasses,
        dist,
        EagerAndRecordGraphs,
        expectedFailureDynamic,
        ExplainWithBackend,
        F,
        flex_attention,
        functools,
        fw_graph,
        HAS_MSGSPEC,
        HAS_OMEGACONG,
        Literal,
        mock,
        msgspec,
        nn,
        OmegaConf,
        os,
        parametrize,
        profile,
        ProfilerActivity,
        requires_cuda,
        serialTest,
        skipIfNotPy312,
        skipIfPy312,
        skipIfWindows,
        SM70OrLater,
        sys,
        TEST_CUDA,
        torch,
        TypedDict,
        typing,
        unittest,
        weakref,
        xfailIfS390X,
    )


class ReproTestsMixin3:
    def test_staticmethod_allow_in_graph(self):
        class MyClass:
            i = 3

            @staticmethod
            def foo_inner(x):
                return torch.mul(x, MyClass.i)

            # if dynamo inlines with fullgraph, will error
            # verify that dynamo doesn't inline
            @staticmethod
            @torch._dynamo.allow_in_graph
            def foo1(x):
                torch._dynamo.graph_break()
                return MyClass.foo_inner(x)

        @torch.compile(backend="eager", fullgraph=True)
        def f_bad(x):
            return MyClass.foo1(x)

        f_bad(torch.ones(2, 2))

    def test_guard_with_tuple_mutation(self):
        class Foo:
            def __init__(self) -> None:
                self.x = 10

        foo = Foo()
        d = {
            "a": 2,
            "b": (foo,),
        }

        def fn(x, d):
            return x * d["a"] * d["b"][0].x

        opt_fn = torch.compile(fn, backend="eager")
        inp = torch.randn(3, 3)
        self.assertEqual(fn(inp, d), opt_fn(inp, d))
        d["b"][0].x = 12
        self.assertEqual(fn(inp, d), opt_fn(inp, d))

    def test_compile_complex_conj(self):
        def f(x):
            return torch.mul(x, 2j)

        x_ref = torch.randn(4, 2, requires_grad=True)
        x_test = x_ref.detach().clone().requires_grad_(True)

        out_ref = f(torch.view_as_complex(x_ref))
        out_test = torch.compile(f, backend="aot_eager")(torch.view_as_complex(x_test))
        self.assertEqual(out_ref, out_test)

        torch.view_as_real(out_ref).sum().backward()
        torch.view_as_real(out_test).sum().backward()
        self.assertEqual(x_ref.grad, x_test.grad)

    @unittest.skipIf(
        not SM70OrLater,
        "Triton only supports devices of CUDA capability >= 7.0",
    )
    def test_add_complex_conj(self):
        def f(x):
            return x + x.conj()

        x = torch.randn(4, dtype=torch.complex64, requires_grad=True)
        out = torch.compile(f, backend="eager")(x)
        expected_complex = (2 * x.real).to(dtype=out.dtype)

        self.assertTrue(out.dtype == torch.complex64)
        self.assertEqual(out, expected_complex)

    def test_partitioner_cse_respects_mutation_boundaries(self):
        set_available = hasattr(torch.ops, "fsdp") and hasattr(torch.ops.fsdp, "set_")
        if not set_available:
            return

        @torch.compile(backend="aot_eager_decomp_partition")
        def f(x, l):
            # z0 and z1 can be CSEd
            z0 = x.sin()
            z1 = x.sin()
            y = x + 1
            torch.ops.fsdp.copy_.default(x, y)
            # z3 and z3 can be CSEd with each other,
            # but *not* with z0/z1 (they cross a mutation boundary)
            z2 = x.sin()
            z3 = x.sin()
            return z0, z1, z2, z3, l**2

        x = torch.randn(3)
        x_clone = x.clone()
        l = torch.randn(3, requires_grad=True)
        z0, z1, z2, z3, _ = f(x, l)

        # the partitioner runs CSE. We expect that of the 4 sin() ops above:
        # - the first 2 are CSE'd
        # - the last 2 are CSE'd
        # - the set_() op in the middle is a mutation barrier, preventing CSE
        self.assertEqual(z0, (x_clone).sin())
        self.assertEqual(z1, (x_clone).sin())
        self.assertEqual(z2, (x_clone + 1).sin())
        self.assertEqual(z3, (x_clone + 1).sin())

    def test_fsdp_set_input_mutation_applied_when_input_gets_no_gradients(self):
        set_available = hasattr(torch.ops, "fsdp") and hasattr(torch.ops.fsdp, "set_")
        if not set_available:
            return

        @torch.compile(backend="aot_eager_decomp_partition")
        def f(x, l):
            z = x.sin()  # noqa: F841
            y = x + 1
            # graph input has its storage mutated
            torch.ops.fsdp.copy_.default(x, y)
            z2 = x.sin()
            return z2, l**2

        x = torch.randn(3)
        x_test = x.clone()
        l = torch.randn(3, requires_grad=True)
        result, _ = f(x, l)
        result_test, _ = torch.compile(f, backend="aot_eager_decomp_partition")(
            x_test, l
        )

        self.assertEqual(result, result_test)
        self.assertEqual(x, x_test)

    def test_inbuilt_nn_module_forward_after_hook_graph_break(self):
        # When a hook causes a graph break on an inbuilt nn.Module, the module's
        # forward should still be traced after the graph break.

        @torch._dynamo.disable
        def my_hook(module, inp):
            return inp

        class Wrapper(nn.Module):
            def __init__(self, lin):
                super().__init__()
                self.lin = lin

            def forward(self, x):
                return self.lin(x)

        lin = nn.Linear(10, 5)
        lin.register_forward_pre_hook(my_hook)
        model = Wrapper(lin)

        backend = EagerAndRecordGraphs()
        torch._dynamo.reset()
        compiled = torch.compile(model, backend=backend)
        output = compiled(torch.randn(3, 10))

        self.assertEqual(output.shape, torch.Size([3, 5]))
        self.assertEqual(len(backend.graphs), 1)
        graph_code = backend.graphs[0].print_readable(print_output=False)
        self.assertIn("torch._C._nn.linear", graph_code)

    def test_aot_autograd_runtime_wrapper_prologue_profiled(self):
        # Names for prologue profiling event
        prologue_name = "AOTDispatcher Runtime Wrapper Prologue"

        # Simple linear op to compile
        mod = torch.nn.Linear(4, 4)
        opt_mod = torch.compile(mod, backend="aot_eager")
        x = torch.randn(4, 4)

        # Run this test with grad and no-grad to test both boolean cases trace_joint
        for c in [contextlib.nullcontext, torch.no_grad]:
            # Run compiled op with profiling
            with c():
                # warmup before profiling
                opt_mod(x)
                with profile(activities=[ProfilerActivity.CPU]) as prof:
                    opt_mod(x)

            # Make sure events are populated then find prologue event and last start time
            events = prof.events()
            self.assertTrue(events is not None)

            prologue_event = None
            last_start_time = 0
            for event in events:
                if hasattr(event, "name") and prologue_name in event.name:
                    prologue_event = event
                if event.time_range.start > last_start_time:
                    last_start_time = event.time_range.start

            # Make sure prologue event exist
            self.assertTrue(prologue_event is not None)

            # Make sure there is at least one other event (compiled function) that starts
            # after prologue starts
            self.assertLess(prologue_event.time_range.end, last_start_time)

    def test_changing_stride(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x, y):
            return x * y

        for i in range(1, 4):
            x = torch.randn(4, i)

            # create a view for i > 1
            if i == 1:
                x1 = x
            else:
                x1 = x[:, 0:1]

            y = torch.randn(4, 1)
            print(x1.shape, y.shape)
            fn(x1, y)

        self.assertTrue(cnt.frame_count <= 2)

    def test_unsqueeze_mul_strides(self):
        # This is a case where we had an input that was marked unbacked:
        # size=[2, u0], stride=[1, 1] which is bad. We want it to actually
        # be size=[2, u0], stride=[u0, 1]. See more in the issue below:
        # https://github.com/pytorch/pytorch/issues/142024

        @torch.compile(backend="eager", fullgraph=True)
        def fn(aot6_sub_58, aot6_mul_170):
            aot6_unsqueeze_14 = torch.ops.aten.unsqueeze.default(aot6_mul_170, 1)
            return torch.ops.aten.mul.Tensor(aot6_sub_58, aot6_unsqueeze_14)

        aot6_sub_58 = torch.randn(2, 1)
        torch._dynamo.decorators.mark_unbacked(aot6_sub_58, 1)
        aot6_mul_170 = torch.randn(2)

        # No assert necessary since this used to crash.
        fn(aot6_sub_58, aot6_mul_170)

    def test_optimized_module_training(self):
        mod = torch.nn.Linear(3, 3)
        mod.eval()

        opt_mod = torch.compile(mod, backend="eager")
        self.assertFalse(opt_mod.training)

        opt_mod.train()
        self.assertTrue(opt_mod.training)
        self.assertTrue(mod.training)

        mod.eval()
        self.assertFalse(opt_mod.training)

    def test_optimized_module_patched_init(self):
        # A regression test for #138157, and the pattern acame from deepspeed.
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.mul(5.0)

        def patch_init(init):
            @functools.wraps(init)
            def wrapper(module, *args, **kwargs):
                if not hasattr(module, "_ds_child_entered"):
                    # child's __init__ was called, since parents all see the same object they can now skip post_init
                    module._ds_child_entered = True
                init(module, *args, **kwargs)

            return wrapper

        def patch_init_for_class(cls):
            if "__init__" in cls.__dict__:
                cls._old_init = cls.__init__
                cls.__init__ = patch_init(cls.__init__)

        patch_init_for_class(MyModule)
        mod = MyModule()
        opt_mod = torch.compile(mod, backend="eager")

        x = torch.rand(10)
        ref = mod(x)
        res = opt_mod(x)

        self.assertEqual(ref, res)

    def test_os_fspath(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            os.fspath(".")
            return torch.sin(x)

        fn(torch.randn(4))

    @requires_cuda
    # test involves custom ops that return unbacked symints
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    # test requires the activation memory budget code to think
    # that j() is banned from recompute
    @torch._functorch.config.patch(activation_memory_budget=0.5)
    def test_partitioner_activation_memory_budget_with_unbacked_symints(self):
        @torch.library.custom_op("test_partitioner::f", mutates_args=[])
        def f(x: torch.Tensor) -> torch.Tensor:
            return x.new_zeros(512, 1)

        @f.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            ctx = torch.library.get_ctx()
            s = ctx.new_dynamic_size()
            return torch.empty(s, 1, device=x.device, dtype=x.dtype)

        @torch.library.custom_op("test_partitioner::g", mutates_args=[])
        def g(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x, x[0].unsqueeze(-1)])

        @g.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x, x[0].unsqueeze(-1)])

        @torch.library.custom_op("test_partitioner::i", mutates_args=[])
        def i(x: torch.Tensor, sz: int) -> torch.Tensor:
            return torch.ones(sz, 1, dtype=x.dtype, device=x.device)

        @i.register_fake
        def _(x: torch.Tensor, sz: int) -> torch.Tensor:
            return torch.empty(sz, 1, dtype=x.dtype, device=x.device)

        @torch.library.custom_op("test_partitioner::j", mutates_args=[])
        def j(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + 1

        @j.register_fake
        def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            sz1 = x.shape[0] - 1
            sz2 = y.numel()
            torch._check(sz1 == sz2)
            # make this a reduction so partitioner bans recompute of it
            return x.sum()

        def f(x, param):
            y = torch.ops.test_partitioner.f(x)
            z = torch.ops.test_partitioner.g(y)
            z2 = torch.ops.test_partitioner.i(x, z.shape[0] - 1)
            z2 = torch.ops.test_partitioner.j(z, z2)
            return torch.matmul(x, param).sin() * z2.sum()

        x = torch.randn(512, 512, device="cuda")
        param = torch.randn(512, 512, device="cuda", requires_grad=True)
        out_ref = f(x, param)
        out_test = torch.compile(f, backend="aot_eager_decomp_partition")(x, param)
        self.assertEqual(out_ref, out_test)

    @requires_cuda
    # This test will fail as flip in combination with particular input lengths
    # produces weird results.
    # This is under investigations in
    # https://github.com/pytorch/pytorch/issues/131805
    @unittest.skip("Skip this flip test for the moment. It is under investigation")
    def test_flip_bad_accuracy(self):
        import torch
        import torch._dynamo.config
        import torch._functorch.config
        import torch._inductor.config
        import torch._inductor.inductor_prims
        import torch.fx.experimental._config

        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, arg0_1):
                rev = torch.ops.prims.rev.default(arg0_1, [0])
                arg0_1 = None
                slice_1 = torch.ops.aten.slice.Tensor(rev, 0, 0, -1, 2)
                slice_2 = torch.ops.aten.slice.Tensor(rev, 0, 1, 9223372036854775807, 2)
                add_1 = torch.ops.aten.add.Tensor(slice_1, slice_2)
                slice_1 = slice_2 = None
                slice_3 = torch.ops.aten.slice.Tensor(add_1, 0, 0, -1, 2)
                slice_4 = torch.ops.aten.slice.Tensor(
                    add_1, 0, 1, 9223372036854775807, 2
                )
                add_2 = torch.ops.aten.add.Tensor(slice_3, slice_4)
                slice_3 = slice_4 = None
                slice_5 = torch.ops.aten.slice.Tensor(add_2, 0, 0, -1, 2)
                slice_6 = torch.ops.aten.slice.Tensor(
                    add_2, 0, 1, 9223372036854775807, 2
                )
                add_3 = torch.ops.aten.add.Tensor(slice_5, slice_6)
                slice_5 = slice_6 = None
                slice_9 = torch.ops.aten.slice.Tensor(add_2, 0, 0, 1)
                add_2 = None
                unsqueeze = torch.ops.aten.unsqueeze.default(slice_9, 1)
                slice_9 = None
                unsqueeze_1 = torch.ops.aten.unsqueeze.default(add_3, 1)
                add_3 = None
                cat = torch.ops.aten.cat.default([unsqueeze, unsqueeze_1], 1)
                unsqueeze = unsqueeze_1 = None
                view = torch.ops.aten.view.default(cat, [2])
                cat = None
                slice_10 = torch.ops.aten.slice.Tensor(view, 0, 0, -1)
                slice_11 = torch.ops.aten.slice.Tensor(
                    add_1, 0, 2, 9223372036854775807, 2
                )
                add_5 = torch.ops.aten.add.Tensor(slice_10, slice_11)
                slice_10 = slice_11 = None
                slice_12 = torch.ops.aten.slice.Tensor(add_1, 0, 0, 1)
                add_1 = None
                cat_1 = torch.ops.aten.cat.default([slice_12, add_5])
                slice_12 = add_5 = None
                unsqueeze_2 = torch.ops.aten.unsqueeze.default(cat_1, 1)
                cat_1 = None
                unsqueeze_3 = torch.ops.aten.unsqueeze.default(view, 1)
                view = None
                cat_2 = torch.ops.aten.cat.default([unsqueeze_2, unsqueeze_3], 1)
                unsqueeze_2 = unsqueeze_3 = None
                view_1 = torch.ops.aten.view.default(cat_2, [4])
                cat_2 = None
                slice_13 = torch.ops.aten.slice.Tensor(
                    rev, 0, 2, 9223372036854775807, 2
                )
                add_6 = torch.ops.aten.add.Tensor(view_1, slice_13)
                slice_13 = None
                slice_14 = torch.ops.aten.slice.Tensor(rev, 0, 0, 1)
                rev = None
                cat_3 = torch.ops.aten.cat.default([slice_14, add_6])
                slice_14 = add_6 = None
                constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
                    view_1, [0, 1], 0.0
                )
                view_1 = None
                unsqueeze_4 = torch.ops.aten.unsqueeze.default(cat_3, 1)
                cat_3 = None
                unsqueeze_5 = torch.ops.aten.unsqueeze.default(constant_pad_nd, 1)
                constant_pad_nd = None
                cat_4 = torch.ops.aten.cat.default([unsqueeze_4, unsqueeze_5], 1)
                unsqueeze_4 = unsqueeze_5 = None
                view_2 = torch.ops.aten.view.default(cat_4, [10])
                cat_4 = None
                slice_15 = torch.ops.aten.slice.Tensor(view_2, 0, 0, 9)
                view_2 = None
                rev_1 = torch.ops.prims.rev.default(slice_15, [0])
                slice_15 = None
                return (rev_1,)

        mod = Repro()
        x = torch.arange(9, device=torch.device("cuda"))

        @torch.compile
        def f(x):
            return mod(x)

        out = f(x)
        self.assertEqual(torch.flip(torch.cumsum(torch.flip(x, [0]), 0), [0]), out[0])

    def test_return_value_duplication_tensor(self) -> None:
        def fn(val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return val * 2, val * 2

        x = torch.randn(2, requires_grad=True)

        expect = fn(x)
        self.assertNotEqual(
            expect[0].untyped_storage().data_ptr(),
            expect[1].untyped_storage().data_ptr(),
        )

        actual = torch.compile(fn, backend="aot_eager")(x)
        self.assertNotEqual(
            actual[0].untyped_storage().data_ptr(),
            actual[1].untyped_storage().data_ptr(),
        )

    def test_return_value_duplication_mixed_grad(self) -> None:
        def fn(val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                out0 = val + 1
            out1 = val + 1
            return out0, out1

        x = torch.randn(2, requires_grad=True)

        with torch.enable_grad():
            expect = fn(x)
            actual = torch.compile(fn, backend="aot_eager")(x)

            self.assertEqual(expect[0].requires_grad, actual[0].requires_grad)
            self.assertEqual(expect[1].requires_grad, actual[1].requires_grad)

    def test_return_value_duplication_scalar(self) -> None:
        def fn(val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x, y = val * 2, val * 2
            return x[0], y[0]

        x = torch.randn(2, requires_grad=True)

        expect = fn(x)
        self.assertNotEqual(
            expect[0].untyped_storage().data_ptr(),
            expect[1].untyped_storage().data_ptr(),
        )

        actual = torch.compile(fn, backend="aot_eager")(x)
        self.assertNotEqual(
            actual[0].untyped_storage().data_ptr(),
            actual[1].untyped_storage().data_ptr(),
        )

    def test_torch_compile_in_compile_frame(self):
        def gn(x, c=None):
            if c is None:
                c = 2
            return c * x

        def outer_func(x):
            return torch.compile(gn, backend="eager")(x)

        compile_outer = torch.compile(outer_func, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = outer_func(x)
        res = compile_outer(x)
        self.assertEqual(ref, res)

    @unittest.skipIf(not HAS_MSGSPEC, "missing msgspec package")
    def test_c_defined_metaclass(self):
        class User(msgspec.Struct):
            """A new type describing a User"""

            name: str
            value: int

        def fn(x):
            u = User("alice", 10)
            return x * u.value

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x), opt_fn(x))

    @unittest.skipIf(not HAS_OMEGACONG, "missing omegaconf package")
    def test_omegaconf_dictconfig(self):
        def fn(cfg, x):
            a = cfg["foo"].a * x
            b = cfg.bar["b"] * a
            cfg.__dict__["baz"] = 4
            return b * cfg.baz

        config = OmegaConf.create({"foo": {"a": 3}, "bar": {"b": 5}})

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        fn(config, x)
        cloned_config = copy.deepcopy(config)
        opt_fn(cloned_config, x)

        self.assertEqual(fn(config, x), opt_fn(config, x))
        self.assertEqual(cloned_config.baz, 4)

    @unittest.skipIf(not HAS_OMEGACONG, "missing omegaconf package")
    def test_omegaconf_listconfig_contains(self):
        def fn(cfg, x):
            if 1 in cfg:
                return torch.sin(x)
            return torch.cos(x)

        config = OmegaConf.create([1, 2, 3, {"key": "value"}])

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(config, x), opt_fn(config, x))

    def test_overwriting_params(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(2, 2)
                self.fc2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        class ZeROOrderedDict(collections.OrderedDict):
            def __init__(self, parent_module=None, *args, **kwargs):
                """A replacement for ``collections.OrderedDict`` to detect external ZeRO params.

                Args:
                    parent_module (``collections.OrderedDict``): the collection to replace
                """

                super().__init__(*args, **kwargs)
                self._parent_module = parent_module

            def __getitem__(self, key):
                param = super().__getitem__(key)

                # Params can be registered as None (e.g., bias)
                if param is None:
                    return param

                # do something here
                return param

        def inject_parameters(module, cls):
            for module in module.modules():  # noqa: B020
                if cls == ZeROOrderedDict:
                    new_param = cls(parent_module=module)
                else:
                    new_param = cls()

                for key, param in module._parameters.items():
                    new_param[key] = param
                module._parameters = new_param

        model = M()

        inject_parameters(model, ZeROOrderedDict)

        model = torch.compile(model, backend="eager", fullgraph=True)

        x = torch.ones(2)
        with torch.no_grad():
            model(x)

    def test_typed_dict(self):
        class LlavaImagePixelInputs(TypedDict):
            type: Literal["pixel_values"]  # noqa: F821
            data: torch.Tensor
            """Shape: `(batch_size, num_channels, height, width)`"""

        def fn(x, y):
            obj = LlavaImagePixelInputs(type=int, data=y)
            out = x * obj["data"]
            obj["data"] = 3
            return out * obj["data"]

        x, y = torch.randn(4), torch.randn(4)
        ref = fn(x, y)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)

        self.assertEqual(ref, res)

    def test_typed_dict_total(self):
        class LlavaImagePixelInputs(TypedDict):
            type: Literal["pixel_values"]  # noqa: F821
            data: torch.Tensor
            """Shape: `(batch_size, num_channels, height, width)`"""

        def fn(x, y):
            obj = LlavaImagePixelInputs(data=y, total=False)
            return x * obj["data"]

        x, y = torch.randn(4), torch.randn(4)
        ref = fn(x, y)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)

        self.assertEqual(ref, res)

    @skipIfPy312  # listcomp bytecode is optimized
    @skipIfWindows(msg="TODO: (xuhancn) fix, AssertionError: Scalars are not equal!")
    def test_listcomp(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._num = 4

            @torch._dynamo.disable(recursive=False)
            def forward(self, x):
                values = [i * torch.cos(x) for i in range(self._num)]
                return sum(values)

        mod = Module()

        def fn(x):
            return mod(x)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.randn(4)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnt.frame_count, 1)
        # Ensure that the listcomp is fully compiled
        self.assertEqual(cnt.op_count, 8)

    def test_distributions_subclass(self):
        import torch
        from torch.distributions import Categorical

        class SubCateg(Categorical):
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def make_dist_and_execute(t, d):
            categ = d(logits=t)
            a = categ.log_prob(categ.sample()) + categ.probs + categ.logits
            return a

        for _ in range(2):
            make_dist_and_execute(torch.randn(10), SubCateg)

    def test_bitwise_print_precedence(self):
        import math

        @torch.compile(fullgraph=True, dynamic=True, backend="eager")
        def f(x):
            torch._check(math.floor((x.size(0) | 3) * 4) == 12)
            return x.sin()

        f(torch.randn(2))

    def test_tensor_split_within_device_cm(self):
        @torch.compile(fullgraph=True, backend="eager")
        def split(x):
            return x.split(4, 0)

        x = torch.zeros(12)
        res = split(x)

        with torch.device("cpu"):
            self.assertEqual(res, split(x))

    def test_method_overriding(self):
        class DilateConv(torch.nn.Module):
            def __init__(
                self,
                dilate_func=None,
            ):
                super().__init__()
                self.dilate_func = dilate_func

            def forward(self, x):
                return self.dilate_func() * torch.sin(x)

        class MainModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = DilateConv(self.dilate_func)
                self.a = 4

            def dilate_func(self):
                return self.a

            def forward(self, x):
                return self.mod(x)

        mod = MainModule()

        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = mod(x)
        res = opt_mod(x)
        self.assertEqual(ref, res)

    def test_symnode_is_op(self):
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x, xs):
            if x.size(0) is xs:
                return x + 1
            else:
                return x * 2

        t = torch.randn(2)
        res = f(t, [1, 2])
        self.assertEqual(t * 2, res)

    def test_compile_copy__int_overload(self):
        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x):
            return x.copy_(1)

        t = torch.zeros(2)
        res = f(t)
        self.assertEqual(torch.ones_like(t), res)

    def test_symnode_is_not_op(self):
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x, xs):
            if x.size(0) is not xs:
                return x + 1
            else:
                return x * 2

        t = torch.randn(2)
        res = f(t, [1, 2])
        self.assertEqual(t + 1, res)

    def test_symint_bitwise(self):
        def fn(x):
            z = x.shape[0]
            z |= z >> 1
            z |= z << 1
            z &= z | (z > 1)
            y = (z > 1) | (z <= 1)
            # test composition with non-bitwise ops
            z = (z | z) % 6
            return y, z

        opt_fn = torch.compile(fn, backend="eager", dynamic=True, fullgraph=True)
        inp = torch.randn(3, 3)
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_bitwise_op_guard(self):
        # attempt evaluating a guard with BitwiseFn_bitwise_[and/or]
        def fn(x):
            if x.shape[0] | x.shape[1] > 4:
                x = x + 1
            if x.shape[0] & x.shape[1] > 2:
                return x + 1
            return x - 1

        opt_fn = torch.compile(fn, backend="eager", dynamic=True, fullgraph=True)
        inp = torch.randn(3, 3)
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_ones_out_dynamic(self):
        def ones_fn(size, out):
            return torch.ones(size, out=out)

        opt_model = torch.compile(ones_fn, backend="eager")

        out1 = torch.empty(2, 3)
        opt_model((2, 3), out1)

        out2 = torch.empty(3, 4)
        opt_model((3, 4), out2)

    def test_zeros_out_dynamic(self):
        def zeros_fn(size, out):
            return torch.zeros(size, out=out)

        opt_model = torch.compile(zeros_fn, backend="eager")

        out1 = torch.empty(2, 3)
        opt_model((2, 3), out1)

        out2 = torch.empty(3, 4)
        opt_model((3, 4), out2)

    def test_empty_out_dynamic(self):
        def empty_fn(size, out):
            return torch.empty(size, out=out)

        opt_model = torch.compile(empty_fn, backend="eager")

        out1 = torch.empty(2, 3)
        opt_model((2, 3), out1)

        out2 = torch.empty(3, 4)
        opt_model((3, 4), out2)

    def test_dataclass_in_module(self):
        @dataclasses.dataclass
        class MyData:
            value: float

        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.my_data = MyData(value=3.14)

            def forward(self, x):
                # Make sure to use the scalar 'value' correctly in tensor operations
                value_tensor = torch.tensor(self.my_data.value)
                return x + value_tensor

        model = MyModel()
        inputs = torch.randn(2, 2)
        expected = model(inputs)
        compiled_model = torch.compile(model, backend="eager")
        actual = compiled_model(inputs)
        self.assertEqual(actual, expected)

    def test_no_tracing_into_eval_frame(self):
        # test that dynamo doesn't trace into nested calls from eval_frame
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x + 1

        orig_fn = torch._dynamo.eval_frame._maybe_set_eval_frame

        def bad(*args, **kwargs):
            torch._dynamo.graph_break()
            return orig_fn(*args, **kwargs)

        with mock.patch("torch._dynamo.eval_frame._maybe_set_eval_frame", bad):
            fn(torch.ones(3))

    @torch._dynamo.config.patch(raise_on_ctx_manager_usage=False)
    def test_no_tracing_into_eval_frame_ctx_manager(self):
        # Test that dynamo doesn't trace into nested calls from eval_frame
        # when using a context manager.
        # Even though we don't officially support Dynamo context managers, we still
        # have tests that use them, so we should still make sure the eval_frame callback
        # is set at the correct places in these cases.
        def fn(x):
            return x + 1

        orig_fn = torch._dynamo.eval_frame._maybe_set_eval_frame

        def bad(*args, **kwargs):
            torch._dynamo.graph_break()
            return orig_fn(*args, **kwargs)

        with mock.patch("torch._dynamo.eval_frame._maybe_set_eval_frame", bad):
            with torch._dynamo.optimize_assert("eager"):
                fn(torch.ones(3))

    @torch._dynamo.config.patch(allow_empty_graphs=True)
    @parametrize("fullgraph", [True, False])
    def test_empty_graph_nested_calls(self, fullgraph):
        def k(x):
            return x

        def g(x):
            return k(x)

        def f(x):
            return g(x)

        # TODO clear this on all tests
        torch._dynamo.eval_frame.clear_dynamo_tls()

        opt_f = torch.compile(f, backend="eager", fullgraph=fullgraph, dynamic=False)
        opt_f(torch.randn(3))
        # we should not be compiling g or h as top-level functions
        self.assertEqual(len(torch._dynamo.eval_frame.dynamo_tls.traced_frame_infos), 1)
        # no recompilation
        opt_f(torch.randn(3))
        self.assertEqual(len(torch._dynamo.eval_frame.dynamo_tls.traced_frame_infos), 1)
        # recompilation
        opt_f(torch.randn(4))
        self.assertEqual(len(torch._dynamo.eval_frame.dynamo_tls.traced_frame_infos), 2)

    def test_torchname(self):
        def fn(obj):
            return torch.typename(obj)

        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(fn(typing.Any), opt_fn(typing.Any))

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    @unittest.skipIf(not dist.is_available(), "test requires distributed")
    # TODO: Remoe this skip once nccl issue if fixed
    @unittest.skip(
        "Failing with ncc update 2.25.1 : https://github.com/pytorch/pytorch/issues/147141"
    )
    def test_ddp_checkpoint(self):
        # https://github.com/pytorch/pytorch/issues/144035
        DIM = 256
        SEQ_LEN = 32

        @torch.compile(backend="eager", fullgraph=True)
        def mlp_forward(x, w1, w2, b1, b2):
            y = F.linear(x, w1, b1)
            y = F.relu(y)
            y = F.linear(y, w2, b2)
            return y

        class MLP(nn.Module):
            def __init__(
                self,
                in_features: int,
                hidden_features: int,
                out_features: int,
            ):
                super().__init__()
                self.w_in = nn.Parameter(torch.randn(hidden_features, in_features))
                self.w_out = nn.Parameter(torch.randn(out_features, hidden_features))
                self.b_in = nn.Parameter(torch.randn(hidden_features))
                self.b_out = nn.Parameter(torch.randn(out_features))

            def forward(self, x):
                result = torch.utils.checkpoint.checkpoint(
                    mlp_forward,
                    x,
                    self.w_in,
                    self.w_out,
                    self.b_in,
                    self.b_out,
                    use_reentrant=False,
                )
                assert isinstance(result, torch.Tensor)  # noqa: S101
                return result

        x = torch.randn(100, SEQ_LEN, DIM)
        y = torch.zeros(100)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
        model = MLP(DIM, 4 * DIM, DIM)

        try:
            # required for DDP wrapper initialization
            prior_master_addr = os.environ.get("MASTER_ADDR", None)
            prior_master_port = os.environ.get("MASTER_PORT", None)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            dist.init_process_group(backend="nccl", world_size=1, rank=0)
            model = model.to("cuda")
            model = nn.parallel.DistributedDataParallel(model)

            for batch in dataloader:
                x, y = batch
                x = x.to("cuda")
                output = model(x)
                loss = output.sum()
                loss.backward()
        finally:
            dist.destroy_process_group()
            if prior_master_addr:
                os.environ["MASTER_ADDR"] = prior_master_addr
            else:
                del os.environ["MASTER_ADDR"]

            if prior_master_port:
                os.environ["MASTER_PORT"] = prior_master_port
            else:
                del os.environ["MASTER_PORT"]

    @torch._dynamo.config.patch(
        recompile_limit=1,
        fail_on_recompile_limit_hit=True,
    )
    def test_compilation_metrics_on_error(self):
        torch._dynamo.utils.clear_compilation_metrics()

        @torch.compile(backend="eager")
        def fn(x):
            # force a recompile in a way friendly to test_dynamic_shapes
            if x.numel() == 100:
                return x.sum()
            elif x.numel() == 10000:
                return x.sum()

        x = torch.randn(10, 10)
        y = torch.randn(100, 100)
        metrics = torch._dynamo.utils._compilation_metrics
        self.assertEqual(len(metrics), 0)

        fn(x)
        self.assertTrue(metrics is torch._dynamo.utils._compilation_metrics)
        self.assertEqual(len(metrics), 1)
        latest_metrics = metrics[-1]
        self.assertTrue(latest_metrics.dynamo_config is not None)
        self.assertTrue(latest_metrics.recompile_reason is None)

        with self.assertRaises(torch._dynamo.exc.FailOnRecompileLimitHit):
            fn(y)
        self.assertTrue(metrics is torch._dynamo.utils._compilation_metrics)
        self.assertEqual(len(metrics), 2)
        latest_metrics = metrics[-1]
        self.assertTrue(latest_metrics.dynamo_config is not None)
        self.assertTrue(latest_metrics.recompile_reason is not None)

        torch._dynamo.utils.clear_compilation_metrics()

    @serialTest()
    def test_dont_dce_rand(self):
        # https://github.com/pytorch/pytorch/issues/143431
        def f(image_latent):
            B = 2
            num_ref = 3
            num_tar = 3
            x = torch.rand(B, 12)
            indices = torch.argsort(torch.rand(*x.shape), dim=-1)[
                :, : num_ref + num_tar
            ]
            return image_latent[torch.arange(B).unsqueeze(-1), indices][:, :num_ref]

        # Generate input once to ensure consistency across runs
        torch.manual_seed(54321)
        torch.cuda.manual_seed_all(54321)
        image_latent = torch.randn((2, 12, 16, 32, 32))

        torch.manual_seed(54321)
        torch.cuda.manual_seed_all(54321)
        expected = f(image_latent).sum()

        # https://github.com/pytorch/pytorch/issues/147171
        with torch._inductor.config.patch(fallback_random=True):
            for backend in ["eager", "aot_eager"]:
                torch.manual_seed(54321)
                torch.cuda.manual_seed_all(54321)
                actual = torch.compile(backend=backend, fullgraph=True)(f)(
                    image_latent
                ).sum()
                self.assertEqual(actual, expected)

    def test_incompatible_configs(self):
        with torch._dynamo.config.patch(
            suppress_errors=False, fail_on_recompile_limit_hit=False
        ):
            torch.compile(lambda: None, backend="eager")

        with torch._dynamo.config.patch(
            suppress_errors=True, fail_on_recompile_limit_hit=False
        ):
            torch.compile(lambda: None, backend="eager")

        with torch._dynamo.config.patch(
            suppress_errors=False, fail_on_recompile_limit_hit=True
        ):
            torch.compile(lambda: None, backend="eager")

        with (
            torch._dynamo.config.patch(
                suppress_errors=True, fail_on_recompile_limit_hit=True
            ),
            self.assertRaises(AssertionError),
        ):
            torch.compile(lambda: None, backend="eager")

    def test_str_isalnum(self):
        def f(x, c):
            str.isalnum(c)
            return x.sin()

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3)
        c = "foobar"
        self.assertEqual(f(x, c), opt_f(x, c))

    def test_nn_param_freevar_codegen(self):
        class Model2(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
                self.batchnorm = nn.BatchNorm2d(num_features=5)
                self.conv_weight = torch.randn(5, 3, 3, 3)
                self.conv_bias = torch.randn(5)

            def forward(self, x):
                self.conv.weight = nn.Parameter(self.conv_weight)
                self.conv.bias = nn.Parameter(self.conv_bias, requires_grad=False)
                self.conv.eval()
                x = self.conv(x)
                x = self.batchnorm(x)
                x = F.relu(x)
                return x

        input_tensor = torch.randn(1, 3, 10, 10)
        func = Model2().to("cpu")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        with torch.no_grad():
            func.train(False)
            v1 = func(input_tensor)
            jit_func = torch.compile(wrapper, backend="eager", fullgraph=True)
            v2 = jit_func(input_tensor)
            self.assertEqual(v1, v2)

    def test_amp_foreach_fake_impl(self):
        inv_scale = torch.full((1,), 0.25)
        found_inf = torch.full((1,), 0.0)
        grads = [torch.ones(10), torch.ones(10)]

        def f():
            res = torch._amp_foreach_non_finite_check_and_unscale_(
                grads, found_inf, inv_scale
            )
            return res

        ref = f()
        res = torch.compile(f, backend="aot_eager")()
        self.assertEqual(ref, res)

    def test_deleted_compile_wrapper_segfault(self):
        def fn(x):
            return x + 1

        opt_fn = torch.compile(fn, backend="eager")
        # This calls cached_backend.clear() which removes any strong references
        # to the callback
        torch._dynamo.reset()
        opt_fn(torch.randn(3))
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.randn(3))  # possible segfault due to first opt_fn deletion

    def test_delete_local_error(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = x + 1
            del y
            z = y + 1  # noqa: F821
            return z

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            fn(torch.ones(3))

    def test_nanmean_out(self):
        def f(x, out):
            torch.nanmean(x, out=out)

        x = torch.randn(4)
        out_ref = torch.tensor(0.0)
        out_res = torch.tensor(0.0)

        f(x, out_ref)
        torch.compile(f, backend="eager", fullgraph=True)(x, out_res)
        self.assertEqual(out_ref, out_res)

    @skipIfNotPy312
    def test_sys_monitoring(self):
        found_dynamo = False
        found_compiled_graph = False
        compiled_graph = None

        def backend(gm, _):
            nonlocal compiled_graph
            compiled_graph = gm
            return gm

        def callback(code, offset):
            nonlocal found_dynamo
            nonlocal found_compiled_graph
            torch._dynamo.graph_break()
            if (
                code
                is torch._dynamo.symbolic_convert.InstructionTranslator.run.__code__
            ):
                found_dynamo = True
            elif compiled_graph and code is compiled_graph.__call__.__code__:
                found_compiled_graph = True

        tool_id = 0
        sys.monitoring.use_tool_id(tool_id, "test")
        old_events = sys.monitoring.get_events(tool_id)
        old_callback = sys.monitoring.register_callback(
            tool_id, sys.monitoring.events.PY_START, callback
        )
        sys.monitoring.set_events(tool_id, sys.monitoring.events.PY_START)
        try:

            @torch.compile(backend=backend, fullgraph=True)
            def fn(x):
                return x + 1

            fn(torch.ones(3))
            # sys.monitoring should still run in Python dynamo
            self.assertTrue(found_dynamo)
            # sys.monitoring should still run on the compiled graph
            self.assertTrue(found_compiled_graph)
        finally:
            sys.monitoring.set_events(tool_id, old_events)
            sys.monitoring.register_callback(
                tool_id, sys.monitoring.events.PY_START, old_callback
            )
            sys.monitoring.free_tool_id(tool_id)

    def test_312_local_cell_overlap(self):
        keys = range(10)
        allowed = [0, 1, 2, 3]

        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            key = [key for key in keys if key in allowed]

            def inner():
                nonlocal key

            return x + key[0]

        self.assertEqual(
            fn(torch.ones(3)), torch.compile(fn, backend="eager")(torch.ones(3))
        )

    def test_cells_unsupported_step_exception(self):
        # This error happened because:
        #  - we were generating cells into a list on the stack
        #  - we encountered an unsupported step, resulting in a step graph break
        #  - we encounter an exception, which pops the stack until it reaches a certain length;
        #    the presence of the list of cells then messes things up.

        cell = 0

        @torch.compile(backend="eager")
        def fn(x):
            x = x + 1 + 2
            torch._dynamo.step_unsupported()
            with contextlib.nullcontext():
                print(cell)
                raise AssertionError

        with self.assertRaises(AssertionError):
            fn(torch.ones(3))

    def test_unbind_copy_out(self):
        def f(eye, out):
            torch.unbind_copy(eye, out=out)

        eye = torch.eye(3)
        out_ref = (torch.zeros(3), torch.zeros(3), torch.zeros(3))
        out_res = (torch.zeros(3), torch.zeros(3), torch.zeros(3))

        f(eye, out_ref)
        torch.compile(f, backend="eager", fullgraph=True)(eye, out_res)
        self.assertEqual(out_ref, out_res)

    def test_setitem_tensor_prop(self):
        # Using the composite implicit of the forward would be incorrect
        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return torch.matmul(x, x.t())

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out

        def fn(x, y):
            x[0] = y[0]
            return MyFn.apply(x)

        def inputs():
            torch.manual_seed(123)
            x = torch.randn(10, 10)
            y = torch.randn(10, 10, requires_grad=True)
            return x, y

        x1, y1 = inputs()
        fn(x1, y1).sum().backward()
        self.assertTrue(x1.requires_grad)

        x2, y2 = inputs()
        torch.compile(fn, backend="eager")(x2, y2).sum().backward()
        self.assertTrue(x2.requires_grad)

        self.assertEqual(y1.grad, y2.grad)

    def test_nn_parameter_ctor_graph_breaks(self):
        def fn():
            param = torch.nn.Parameter(torch.ones(10))
            return param * 2

        self.maxDiff = None
        eb = ExplainWithBackend("eager")
        optimized_fn = torch.compile(fn, backend=eb)
        _ = optimized_fn()
        explain_output = eb.output()
        self.assertEqual(explain_output.graph_break_count, 1)
        expected_msg = (
            "Attempted to use `torch.nn.Parameter()` constructor with Dynamo\n"
            "  Explanation: Dynamo does not support this\n"
            "  Hint: Try to construct `torch.nn.Parameter()` outside the compiled region.\n"
            "  Hint: If this is not possible, turn `graph_break_on_nn_param_ctor` off\n"
            "  Hint: It may be possible to write Dynamo tracing rules for this code. "
            "Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.\n\n"
            "  Developer debug context: \n\n"
            " For more details about this graph break, please visit: "
            "https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0264.html"
        )
        self.assertEqual(explain_output.break_reasons[0].reason, expected_msg)

    @parametrize("backend", ["eager", "inductor"])
    def test_issue164247(self, backend: str):
        if backend == "inductor" and torch._dynamo.config.dynamic_shapes:
            raise unittest.SkipTest(
                "Skip only in dynamic-shapes wrapper (known issue #157612)"
            )

        class MixedFakeModeModel(nn.Module):
            def __init__(self, dim=64):
                super().__init__()
                self.dim = dim
                self.lin = torch.nn.Linear(64, 64)

            def forward(self, x):
                batch_size, seq_len, _ = x.shape

                # Process input first - this creates fake tensors in export's fake mode
                processed = self.lin(x)

                # Create some computation that depends on processed tensor
                intermediate = processed.sum(dim=-1).detach()  # Shape: (batch, seq_len)

                def dynamic_mask_function(batch_idx, head_idx, q_idx, kv_idx):
                    threshold = intermediate[
                        batch_idx, q_idx % seq_len
                    ]  # Access the captured tensor
                    return (kv_idx <= q_idx) & (threshold > 0)

                block_mask = create_block_mask(
                    mask_mod=dynamic_mask_function,
                    B=batch_size,
                    H=None,
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    device=x.device,
                    _compile=False,
                )
                q = processed.view(batch_size, 1, seq_len, self.dim).detach()
                k = processed.view(batch_size, 1, seq_len, self.dim).detach()
                v = processed.view(batch_size, 1, seq_len, self.dim).detach()

                out = torch.compile(flex_attention)(q, k, v, block_mask=block_mask)
                out = flex_attention(q, k, v, block_mask=block_mask)

                return out

        backend_counter = CompileCounterWithBackend(backend)
        model = MixedFakeModeModel()
        compiled = torch.compile(model, backend=backend_counter, fullgraph=True)

        if backend == "inductor":
            # A known InductorError Issue https://github.com/pytorch/pytorch/issues/157612
            with self.assertRaises(RuntimeError):
                compiled(torch.randn(2, 128, 64))
        else:
            compiled(torch.randn(2, 128, 64))

        # One graph, so no graph breaks
        self.assertEqual(backend_counter.frame_count, 1)
        self.assertEqual(len(backend_counter.graphs), 1)

    def test_guard_same_frame_fail_message(self):
        import torch._dynamo.guards as g

        # deterministically fail check on the same frame to verify error message correctness
        # the other example of fail might be datetime.now() until patched - see issue #164990
        compile_check_fn = g.CheckFunctionManager.compile_check_fn

        def wrapper(self, builder, sorted_guards, guard_fail_fn):
            compile_check_fn(self, builder, sorted_guards, guard_fail_fn)

            def check(x):
                return False

            self.guard_manager.check = check

        with mock.patch.object(g.CheckFunctionManager, "compile_check_fn", new=wrapper):

            class Model(nn.Module):
                def forward(self, x):
                    return x + 1

            model = Model()
            x = torch.randn(5)

            with self.assertRaises(AssertionError) as e:
                torch.compile(model, backend="eager")(x)

        msg = str(e.exception)
        self.assertIn(
            "Guard failed on the same frame it was created. This is a bug - please create an issue."
            "Guard fail reason: ",
            msg,
        )

    @xfailIfS390X
    @unittest.skipIf(
        sys.version_info < (3, 12) or sys.version_info >= (3, 14),
        "only 3.12, 3.13 affected by c recursion limit",
    )
    def test_dynamo_set_recursion_limit(self):
        old_recursion_limit = sys.getrecursionlimit()
        old_dynamo_recursion_limit = torch._dynamo.get_recursion_limit()
        try:

            def fn(x, n):
                if n == 0:
                    return x
                return fn(x, n - 1) + 1

            sys.setrecursionlimit(100)

            with self.assertRaises(RecursionError):
                fn(torch.ones(3), 500)

            sys.setrecursionlimit(1000)

            fn(torch.ones(3), 500)
            opt_fn = torch.compile(fn, backend="eager", dynamic=False)
            sys.setrecursionlimit(20000)
            with self.assertRaises(Exception):
                opt_fn(torch.ones(3), 500)

            torch._dynamo.set_recursion_limit(20000)
            self.assertEqual(fn(torch.ones(3), 500), opt_fn(torch.ones(3), 500))
        finally:
            torch._dynamo.set_recursion_limit(old_dynamo_recursion_limit)
            sys.setrecursionlimit(old_recursion_limit)

    @unittest.skipIf(
        sys.version_info < (3, 12) or sys.version_info >= (3, 14),
        "only 3.12, 3.13 affected by c recursion limit",
    )
    def test_dynamo_set_recursion_limit_usage(self):
        old_dynamo_recursion_limit = torch._dynamo.get_recursion_limit()
        try:
            torch._dynamo.set_recursion_limit(500)
            self.assertEqual(torch._dynamo.get_recursion_limit(), 500)

            @torch.compile(backend="eager", dynamic=False)
            def fn(x, n):
                if n == 0:
                    return x
                return fn(x, n - 1) + 1

            # a limit of 500 should be lower than the default limit
            with self.assertWarnsRegex(RuntimeWarning, "new c_recursion limit"):
                fn(torch.ones(3), 5)

            with self.assertRaisesRegex(ValueError, "recursion limit"):
                torch._dynamo.set_recursion_limit(0)

            self.assertEqual(torch._dynamo.get_recursion_limit(), 500)
        finally:
            torch._dynamo.set_recursion_limit(old_dynamo_recursion_limit)

    @expectedFailureDynamic
    def test_dynamo_default_lru_cache_behavior(self):
        @torch.compile(backend="eager")
        def fn(x):
            return x + 10

        torch._dynamo.reset()
        if torch._C._dynamo.eval_frame._debug_get_cache_entry_list(
            fn._torchdynamo_orig_callable.__code__
        ):
            raise AssertionError("Expected no cache entries after reset")

        # Step 1: Compile a static shapes graph
        x = torch.randn(10, 10)
        fn(x)
        a = torch._C._dynamo.eval_frame._debug_get_cache_entry_list(
            fn._torchdynamo_orig_callable.__code__
        )
        self.assertEqual(len(a), 1)
        static_shapes_cache_entry = a[0]

        # Step 2: Compile a dynamic shapes graph
        y = torch.randn(20, 20)
        fn(y)
        b = torch._C._dynamo.eval_frame._debug_get_cache_entry_list(
            fn._torchdynamo_orig_callable.__code__
        )
        self.assertEqual(len(b), 2)
        self.assertEqual(b[1], static_shapes_cache_entry)
        dynamic_shapes_cache_entry = b[0]

        # Step 3: Run with Step 1's inputs
        # LRU cache will match against dynamic shape graph first
        fn(x)
        c = torch._C._dynamo.eval_frame._debug_get_cache_entry_list(
            fn._torchdynamo_orig_callable.__code__
        )
        self.assertEqual(len(c), 2)
        self.assertEqual(c[0], dynamic_shapes_cache_entry)
        self.assertEqual(c[1], static_shapes_cache_entry)

    @expectedFailureDynamic
    def test_dynamo_disable_lru_cache_behavior(self):
        @torch.compile(backend="eager")
        def fn(x):
            return x + 10

        def run():
            torch._dynamo.reset()
            if torch._C._dynamo.eval_frame._debug_get_cache_entry_list(
                fn._torchdynamo_orig_callable.__code__
            ):
                raise AssertionError("Expected no cache entries after reset")

            # Step 1: Compile a static shapes graph
            x = torch.randn(10, 10)
            fn(x)
            a = torch._C._dynamo.eval_frame._debug_get_cache_entry_list(
                fn._torchdynamo_orig_callable.__code__
            )
            self.assertEqual(len(a), 1)
            static_shapes_cache_entry = a[0]

            # Step 2: Compile a dynamic shapes graph
            y = torch.randn(20, 20)
            fn(y)
            b = torch._C._dynamo.eval_frame._debug_get_cache_entry_list(
                fn._torchdynamo_orig_callable.__code__
            )
            self.assertEqual(len(b), 2)
            self.assertEqual(b[0], static_shapes_cache_entry)
            dynamic_shapes_cache_entry = b[1]

            # Step 3: Run with Step 1's inputs
            # LRU cache is disabled, we should still have static entry first
            fn(x)
            c = torch._C._dynamo.eval_frame._debug_get_cache_entry_list(
                fn._torchdynamo_orig_callable.__code__
            )
            self.assertEqual(len(c), 2)
            self.assertEqual(c[0], static_shapes_cache_entry)
            self.assertEqual(c[1], dynamic_shapes_cache_entry)

        try:
            torch._C._dynamo.eval_frame._set_lru_cache(False)
            run()
        finally:
            torch._C._dynamo.eval_frame._set_lru_cache(True)

    def test_patch_track_step_called_skipped(self):
        # Regression test for patch_track_step_called being ignored by dynamo
        # We need to clear FORCE_SKIP_FILES to test that the function name check
        # properly ignores patch_track_step_called even when lr_scheduler.py is not
        # in FORCE_SKIP_FILES
        import torch._dynamo.trace_rules as trace_rules

        old_force_skip_files = trace_rules.FORCE_SKIP_FILES
        try:
            trace_rules.FORCE_SKIP_FILES = set()

            cnt = CompileCounter()

            @torch.compile(backend=cnt, fullgraph=True)
            def fn(x, optimizer):
                # Create an LR scheduler which internally calls patch_track_step_called
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
                return x * 2, scheduler

            model = torch.nn.Linear(10, 10)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            x = torch.randn(10, 10)

            result, _ = fn(x, optimizer)
            expected = x * 2
            self.assertEqual(result, expected)
            self.assertEqual(cnt.frame_count, 1)
        finally:
            trace_rules.FORCE_SKIP_FILES = old_force_skip_files

    @parametrize("set_type", [set, frozenset], name_fn=lambda t: t.__name__)
    def test_set_doesnt_recompile_with_ac(self, set_type):
        import torch

        with torch._dynamo.config.patch({"error_on_recompile": True}):
            import functools

            from torch.utils.checkpoint import (
                checkpoint,
                CheckpointPolicy,
                create_selective_checkpoint_contexts,
            )

            def policy(compute_heavy_ops, ctx, func, *args, **kwargs):
                if func in compute_heavy_ops:
                    return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE

            def g(x):
                return torch.mm(x, x).sin().exp()

            @torch.compile(fullgraph=True, backend="eager")
            def f(x, policy):
                return checkpoint(g, x, use_reentrant=False, context_fn=policy)

            x = torch.randn(4, 4, requires_grad=True)
            f(
                x,
                functools.partial(
                    create_selective_checkpoint_contexts,
                    functools.partial(policy, set_type([torch.ops.aten.mm.default])),
                ),
            )
            f(
                x,
                functools.partial(
                    create_selective_checkpoint_contexts,
                    functools.partial(policy, set_type([torch.ops.aten.mm.default])),
                ),
            )

    def test_select_scatter_mixed_dtype(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                src = torch.tensor([0])
                out = torch.select_scatter(x, src, 1, 0)
                return out

        model = Model()
        x = torch.randn(1, 10)
        inputs = [x]

        compiled_model = torch.compile(model, backend="eager")

        self.assertEqual(model(*inputs), compiled_model(*inputs))

    @requires_cuda
    def test_diagonal_scatter_single_elem_cpu_with_cuda_tensor(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.ones(x.size(0))
                x = torch.diagonal_scatter(x, y)
                return x

        model = Model()

        x = torch.rand(1, 2)
        inputs = [x]

        device = "cuda"
        model = model.to(device)
        inputs = [x.to(device) for x in inputs]

        compiled_model = torch.compile(model, backend="eager")

        self.assertEqual(model(*inputs), compiled_model(*inputs))

    def test_autograd_function_ctx_stash_no_vc_check(self):
        # Test that tensors stashed directly on ctx (e.g., ctx.x = x) in an
        # autograd.Function don't trigger version counter checks, while tensors
        # saved via save_for_backward do.
        class MutatingFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, a, b, c, x, y, z):
                # Stash b and y directly on ctx (no VC check)
                ctx.b = b
                ctx.y = y
                # Save a, c, x via save_for_backward (with VC check)
                ctx.save_for_backward(a, c, x)
                return z + 1

            @staticmethod
            def backward(ctx, grad_output):
                a, c, x = ctx.saved_tensors
                b = ctx.b
                y = ctx.y
                # Mutate the stashed tensors in backward
                # This would fail with VC check if they went through save_for_backward
                b.mul_(2)
                y.mul_(3)
                return None, None, None, None, None, grad_output + 2 + a + c + x

        def my_func(*args):
            return MutatingFunction.apply(*args)

        compiled_func = torch.compile(my_func, backend=aot_graph_capture_backend)

        # Create tensors - only z requires grad
        a = torch.zeros(4, requires_grad=False)
        b = torch.zeros(4, requires_grad=False)
        c = torch.zeros(4, requires_grad=False)
        x = torch.zeros(4, requires_grad=False)
        y = torch.zeros(4, requires_grad=False)
        z1 = torch.randn(4, requires_grad=True)
        z2 = torch.randn(4, requires_grad=True)

        # Two forward calls that save b and y
        out1 = compiled_func(a, b, c, x, y, z1)
        out2 = compiled_func(a, b, c, x, y, z2)

        # First backward mutates b and y
        out1.sum().backward()

        # Second backward should NOT error even though b and y were mutated
        # because they were stashed on ctx, not saved via save_for_backward
        out2.sum().backward()
        # If we got here without error, the test passed
        # Also, assert that the AOTAutograd output descriptors on the fw graph show up
        # Of 5 total activations, 2 of them are smuggled through ctx without VC checks
        # (b and y via ctx.b = b, ctx.y = y) while 3 are saved via save_for_backward
        # (a, c, x via ctx.save_for_backward(a, c, x))
        # In dynamic shapes mode, there's also a symint saved for backward.
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                "\n".join(
                    [
                        str(x)
                        for x in fw_graph[0]
                        .graph.find_nodes(op="output")[0]
                        .meta["desc"]
                    ]
                ),
                """\
PlainAOTOutput(idx=0)
SavedForBackwardsAOTOutput(idx=0)
SavedForBackwardsAOTOutput(idx=1)
SavedForBackwardsAOTOutput(idx=2)
SavedForBackwardsNoVcCheckAOTOutput(idx=3)
SavedForBackwardsNoVcCheckAOTOutput(idx=4)""",
            )
        else:
            self.assertExpectedInline(
                "\n".join(
                    [
                        str(x)
                        for x in fw_graph[0]
                        .graph.find_nodes(op="output")[0]
                        .meta["desc"]
                    ]
                ),
                """\
PlainAOTOutput(idx=0)
SavedForBackwardsAOTOutput(idx=0)
SavedForBackwardsAOTOutput(idx=1)
SavedForBackwardsAOTOutput(idx=2)
SavedForBackwardsNoVcCheckAOTOutput(idx=3)
SavedForBackwardsNoVcCheckAOTOutput(idx=4)
SavedForBackwardsAOTOutput(idx=5)""",
            )

    def test_move_tensor_subclass_parameter_after_compile(self):
        aten = torch.ops.aten

        class Subclass(torch.Tensor):
            def __new__(cls, data):
                return torch.Tensor._make_wrapper_subclass(
                    cls, data.shape, dtype=data.dtype, device=data.device
                )

            def __init__(self, data):
                self._data = data

            def __repr__(self):
                return f"{self.__class__.__name__}(data={self._data})"

            def __tensor_flatten__(self):
                return ["_data"], []

            @classmethod
            def __tensor_unflatten__(cls, inner_tensors, ctx, outer_size, outer_stride):
                return cls(inner_tensors["_data"])

            def __torch_function__(self, func, types, args, kwargs=None):
                if func == torch.nn.functional.linear:
                    return func(args[0], args[1]._data, *args[2:])

                with torch._C.DisableTorchFunctionSubclass():
                    return func(*args, **(kwargs or dict()))

            def __torch_dispatch__(self, func, types, args, kwargs):
                if func in (aten._to_copy.default, aten.detach.default):
                    args = [x._data if isinstance(x, Subclass) else x for x in args]
                    out = func(*args, **kwargs)
                    return Subclass(out)

                raise NotImplementedError(f"{func=}")

        # Compile on CPU
        linear = torch.nn.Linear(2, 2)
        linear.weight = torch.nn.Parameter(Subclass(linear.weight.detach()))
        linear.compile()
        linear(torch.randn(1, 2))

        # Check that weakrefs are cleared after compile
        t1 = linear.weight
        self.assertEqual(len(weakref.getweakrefs(t1)), 0)

        # Already on CPU, so should just clear weakrefs
        linear.cpu()

        # Check that there is no recompile
        with torch._dynamo.config.patch(error_on_recompile=True):
            linear(torch.randn(1, 2, device="cpu"))

    def test_property_setter_with_dict_get_176608(self):
        """
        Test that property setters work correctly with __dict__.get() in compiled functions.
        Regression test for https://github.com/pytorch/pytorch/issues/176608
        """
        from torch.compiler import disable

        class Container:
            def __init__(self):
                self._len_value = 0

            @property
            def _len(self):
                # Using __dict__.get instead of self._len_value triggers the bug
                return self.__dict__.get("_len_value", 0)

            @_len.setter
            def _len(self, value):
                self._len_value = value

            def add(self, n):
                self._len = self._len + n

            @disable()
            def __len__(self):
                return self._len

        c = Container()

        @torch.compile(backend="eager")
        def f(x):
            c.add(x.shape[0])  # mutates c._len_value via property setter
            return len(c)  # reads c._len via property getter -> __dict__.get

        result = f(torch.randn(5))
        self.assertEqual(result, 5)

    def test_one_hot_bounds_check_compiled(self):
        # https://github.com/pytorch/pytorch/issues/144211
        # torch.compile(one_hot) should raise on out-of-bounds indices,
        # not silently produce wrong results.
        one_hot = torch.compile(torch.nn.functional.one_hot, fullgraph=True)

        a = torch.arange(0, 5) % 3  # [0, 1, 2, 0, 1]
        with self.assertRaises(RuntimeError):
            one_hot(a, 1)

        torch._dynamo.reset()
        with self.assertRaises(RuntimeError):
            one_hot(torch.tensor([-1, 0, 1]), 3)

        torch._dynamo.reset()
        expected = torch.nn.functional.one_hot(a, 3)
        self.assertEqual(one_hot(a, 3), expected)

    @unittest.expectedFailure
    def test_method_dunder_dict_setitem(self):
        # Reproducer for: getattr(obj, method_name).__dict__['key'] = value
        # method.__dict__ is handled specially by CPython at C level (no
        # tp_dictoffset, no Python-visible descriptor), which caused
        # object.__getattribute__(method, "__dict__") to raise AttributeError.

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            getattr(self, self._testMethodName).__dict__["slow_test"] = True
            return x.sin()

        x = torch.randn(2)
        _ = fn(x)
        self.assertTrue(getattr(self, self._testMethodName).__dict__.get("slow_test"))
