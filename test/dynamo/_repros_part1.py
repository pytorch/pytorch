try:
    from ._repros_common import (
        _do_paste_mask,
        _get_min_chunk_len,
        _get_sorted_bucket_idx_and_undo_sorted_bucket_idx,
        _merge_criteria_processor_list,
        _validate_model_kwargs,
        ABC,
        BatchNormAct2d,
        Boxes,
        ChunkReformerFeedForward,
        collections,
        CompileCounter,
        contextlib,
        convert_boxes_to_pooler_format,
        copy,
        create_rand_mask_from_inputs,
        CustomList1,
        CustomList2,
        DummyConfig,
        Enum,
        gc,
        get_parameter_dtype,
        has_detectron2,
        is_fx_tracing_test,
        itertools,
        ListConfig,
        logging,
        longformer_chunk,
        make_logging_test,
        MockModule,
        ModelOutput,
        nn,
        np,
        PartialMaml,
        PartialT5,
        rand_strided,
        random,
        ReformerEncoder,
        requires_cuda,
        same,
        same_two_models,
        SequentialAppendList,
        skipIfWindows,
        torch,
        TransformerEncoderLayer,
        TwoTensor,
        unittest,
        weakref,
    )
except ImportError:
    from _repros_common import (
        _do_paste_mask,
        _get_min_chunk_len,
        _get_sorted_bucket_idx_and_undo_sorted_bucket_idx,
        _merge_criteria_processor_list,
        _validate_model_kwargs,
        ABC,
        BatchNormAct2d,
        Boxes,
        ChunkReformerFeedForward,
        collections,
        CompileCounter,
        contextlib,
        convert_boxes_to_pooler_format,
        copy,
        create_rand_mask_from_inputs,
        CustomList1,
        CustomList2,
        DummyConfig,
        Enum,
        gc,
        get_parameter_dtype,
        has_detectron2,
        is_fx_tracing_test,
        itertools,
        ListConfig,
        logging,
        longformer_chunk,
        make_logging_test,
        MockModule,
        ModelOutput,
        nn,
        np,
        PartialMaml,
        PartialT5,
        rand_strided,
        random,
        ReformerEncoder,
        requires_cuda,
        same,
        same_two_models,
        SequentialAppendList,
        skipIfWindows,
        torch,
        TransformerEncoderLayer,
        TwoTensor,
        unittest,
        weakref,
    )


class LRUCacheWarningTestsMixin:
    @requires_cuda
    @make_logging_test(dynamo=logging.DEBUG)
    def test_lru_cache_warning_issued_during_tracing(self, records):
        prev_default = torch._C._get_default_device()

        def _restore_default_device():
            if prev_default == "cpu":
                torch.set_default_device(None)
            else:
                torch.set_default_device(prev_default)

        self.addCleanup(_restore_default_device)
        torch.set_default_device("cuda")

        @torch.compile(backend="eager")
        def f(x):
            torch.get_device_module()
            x = x.cos().sin()
            return x

        result = f(torch.randn(1024))
        self.assertIsInstance(result, torch.Tensor)

        for record in records:
            if "call to a lru_cache wrapped function at:" in record.getMessage():
                self.fail("lru_cache warning was incorrectly logged")


class ReproTestsMixin1:
    def setUp(self) -> None:
        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        self.exit_stack = contextlib.ExitStack()
        self.exit_stack.enter_context(
            install_guard_manager_testing_hook(self.guard_manager_clone_hook_fn)
        )
        super().setUp()

    def tearDown(self) -> None:
        self.exit_stack.close()
        super().tearDown()

    def test_compiled_module_truthiness(self):
        # Test with empty ModuleList
        original_empty = nn.ModuleList()
        compiled_empty = torch.compile(original_empty, backend="eager")
        self.assertEqual(bool(original_empty), bool(compiled_empty))
        self.assertFalse(bool(compiled_empty))
        # Test with non-empty ModuleList
        original_filled = nn.ModuleList([nn.Linear(10, 5)])
        compiled_filled = torch.compile(original_filled, backend="eager")
        self.assertEqual(bool(original_filled), bool(compiled_filled))
        self.assertTrue(bool(compiled_filled))

    def guard_manager_clone_hook_fn(self, guard_manager_wrapper, f_locals, builder):
        root = guard_manager_wrapper.root
        cloned_root = root.clone_manager(lambda x: True)
        cloned_wrapper = torch._dynamo.guards.GuardManagerWrapper(cloned_root)
        self.assertEqual(str(guard_manager_wrapper), str(cloned_wrapper))
        self.assertTrue(cloned_root.check(f_locals))
        if guard_manager_wrapper.diff_guard_root:
            self.assertTrue(guard_manager_wrapper.diff_guard_root.check(f_locals))

    def test_swap_tensors_subclass_not_blocked_by_metaconverter_refcycle(self):
        """Checks that MetaConverter doesn't create refcycles of itself that blocks swap_tensor on subclass tensors"""
        was_gc_enabled = gc.isenabled()
        gc.disable()
        try:
            from torch._subclasses.meta_utils import MetaConverter
            from torch.utils.weak import WeakIdRef

            w_a = torch.randn(4, 4)
            w_b = w_a.clone()
            # For custom tensor parameters, `nn.Parameter` returns the same custom tensor type
            # and sets `_is_param`, which is what Module._apply uses to decide swap_tensors.
            w = nn.Parameter(TwoTensor(w_a, w_b).requires_grad_())

            def weak_id_ref_count(t: torch.Tensor) -> int:
                return sum(isinstance(r, WeakIdRef) for r in weakref.getweakrefs(t))

            baseline_weak_id_refs = weak_id_ref_count(w)
            converter = MetaConverter()
            _ = converter(w)
            after_convert_weak_id_refs = weak_id_ref_count(w)
            self.assertGreater(after_convert_weak_id_refs, baseline_weak_id_refs)

            # If MetaConverter is collectable by refcount, the weakrefs go away immediately.
            # If it's kept alive only by a refcycle, this weakref will persist while gc is
            # disabled.
            del _
            del converter
            self.assertEqual(weak_id_ref_count(w), baseline_weak_id_refs)
        finally:
            if was_gc_enabled:
                gc.enable()
            gc.collect()

    def test_do_paste_mask(self):
        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()
        opt__do_paste_mask = torch.compile(_do_paste_mask, backend=cnt)
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 1,
            427,
            640,
            True,
        )
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 2,
            427,
            640,
            True,
        )
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 3,
            612,
            612,
            True,
        )
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 4,
            612,
            612,
            True,
        )
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 2,
            427,
            640,
            False,
        )
        # (dynamic shapes, static shapes)
        self.assertIn(cnt.frame_count, (5, 7))
        self.assertIn(cnt.op_count, (92, 106, 119))

    def test_convert_boxes_to_pooler_format(self):
        boxes1 = [
            Boxes(torch.arange(0, 8).reshape((2, 4))),
            Boxes(torch.arange(8, 16).reshape((2, 4))),
        ]
        boxes2 = [
            Boxes(torch.arange(16, 20).reshape((1, 4))),
            Boxes(torch.arange(20, 24).reshape((1, 4))),
        ]
        correct1 = convert_boxes_to_pooler_format(boxes1)
        correct2 = convert_boxes_to_pooler_format(boxes2)
        fn = convert_boxes_to_pooler_format
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        self.assertTrue(same(opt_fn(boxes1), correct1))
        self.assertTrue(same(opt_fn(boxes2), correct2))

        # repeat_interleave is a dynamic shape operator we do not execute/
        # In the future, we could reduce the frame_count down to 1
        # by guarding on the exact values of `Tensor repeats` arg
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """4""")
            self.assertExpectedInline(cnt.op_count, """10""")
        else:
            self.assertExpectedInline(cnt.frame_count, """4""")
            self.assertExpectedInline(cnt.op_count, """14""")

    def test_boxes_len(self):
        def fn(boxes):
            return len(boxes) + boxes.__len__() + boxes.tensor

        boxes1 = Boxes(torch.arange(0, 8).reshape((2, 4)))
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(boxes1), boxes1.tensor + 4.0))

        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """1""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """2""")

    def _reformer(self, nopython):
        input = torch.randn([1, 64, 256])
        model = ReformerEncoder()
        torch.manual_seed(1337)
        correct = copy.deepcopy(model)(input)
        cnt = torch._dynamo.testing.CompileCounter()
        torch.manual_seed(1337)
        opt_model = torch.compile(model, backend=cnt, fullgraph=nopython)
        self.assertTrue(same(opt_model(input), correct))
        return cnt

    def test_out_overload_non_contiguous(self):
        def f(x, y):
            return torch.abs(x, out=y.T)

        f_compiled = torch.compile(f, backend="aot_eager")

        x_ref = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        y_ref = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        x_test = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        y_test = torch.arange(4, dtype=torch.float32).reshape(2, 2)

        out_ref = f(x_ref, y_ref)
        out_test = f_compiled(x_test, y_test)
        self.assertEqual(out_ref, out_test)
        self.assertEqual(y_ref, y_test)

    def test_index_select_contiguous_with_compile(self):
        def fn(x):
            x = x.permute(1, 2, 0)
            x = x.unsqueeze(2)
            idx = torch.tensor([0], device=x.device)
            x = torch.index_select(x, 0, idx)
            return x

        x = torch.randn(6, 3, 6)
        eager_out = fn(x)
        self.assertTrue(eager_out.is_contiguous())
        compiled_fn = torch.compile(fn, backend="aot_eager_decomp_partition")
        compiled_out = compiled_fn(x)
        self.assertTrue(compiled_out.is_contiguous())
        self.assertEqual(eager_out, compiled_out)
        self.assertEqual(eager_out.stride(), compiled_out.stride())

    def test_roll_as_strided_channels_last(self):
        def fn(x):
            x = torch.roll(x, shifts=-1, dims=0)
            return torch.as_strided(x, (5, 10), (4, 1), 0)

        x = torch.arange(100).reshape(2, 5, 2, 5).to(memory_format=torch.channels_last)
        eager_out = fn(x)
        compiled_fn = torch.compile(fn, backend="aot_eager_decomp_partition")
        compiled_out = compiled_fn(x)
        self.assertEqual(eager_out, compiled_out)

    def test_inplace_add_from_meta_tensor_factory(self):
        def fn(x):
            log_det = torch.zeros(x.size(0), device=x.device)
            log_det += torch.zeros(x.size(0), device="meta")
            return log_det

        x = torch.randn(2, 4)
        eager_out = fn(x)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled_out = compiled_fn(x)

        self.assertEqual(eager_out.device, compiled_out.device)
        self.assertEqual(eager_out, compiled_out)

    def test_view_dtype_overload(self):
        def f(x):
            return x.view(torch.int32)

        f_compiled = torch.compile(f, backend="aot_eager")

        x1 = torch.ones(4, requires_grad=True)
        out_ref = f(x1)
        out_test = f_compiled(x1)
        self.assertEqual(out_ref, out_test)

        x2 = torch.ones(4, requires_grad=False)
        out_ref = f(x2)
        out_test = f_compiled(x2)
        self.assertEqual(out_ref, out_test)

    def test_intermediate_leaf_requires_grad(self):
        def f(x):
            leaf = torch.ones(2, requires_grad=True)
            return leaf, leaf * 2

        f_compiled = torch.compile(f, backend="aot_eager")
        x = torch.arange(4, dtype=torch.float32).reshape(2, 2)

        leaf, out = f(x)
        leaf_test, out_test = f_compiled(x)
        out.sum().backward()
        out_test.sum().backward()
        self.assertEqual(leaf.grad, leaf_test.grad)

    def test_unpack_hooks_dont_run_during_tracing(self):
        def f(x, y):
            return x * y

        f_compiled = torch.compile(f, backend="aot_eager")

        pack_count = 0
        unpack_count = 0

        def pack_hook(x):
            nonlocal pack_count
            pack_count += 1
            return x

        # unpack hook shouldn't run during compilation, while we trace the forward
        def unpack_hook(x):
            nonlocal unpack_count
            unpack_count += 1
            return x

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=False)
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            out_test = f_compiled(x, y)
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 0)
            out_test.sum().backward()
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 1)

    def test_unpack_hooks_can_be_disabled(self):
        def f(x, y):
            return x * y

        f_compiled = torch.compile(f, backend="aot_eager")

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=False)
        with torch.autograd.graph.disable_saved_tensors_hooks("hooks are disabled"):
            out_test = f_compiled(x, y)
            out_test.sum().backward()

    def test_disabling_unpack_hooks_within_compiled_region(self):
        def g(z):
            with torch.autograd.graph.disable_saved_tensors_hooks("hooks are disabled"):
                return z + 5

        def f(x, y):
            z = x * y
            return g(z)

        f_compiled = torch.compile(f, backend="aot_eager")

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=False)
        out_test = f_compiled(x, y)
        out_test.sum().backward()

    def test_gan_repro_trying_to_backward_through_the_graph_a_second_time(self):
        def f(a, b):
            c = torch.ones(2, 2)
            d = torch.ones(2, 2)
            e = torch.matmul(a, c)
            g_loss = torch.abs(e - d).mean()
            g_loss.backward()
            fake_d_pred = torch.matmul(b, e.detach())
            d_loss = fake_d_pred.mean()
            d_loss.backward()

        a_ref = torch.randn(2, 2, requires_grad=True)
        b_ref = torch.randn(2, 2, requires_grad=True)
        out_ref = f(a_ref, b_ref)

        a_test = a_ref.detach().clone().requires_grad_(True)
        b_test = b_ref.detach().clone().requires_grad_(True)
        out_test = torch.compile(f, backend="aot_eager")(a_test, b_test)

        self.assertEqual(out_ref, out_test)
        self.assertEqual(a_ref.grad, a_test.grad)
        self.assertEqual(b_ref.grad, b_test.grad)

    def test_tuple_enum_as_key_dict(self):
        class MyEnum(Enum):
            A = "a"

        class SomeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, x) -> torch.Tensor:
                return self.linear(x[MyEnum.A])

        x = {MyEnum.A: torch.rand(8, 1)}
        model_pytorch = SomeModel()
        model = torch.compile(model_pytorch, backend="eager")
        # Executing twice works
        model(x)
        y = model(x)
        self.assertEqual(y, model_pytorch(x))

    def test_embedding_backward_broadcasting_decomp(self):
        def f(grad_output, indices):
            num_weights = 10
            padding_idx = 1
            scale_grad_by_freq = True
            return torch.ops.aten.embedding_dense_backward(
                grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
            )

        f_compiled = torch.compile(f, backend="aot_eager")

        grad_output = torch.ones(2, 4, 3, dtype=torch.float16)
        indices = torch.ones(2, 4, dtype=torch.int64)

        out_ref = f(grad_output, indices)
        out_test = f_compiled(grad_output, indices)

        self.assertEqual(out_ref, out_test)

    def test_reformer_eval(self):
        with torch.no_grad():
            cnt = self._reformer(nopython=True)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 10)

    def test_reformer_train(self):
        with torch.enable_grad():
            cnt = self._reformer(nopython=False)
        self.assertExpectedInline(cnt.frame_count, """1""")
        self.assertExpectedInline(cnt.op_count, """10""")

    def test_longformer_chunk(self):
        input1 = torch.randn([1, 4096, 1])
        input2 = torch.randn([12, 4096, 64])
        correct1 = longformer_chunk(input1)
        correct2 = longformer_chunk(input2)
        fn = longformer_chunk
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(input1), correct1))
        self.assertTrue(same(opt_fn(input2), correct2))
        self.assertTrue(same(opt_fn(input1), correct1))
        self.assertTrue(same(opt_fn(input2), correct2))

        if torch._dynamo.config.assume_static_by_default:
            if torch._dynamo.config.automatic_dynamic_shapes:
                self.assertExpectedInline(cnt.frame_count, """2""")
                self.assertExpectedInline(cnt.op_count, """8""")
            else:
                self.assertExpectedInline(cnt.frame_count, """2""")
                self.assertExpectedInline(cnt.op_count, """4""")
        else:
            self.assertExpectedInline(cnt.frame_count, """2""")
            self.assertExpectedInline(cnt.op_count, """19""")

    def test_hf_t5_forward(self):
        input = torch.randn([1, 2048, 512])
        model = PartialT5()
        correct = model(input)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        self.assertTrue(same(opt_model(input), correct))

        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """11""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """11""")

    def test_module_in_skipfiles(self):
        model = nn.Linear(10, 10)
        cnt = torch._dynamo.testing.CompileCounter()
        torch.compile(model, backend=cnt, fullgraph=True)(torch.randn([5, 10]))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_function_in_skipfiles(self):
        cnt = torch._dynamo.testing.CompileCounter()
        torch.compile(torch.sin, backend=cnt, fullgraph=True)(torch.randn([5, 10]))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_slicing_dynamic_shape(self):
        def fn(y):
            x = torch.ones(8)
            idx = y[0]
            out = x[idx:]
            return (out + 3) * 5

        counter = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)
        out = opt_fn(torch.ones(10, dtype=torch.long))
        # idx should be 1 -> slicing off [1:] of 8 elem tensor
        self.assertEqual(list(out.shape), [7])

        self.assertEqual(counter.op_count, 2)
        self.assertEqual(counter.frame_count, 1)

        self.assertEqual(list(opt_fn(torch.tensor([4])).shape), [4])

    def test_slicing_dynamic_shape_setitem(self):
        def fn(input_lengths: torch.Tensor, new_ones_1):
            getitem_13 = input_lengths[3]
            new_ones_1[(3, slice(getitem_13, None, None))] = 0
            setitem_13 = new_ones_1
            return (setitem_13,)

        x = torch.randn(10).to(dtype=torch.int64)
        y = torch.randn(10, 204)
        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="aot_eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    @torch._dynamo.config.patch(error_on_recompile=True)
    @torch.fx.experimental._config.patch(use_duck_shape=False)
    def test_dynamic_shape_disable_duck_size(self):
        # noqa: F841

        class TestModel(nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, x: torch.Tensor, val: int) -> torch.Tensor:
                return x + val

        main_model = TestModel().to(memory_format=torch.channels_last)
        opt_model = torch.compile(main_model, backend="eager", dynamic=True)

        x1 = torch.rand(2, 5, 10, 10).to(memory_format=torch.channels_last)
        x2 = torch.rand(2, 5, 4, 8).to(memory_format=torch.channels_last)

        main_model(x1, 4)
        opt_model(x1, 4)

        main_model(x2, 20)
        opt_model(x2, 20)

    def test_chunk_reformer_ff(self):
        input = torch.randn([1, 4096, 256])
        model = ChunkReformerFeedForward()
        correct = model(input)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        self.assertTrue(same(opt_model(input), correct))

        self.assertEqual(cnt.frame_count, 1)
        self.assertLessEqual(cnt.op_count, 10)

    @unittest.expectedFailure
    @torch._dynamo.config.patch(
        fake_tensor_propagation=True, capture_scalar_outputs=True
    )
    def test_maml_item_capture(self):
        a = torch.randn(5, 1, 28, 28)
        b = torch.zeros(5, dtype=torch.int64)
        c = torch.randn(75, 1, 28, 28)
        d = torch.zeros(75, dtype=torch.int64)
        model = PartialMaml()
        correct = model(a, b, c, d)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch.compile(model, backend=cnt)
        for _ in range(10):
            self.assertTrue(same(opt_model(a, b, c, d), correct))

        # if torch._dynamo.config.assume_static_by_default:
        #     self.assertExpectedInline(cnt.frame_count, """2""")
        # else:
        #     self.assertExpectedInline(cnt.frame_count, """3""")
        # TODO(jansel): figure out why op count depends on imports
        self.assertIn(cnt.op_count, (36, 35, 34, 29, 28, 27))

    @torch._dynamo.config.patch(capture_scalar_outputs=False)
    def test_maml_no_item_capture(self):
        a = torch.randn(5, 1, 28, 28)
        b = torch.zeros(5, dtype=torch.int64)
        c = torch.randn(75, 1, 28, 28)
        d = torch.zeros(75, dtype=torch.int64)
        model = PartialMaml()
        correct = model(a, b, c, d)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch.compile(model, backend=cnt)
        for _ in range(10):
            self.assertTrue(same(opt_model(a, b, c, d), correct))

        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """2""")
        else:
            self.assertExpectedInline(cnt.frame_count, """3""")

    def test_hf_model_output(self):
        ex = ModelOutput(a=torch.randn(10), b=torch.randn(10), c=torch.randn(10))

        def fn1(x):
            return x["a"] + 1

        def fn2(x):
            return x.a + 1

        def fn3(x):
            return x.to_tuple()[0] + 1

        def fn4(x):
            return x[0] + 1

        cnt = torch._dynamo.testing.CompileCounter()
        for fn in (fn1, fn2, fn3, fn4):
            cnt.clear()
            opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
            self.assertTrue(same(opt_fn(ex), ex.a + 1))
            self.assertEqual(cnt.frame_count, 1)
            self.assertEqual(cnt.op_count, 1)

    def test_create_rand_mask_from_inputs(self):
        args = [
            torch.randn([1, 64, 64]),
            torch.randn([1, 64, 64]),
            torch.zeros([1, 12, 62, 3], dtype=torch.int64),
            12,
            3,
            1,
            4096,
            64,
        ]
        correct = create_rand_mask_from_inputs(*args)
        fn = create_rand_mask_from_inputs

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(*args), correct))
        self.assertEqual(cnt.frame_count, 1)
        # Op count may vary between 8-10 in static mode.
        if torch._dynamo.config.assume_static_by_default:
            self.assertIn(cnt.op_count, range(8, 11))
        else:
            self.assertExpectedInline(cnt.op_count, """11""")

    def test_rng_state(self):
        def fn():
            state = torch.get_rng_state()
            before = torch.rand(1000)
            torch.set_rng_state(state)
            after = torch.rand(1000)
            return before, after

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)

        before, after = opt_fn()
        self.assertTrue(same(before, after))
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)  # rand, rand
        try:
            _, _ = torch._dynamo.export(fn)()
            # See https://github.com/pytorch/pytorch/pull/87490
            self.fail("unexpected export success")
        except torch._dynamo.exc.Unsupported:
            pass

    def test_threading_local(self):
        import threading

        foo = threading.local()
        foo.x = torch.rand(1)

        def f(x):
            return torch.cat([x, foo.x])

        cnt = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=cnt, fullgraph=True)

        inp = torch.ones(1)
        out = f(inp)
        opt_out = opt_f(inp)
        self.assertEqual(opt_out, out)
        self.assertEqual(cnt.frame_count, 1)

    def test_seq_append_list(self):
        x = torch.randn(4, 10)
        model = SequentialAppendList(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )
        # this one is tricky because it mutates the list provided as an input
        l1 = [x]
        l2 = [x]
        correct, _ = model(x, l1)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        result, l3 = opt_model(x, l2)
        self.assertTrue(same(result, correct))
        self.assertTrue(same(l1, l2))
        self.assertIs(l2, l3)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 5)

    def test_batch_norm_act(self):
        a = torch.randn(5, 1, 28, 28)
        model = BatchNormAct2d(1).eval()
        correct = model(a)
        cnt = torch._dynamo.testing.CompileCounter()
        if not torch._dynamo.config.specialize_int:
            # _local_scalar_dense causes graph break w 0-dim tensor
            opt_model = torch.compile(model, backend=cnt)
            self.assertTrue(same(opt_model(a), correct))
            return

        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        self.assertTrue(same(opt_model(a), correct))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_get_parameter_dtype(self):
        model = SequentialAppendList(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )

        def fn(model, x):
            return x + torch.randn(
                10, dtype=get_parameter_dtype(model), device=x.device
            )

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertEqual(opt_fn(model, torch.randn(10)).dtype, torch.float32)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_nn_parameter(self):
        def test_fn():
            a = torch.nn.Parameter(torch.randn(5, 5))
            # Checks that TensorVariable stores the type information correctly
            self.assertTrue(isinstance(a, torch.nn.Parameter))
            return a

        cnt = torch._dynamo.testing.CompileCounter()
        opt_test_fn = torch.compile(test_fn, backend=cnt)
        out = opt_test_fn()
        self.assertTrue(isinstance(out, torch.nn.Parameter))

    def test_Size(self):
        def test_fn():
            a = torch.randn(4)
            x = torch.Size([1, 2, 3])
            # Checks that SizeVariable return torch.Size object
            assert isinstance(x, torch.Size)  # noqa: S101
            # Causes graph breaks and checks reconstruction of SizeVariable
            # object
            self.assertIsInstance(x, torch.Size)
            return a

        cnt = torch._dynamo.testing.CompileCounter()
        opt_test_fn = torch.compile(test_fn, backend=cnt)
        opt_test_fn()

    def test_copy_weird_strides(self):
        # This test requires inductor's copy() decomp to preserve strides properly.
        def test_fn(a):
            b = torch.zeros(48, 4, 256, 513)
            b[:, 0, 1:256, 1:256] = a
            c = b.view(4, 12, 1024, 513)
            d = c.transpose(2, 1)
            d.add_(1)
            return d

        sh, st, dt, dev, rg = (
            (48, 255, 255),
            (787968, 513, 1),
            torch.float16,
            "cpu",
            True,
        )
        a = rand_strided(sh, st, dt, dev).requires_grad_(rg)
        compiled_f = torch.compile(test_fn, backend="aot_eager_decomp_partition")
        out1 = test_fn(a)
        out2 = compiled_f(a)
        self.assertEqual(out1, out2)

    def test_indexing_with_list(self):
        def test_fn():
            def run_test(tensor, *idx):
                npt = tensor.numpy()
                assert npt[idx].shape == tensor[idx].shape  # noqa: S101

            x = torch.arange(0, 10)
            cases = [
                [None, None],
                [1, None],
            ]

            for case in cases:
                run_test(x, *case)

            return torch.randn(4)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_test_fn = torch.compile(test_fn, backend=cnt)
        opt_test_fn()

    def test_foreach_decomp_arg_names(self):
        # https://github.com/pytorch/pytorch/issues/138698

        @torch.compile(fullgraph=True, backend="eager")
        def foreach_pow(**kwargs):
            return torch._foreach_pow(**kwargs)

        foreach_pow(self=[torch.ones(2, 2, device="cpu")], exponent=2.7)

        @torch.compile(fullgraph=True, backend="eager")
        def foreach_lerp_(**kwargs):
            return torch._foreach_lerp_(**kwargs)

        foreach_lerp_(
            self=[torch.ones(2, 2, device="cpu")],
            tensors1=[torch.ones(2, 2, device="cpu")],
            weights=[torch.ones(2, 2, device="cpu")],
        )

    def test_reformer_min_chunk_len(self):
        def fn(cfg):
            t = torch.empty(10)
            t.fill_(_get_min_chunk_len(cfg))
            return t[0]

        cfg = DummyConfig()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertEqual(opt_fn(cfg), 64)
        # With unspec int, maximum computation is preserved
        self.assertExpectedInline(cnt.frame_count, """1""")
        if torch._dynamo.config.automatic_dynamic_shapes:
            if not torch._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(cnt.op_count, """4""")
            else:
                self.assertExpectedInline(cnt.op_count, """3""")
        else:
            self.assertExpectedInline(cnt.op_count, """3""")

    def test_reformer_sorting(self):
        x = torch.zeros([1, 12, 4096], dtype=torch.int64)
        correct = _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(x)
        fn = _get_sorted_bucket_idx_and_undo_sorted_bucket_idx

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(x), correct))
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """14""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """16""")

    def test_recursive_map(self):
        # https://github.com/pytorch/torchdynamo/issues/132
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = v

        def toy_example(a, b, v):
            x = a / (torch.abs(a) + 1)
            if v is not None:
                _recursive_map(v)
            return x * b

        cnt = torch._dynamo.testing.CompileCounter()
        opt_toy_example = torch.compile(toy_example, backend=cnt)
        opt_toy_example(
            torch.randn(10),
            torch.randn(10),
            {"layer0": {"memory_keys": torch.randn(10)}},
        )
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 4)

    def test_issue114171(self):
        device = torch.device("cpu")

        def fcnn(in_dim, out_dim, hidden_dim, activation=torch.nn.GELU):
            layers = [
                torch.nn.Linear(in_dim, hidden_dim, device=device),
                activation(),
                torch.nn.Linear(hidden_dim, out_dim, device=device),
            ]
            return torch.nn.Sequential(*layers)

        class testmodel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.interaction_networks = torch.nn.ModuleList(
                    [fcnn(262, 1174, 400) for _ in range(4)]
                )

            def interact(self, x, cycle):
                return self.interaction_networks[cycle](x)

        model = testmodel()
        forward_aot = torch.compile(
            model.interact, fullgraph=True, dynamic=True, backend="eager"
        )

        x = torch.rand([111, 262], device=device)
        forward_aot(x, 2)  # previously failed

    def test_issue175(self):
        n_heads = 2
        d_model = 64
        model = TransformerEncoderLayer(d_model, n_heads)
        inp = torch.randn(1, d_model)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch.compile(model, backend=cnt, fullgraph=True)
        opt_model(inp)
        opt_model(inp)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(12, cnt.op_count)

    def test_exec_import(self):
        def fn1():
            exec("import math")

        def fn2():
            try:
                math.sqrt(4)
                return False
            except NameError:
                return True

        def fn3():
            fn1()
            return fn2()

        self.assertTrue(fn3())
        opt_fn3 = torch.compile(fn3, backend="eager")
        self.assertTrue(opt_fn3())

    def test_exec_wildcard_import(self):
        # Test that globals are not carried over from frame to frame
        def fn1():
            exec("from torch import *")

        def fn2():
            x = torch.zeros(4)
            for i in range(5):
                x = x + i
            return x

        def fn3():
            fn1()
            return fn2()

        ref = fn3()
        opt_fn3 = torch.compile(fn3, backend="eager")
        res = opt_fn3()
        self.assertTrue(same(ref, res))

    def test_with_on_graph_break_inst(self):
        def reversible(x):
            print("Hello world")  # Cause graph break so inline fails
            return torch.sin(torch.cos(x))

        def fn(x):
            with torch.enable_grad():
                a = torch.sin(x)
                b = reversible(a)
                c = torch.sigmoid(b)
                c.sum().backward()
                return x.grad

        x = torch.randn(3, requires_grad=True)
        x.grad = None
        with torch.no_grad():
            ref = fn(x)

        x.grad = None
        opt_fn = torch.compile(fn, backend="eager")
        with torch.no_grad():
            res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_with_on_graph_break_nested(self):
        def reversible(x):
            torch._dynamo.graph_break()  # Cause graph break so inline fails
            return torch.sin(torch.cos(x))

        def fn(x):
            # nested context manager failed previously
            with torch.no_grad():
                with torch.enable_grad():
                    a = torch.sin(x)
                    b = reversible(a)
                    c = torch.sigmoid(b)
                    c.sum().backward()
                    return x.grad

        x = torch.randn(3, requires_grad=True)
        x.grad = None
        with torch.no_grad():
            ref = fn(x)

        x.grad = None
        opt_fn = torch.compile(fn, backend="eager")
        with torch.no_grad():
            res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_grad_mode_carrying_correct_state_after_graph_break(self):
        def fn(x):
            with torch.no_grad():
                y = x * 3
                print("Break")
                z = x + 2
            return y, z

        x = torch.randn(3, requires_grad=True)
        opt_fn = torch.compile(fn, backend="eager")
        y, z = opt_fn(x)
        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)

    def test_locals_traced_correctly_under_compile(self):
        def fn(x):
            if x.dim() > 2:
                batch_size, seq_len, hidden_dim = x.shape
                x = x.view(-1, hidden_dim)

            x = x + 1

            if "batch_size" in locals() and "seq_len" in locals():
                x = x.view(batch_size, seq_len, -1)
            return x

        x = torch.randn(2, 3, 4)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertTrue(same(fn(x), opt_fn(x)))

    def test_vars_traced_correctly_under_compile(self):
        def fn(x):
            if x.dim() > 2:
                batch_size, seq_len, hidden_dim = x.shape
                x = x.view(-1, hidden_dim)

            x = x + 1

            if "batch_size" in vars() and "seq_len" in vars():
                x = x.view(batch_size, seq_len, -1)
            return x

        x = torch.randn(2, 3, 4)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertTrue(same(fn(x), opt_fn(x)))

    def test_abc_setattr(self):
        # tests that we correctly bail out of __setattr__ calls

        # TODO: does not ensure ABC classes are correctly inferred as ClassVariables
        # (doesn't test the fix for 'super()')

        class BaseModule(torch.nn.Module, ABC):
            def blah(self, x):
                return x + 1

        class Derived(BaseModule):
            def __setattr__(self, name, value) -> None:
                super().__setattr__(name, value)

            def forward(self, x):
                # expect a graph break on __setattr__
                self.foo = 0
                return self.blah(x)

            def blah(self, x):
                return super().blah(x)

        x = torch.randn(3, requires_grad=True)
        mod = Derived()
        opt_mod = torch.compile(mod, backend="eager")
        opt_mod(x)

        # Not sure what this test is testing. It was earlier graph breaking on
        # __dict__, so the counter >= 2. With __dict__ support, there is no
        # graph break.
        self.assertGreaterEqual(torch._dynamo.utils.counters["frames"]["ok"], 1)
        self.assertGreaterEqual(torch._dynamo.utils.counters["frames"]["total"], 1)

    @torch._dynamo.config.patch("suppress_errors", True)
    def test_guard_fail_tensor_bool(self):
        @torch._dynamo.disable(recursive=False)
        def fn():
            condition_shape = (5, 5)
            dtypes = (torch.bool,)
            shapes = (
                (),
                (5,),
                (1, 5),
            )

            tensors = [
                torch.empty(shape, dtype=dtype).fill_(17)
                for shape, dtype in itertools.product(shapes, dtypes)
            ]

            x_vals = (5.0, *tensors)
            y_vals = (6.0, *tensors)

            @torch._dynamo.disable
            def get_expected(condition, x, y):
                x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
                return torch.from_numpy(
                    np.where(condition.cpu().numpy(), x_np, y_np)
                ).to(common_dtype)

            for x, y in zip(x_vals, y_vals):
                condition = torch.empty(*condition_shape, dtype=torch.bool).bernoulli_()
                common_dtype = torch.result_type(x, y)

                def check_equal(condition, x, y):
                    # NumPy aggressively promotes to double, hence cast to output to correct dtype
                    expected = get_expected(condition, x, y)
                    result = torch.where(condition, x, y)
                    assert torch.allclose(expected, result)  # noqa: S101

                check_equal(condition, x, y)
                check_equal(condition, y, x)

        fn()
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn()

    def test_guard_fail_nested_tuple(self):
        def fn(args):
            return torch.ones(()), args[0] * 2

        # This adds a tensor check on args[1][0] and args[1][1]
        args1 = (torch.ones(1), (torch.ones(1), torch.ones(1)))
        args2 = (torch.ones(1), torch.ones(1))
        opt_fn = torch.compile(fn, backend="eager")
        ref = opt_fn(args1)
        res = opt_fn(args2)

        self.assertTrue(same(ref, res))

    def test_nullcontext1(self):
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x, ctx):
            x = x.sin()
            with ctx:
                x = x.cos()
            x = x.sin()
            return x

        y = torch.randn(10)
        self.assertTrue(same(fn(y, contextlib.nullcontext()), y.sin().cos().sin()))

    def test_nullcontext2(self):
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x, ctx):
            x = x.sin()
            with ctx():
                x = x.cos()
            x = x.sin()
            return x

        y = torch.randn(10)
        self.assertTrue(same(fn(y, contextlib.nullcontext), y.sin().cos().sin()))

    def test_no_grad_inline(self):
        @torch.no_grad()
        def a(x):
            return x.sin()

        @torch.compile(backend="eager", fullgraph=True)
        def b(x):
            return a(x).cos()

        y = torch.randn(10)
        self.assertTrue(same(b(y), y.sin().cos()))

    @skipIfWindows(
        msg="torch._dynamo.exc.TorchRuntimeError: Failed running call_function <class 'torch.LongTensor'>(*(FakeTensor(..., size=(10,), dtype=torch.int32),), **{}):"  # noqa: B950
    )
    def test_longtensor_list(self):
        for partition in [0, 5, 10]:

            @torch._dynamo.disable
            def rand_gen():
                rand_vals = [random.randint(5, 10) for _ in range(10)]
                # List of tensors mixed with np.arrays
                return list(np.array(rand_vals[:partition])) + [
                    torch.tensor(val) for val in rand_vals[partition:]
                ]

            def fn(x):
                random_list = rand_gen()
                z = torch.LongTensor(random_list)
                return x * z

            x = torch.ones(10) * 2

            random.seed(0)
            ref0 = fn(x)
            ref1 = fn(x)

            opt_fn = torch.compile(fn, backend="eager")
            # Especially for internal usage, there are many calls to random functions
            # on first compile, e.g., from various library initializations. Run once
            # to get that out of the way before resetting the seed:
            opt_fn(x)

            random.seed(0)
            res0 = opt_fn(x)
            res1 = opt_fn(x)

            self.assertTrue(same(ref0, res0))
            self.assertTrue(same(ref1, res1))

    def test_primtorch(self):
        @torch.compile(backend="eager")
        def fn(x):
            torch._refs.abs(x)

        fn(torch.randn(3))

    @unittest.expectedFailure
    # inline_call [('inline in skipfiles: bind ...python3.10/inspect.py', 1)]
    def test_primtorch_no_graph_break(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            torch._refs.abs(x)

        fn(torch.randn(3))

    def test_torch_tensor_ops_no_graph_break(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            torch.Tensor.abs_(x)

        fn(torch.randn(3))

    @unittest.skipIf(
        not isinstance(torch.ops.aten.abs, torch._ops.OpOverloadPacket),
        "old pt doesn't work",
    )
    def test_torch_ops_aten(self):
        # Picked an op that doesn't show up in the default list
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return torch.ops.aten.absolute(x)

        fn(torch.randn(3))

    def test_hf_gelu_inline(self):
        class GELUActivation(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.act = nn.functional.gelu

            def forward(self, input):
                return self.act(input)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return GELUActivation()(x)

        y = torch.randn(10)
        self.assertTrue(same(fn(y), nn.functional.gelu(y)))

        @torch.compile(backend="eager", fullgraph=True)
        def fn_returns(x):
            return GELUActivation(), x + 1

        act, _ = fn_returns(y)
        self.assertIsInstance(act, GELUActivation)
        self.assertIs(act.act, nn.functional.gelu)
        self.assertTrue(hasattr(act, "_buffers"))  # check that __init__ got called

    def test_dropout_inline(self):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.nn.Dropout(0.1)(x)

        y = torch.randn(10)
        torch.manual_seed(1337)
        ref = nn.functional.dropout(y, 0.1)
        torch.manual_seed(1337)
        res = fn(y)
        self.assertTrue(same(ref, res))

    def test_setitem_boolean_mask_diff(self):
        def fn(x, b, y):
            x = x.clone()
            x[b] = y
            return x

        opt_fn = torch.compile(fn, backend="aot_eager")
        x = torch.randn(4, requires_grad=True)
        b = torch.tensor([True, False, True, False])
        y = torch.randn(2, requires_grad=True)
        opt_fn(x, b, y)

    def test_setitem_tuple_boolean_mask_diff(self):
        def fn(x, b, y):
            x = x.clone()
            x[:, b] = y
            return x

        opt_fn = torch.compile(fn, backend="aot_eager")
        x = torch.randn(8, 4, requires_grad=True)
        b = torch.tensor([True, False, True, False])
        y = torch.randn(2, requires_grad=True)
        opt_fn(x, b, y)

    def test_torch_tensor_ops(self):
        def fn(x):
            return torch.Tensor.abs_(x)

        x = torch.randn(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        y = fn(x)
        y_ = opt_fn(x)
        self.assertTrue(same(y, y_))

    def test_guard_ordering_shape_fail(self):
        # If a function which takes a tensor has an inner function which
        # is compiled and generates a guard on its shape,
        # they are evaluated in the wrong order. So if on a subsequent call
        # an int is passed instead of a tensor, guard evaluation will crash
        # with a "no attribute: shape" error
        m = MockModule()
        opt_m = torch.compile(m, backend="eager")
        opt_m.fn(torch.ones((5, 5)))
        opt_m.fn(-3)

    def test_tensor_isinstance_tuple(self):
        @torch.compile(backend="eager")
        def fn():
            t = torch.ones(5, 5)
            if not isinstance(t, (int, torch.Tensor)):
                msg = str.format(
                    "{0} is not an instance of {1}",
                    type(t),
                    (int, torch.Tensor),
                )
                raise ValueError(msg)
            return True

        fn()

    def test_isinstance_dtype(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            isinstance(torch.bfloat16, torch.dtype)
            return x

        fn(torch.randn(3))

    def test_isinstance_storage(self):
        @torch.compile(backend="eager")
        def fn(x):
            f = bytearray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40])
            bools = torch.BoolStorage.from_buffer(f, "big")
            assert isinstance(bools, torch.BoolStorage)  # noqa: S101
            return x

        fn(torch.randn(3))

    def test_issue111522(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x, y):
            return x + y.a

        class A:
            a = 2

        self.assertEqual(f(torch.zeros(2), A()), torch.full([2], 2.0))

        del A.a

        # graph break on missing attr
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            f(torch.zeros(2), A())

    def test_sort_out(self):
        dtype = torch.float32
        device = "cpu"

        def fn():
            tensor = torch.randn((3, 5), dtype=dtype, device=device)[:, 0]
            values1 = torch.tensor(0, dtype=dtype, device=device)
            indices1 = torch.tensor(0, dtype=torch.long, device=device)
            torch.sort(tensor, out=(values1, indices1))
            self.assertEqual(values1.stride(), (1,))
            self.assertEqual(indices1.stride(), (1,))

        fn()
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn()

    def test_sort_out2(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sorted = torch.nn.Buffer(torch.ones(4, 4))
                self.indices = torch.nn.Buffer(torch.ones(4, 4, dtype=torch.long))

            def forward(self, x):
                torch.sort(x, out=(self.sorted, self.indices))
                return (x + 1, self.sorted, self.indices)

        x = torch.randn(4, 4)
        m = MyModule()
        ref = m(x)
        opt_m = torch.compile(m, backend="eager")
        res = opt_m(x)
        self.assertTrue(same(ref, res))

    def test_sigmoid_out(self):
        dtype = torch.float32
        device = "cpu"

        def fn():
            inp = torch.randn((3, 5), dtype=dtype, device=device)
            out1 = torch.tensor(0, dtype=dtype, device=device)
            torch.sigmoid(inp, out=out1)
            self.assertEqual(out1.numel(), 15)

        fn()
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn()

    def test_sigmoid_out2(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = torch.nn.Buffer(torch.ones(4, 4))

            def forward(self, x):
                torch.sigmoid(x, out=self.base)
                return x + self.base

        x = torch.randn(4, 4)
        m = MyModule()
        ref = m(x)
        opt_m = torch.compile(m, backend="eager")
        res = opt_m(x)
        self.assertTrue(same(ref, res))

    def test_out_root_cell_shape_change(self):
        @torch.compile(backend="eager")
        def fn():
            out = torch.empty(0)

            def run():
                x = torch.zeros(3, 5)
                torch.sigmoid(x, out=out)
                return out.size()

            return run()

        res = fn()
        self.assertEqual((3, 5), res)

    def test_out_nested_cell_shape_change(self):
        @torch.compile(backend="eager")
        def fn():
            def run():
                x = torch.zeros(3, 5)
                out = torch.empty(0)

                def capture():
                    return out  # Force `out` to be a nested cell

                torch.sigmoid(x, out=out)
                return out.size()

            return run()

        res = fn()
        self.assertEqual((3, 5), res)

    def test_out_root_cell_tuple_shape_change(self):
        @torch.compile(backend="eager")
        def fn():
            out1 = torch.empty(0)
            out2 = torch.empty(0, dtype=torch.long)

            def run():
                x = torch.zeros(3, 5)
                torch.sort(x, out=(out1, out2))
                return out1.size(), out2.size()

            return run()

        res = fn()
        self.assertEqual(((3, 5), (3, 5)), res)

    def test_out_nested_cell_tuple_shape_change(self):
        @torch.compile(backend="eager")
        def fn():
            def run():
                x = torch.zeros(3, 5)
                out1 = torch.empty(0)
                out2 = torch.empty(0, dtype=torch.long)

                def capture():
                    # Force `out1` and `out2` to be nested cells
                    return out1, out2

                torch.sort(x, out=(out1, out2))
                return out1.size(), out2.size()

            return run()

        res = fn()
        self.assertEqual(((3, 5), (3, 5)), res)

    def test_slice_into_list_mutable(self):
        class Mod(torch.nn.Module):
            def forward(self, listy):
                x = listy[3:5]
                for _ in range(10):
                    z = torch.abs(torch.randn(10)) + 1
                    x[0] = z
                return x

        m = Mod()
        listy = [torch.randn(10)] * 10

        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch.compile(m, backend=cnt, fullgraph=True)
        opt_m.forward(listy)

        self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_issue111918(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, dynamic=True)
        def fn(x):
            x = x + 1
            y = x.item()
            if y > 2:
                return x * 2
            else:
                return x * 3

        x = torch.tensor([3.0])
        fn(x)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 4)

        torch._dynamo.reset()
        fn = torch.compile(fn, fullgraph=True, backend="eager")
        with self.assertRaises(torch._dynamo.exc.UserError):
            fn(x)

    def test_vdd_duplicate_error(self):
        def fn(a, dt):
            keys = list(dt._jt_dict.keys())
            p = torch.cos(dt._jt_dict[keys[0]]._value)
            q = torch.sin(a)
            r = torch.sigmoid(dt._jt_dict[keys[0]]._value)
            return p + q + r

        class Value:
            def __init__(self) -> None:
                self._value = torch.randn(4)

        class Sample:
            def __init__(self) -> None:
                self._jt_dict = {}
                self._jt_dict["POSITION_ID"] = Value()

        a = torch.randn(4)
        sample = Sample()

        ref = fn(a, sample)

        optimized_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = optimized_fn(a, sample)

        self.assertTrue(same(ref, res))

    def test_specialized_stride(self):
        def f():
            e = torch.empty(4)
            x = e[::2]
            return x.stride()

        self.assertEqual(f(), torch.compile(f, backend="eager")())

    def test_out_none(self):
        # https://github.com/pytorch/pytorch/issues/92814
        def fn(input):
            return torch.nn.functional.normalize(input, dim=0, out=None)

        x = torch.rand([1])
        self.assertEqual(fn(x), torch.compile(fn, backend="eager")(x))

    def test_multi_import(self):
        if not has_detectron2():
            raise unittest.SkipTest("requires detectron2")

        @torch.compile(backend="eager", fullgraph=True)
        def to_bitmasks(boxes):
            from detectron2.layers.mask_ops import (
                _paste_masks_tensor_shape,
                paste_masks_in_image,
            )

            if (
                paste_masks_in_image is not None
                and _paste_masks_tensor_shape is not None
            ):
                return boxes + 1

        self.assertTrue((to_bitmasks(torch.zeros(10)) == torch.ones(10)).all())

    def test_multi_dot_import(self):
        def fn1(x):
            return torch.sin(x)

        def fn(x):
            import torch.fx

            _ = torch.fx.symbolic_trace(fn1)
            return x * 2

        x = torch.randn(10)
        fn(x)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_relative_import(self):
        try:
            from . import utils as _  # noqa: F401

            def fn(x):
                from .utils import tensor_for_import_testing

                return x * 2 * tensor_for_import_testing

        except ImportError:

            def fn(x):
                from utils import tensor_for_import_testing

                return x * 2 * tensor_for_import_testing

        x = torch.randn(10, device="cpu")
        fn(x)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_relative_import_no_modulename(self):
        try:
            from . import utils as _  # noqa: F401

            def fn(x):
                from . import utils

                return x * 2 * utils.tensor_for_import_testing

        except ImportError:

            def fn(x):
                import utils

                return x * 2 * utils.tensor_for_import_testing

        x = torch.randn(10, device="cpu")
        fn(x)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_bigbird_unsqueeze_inplace(self):
        def fn(reshape_2):
            view_2 = reshape_2.clone()
            view_2.unsqueeze_(2)
            cat_11 = torch.cat([view_2], dim=2)
            view_13 = cat_11.view((2, 12, 64, -1))
            return (view_13,)

        x = torch.randn(2, 12, 64, 64, requires_grad=True)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="aot_eager")
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_issue1466_size_aot_autograd(self):
        def fn(x):
            # do a tensor op and a size compute
            y = x * 2
            x_size = x.size()
            # trigger a graph break
            print("arf")
            # use the tensor op and size compute
            z = y.view(x_size) + 1
            return z

        x = torch.randn(2, 3, requires_grad=True)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="aot_eager")
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_ellipsis(self):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lnorm = torch.nn.LayerNorm(
                    (256,), eps=1e-06, elementwise_affine=True
                )
                self.linear = torch.nn.Linear(
                    in_features=256, out_features=256, bias=True
                )

            def forward(self, cat_10):
                lnorm = self.lnorm(cat_10)
                getitem_64 = lnorm[
                    (slice(None, None, None), slice(0, 1, None), Ellipsis)
                ]
                linear = self.linear(getitem_64)
                return (linear,)

        args = [torch.randn(2, 197, 256)]

        mod = Repro()
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)

        self.assertTrue(same(mod(*args), opt_mod(*args)))

    def test_reinplacing(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.self_layoutlm_embeddings_x_position_embeddings = (
                    torch.nn.Embedding(1024, 768)
                )
                self.self_layoutlm_embeddings_y_position_embeddings = (
                    torch.nn.Embedding(1024, 768)
                )

            def forward(self, getitem_1, getitem_2, add):
                self_layoutlm_embeddings_x_position_embeddings = (
                    self.self_layoutlm_embeddings_x_position_embeddings(getitem_1)
                )
                self_layoutlm_embeddings_y_position_embeddings = (
                    self.self_layoutlm_embeddings_y_position_embeddings(getitem_2)
                )
                add_1 = add + self_layoutlm_embeddings_x_position_embeddings
                add_2 = add_1 + self_layoutlm_embeddings_y_position_embeddings
                return (add_2,)

        mod = MockModule()
        opt_mod = torch.compile(mod, backend="aot_eager_decomp_partition")

        args = [
            ((2, 512), (2048, 4), torch.int64, "cpu", False),
            ((2, 512), (2048, 4), torch.int64, "cpu", False),
            ((2, 512, 768), (393216, 768, 1), torch.float32, "cpu", True),
        ]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]
        self.assertTrue(same_two_models(mod, opt_mod, args))

    def test_optimized_deepcopy(self):
        # See https://github.com/pytorch/pytorch/pull/88629
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(in_features=2, out_features=3, bias=True)

            def forward(self, x):
                return self.fc(x)

        mod = Foo()
        opt_mod = torch.compile(mod, backend="eager")
        args = [torch.randn(1, 2)]
        self.assertTrue(same_two_models(mod, opt_mod, args))

    def test_class_member(self):
        class Foo(torch.nn.Module):
            a = 4
            b = torch.ones(3, 4)

            def __init__(self) -> None:
                super().__init__()
                self.c = 4

            def forward(self, x):
                return x.cos() + self.a + self.b + self.c

        mod = Foo()
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        args = (torch.randn(3, 4),)
        self.assertTrue(same(mod(*args), opt_mod(*args)))

    def test_named_buffers(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = torch.nn.Buffer(torch.ones(3))
                self.y = torch.nn.Buffer(torch.ones(3))

            def forward(self, inp):
                res = 0
                for _, buffer in self.named_buffers():
                    res += buffer.sum()

                return inp.cos() + res

        mod = Foo()
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        args = (torch.randn(3, 4),)
        self.assertTrue(same(mod(*args), opt_mod(*args)))

    def test_requires_grad_guards_with_grad_mode1(self):
        def f(x):
            if x.requires_grad:
                return x + 1
            else:
                return x + 2

        x = torch.ones(2, requires_grad=True)

        f_compiled = torch.compile(f, backend="eager")
        with torch.no_grad():
            # compile an inference graph
            f_compiled(x)

        # Test: we should fail guards and recompile (even though it's still an inference graph)
        out_ref = f(x.detach())
        out = f_compiled(x.detach())

        self.assertEqual(out_ref, out)
        self.assertEqual(out_ref.requires_grad, out.requires_grad)

    def test_requires_grad_guards_with_grad_mode2(self):
        x = torch.ones(2, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        m = torch.nn.Linear(2, 2)
        m_compiled = torch.compile(m, backend="eager")

        with torch.no_grad():
            # compile an inference graph
            m_compiled(x)

        # Test: we should fail guards and recompile a training graph
        out_ref = m(x_ref)
        out = m_compiled(x)
        self.assertEqual(out_ref, out)
        self.assertEqual(out_ref.requires_grad, out.requires_grad)

    def test_is_symbolic_tracing(self):
        # Ensure no graph break here
        def fn(x):
            if is_fx_tracing_test():
                return x * 2
            return x * 4

        a = torch.randn(4)
        ref = fn(a)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(a)
        self.assertTrue(same(ref, res))

    def test_tokenization(self):
        from collections import UserDict

        class BatchEncoding(UserDict):
            """
            Copied from tokenization
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

        def tokenization(x):
            encoding = BatchEncoding({"key": x})
            return encoding["key"]

        opt_fn = torch.compile(tokenization, backend="eager")
        x = torch.rand((1, 4))
        ref = tokenization(x)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_modules(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(4, 3)

            def forward(self, inp):
                res = torch.zeros(3, 3)
                for _ in self.modules():
                    res += self.fc(inp)
                return res

        mod = Foo()
        args = (torch.ones(3, 4),)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_mod = torch.compile(mod, backend=cnt, fullgraph=True)
        self.assertTrue(same(mod(*args), opt_mod(*args)))
        self.assertEqual(cnt.op_count, 5)
        self.assertEqual(cnt.frame_count, 1)

    def test_omegaconf_listconfig_iter(self):
        obj = ListConfig()
        x = torch.zeros(2)

        def fn():
            y = x
            for i in obj:
                y += i
            return y

        expected = fn()
        actual = torch.compile(fn, fullgraph=True, backend="eager")()
        self.assertEqual(actual, expected)

    def test_user_defined_iter(self):
        class MyIter:
            def __init__(self) -> None:
                self.i = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.i < 3:
                    self.i += 1
                    return self.i
                raise StopIteration

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            for i in MyIter():
                x += i
            return x

        self.assertEqual(fn(torch.zeros(1)), torch.full([1], 6.0))

    def test_stop_iteration_reconstruct(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x.sin(), StopIteration(1, 2, 3)

        _, res = fn(torch.ones(1))
        self.assertEqual(str(res), str(StopIteration(1, 2, 3)))

    def test_tensor_data_kwarg(self):
        # https://github.com/pytorch/pytorch/issues/96278
        def f():
            return torch.tensor(data=[[1.0, -1.0]])

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(f, backend=cnt, fullgraph=True)
        self.assertTrue(same(f(), opt_fn()))
        self.assertEqual(cnt.frame_count, 1)

    def test_for_loop_graph_break(self):
        def inner(x):
            return torch.sin(x)

        def fn(x):
            for _ in range(100):
                inner(x)
                torch._dynamo.graph_break()
            return x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_for_loop_graph_break_before(self):
        # Checks that the backedge is calculated correctly
        def inner(x):
            return torch.sin(x)

        def fn(x):
            torch._dynamo.graph_break()
            for _ in range(100):
                inner(x)
            return x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 100)

    def test_avoid_dupe_specialization(self):
        def f(x, y):
            return (x + y) * 1

        opt_f = torch.compile(f, backend="aot_eager")

        for b in [True, False]:
            x = torch.randn(4, requires_grad=b)
            y = torch.randn(4, requires_grad=b)
            self.assertEqual(f(x, x), opt_f(x, x))
            self.assertEqual(f(x, y), opt_f(x, y))

    def test_validate_model_kwargs(self):
        cnt = CompileCounter()

        def f1(a, b):
            return torch.sin(a) + torch.cos(b)

        @torch.compile(backend=cnt, fullgraph=True)
        def f2(**kwargs):
            _validate_model_kwargs(f1, kwargs)
            return f1(**kwargs)

        x = torch.randn(10)
        y = torch.randn(10)

        self.assertEqual(f2(a=x, b=y), f1(x, y))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)

    def test_swin_base_tensor_attr(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # NB: not a parameter or buffer
                self.t = torch.randn(3)

            def forward(self, x):
                return x + torch.cat((self.t, self.t))

        mod = Foo()
        opt_mod = torch.compile(mod, backend="eager")
        args = [torch.randn(6)]
        self.assertTrue(same_two_models(mod, opt_mod, args))
        opt_mod(*args)

    def test_pointless_graph_removal(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            with torch.no_grad():
                torch._dynamo.graph_break()
                return x + 1

        fn(torch.randn(4))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)

    def test_output_aliases_intermediate(self):
        def f(x):
            intermediate = x.mul(2)
            return intermediate.view(-1), intermediate

        opt_f = torch.compile(f, backend="aot_eager")

        for b in [True, False]:
            x = torch.randn(4, requires_grad=b)
            out = f(x)
            out_test = opt_f(x)
            self.assertEqual(out[0], out_test[0])
            self.assertEqual(out[1], out_test[1])
            self.assertEqual(out[0].requires_grad, out_test[0].requires_grad)
            self.assertEqual(out[1].requires_grad, out_test[1].requires_grad)
            # test that the aliasing relationship of outputs is preserved
            out[0].mul_(2)
            out_test[0].mul_(2)
            self.assertEqual(out[0], out_test[0])
            self.assertEqual(out[1], out_test[1])

    def test_while_loop_graph_break(self):
        # Repro of tacotron2 cache_size_recompilation
        def inner(x):
            return torch.sin(x)

        def fn(x):
            i = 20
            while i > 10:
                x = inner(x)
                i -= 1
                torch._dynamo.graph_break()
            return x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_nested_while_loop_graph_break(self):
        def inner_loop(x):
            i = 3
            while i > 0:
                i -= 1
                x += 1
                torch._dynamo.graph_break()
            return x

        def inner(x):
            inner_loop(x)
            return torch.sin(x)

        def fn(x):
            i = 20
            while i > 10:
                x = inner(x)
                i -= 1
                torch._dynamo.graph_break()
            return x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_while_loop_graph_break_inside_call_function(self):
        # Repro of huggingface graph break inside loop in `get_parameter_dtype`.
        # Skip only the inner frame that has loop that contains graph break.
        def inner(x):
            for _ in range(3):
                x += 1
                torch._dynamo.graph_break()
            return x

        def fn(x):
            x += 2
            inner(x)
            x += 3
            return x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)

    def test_exception_in_dynamo_handling(self):
        hit_handler = False

        # See https://github.com/pytorch/pytorch/pull/96488
        @contextlib.contextmanager
        def ctx():
            try:
                yield
            except RuntimeError:
                nonlocal hit_handler
                hit_handler = True

        @torch.compile(backend="eager")
        def f():
            with ctx():
                h()

        def h():
            raise RuntimeError("boof")

        # Should not error
        f()
        self.assertTrue(hit_handler)

    def test_generator_dealloc(self):
        # See https://github.com/pytorch/pytorch/pull/96488
        #
        # NB: yes, [(...)] is intentional, this is a list containing a
        # generator
        generator_box = [(x for x in [1, 2, 3])]

        counter = torch._dynamo.testing.CompileCounter()

        def g(x):
            return x + 2

        # TODO: This test is pretty delicate.  To test if it's actually doing
        # anything, rebuild eval_frame.c with '#define TORCHDYNAMO_DEBUG 1'
        # and then look at the logs for:
        #
        # TRACE[_custom_eval_frame:650] begin <genexpr> test_repros.py 2276 -1 0 0
        # TRACE[_custom_eval_frame:664] throw <genexpr>
        #
        # This means we're actually hitting the relevant codepath

        # NB: Make sure we don't actually Dynamo this frame; if we do Dynamo
        # this frame, Dynamo actually DOES understand list.clear and will
        # arrange for the generator deallocation to happen when the eval frame
        # handler is disabled, which will prevent the bug from happening (we
        # specifically want to trigger the generator deallocation WHILE the
        # dynamo eval frame handler is active), as that will cause the
        # generator to become exhausted and trigger the throw_flag == TRUE
        # case.
        @torch._dynamo.disable(recursive=False)
        def f(x):
            generator_box.clear()
            return g(x)

        self.assertNoUnraisable(
            lambda: torch.compile(f, backend=counter)(torch.randn(3))
        )

        # Make sure the x + 2 is captured (a previous incorrect implementation
        # of this fix would have disabled the eval frame callback, which means
        # g wouldn't get traced
        self.assertEqual(counter.op_count, 1)

    def test_error_return_without_exception_set(self):
        # https://github.com/pytorch/pytorch/issues/93781
        @torch.compile(backend="eager")
        def f():
            _generator_type = type(_ for _ in ())

        self.assertNoUnraisable(f)

    def common_merge_criteria_processor_list(self, list_cls, fullgraph):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=fullgraph)
        def f(x, left, right):
            combined = _merge_criteria_processor_list(left, right)
            return combined(x)

        l1 = list_cls([torch.nn.ReLU(), torch.nn.Sigmoid()])
        l2 = list_cls([])
        input = torch.randn(16)
        result = f(input, l1, l2)
        self.assertEqual(result, l1(input))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

        cnt.clear()
        l3 = list_cls([torch.nn.SiLU()])
        expected = l3(l1(input))
        result = f(input, l1, l3)
        self.assertEqual(len(l1), 3)
        self.assertEqual(result, expected)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)

    def test_merge_criteria_processor_list1(self):
        self.common_merge_criteria_processor_list(CustomList1, False)

    def test_merge_criteria_processor_list2(self):
        self.common_merge_criteria_processor_list(CustomList2, True)

    def test_restricted_list_subclass1(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            l = CustomList2()
            l.extend([True])
            l.append(a)
            l.extend([b])
            l.pop(0)
            l.append(l.length_times_10())
            return sum(l)

        x = torch.randn(10)
        y = torch.randn(10)
        self.assertEqual(fn(x, y), x + y + 20)
        self.assertEqual(cnt.op_count, 3)

    def test_restricted_list_subclass2(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            l1 = CustomList2([a + 1])
            l2 = CustomList2([b + 2])
            l1.extend(l2)
            return l1

        x = torch.randn(10)
        y = torch.randn(10)
        z = fn(x, y)
        self.assertEqual(type(z), CustomList2)
        self.assertEqual(len(z), 2)
        self.assertEqual(z.length_times_10(), 20)
        self.assertEqual(list(z), [x + 1, y + 2])

    def test_restricted_list_subclass3(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a: CustomList2, b: CustomList2):
            a.extend(b)
            a.append_twice(b[2] + 1)
            a.append(b[3] + 2)
            return b

        x = torch.randn(10)
        y = torch.randn(10)
        l = CustomList2([x, y])
        self.assertIs(fn(l, l), l)
        self.assertEqual(len(l), 7)
        self.assertIs(l[0], x)
        self.assertIs(l[1], y)
        self.assertIs(l[2], x)
        self.assertIs(l[3], y)
        self.assertEqual(l[4], x + 1)
        self.assertIs(l[5], l[4])
        self.assertEqual(l[6], y + 2)

    def test_rewrite_assert_with_msg(self):
        def f(x):
            b = x.sin()
            assert x[0] == 3, "First dim need to be 3"  # noqa: S101
            return x.cos() + b

        args = (torch.Tensor([3, 4, 5]),)
        cnt = torch._dynamo.testing.CompileCounter()

        opt_f = torch.compile(f, backend=cnt, fullgraph=True)
        self.assertTrue(same(f(*args), opt_f(*args)))
        self.assertEqual(cnt.op_count, 6)
        self.assertEqual(cnt.frame_count, 1)

        exported, _ = torch._dynamo.export(f)(torch.Tensor([3, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

    def test_list_aliasing(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a):
            a.append(torch.sin(a[0]))
            return a

        x = torch.randn(10)
        l = [x]
        self.assertIs(fn(l), l)
        self.assertEqual(len(l), 2)
        self.assertIs(l[0], x)
        self.assertEqual(l[1], torch.sin(x))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_not_rewrite_assert_for_other_errors(self):
        def f(x):
            b = x.sin()
            if not x.sum() <= 3:
                raise ValueError("input sum needs to be 3")
            return x.cos() + b

        args = (torch.Tensor([3, 4, 5]),)
        opt_fn = torch.compile(f, backend="eager")
        with self.assertRaisesRegex(ValueError, "input sum needs to be 3"):
            opt_fn(*args)

    def test_rewrite_assert_dont_change_bytecode(self):
        def fn(x):
            with torch.no_grad():
                assert x.max() < 5, f"invalid max {x.max()}"  # noqa: S101
                x = torch.sin(x)
            return x

        x = torch.ones(4)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertTrue(same(fn(x), opt_fn(x)))

    def test_rewrite_assert_without_msg(self):
        def f(x):
            b = x.sin()
            assert x[0] == 3  # noqa: S101
            return x.cos() + b

        args = (torch.Tensor([3, 4, 5]),)
        exported, _ = torch._dynamo.export(f)(torch.Tensor([3, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

        with self.assertRaisesRegex(RuntimeError, "assertion error"):
            exported(torch.Tensor([5, 6, 7]))

    def test_rewrite_assert_with_non_string_msg(self):
        def f(x):
            b = x.sin()
            assert x[0] == 2, x  # noqa: S101
            return x.cos() + b

        torch._dynamo.utils.counters.clear()
        args = torch.Tensor([3, 4, 5])
        opt_f = torch.compile(f, backend="eager")
        with self.assertRaisesRegex(AssertionError, "tensor"):
            opt_f(args)
        for gb, cnt in torch._dynamo.utils.counters["graph_break"].items():
            if "assert with non-string message" in gb:
                self.assertEqual(cnt, 1)
                break
        else:
            # graph break not found
            self.assertTrue(False)

    def test_rewrite_assert_noop(self):
        def f(x):
            b = x.sin()
            assert True  # noqa: S101
            assert x.dtype == torch.float32  # noqa: S101
            return x.cos() + b

        args = (torch.Tensor([3, 4, 5]),)
        exported, _ = torch._dynamo.export(f)(torch.Tensor([3, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

        cnt = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=cnt, fullgraph=True)
        self.assertTrue(same(f(*args), opt_f(*args)))
        # torch._assert shouldn't be in the graph
        self.assertEqual(cnt.op_count, 3)
        self.assertEqual(cnt.frame_count, 1)

        exported, _ = torch._dynamo.export(f)(torch.Tensor([4, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

    def test_size_typematch(self):
        def f(x, y):
            if isinstance(x, torch.Size):
                return y + 1
            else:
                return y + 2

        y = torch.zeros(1)
        x1 = torch.Size((3,))
        x2 = (3,)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=cnt, fullgraph=True)
        self.assertTrue(same(f(x1, y), opt_f(x1, y)))
        self.assertTrue(same(f(x2, y), opt_f(x2, y)))
        self.assertEqual(cnt.frame_count, 2)

    def test_hf_classinstantier(self):
        # hf activations.py
        class ClassInstantier(collections.OrderedDict):
            def __getitem__(self, key):
                content = super().__getitem__(key)
                cls, kwargs = content if isinstance(content, tuple) else (content, {})
                return cls(**kwargs)

        ACT2CLS = ClassInstantier(
            {
                "relu": (nn.ReLU, {"inplace": False}),
                "tanh": nn.Tanh,
            }
        )

        @torch.compile(fullgraph=True, backend="eager")
        def f(x, act):
            return ACT2CLS[act](x)

        y = torch.randn(10)
        self.assertTrue(same(f(y, "tanh"), torch.tanh(y)))
        self.assertTrue(same(f(y, "relu"), torch.relu(y)))

    def test_ephemeral_module(self):
        # hf activations.py
        class ReLUSquaredActivation(nn.Module):
            def forward(self, input):
                relu_applied = torch.nn.functional.relu(input)
                squared = torch.square(relu_applied)
                return squared

        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            x = x + 0.2
            x = ReLUSquaredActivation()(x)
            x = x + 1
            return x

        y = torch.randn(10)
        self.assertTrue(same(f(y), ReLUSquaredActivation()(y + 0.2) + 1))

    def test_inplace_unsqueeze_input(self):
        def backend(gm, example_inputs):
            self.assertEqual(example_inputs[-1].size(), torch.Size([1, 3, 4]))
            return gm

        @torch.compile(backend=backend)
        def fn(x):
            x.unsqueeze_(0)
            return x + 1

        inputs = [torch.randn(3, 4)]
        self.assertEqual(fn(*inputs).size(), torch.Size([1, 3, 4]))
        self.assertEqual(inputs[0].size(), torch.Size([1, 3, 4]))
