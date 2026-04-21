# Owner(s): ["module: dynamo"]
# flake8: noqa: B001,B006,B020,B021,B950,C405,C416,E711,E721,E722,E731,F401,F403,F405,F541,F821,F823
# ruff: noqa: B006,E731,F401,F403,F405,F823,F841,PIE804,RSE102,TRY002,UP004,UP008,UP028
try:
    from .test_misc import *
except ImportError:
    from test_misc import *


class UserDefinedObjectTests(torch._inductor.test_case.TestCase):
    def test_module_not_callable(self):
        def fn(x):
            return torch.fft(x)

        counter = CompileCounter()
        a = torch.randn(10, 10)
        opt_fn = torch.compile(fn, backend=counter)
        self.assertRaisesRegex(
            TypeError, "'module' object is not callable", lambda: opt_fn(a)
        )

    def test_user_defined_setattr1(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            obj.y = obj.x + 1

        obj = UserDefineSetAttr()
        with patch.object(UserDefineSetAttr, "setup", True):
            obj.x = torch.randn(8)
        fn(obj)
        with patch.object(UserDefineSetAttr, "setup", True):
            self.assertEqual(obj.y, obj.x + 1)
        self.assertEqual(obj.__dict__.keys(), {"pfx_x", "pfx_y"})

    def test_user_defined_setattr2(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            obj = UserDefineSetAttr()
            obj.x = x
            obj.y = obj.x + 1
            return obj

        x = torch.randn(8)
        obj = fn(x)
        with patch.object(UserDefineSetAttr, "setup", True):
            self.assertIs(obj.x, x)
            self.assertEqual(obj.y, x + 1)
        self.assertEqual(obj.__dict__.keys(), {"pfx_x", "pfx_y"})

    def test_user_defined_binop(self):
        class MyClass:
            def __init__(self, value):
                self.value = value

            def __radd__(self, other):
                return self.value + other

        def fn(x, c):
            y = x.shape[0] + c
            return x + y

        counts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=counts)

        x = torch.randn(3)
        c = MyClass(4)
        ref = fn(x, c)
        res = opt_fn(x, c)

        self.assertTrue(same(ref, res))
        self.assertEqual(counts.frame_count, 1)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(counts.op_count, """1""")
        else:
            self.assertExpectedInline(counts.op_count, """2""")

    def test_user_defined_iter(self):
        class Mod:
            def __init__(self) -> None:
                self.a = [torch.randn(2, 2), torch.randn(2, 2)]

            def __iter__(self):
                return iter(self.a)

        def f(mod):
            ret = []
            for x in mod:
                ret.append(x + 1)
            return ret

        mod = Mod()
        counts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(f, backend=counts, fullgraph=True)
        ref = f(mod)
        res = opt_fn(mod)
        res = opt_fn(mod)
        res = opt_fn(mod)
        res = opt_fn(mod)
        self.assertTrue(same(ref, res))
        self.assertEqual(counts.frame_count, 1)

        mod.a.append(torch.randn(2, 2))
        # `for x in mod` is inlined, where iter(m.a) creates a guard on the list length of m.a
        # Mutating length of mod.a causes a re-compilation.
        ref2 = f(mod)
        res2 = opt_fn(mod)
        res2 = opt_fn(mod)
        res2 = opt_fn(mod)
        res2 = opt_fn(mod)
        self.assertTrue(same(ref2, res2))
        self.assertEqual(counts.frame_count, 2)

    def test_os_environ_get(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x):
            if os.environ.get("OS_ENVIRON_TEST") == "1":
                return x + 1
            else:
                return x - 1

        x = torch.ones(2, 3)
        try:
            original = os.environ.get("OS_ENVIRON_TEST", None)

            os.environ["OS_ENVIRON_TEST"] = "1"
            res1 = fn(x)
            self.assertEqual(res1, x + 1)
            self.assertEqual(cnts.frame_count, 1)
            os.environ["OS_ENVIRON_TEST"] = "0"
            res2 = fn(x)
            self.assertEqual(res2, x - 1)
            # Ensure re-compile if os.environ items updated
            self.assertEqual(cnts.frame_count, 2)
        finally:
            if original is None:
                del os.environ["OS_ENVIRON_TEST"]
            else:
                os.environ["OS_ENVIRON_TEST"] = original

    def test_os_environ_set_graph_break(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=False)
        def fn(x):
            x = x + 1
            os.environ["OS_ENVIRON_TEST"] = "0"
            return torch.sin(x)

        x = torch.ones(2, 3)
        try:
            original = os.environ.get("OS_ENVIRON_TEST", None)

            os.environ["OS_ENVIRON_TEST"] = "1"
            res1 = fn(x)
            self.assertEqual(res1, torch.sin(x + 1))
            self.assertEqual(os.environ["OS_ENVIRON_TEST"], "0")
            # Ensure we graph break on os.environ.__setitem__
            self.assertEqual(cnts.frame_count, 2)
        finally:
            if original is None:
                del os.environ["OS_ENVIRON_TEST"]
            else:
                os.environ["OS_ENVIRON_TEST"] = original

    def test_sys_modules(self):
        def fn(x, y):
            mod_a = sys.modules.get("aaaaaaaa")
            assert mod_a is None  # noqa: S101
            assert "bbbbbbbb" not in sys.modules  # noqa: S101

            assert "operator" in sys.modules  # noqa: S101
            operator = sys.modules["operator"]
            builtins = sys.modules.get("builtins")
            operator2 = sys.modules.get("cccccccc", operator)

            return operator.add(x, y), operator2.neg(builtins.abs(x))

        torch._dynamo.testing.standard_test(self, fn, 2, expected_ops=3)

        x = torch.randn(10, 10)
        _, guards = torch._dynamo.export(fn, x, x)
        guard_code = []
        for guard in guards:
            if guard.code_list:
                guard_code += guard.code_list

        # Filter out id-matches that won't reproduce run to run
        guard_code = filter(
            lambda line: "id" not in line and "lookup_backend" not in line,
            guard_code,
        )
        guard_code_str = "\n".join(guard_code)

        # Make sure that the dict_contains are present in the order of added
        self.assertExpectedInline(
            guard_code_str,
            """\
L['x'].size()[1] == L['x'].size()[0]
L['x'].storage_offset() == 0
2 <= L['x'].size()[0]
utils_device.CURRENT_DEVICE == None
str(L['x'].dtype) == 'torch.float32'
str(L['x'].device) == 'cpu'
L['x'].requires_grad == False
L['x'].ndimension() == 2
hasattr(L['x'], '_dynamo_dynamic_indices') == False
L['x'] is L['y']
not ___dict_contains('aaaaaaaa', G['sys'].modules)
not ___dict_contains('bbbbbbbb', G['sys'].modules)
___dict_contains('operator', G['sys'].modules)
not ___dict_contains('cccccccc', G['sys'].modules)""",
        )

    def test_getattr_dict(self):
        def fn(x):
            from torch.masked.maskedtensor._ops_refs import _MASKEDTENSOR_FUNCTION_TABLE

            return x * len(_MASKEDTENSOR_FUNCTION_TABLE)

        i = torch.randn(5)
        r1 = fn(i)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        r2 = opt_fn(i)
        self.assertEqual(r1, r2)

    def test_mro_type_tensor_no_source(self):
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            z = []
            input_type = type(torch.ones(2, 2))
            for cls in input_type.__mro__:
                z.append(cls.__name__)

            return x, input_type, z

        inp = torch.ones(2, 2)
        fn(inp)

    def test_typing_dict(self):
        def fn(d):
            return d[T]

        d = {T: torch.randn(3)}
        r1 = fn(d)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        r2 = opt_fn(d)
        self.assertEqual(r1, r2)

    @torch._dynamo.config.patch(specialize_float=True)
    def test_config_obj(self):
        class Cfg:
            def __init__(self) -> None:
                self.val = 0.5
                self.count = 3

        def fn(x, cfg):
            for i in range(cfg.count):
                x = x + cfg.val
            return x

        cfg1 = Cfg()
        cfg1.val = 1.0
        cfg2 = Cfg()
        v = torch.zeros(1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        v = opt_fn(v, cfg1)  # 3
        v = opt_fn(v, cfg2)  # 4.5
        cfg2.count = 1
        v = opt_fn(v, cfg2)  # 5
        cfg2.val = 2.0
        v = opt_fn(v, cfg2)  # 7
        self.assertEqual(v[0], 7)
        self.assertEqual(cnts.op_count, 8)

    def test_config_getattr_default(self):
        class Cfg:
            def __init__(self) -> None:
                self.val = 0.5
                self.count = 10

        def fn(x, cfg):
            if getattr(cfg, "just_add_7", False):
                return x + 7
            for i in range(cfg.count):
                x = x + cfg.val
            return x

        cfg1 = Cfg()
        v = torch.zeros(1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v, cfg1)[0], 5)
        self.assertEqual(opt_fn(v, cfg1)[0], 5)
        cfg1.just_add_7 = True
        self.assertEqual(opt_fn(v, cfg1)[0], 7)
        self.assertEqual(opt_fn(v, cfg1)[0], 7)
        cfg1.just_add_7 = False
        self.assertEqual(opt_fn(v, cfg1)[0], 5)
        self.assertEqual(opt_fn(v, cfg1)[0], 5)
        self.assertEqual(cnts.frame_count, 3)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_user_code_statically_known(self):
        from torch.fx.experimental.symbolic_shapes import (
            has_static_value,
            statically_known_true,
        )

        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            # At this point, this isn't statically known, only the hint says so.
            if statically_known_true(x.shape[0] > 9):
                raise Exception()
            torch._check(x.shape[0] >= 10)
            # But now it is.
            return statically_known_true(x.shape[0] > 9), has_static_value(x.shape[0])

        x = torch.zeros(10)
        torch._dynamo.mark_dynamic(x, 0)
        self.assertEqual(f(x), (True, False))

        @torch.compile(fullgraph=True, dynamic=True, backend="eager")
        def g(x, y):
            n = x.item()
            torch._check(n == 3)
            return has_static_value(4.0), has_static_value(n)

        out = g(torch.tensor([3]), torch.zeros(1))
        self.assertEqual(out, (True, True))

    def test_namedtuple1(self):
        def fn(a, b):
            tmp = MyTuple(a, b, a + b)
            return MyTuple(tmp.a, tmp[1], tmp.ab + b)

        v1 = torch.Tensor([10])
        v2 = torch.Tensor([20])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v1, v2).ab, 50)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_namedtuple2(self):
        def fn(packed):
            a, b, c = packed
            if hasattr(packed, "b"):
                b = packed.b + 1
            c = packed[2]
            d = len(packed._fields)
            return a + b + c + d

        v1 = torch.Tensor([1])
        v2 = torch.Tensor([2])
        v3 = torch.Tensor([3])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(MyTuple(v1, v2, v3))[0], 10)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    def test_namedtuple3(self):
        def fn(x, packed):
            if isinstance(packed, MyTuple):
                return x + 1
            else:
                return x - 1

        x = torch.rand([2, 3])
        packed = MyTuple(1, 2, 3)
        ref = fn(x, packed)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, packed)
        self.assertTrue(same(ref, res))

    def test_namedtuple_with_custom_getitem(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(my_tuple):
            return my_tuple.a + 1

        class MyTuple(typing.NamedTuple):
            a: torch.Tensor
            b: torch.Tensor

            def __getitem__(self, index):
                return MyTuple(a[index], b[index])

        a = torch.randn(2)
        b = torch.randn(2)

        out = f(MyTuple(a, b))
        self.assertTrue(same(a + 1, out))

        # Test guard evaluation in the second call
        out = f(MyTuple(a, b))
        self.assertTrue(same(a + 1, out))

    def test_namedtuple_source_dynamic_attributes(self):
        class MyNamedTuple(typing.NamedTuple):
            a: torch.Tensor
            b: torch.Tensor

        class MyNamedTupleSubclass(MyNamedTuple):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def f(tup):
            c = torch.tensor(3.0)
            tup.c = c  # Add dynamic attribute
            return tup

        extended_tup = MyNamedTupleSubclass(a=torch.tensor([1.0]), b=torch.tensor(2.0))
        result = f(extended_tup)
        # Verify the tuple has the expected structure
        self.assertEqual(result.a, torch.tensor([1.0]))
        self.assertEqual(result.b, torch.tensor(2.0))
        self.assertTrue(hasattr(result, "c"))
        self.assertEqual(result.c, torch.tensor(3.0))

    def test_namedtuple_sourceless_dynamic_attributes(self):
        class MyNamedTuple(typing.NamedTuple):
            a: torch.Tensor
            b: torch.Tensor

        class MyNamedTupleSubclass(MyNamedTuple):
            pass

        @torch.compile(backend="eager")
        def f():
            # Create namedtuple inside function (sourceless)
            tup = MyNamedTupleSubclass(a=torch.tensor([1.0]), b=torch.tensor(2.0))
            # Add dynamic attribute
            tup.c = torch.tensor(3.0)
            return tup

        result = f()
        # Verify the tuple has the expected structure
        self.assertEqual(result.a, torch.tensor([1.0]))
        self.assertEqual(result.b, torch.tensor(2.0))
        # Verify the dynamic attribute is preserved
        self.assertTrue(hasattr(result, "c"))
        self.assertEqual(result.c, torch.tensor(3.0))

    def test_namedtuple___eq__(self):
        class MyNamedTuple(typing.NamedTuple):
            a: int
            b: int

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            t1 = MyNamedTuple(a=1, b=2)
            t2 = (1, 2)
            return x.sin(), (t1 == t2)

        x = torch.randn(2)
        res = f(x)
        self.assertTrue(res[1])

    def test_structseq1(self):
        def fn(x, y):
            return torch.return_types.max((x, y))

        x = torch.randn(3, 2)
        y = torch.randn(2, 4)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

    def test_structseq2(self):
        def fn(x, y):
            return tuple(torch.return_types.qr((2 * x, y - 1)))

        x = torch.randn(3, 2)
        y = torch.randn(2, 4)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

    def test_structseq_repr(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            result = torch.max(x, dim=0)
            s = repr(result)
            return result.values

        x = torch.randn(3, 2)

        # Verify that fullgraph=True fails (confirms graph break occurs)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, fullgraph=True, backend="eager")(x)

        # Verify that it works without fullgraph
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        result = opt_fn(x)
        self.assertEqual(cnts.frame_count, 1)

    def test_class_binop(self):
        class Foo:
            def __init__(self, x):
                self.x = x

            def __add__(self, other):
                return Foo(self.x + other.x)

        def fn(a, b):
            return a + b

        x = torch.randn(2)
        a, b = Foo(x), Foo(x + 1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(a, b).x, 2 * x + 1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

        def fn(a, b):
            return a - b

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertRaises(torch._dynamo.exc.Unsupported, opt_fn, a, b)

    def test_user_getattr1(self):
        class MyConfig(dict):
            def __getattr__(self, name):
                return self[name]

        def fn(cfg, x, y):
            return x + y + cfg.offset

        x = torch.randn(10)
        cfg = MyConfig(offset=5)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(cfg, x, x), 2 * x + 5))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_user_getattr2(self):
        class MyConfig:
            defined_on_class = 1

            def __init__(self) -> None:
                self.defined_on_object = 2

            def __getattr__(self, name):
                return 3

        def fn(cfg, x):
            return x + cfg.defined_on_class - cfg.defined_on_object + cfg.not_defined

        x = torch.randn(10)
        cfg = MyConfig()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(cfg, x), x + 1 - 2 + 3))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    def test_getset_descriptor(self):
        def fn(g, x):
            # Just to make Dynamo not skip the frame
            torch.sin(x)
            return g.__get__(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)
        g = torch.Tensor.shape

        res = opt_fn(g, torch.ones(2, 2))
        exp_res = fn(g, torch.ones(2, 2))
        self.assertEqual(res, exp_res)

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            res = opt_fn(g, torch.ones(2, 2))

    def test_dict_with_descriptor(self):
        class MyDescriptor:
            def __get__(self, obj, objtype=None):
                if obj is None:
                    return self
                return obj.__dict__.get("_value", 0)

            def __set__(self, obj, value):
                obj.__dict__["_value"] = value

        class MyClass:
            prop = MyDescriptor()

            def __init__(self):
                self.prop = 42

        def fn(obj):
            obj.__dict__["extra"] = 99
            return obj.prop + obj.__dict__["extra"]

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        obj = MyClass()
        res = fn(obj)

        obj2 = MyClass()
        out = opt_fn(obj2)
        self.assertEqual(res, out)
        self.assertEqual(obj2.__dict__["_value"], 42)
        self.assertEqual(obj2.__dict__["extra"], 99)

    def test_dict_with_slots(self):
        class SlottedClass:
            __slots__ = ("x",)

            def __init__(self, x):
                self.x = x

        def fn(obj):
            # SlottedClass doesn't have __dict__, so this should fail or be handled
            return obj.x * 2

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        obj = SlottedClass(5)
        res = fn(obj)
        out = opt_fn(obj)
        self.assertEqual(res, out)

    def test_dict_with_slots_and_dict(self):
        class SlottedWithDict:
            __slots__ = ("x", "__dict__")

            def __init__(self, x):
                self.x = x

        def fn(obj):
            obj.__dict__["custom"] = 100
            return obj.x + obj.__dict__["custom"]

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        obj = SlottedWithDict(7)
        res = fn(obj)

        obj2 = SlottedWithDict(7)
        out = opt_fn(obj2)
        self.assertEqual(res, out)
        self.assertEqual(obj2.__dict__["custom"], 100)

    def test_dict_descriptor_interaction(self):
        class DescriptorWithDict:
            def __get__(self, obj, objtype=None):
                if obj is None:
                    return self
                return obj.__dict__.get("_internal", "default")

            def __set__(self, obj, value):
                obj.__dict__["_internal"] = f"desc_{value}"

        class MyClass:
            desc = DescriptorWithDict()

            def __init__(self):
                pass

        def fn(obj):
            obj.desc = "hello"
            obj.__dict__["_internal"] = "direct"
            return obj.desc

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        obj = MyClass()
        ref = fn(obj)

        obj2 = MyClass()
        out = opt_fn(obj2)
        self.assertEqual(ref, out)
        self.assertEqual(out, "direct")

    def test_mutable_mapping_dict_update(self):
        """Test that MutableMappingVariable handles __dict__ updates correctly."""
        from collections import OrderedDict

        class CustomMapping(OrderedDict):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def fn(mapping):
            mapping.__dict__["custom_attr"] = 42
            return mapping.__dict__["custom_attr"]

        mapping = CustomMapping()
        out = fn(mapping)
        self.assertEqual(out, 42)
        self.assertIn("custom_attr", mapping.__dict__)
        self.assertEqual(mapping.__dict__["custom_attr"], 42)

    def test_mutable_mapping_dict_access_pattern(self):
        """Test accessing attributes through __dict__ on MutableMapping subclasses."""
        from collections import OrderedDict

        class TrackedDict(OrderedDict):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def fn(d):
            # Pattern: check if attribute exists in __dict__
            if "tracker" not in d.__dict__:
                d.__dict__["tracker"] = []
            d.__dict__["tracker"].append(1)
            return len(d.__dict__["tracker"])

        d = TrackedDict()
        out = fn(d)
        self.assertEqual(out, 1)
        self.assertIn("tracker", d.__dict__)
        self.assertEqual(d.__dict__["tracker"], [1])

    def test_mutable_mapping_lazy_dict_initialization(self):
        """Test lazy initialization pattern with MutableMapping __dict__."""
        from collections import defaultdict

        class LazyMapping(dict):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def fn(mapping):
            # Lazy initialization in __dict__
            if not hasattr(mapping, "_cache"):
                mapping.__dict__["_cache"] = {}
            mapping.__dict__["_cache"]["key"] = "value"
            return mapping.__dict__["_cache"]["key"]

        m = LazyMapping()
        out = fn(m)
        self.assertEqual(out, "value")
        self.assertIn("_cache", m.__dict__)
        self.assertEqual(m.__dict__["_cache"]["key"], "value")

    def test_mutable_mapping_dict_with_property_setter(self):
        """Test MutableMapping with property setters that access __dict__."""
        from collections import OrderedDict

        class PropertyMapping(OrderedDict):
            def __init__(self):
                super().__init__()
                self.__dict__["_value"] = 0

            @property
            def value(self):
                return self.__dict__.get("_value", 0)

            @value.setter
            def value(self, v):
                self.__dict__["_value"] = v

        @torch.compile(fullgraph=True, backend="eager")
        def fn(m):
            m.value = 100
            return m.value

        m = PropertyMapping()
        out = fn(m)
        self.assertEqual(out, 100)
        self.assertIn("_value", m.__dict__)
        self.assertEqual(m.__dict__["_value"], 100)

    def test_mutable_mapping_dict_multiple_accesses(self):
        """Test multiple accesses and mutations to MutableMapping __dict__."""
        from collections import OrderedDict

        class MultiAccessMapping(OrderedDict):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def fn(mapping):
            # Multiple accesses to __dict__
            mapping.__dict__["a"] = 1
            mapping.__dict__["b"] = mapping.__dict__["a"] + 1
            mapping.__dict__["c"] = mapping.__dict__["b"] + 1
            return mapping.__dict__["a"] + mapping.__dict__["b"] + mapping.__dict__["c"]

        m = MultiAccessMapping()
        out = fn(m)
        self.assertEqual(out, 6)  # 1 + 2 + 3
        self.assertIn("a", m.__dict__)
        self.assertIn("b", m.__dict__)
        self.assertIn("c", m.__dict__)
        self.assertEqual(m.__dict__["a"], 1)
        self.assertEqual(m.__dict__["b"], 2)
        self.assertEqual(m.__dict__["c"], 3)

    def test_get_attr_function(self):
        def fn(g, x):
            return g(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        g = torch.Tensor.shape.__get__

        res = opt_fn(g, torch.ones(2, 2))
        exp_res = fn(g, torch.ones(2, 2))
        self.assertEqual(res, exp_res)

    def test_user_getattribute(self):
        class MyObject:
            def __init__(self) -> None:
                self.custom_dict = {"a": torch.rand((2, 2))}
                self.my_number = 42

            def __getattribute__(self, name):
                custom_dict = super().__getattribute__("custom_dict")
                if name in custom_dict:
                    return custom_dict[name]
                return super().__getattribute__(name)

            def run(self, x):
                return self.my_number * x + self.a * x

        def fn(obj, x):
            return obj.run(x)

        obj = MyObject()
        x = torch.rand((2, 2))
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(obj, x), fn(obj, x)))

    def test_nn_module_getattr(self):
        class MyMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.custom_dict = {"queue": [torch.rand((2, 2)) for _ in range(3)]}
                self.other_attr = torch.rand((2, 2))

            def __getattr__(self, name):
                custom_dict = self.custom_dict
                if name in custom_dict:
                    return custom_dict[name]
                return super().__getattr__(name)

            def forward(self, x):
                return x @ self.other_attr + self.queue[-1]

        x = torch.rand((2, 2))
        mod = MyMod()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_mod = torch.compile(mod, backend=cnts)
        self.assertTrue(same(opt_mod(x), mod(x)))
        self.assertTrue(cnts.frame_count, 1)
        self.assertTrue(cnts.op_count, 2)

    def test_nn_module_getattribute(self):
        class MyMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.my_number = 42

            def __getattribute__(self, name):
                if name == "special_attr":
                    return torch.tensor([[1, 2], [3, 4]])
                return super().__getattribute__(name)

            def forward(self, x):
                return self.my_number * x + self.special_attr * x

        def fn(mod, x):
            return mod(x)

        mod = MyMod()
        x = torch.rand((2, 2))
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(mod, x), fn(mod, x)))

    def test_nn_module_getattribute_simple_delegation(self):
        # Test that nn.Module with __getattribute__ that overrides a
        # single attribute name compiles without graph break.
        class MyMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = 3.0

            def __getattribute__(self, name):
                if name == "my_scale":
                    return super().__getattribute__("scale")
                return super().__getattribute__(name)

            def forward(self, x):
                return x * self.my_scale

        mod = MyMod()
        x = torch.randn(2, 4)
        expected = mod(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(mod, backend=cnts)
        result = opt_fn(x)
        self.assertTrue(same(result, expected))

    def test_nn_module_getattribute_graph_break(self):
        # __getattribute__ that Dynamo cannot trace produces correct results
        # via eager fallback instead of crashing.
        class MyMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def __getattribute__(self, name):
                if name == "my_attr":
                    return eval("42")  # eval is untraceable
                return super().__getattribute__(name)

            def forward(self, x):
                a = self.my_attr
                return self.linear(x) + a

        mod = MyMod()
        x = torch.randn(2, 4)
        expected = mod(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(mod, backend=cnts)
        result = opt_fn(x)
        self.assertTrue(same(result, expected))

    def test_constant_getattr(self):
        # https://github.com/pytorch/pytorch/issues/97480
        def fn():
            return getattr(None, "arg", 3)

        cnt = torch._dynamo.testing.CompileCounter()
        optimized_fn = torch.compile(fn, backend=cnt)
        res = optimized_fn()
        self.assertTrue(same(res, 3))

    def test_user_property(self):
        class MyConfig:
            @property
            def prop5(self):
                return 5

        def fn(cfg, x, y):
            return x + y + cfg.prop5

        x = torch.randn(10)
        cfg = MyConfig()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(cfg, x, x), 2 * x + 5))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_dataclass_fields(self):
        @dataclasses.dataclass
        class MyDataClass:
            a: torch.Tensor
            b: torch.Tensor = None
            c: torch.Tensor = None
            d: torch.Tensor = None
            e: torch.Tensor = None

        def fn(obj):
            class_fields = dataclasses.fields(obj)
            assert len(class_fields)  # noqa: S101
            assert all(field.default is None for field in class_fields[1:])  # noqa: S101
            other_fields_are_none = all(
                getattr(obj, field.name) is None for field in class_fields[1:]
            )
            assert not other_fields_are_none  # noqa: S101

            if not hasattr(obj, "a"):
                return -1
            if hasattr(obj, "z"):
                return -2

            total = getattr(obj, class_fields[0].name)
            for field in class_fields[1:]:
                v = getattr(obj, field.name)
                if v is not None:
                    total += v

            return total

        obj1 = MyDataClass(torch.randn(10), torch.randn(10), torch.randn(10))
        obj2 = MyDataClass(torch.randn(10), e=torch.randn(10))
        correct1 = fn(obj1)
        correct2 = fn(obj2)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(obj1), correct1))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(obj2), correct2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

        # guard failure
        obj2.z = True
        self.assertEqual(opt_fn(obj2), -2)

    def test_dataclass_local_hasattr(self):
        cnt = CompileCounter()
        x = torch.randn(10)

        @dataclasses.dataclass
        class MyDataClass:
            a: torch.Tensor
            b: torch.Tensor

        @torch.compile(backend=cnt, fullgraph=True)
        def fn():
            obj = MyDataClass(x + 1, x - 1)
            if not hasattr(obj, "a"):
                return -1
            if hasattr(obj, "z"):
                return -2
            return obj

        result = fn()
        self.assertIsInstance(result, MyDataClass)
        self.assertEqual(result.a, x + 1)
        self.assertEqual(result.b, x - 1)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_catch_watchings1(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            with warnings.catch_warnings(record=True):
                return x.sin()

        x = torch.randn(8)
        self.assertEqual(fn(x), x.sin())
        self.assertEqual(cnt.frame_count, 1)

    def test_catch_watchings2(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            return x.sin(), warnings.catch_warnings(record=True)

        x = torch.randn(8)
        _, a = fn(x)
        _, b = fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertIsInstance(a, warnings.catch_warnings)
        self.assertIsInstance(b, warnings.catch_warnings)
        self.assertIsNot(a, b)

    def test_mutable_mapping_multiple_inheritance(self):
        class MyWeirdDict(collections.abc.MutableMapping, torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self._items = kwargs

            def keys(self):
                return self._items.keys()

            def __getitem__(self, item):
                return self._items[item]

            def __setitem__(self, key, value):
                self._items[key] = value

            def __delitem__(self, item):
                del self._items[item]

            def __len__(self):
                return len(self._items)

            def __iter__(self):
                yield from self._items

            def __hash__(self):
                return hash(id(self))

            def items(self):
                for k, v in self._items.items():
                    yield (k, v)

        @torch.compile(fullgraph=True, backend="eager")
        def to_weird_dict(td):
            return MyWeirdDict(**td)

        d = MyWeirdDict(a=1, b=2, c=3)
        res = to_weird_dict(d)
        self.assertEqual(tuple(d.items()), tuple(res.items()))

    def test_dunder_new_function_inlining(self):
        # https://github.com/pytorch/pytorch/issues/107460

        counters.clear()

        class ModelA(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.tanh(x + 1)

        class ModelB(torch.nn.Module):
            def __new__(cls):
                return ModelA()

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.Linear(2, 2)

            def forward(self, x):
                other = ModelB()
                return self.layer(x) + other(x)

        x = torch.rand(2, 2)
        m = Model()

        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        ref = m(x)
        res = opt_m(x)
        self.assertTrue(same(ref, res))

    def test_dunder_new_function_inlining1(self):
        class Mock:
            def __new__(cls):
                return super().__new__(cls)

            def __init__(self):
                self.c = 5

            def run(self, x):
                return x * self.c

        def fn(x):
            mock = Mock()
            return mock.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)

        self.assertEqual(fn(x), opt_fn(x))

    def test_dunder_new_function_inlining2(self):
        class Vehicle:
            def __new__(cls, *args, **kwargs):
                return super(Vehicle, cls).__new__(cls)

            def __init__(self, make, model, year):
                self.make = make
                self.model = model
                self.year = year

        class Car(Vehicle):
            def __new__(cls, *args, **kwargs):
                return super(Car, cls).__new__(cls)

            def __init__(self, make, model, year, num_doors):
                super(Car, self).__init__(make, model, year)
                self.num_doors = num_doors

        class ElectricCar(Car):
            def __new__(cls, *args, **kwargs):
                return super(ElectricCar, cls).__new__(cls)

            def __init__(self, make, model, year, num_doors, battery_capacity):
                super(ElectricCar, self).__init__(make, model, year, num_doors)
                self.battery_capacity = battery_capacity

            def run(self, x):
                return torch.sin(x)

        def fn(x):
            ev = ElectricCar("Tesla", "Model S", 2022, 4, "100 kWh")
            return ev.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.randn(4)

        self.assertEqual(fn(x), opt_fn(x))

    def test_dunder_new_function_inlining3(self):
        class Foo:
            def __new__(cls):
                instance = object.__new__(cls)
                instance.a = 3
                return instance

            def __init__(self):
                self.a = 5

            def run(self, x):
                return torch.sin(x) * self.a

        class Bar:
            def __new__(cls):
                instance = object.__new__(Foo)  # not returning a new instance of Bar
                instance.a = 7
                return instance

            def __init__(self):
                self.a = 11  # not called in Bar()

            def run(self, x):
                return torch.sin(x) * self.a

        def fn(x):
            bar = Bar()
            return bar.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_dunder_new_function_inlining4(self):
        class Mock(object):
            def __new__(cls, *args):
                return object.__new__(cls)

            def __init__(self):
                self.a = 5

            def run(self, x):
                return torch.sin(x) * self.a

        def fn(x):
            mock = Mock()
            return mock.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_user_defined_object_class_interaction(self):
        class Foo:
            x = 5

        class Mock:
            # This is a class variable
            class_variable = Foo()

            @classmethod
            def get_class_variable(cls):
                # Accessing the class variable using the cls parameter
                return cls.class_variable.x

            def run(self, x):
                return self.get_class_variable() * x

        def fn(x):
            mock = Mock()
            return mock.run(x)

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_multiple_inheritance(self):
        class Base1:
            def __new__(cls):
                return super().__new__(cls)

            def __init__(self):
                super().__init__()
                if not hasattr(self, "base2"):
                    raise ValueError("Wrong MRO tracing")
                self.base1 = 3

        class Base2:
            def __new__(cls):
                return super().__new__(cls)

            def __init__(self):
                super().__init__()
                self.base2 = 5

        class Derived(Base1, Base2):
            def __new__(cls):
                return super().__new__(cls)

            def __init__(self):
                super().__init__()
                self.derived = 7

            def run(self, x):
                return self.base1 * self.base2 * self.derived * x

        def fn(x):
            o = Derived()
            return o.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_class_duner_mro(self):
        class ModuleA(torch.nn.Module):
            pass

        class ModuleB(ModuleA):
            pass

        def fn(x, mod):
            if ModuleA in type(mod).__mro__:
                return x + 1
            else:
                return x - 1

        x = torch.rand(2, 3)
        mod = ModuleB()
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        ref = fn(x, mod)
        res = opt_fn(x, mod)
        self.assertTrue(same(ref, res))

    def test_class_duner_flags(self):
        class ModuleA(torch.nn.ModuleDict, collections.abc.MutableMapping):
            def __hash__(self):
                return id(self)

        def fn(x, mod_class):
            if mod_class.__flags__ & TPFLAGS_MAPPING:
                return x + 1
            else:
                return x - 1

        x = torch.rand(2, 3)
        mod_class = ModuleA
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        ref = fn(x, mod_class)
        res = opt_fn(x, mod_class)
        self.assertTrue(same(ref, res))

        def fn(x, mod):
            if type(mod).__flags__ & TPFLAGS_MAPPING:
                return x + 1
            else:
                return x - 1

        x = torch.rand(2, 3)
        mod = ModuleA()
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        ref = fn(x, mod)
        res = opt_fn(x, mod)
        self.assertTrue(same(ref, res))

    @torch._dynamo.config.patch(nested_graph_breaks=False)
    def test_module_deepcopy(self):
        m1 = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )
        m2 = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )

        def fn(m, x):
            m_copy = copy.deepcopy(m)
            return m_copy(x)

        v = torch.randn(10)
        correct1 = fn(m1, v)
        correct2 = fn(m2, v)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            self.assertTrue(same(opt_fn(m1, v), correct1))
        for _ in range(10):
            self.assertTrue(same(opt_fn(m2, v), correct2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    def test_deepcopy_dict(self):
        MY_DICT = {"a": 1, "b": 2.0, "c": None}

        def fn(x):
            d = copy.deepcopy(MY_DICT)
            d["b"] = 3.0
            return x + d["b"]

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(result, correct)

    def test_deepcopy_nested_dict(self):
        NESTED = {"a": {"b": 1.0}, "c": [2.0, 3.0]}

        def fn(x):
            d = copy.deepcopy(NESTED)
            return x + d["a"]["b"] + d["c"][0]

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(result, correct)

    def test_deepcopy_list(self):
        MY_LIST = [1.0, 2.0, 3.0]

        def fn(x):
            lst = copy.deepcopy(MY_LIST)
            lst[0] = 5.0
            return x + lst[0]

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(result, correct)

    def test_deepcopy_user_defined_object(self):
        class MyConfig:
            def __init__(self, hidden_size=64):
                self.hidden_size = hidden_size

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MyConfig()
                self.linear = torch.nn.Linear(64, 64)

            def forward(self, x):
                cfg = copy.deepcopy(self.config)
                return self.linear(x) * cfg.hidden_size

        m = MyModule()
        x = torch.randn(2, 64)
        result = torch.compile(m, fullgraph=True, backend="eager")(x)
        self.assertEqual(m(x), result)

    def test_deepcopy_user_defined_object_with_containers(self):
        class Config:
            def __init__(self):
                self.sizes = [1, 2, 3]
                self.mapping = {"a": 10, "b": 20}
                self.flags = (True, False)

        def fn(x, cfg):
            c = copy.deepcopy(cfg)
            c.sizes[0] = 99
            c.mapping["a"] = 77
            return x + c.sizes[0] + c.mapping["a"]

        cfg = Config()
        x = torch.randn(4)
        correct = fn(x, cfg)
        result = torch.compile(fn, fullgraph=True, backend="eager")(x, cfg)
        self.assertEqual(result, correct)
        # Verify deepcopy didn't mutate original
        self.assertEqual(cfg.sizes[0], 1)
        self.assertEqual(cfg.mapping["a"], 10)

    def test_deepcopy_set(self):
        MY_SET = {1, 2, 3}

        def fn(x):
            s = copy.deepcopy(MY_SET)
            return x + len(s)

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(result, correct)

    def test_deepcopy_frozenset(self):
        MY_FROZENSET = frozenset([1, 2, 3])

        def fn(x):
            s = copy.deepcopy(MY_FROZENSET)
            return x + len(s)

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(result, correct)

    def test_deepcopy_user_defined_object_with_method(self):
        class MyConfig:
            def __init__(self, hidden_size=64):
                self.hidden_size = hidden_size

            def get_size(self):
                return self.hidden_size

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MyConfig()
                self.linear = torch.nn.Linear(64, 64)

            def forward(self, x):
                cfg = copy.deepcopy(self.config)
                return self.linear(x) * cfg.get_size()

        m = MyModule()
        x = torch.randn(2, 64)
        correct = m(x)
        result = torch.compile(m, fullgraph=True, backend="eager")(x)
        self.assertEqual(result, correct)

    def test_deepcopy_nested_user_defined_object(self):
        class Inner:
            def __init__(self, scale):
                self.scale = scale

        class Outer:
            def __init__(self):
                self.inner = Inner(2.0)
                self.bias = 1.0

        def fn(x, cfg):
            c = copy.deepcopy(cfg)
            c.inner.scale = 3.0
            return x * c.inner.scale + c.bias

        cfg = Outer()
        x = torch.randn(4)
        correct = fn(x, cfg)
        result = torch.compile(fn, fullgraph=True, backend="eager")(x, cfg)
        self.assertEqual(result, correct)
        # Verify deepcopy didn't mutate original
        self.assertEqual(cfg.inner.scale, 2.0)

    def test_deepcopy_with_getattribute_override(self):
        # Regression test: classes that override __getattribute__ (like
        # HuggingFace PretrainedConfig) caused a graph break on
        # __reduce_ex__ because SuperVariable.call_method for
        # object.__getattribute__ bypassed the polyfill detection in
        # resolve_type_attr.
        class Config:
            attribute_map = {}

            def __init__(self, hidden_size=768, num_layers=6):
                self.hidden_size = hidden_size
                self.num_layers = num_layers

            def __getattribute__(self, key):
                if key != "attribute_map" and key in super().__getattribute__(
                    "attribute_map"
                ):
                    key = super().__getattribute__("attribute_map")[key]
                return super().__getattribute__(key)

        def fn(x, config):
            c = copy.deepcopy(config)
            return x * c.hidden_size + c.num_layers

        x = torch.randn(3)
        config = Config()
        correct = fn(x, config)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x, config)
        self.assertEqual(result, correct)

    def test_type_copy(self):
        def fn(seq):
            a, b = seq
            return type(seq)([a + 1, b + 2, a + b])

        args1 = [torch.randn(10), torch.randn(10)]
        args2 = (torch.randn(10), torch.randn(10))
        correct1 = fn(args1)
        correct2 = fn(args2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(args1), correct1))
        self.assertTrue(same(opt_fn(args2), correct2))
        self.assertIsInstance(opt_fn(args1), list)
        self.assertIsInstance(opt_fn(args2), tuple)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 6)

    def test_nesteduserfunction_setattr(self):
        x = 0

        def update(y):
            def wrapper():
                x += y

            return wrapper

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            w = update(123)
            w.__wrapped__ = x
            return t.sin(), w

        t = torch.randn(2)
        y, w = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(w.__wrapped__, x)

    def test_object_setattr(self):
        @dataclasses.dataclass
        class A:
            x: torch.Tensor

        def fn1(x) -> None:
            a = A(x)
            object.__setattr__(a, "x", x + 2)
            return a

        x1 = torch.randn(10)
        obj11 = fn1(x1.clone())

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch.compile(fn1, backend=cnts, fullgraph=True)
        obj12 = opt_fn1(x1.clone())
        self.assertTrue(same(obj11.x, x1 + 2))
        self.assertTrue(same(obj12.x, x1 + 2))
        self.assertTrue(same(obj11.x, obj12.x))
        self.assertEqual(cnts.frame_count, 1)

        @dataclasses.dataclass(frozen=True)
        class B:
            x: torch.Tensor

        def fn2(x) -> None:
            b = B(x)
            return b

        x2 = torch.randn(10)
        obj21 = fn2(x2.clone())

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn2 = torch.compile(fn2, backend=cnts, fullgraph=True)
        obj22 = opt_fn2(x2.clone())
        self.assertTrue(same(obj21.x, x2))
        self.assertTrue(same(obj22.x, x2))
        self.assertTrue(same(obj21.x, obj22.x))
        self.assertEqual(cnts.frame_count, 0)

        @dataclasses.dataclass(frozen=True)
        class C:
            x: torch.Tensor

        def fn3(x) -> None:
            c = C(x)
            object.__setattr__(c, "x", x + 2)
            return c

        x3 = torch.randn(10)
        obj31 = fn3(x3.clone())

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn3 = torch.compile(fn3, backend=cnts, fullgraph=True)
        obj32 = opt_fn3(x3.clone())
        self.assertTrue(same(obj31.x, x3 + 2))
        self.assertTrue(same(obj32.x, x3 + 2))
        self.assertTrue(same(obj31.x, obj32.x))
        self.assertEqual(cnts.frame_count, 1)

        @dataclasses.dataclass(frozen=True)
        class D:
            x: torch.Tensor

            def __post_init__(self):
                object.__setattr__(self, "y", self.x + 2)

        def fn4(x) -> None:
            d = D(x)
            return d

        x4 = torch.randn(10)
        obj41 = fn4(x4.clone())

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn4 = torch.compile(fn4, backend=cnts, fullgraph=True)
        obj42 = opt_fn4(x4.clone())
        self.assertTrue(same(obj41.x, x4))
        self.assertTrue(same(obj42.x, x4))
        self.assertTrue(same(obj41.x, obj42.x))
        self.assertTrue(same(obj41.y, x4 + 2))
        self.assertTrue(same(obj42.y, x4 + 2))
        self.assertTrue(same(obj41.y, obj42.y))
        self.assertEqual(cnts.frame_count, 1)

    def test_thread_local_setattr(self):
        from threading import local

        loc = local()

        @torch.compile(fullgraph=True, backend="eager")
        def fn(x, l):
            l.x = x
            return x + 1

        x = torch.ones(2, 2)
        fn(x, loc)

        self.assertTrue(loc.x is x)

    def test_user_defined_class_name(self):
        class MyClassFoo:
            pass

        def fn1(a, b, c):
            tmp = MyClassFoo()
            if tmp.__class__.__name__ == "MyClassFoo":
                return a - b / c

        torch._dynamo.testing.standard_test(self, fn=fn1, nargs=3)

    def test_class_reassignment_graph_break(self):
        class BaseClass:
            def __init__(self, x):
                self.x = x

        class DerivedClass(BaseClass):
            def __init__(self, x):
                super().__init__(x)
                self.y = x * 2

        def fn(x):
            obj = BaseClass(5)
            obj.__class__ = DerivedClass
            is_derived = isinstance(obj, DerivedClass)
            has_y = hasattr(obj, "y")
            return x + 1, is_derived, has_y

        x = torch.ones(1)
        eager_result = fn(x)
        compiled_result = torch.compile(fn, backend="eager")(x)
        self.assertEqual(eager_result, compiled_result)

    def test_user_defined_class_python_type(self):
        class MyClass1:
            pass

        class ExampleMeta(type):
            pass

        class MyClass2(metaclass=ExampleMeta):
            pass

        def fn(x, c):
            if isinstance(c, MyClass1):
                return x + 1
            elif isinstance(c, MyClass2):
                return x + 2
            else:
                return x + 3

        x = torch.rand(3)
        opt_fn = torch.compile(fn, backend="eager")
        for c in [MyClass1, MyClass2]:
            ref = fn(x, c)
            res = opt_fn(x, c)
            self.assertTrue(same(ref, res))

    def test_super_calling_with_metaclass(self):
        class ExampleMeta(type):
            pass

        class MyClass1(metaclass=ExampleMeta):
            coeff = 4  # Force the constant guard to test source in guards

            @classmethod
            def add(cls, x):
                return x + 1

        class MyClass2(MyClass1):
            @classmethod
            def add(cls, x):
                torch._dynamo.graph_break()
                return x + super().add(x) + super().coeff

        def fn(x, obj):
            return x + obj.add(x)

        x = torch.rand(3)
        obj = MyClass2()
        opt_fn = torch.compile(fn, backend="eager")
        ref = fn(x, obj)
        res = opt_fn(x, obj)
        self.assertTrue(same(ref, res))

    def test_usr_cls_staticmethod(self):
        class Foo:
            @staticmethod
            def bar(a, b):
                return a + b

        def fn(a, b):
            return Foo.bar(a, b) - 1

        torch._dynamo.testing.standard_test(self, fn=fn, nargs=2)

    def test_usr_cls_classmethod(self):
        class Foo:
            @classmethod
            def bar(cls, a, b):
                return a + b

        def fn(a, b):
            return Foo.bar(a, b) - 1

        torch._dynamo.testing.standard_test(self, fn=fn, nargs=2)

    def test_dunder_methods(self):
        class Foo:
            def __init__(self, val):
                super().__init__()
                self.val = val

            def __add__(self, other):
                return Foo(self.val + other.val)

            def __mul__(self, other):
                return Foo(self.val * other.val)

            def __truediv__(self, other):
                return Foo(self.val / other.val)

            def __sub__(self, other):
                return Foo(self.val - other.val)

        def fn(a, b, c):
            return Foo(a) + Foo(b) * Foo(c) / Foo(a) - Foo(b)

        torch._dynamo.testing.standard_test(self, fn=fn, nargs=3, expected_ops=4)

    def test_function_annotation(self):
        class Variable:
            pass

        def fn(x):
            x = x / 3.0

            def inner(y: typing.List[Variable]):
                return x + 1

            return inner

        x1 = torch.randn(10)
        obj2 = fn(x1)([])

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        opt_fn_inner = torch._dynamo.optimize_assert(cnts)(opt_fn(x1))
        obj1 = opt_fn_inner([])
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 2)

    def test_function_generic_alias_annotation(self):
        class Variable:
            pass

        def fn(x):
            x = x / 3.0

            def inner(y: list[Variable]):
                return x + 1

            return inner

        x1 = torch.randn(10)
        obj2 = fn(x1)([])

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        opt_fn_inner = torch._dynamo.optimize_assert(cnts)(opt_fn(x1))
        obj1 = opt_fn_inner([])
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 2)

    def test_top_package_import(self):
        def fn(x):
            import torch.fx

            assert not isinstance(x, torch.fx.Proxy)  # noqa: S101
            return torch.sin(x)

        x = torch.randn(4, 5)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_typing_typevar(self):
        def fn(x):
            def sumt(y: torch.Tensor) -> torch.Tensor:
                return torch.sum(y)

            def foo(c: typing.Callable[[T], T], y: T) -> T:
                return c(y)

            return foo(sumt, x)

        x = torch.randn(3)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)

    def test_typing_union_and_optional(self):
        def fn(x):
            a = torch.jit.annotate(typing.Dict[str, typing.Optional[torch.Tensor]], {})
            b = torch.jit.annotate(
                typing.Dict[str, typing.Union[torch.Tensor, None]], {}
            )
            return a, b, x + 1

        x = torch.randn(3)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=False)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_tying_union_new_syntax(self):
        def fn(x):
            def inner1(y: torch.Tensor | None):
                return y

            def inner2(y: None | torch.Tensor):
                return y

            def inner3(y: torch.Tensor | list[int]):
                return y

            return x + 1

        torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(3))

    @unittest.expectedFailure
    def test_typing_union_new_syntax_reconstruct(self):
        def fn(x):
            return (
                x + 1,
                torch.Tensor | None,
                None | torch.Tensor,
                torch.Tensor | list[int],
            )

        torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(3))

    def test_const_dict_variable_python_type(self):
        from torch._dynamo.variables import ConstantVariable, ConstDictVariable

        make_key = ConstantVariable.create

        d1 = {
            make_key("a"): ConstantVariable.create(10),
            make_key("b"): ConstantVariable.create(20),
        }
        d2 = collections.OrderedDict(
            [
                (make_key("x"), ConstantVariable.create(12)),
                (make_key("y"), ConstantVariable.create(22)),
            ]
        )
        self.assertEqual(ConstDictVariable(d1).python_type(), dict)
        self.assertEqual(
            ConstDictVariable(d2, collections.OrderedDict).python_type(),
            collections.OrderedDict,
        )

    def test_function_return_none_creates_constant_variable(self):
        """
        Test that functions returning None properly return ConstantVariable.create(None)
        instead of raw None, which would violate the stack's type contract.

        Regression test for: Avoid using Optional[VariableTracker]
        """

        def gn(x):
            return

        torch._dynamo.config.reorderable_logging_functions.add(gn)

        @torch.compile(backend="eager")
        def fn(x):
            x = x + 1
            if gn(x) is None:
                return x + 2
            return x + 4

        # If this doesn't crash, the test passes
        fn(torch.ones(3))

    @unittest.skipIf(not torch.distributed.is_available(), "requires distributed")
    def test_or_union_type_opaque_class(self):
        # Test that or_ on opaque class types (e.g. Shard | _StridedShard)
        # doesn't cause a graph break.
        from torch.distributed.tensor import Shard
        from torch.distributed.tensor.placement_types import _StridedShard

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            _ = Shard | _StridedShard
            return x + 1

        x = torch.randn(4)
        result = fn(x)
        self.assertEqual(result, x + 1)
        self.assertEqual(cnt.frame_count, 1)

    def test_nn_functional_reduction(self):
        def fn(loss, reduction):
            reduction_enum = F._Reduction.get_enum(reduction)
            if reduction_enum == 0:
                return loss
            elif reduction_enum == 1:
                return loss.mean()
            elif reduction_enum == 2:
                return loss.sum()

        x = torch.rand([3, 5])
        y = "mean"
        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertTrue(torch.allclose(ref, res))

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_module_complex_iter(self):
        n_embd = 768
        block_size = 128
        vocab_size = 65
        embd_pdrop = 0.1

        class FakeGPT(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
                self.pos_emb = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
                self.drop = torch.nn.Dropout(embd_pdrop)
                self.ln_f = torch.nn.LayerNorm(n_embd)
                self.head = torch.nn.Linear(n_embd, vocab_size, bias=False)

                self.block_size = block_size
                self.names = []

            def forward(self, idx, targets=None):
                b, t = idx.size()
                assert t <= self.block_size, (  # noqa: S101
                    "Cannot forward, model block size is exhausted."
                )

                # forward the GPT model
                token_embeddings = self.tok_emb(
                    idx
                )  # each index maps to a (learnable) vector
                position_embeddings = self.pos_emb[
                    :, :t, :
                ]  # each position maps to a (learnable) vector
                x = self.drop(token_embeddings + position_embeddings)
                x = self.blocks(x)
                x = self.ln_f(x)
                logits = self.head(x)

                # if we are given some desired targets also calculate the loss
                loss = None
                if targets is not None:
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1)
                    )

                return logits, loss

            def foo(self, memo=None, prefix="", remove_duplicate=False):
                for mn, m in self.named_modules(
                    memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
                ):
                    for pn, p in self.named_parameters():
                        fpn = f"{mn}.{pn}" if mn else pn
                        self.names.append(fpn)

        # Test plain recurse
        model_a = FakeGPT()
        model_a.foo()
        a_names = model_a.names

        model_b = FakeGPT()
        opt_model_b = torch.compile(model_b, backend="eager", fullgraph=True)
        opt_model_b.foo()

        self.assertEqual(a_names, model_b.names)

        # Test with prefix
        model_a = FakeGPT()
        model_a.foo(prefix="abc")
        a_names = model_a.names

        model_b = FakeGPT()
        opt_model_b = torch.compile(model_b, backend="eager", fullgraph=True)
        opt_model_b.foo(prefix="abc")

        self.assertEqual(a_names, model_b.names)

    def test_object_classmethod(self):
        class C:
            @classmethod
            def fn(cls, x):
                return x + x

        @torch.compile(backend="eager", fullgraph=True)
        def f():
            return C().fn(torch.ones(2, 3))

        self.assertTrue(torch.allclose(f(), torch.tensor([2.0])))

    def test_object_staticmethod(self):
        class C:
            @staticmethod
            def fn(x):
                return x + x

        @torch.compile(backend="eager", fullgraph=True)
        def f():
            return C().fn(torch.ones(2, 3))

        self.assertTrue(torch.allclose(f(), torch.tensor([2.0])))

    def test_user_function_variable_supports_type_abcmeta_argument(self):
        class Foo(metaclass=abc.ABCMeta):
            @abc.abstractclassmethod
            def read(self):  # noqa: B027
                pass

        class Bar(Foo):
            def read(self):
                return "Hello World!"

        class Baz:
            pass

        def gn(x, tys=(Bar, Baz)):
            if Bar in tys:
                return x - 1
            else:
                return x + 1

        def fn(x):
            return gn(x)

        x = torch.randn(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_user_function_variable_supports_function_argument(self):
        # Test user defined function default arguments can be:
        # 1, user defined functions (e.g, add1)
        # 2, torch functions (e.g, torch.sin)
        # 3, python builtin functions (e.g, operator.neg)
        def add1(x):
            return x + 1

        def gn(x, f1=add1, f2=torch.sin, f3=operator.neg):
            return f3(f2(f1(x)))

        def fn(x):
            return gn(x)

        x = torch.randn(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_typing_variable_isinstance(self):
        def fn(x, m):
            if isinstance(m, typing.Mapping):
                return x + 1
            else:
                return x - 1

        x = torch.randn(2, 3)
        m = {"x": torch.randn(3)}
        ref = fn(x, m)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, m)
        self.assertTrue(torch.allclose(ref, res))

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_nn_sequential_invocation(self):
        with freeze_rng_state():

            class TestModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linears = torch.nn.Sequential(
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                    )

                def forward(self, x):
                    all_but_last = self.linears[:-1]
                    return all_but_last(x)

            m = TestModel()
            x = torch.rand((2, 2))
            real = m(x)
            graph, _ = torch._dynamo.export(m)(x)
            dynamo_result = graph(x)
            self.assertTrue(same(real, dynamo_result))

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_nn_sequential_invocation_reposition_indices(self):
        with freeze_rng_state():

            class TestModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linears = torch.nn.Sequential(
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                    )

                def forward(self, x):
                    all_but_last = self.linears[1:3]
                    return all_but_last(x)

            m = TestModel()
            x = torch.rand((2, 2))
            real = m(x)
            graph, _ = torch._dynamo.export(m)(x)
            dynamo_result = graph(x)
            self.assertTrue(same(real, dynamo_result))

    def test_if_cond_nn_mod1(self):
        class MockModule(torch.nn.Module):
            def __init__(self, output_relu=True):
                super().__init__()
                self.relu = torch.nn.ReLU() if output_relu else None

            def forward(self, x):
                x = torch.sin(x)
                if self.relu:
                    x = self.relu(x)
                return x

        model = MockModule()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)

        x = torch.rand(4)
        ref = model(x)
        res = opt_model(x)
        self.assertTrue(same(ref, res))

        model = MockModule(output_relu=False)
        opt_model = torch.compile(model, backend="eager", fullgraph=True)

        x = torch.rand(4)
        ref = model(x)
        res = opt_model(x)
        self.assertTrue(same(ref, res))

    def test_if_cond_nn_mod2(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.Sequential()

            def forward(self, x):
                if self.layer:
                    return x + 1
                else:
                    return x - 1

        model = MockModule()
        x = torch.rand(4)
        ref = model(x)
        opt_model = torch.compile(backend="eager")(model)
        res = opt_model(x)
        self.assertTrue(same(ref, res))

    def test_if_cond_nn_mod3(self):
        def fn(x):
            if torch.nn.ModuleList():
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        ref = fn(x)
        opt_fn = torch.compile(backend="eager")(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_if_cond_user_defined_object(self):
        # obj.__bool__ is not existed
        class A:  # noqa: B903
            def __init__(self, x):
                self.x = x

        # obj.__bool__ is function and returns bool type
        class B:
            def __init__(self, x):
                self.x = x

            def __bool__(self):
                return self.x > 0

        # obj.__bool__ is non-function
        class C:
            def __init__(self, x):
                self.x = x
                self.__bool__ = False

        def fn(x, obj):
            if not obj:
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        obj1 = A(0.5)
        obj2 = B(0.5)
        obj3 = B(-0.5)
        obj4 = C(0.5)
        for obj in [obj1, obj2, obj3, obj4, obj3, obj2]:
            ref = fn(x, obj)
            res = opt_fn(x, obj)
            self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 4)

    def test_if_cond_user_defined_object2(self):
        # obj.__bool__ is function and returns non-bool type
        class MyObj:
            def __init__(self, x):
                self.x = x

            def __bool__(self):
                self.x = 1.2
                return self.x

        def fn(a, obj):
            if not obj:
                return a + obj.x
            else:
                return a - obj.x

        x = torch.rand(4)
        obj = MyObj(0.5)
        opt_fn = torch.compile(fn, backend="eager")
        try:
            opt_fn(x, obj)
            self.assertFalse(True)
        except TypeError as e:
            self.assertIn("__bool__ should return bool, returned float", str(e))

    def test_if_cond_user_defined_object3(self):
        # obj.__bool__ is not existed, but obj.__len__ exists
        class A:  # noqa: B903
            def __init__(self, x):
                self.x = x

            def __len__(self):
                return len(self.x)

        # obj.__bool__ takes precedence over obj.__len__
        class B:
            def __init__(self, x):
                self.x = x

            def __bool__(self):
                return False

            def __len__(self):
                return len(self.x)

        def fn(x, obj):
            if not obj:
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        obj1 = A([1, 2, 3])
        obj2 = A([])
        obj3 = B([1, 2, 3])
        obj4 = B([])
        for obj in [obj1, obj2, obj3, obj4]:
            ref = fn(x, obj)
            res = opt_fn(x, obj)
            self.assertTrue(same(ref, res))

    def test_class_has_instancecheck_method(self):
        class A:
            pass

        class ExampleMeta(type):
            def __instancecheck__(cls, instance):
                return True

        class B(metaclass=ExampleMeta):
            pass

        def fn(x, obj):
            if isinstance(obj, B):
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        obj = A()
        ref = fn(x, obj)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, obj)
        self.assertTrue(same(ref, res))

    def test_custom_instancecheck_does_not_cause_extra_init(self):
        # When __new__ returns an object whose type is not a subclass of cls,
        # CPython's type.__call__ skips __init__. The polyfill
        # instantiate_user_defined_class_object must match this behavior even
        # when the metaclass defines a custom __instancecheck__ that would
        # return True for isinstance().
        class Meta(type):
            def __instancecheck__(cls, instance):
                return isinstance(instance, Base) and instance.tag == cls._tag

        class Base:
            def __init__(self, tag="default"):
                self.tag = tag

        class Child(Base, metaclass=Meta):
            _tag = "child"

            def __new__(cls):
                # Returns a Base (not a Child), like ByteStorage.__new__
                return Base(tag="child")

        def fn():
            obj = Child()
            return obj.tag

        ref = fn()
        self.assertEqual(ref, "child")

        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn()
        self.assertEqual(res, "child")

    def test_custom_instancecheck_init_not_called(self):
        class AlwaysTrueMeta(type):
            def __instancecheck__(cls, instance):
                return True

        class Child(metaclass=AlwaysTrueMeta):
            def __new__(cls):
                return object()

            def __init__(self):
                raise AssertionError("should NOT be called")

        def fn():
            return Child()

        ref = fn()
        self.assertIsInstance(ref, object)

        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn()
        self.assertIsInstance(res, object)

    def test_call_parent_non_class_methods_from_child(self):
        class A:
            a = 4

            def add(self, x):
                return x + 10

            def mul(self, x):
                return x * 0.1

        class B(A):
            coeff = 4

            def add(self, x):
                return x + 20

            @classmethod
            def cube(cls, x):
                return cls.coeff * x * x * x

            def mul(self, x):
                return super().mul(x) * x * 0.2

        class C(B):
            def add(self, x):
                b = super().cube(x)
                c = A.add(self, x)
                d = B.mul(self, x)
                e = super(B, self).add(x)
                f = super().a * x
                return b + c + d + e + f

        x = torch.rand(4)
        fn = C().add
        ref = fn(x)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnt.frame_count, 1)

        # Check recompilation
        A.a = 5
        ref = fn(x)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        # Ensure that super guard checks are working as expected
        res = opt_fn(x)
        self.assertEqual(cnt.frame_count, 2)

    def test_builder_for_class_with_metaclass(self):
        class ExampleMeta(type):
            pass

        class MyClass(metaclass=ExampleMeta):
            pass

        def fn(x, y):
            if isinstance(y, MyClass):
                return x + 1
            else:
                return x - 1

        x = torch.rand([4, 4])
        y = MyClass()
        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_assigning_function_to_object_attribute(self):
        # user-defined functions which are object's attributes are not converted to bound methods
        def my_add(*args):
            a, b = args
            return a + b

        class MyClass:
            def __init__(self, func):
                self.add = func

        obj = MyClass(my_add)

        def fn(x):
            return obj.add(x, 2)

        x = torch.rand(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(backend="eager")(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_assigning_function_to_class_attribute(self):
        # user-defined functions which are class's attributes are converted to bound methods
        def my_add(*args):
            obj, a, b = args
            return obj.x + a + b

        class MyClass:
            add = my_add

            def __init__(self, x):
                self.x = x

        obj = MyClass(0.5)

        def fn(x):
            return obj.add(x, 2)

        x = torch.rand(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(backend="eager")(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_fn_hasattr__name__1(self):
        def fn():
            foo = lambda x: x + 1
            return hasattr(foo, "__name__")

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertEqual(fn_out, compiled_out)
        self.assertTrue(fn_out)

    def test_fn_hasattr__name__2(self):
        def bar(x):
            return torch.sin(x)

        def fn():
            return hasattr(bar, "__name__")

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertEqual(fn_out, compiled_out)
        self.assertTrue(fn_out)

    def test_fn_hasattr__name__3(self):
        def bar(x, y):
            return torch.sin(x) + torch.cos(y)

        baz = functools.partial(bar, y=4)

        def fn():
            return hasattr(baz, "__name__")

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertEqual(fn_out, compiled_out)
        self.assertFalse(fn_out)

    def test_constant_hasattr_returns_bool(self):
        """Test that hasattr on constant values properly returns boolean ConstantVariable."""

        # Test various constant types
        def fn():
            # String constant
            s = "hello"
            result1 = hasattr(s, "upper")  # True
            result2 = hasattr(s, "nonexistent")  # False

            # Integer constant
            i = 42
            result3 = hasattr(i, "bit_length")  # True
            result4 = hasattr(i, "fake_method")  # False

            # Float constant
            f = 3.14
            result5 = hasattr(f, "is_integer")  # True
            result6 = hasattr(f, "missing_attr")  # False

            # Use all results to ensure they're compiled
            return (result1, result2, result3, result4, result5, result6)

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertEqual(fn_out, compiled_out)
        self.assertEqual(fn_out, (True, False, True, False, True, False))

    def test_class_hasattr_sourceless_descriptor(self):
        """Test that hasattr on sourceless UserDefinedClassVariable does not graph break."""

        class FlagDescriptor:
            def __get__(self, instance, owner):
                if hasattr(owner, "flag"):
                    return 1
                return 0

        class WithFlag:
            flag = True
            prop = FlagDescriptor()

        class WithoutFlag:
            prop = FlagDescriptor()

        def fn(x, obj):
            return x + obj.prop

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        x = torch.randn(3)
        self.assertEqual(fn(x, WithFlag()), compiled_fn(x, WithFlag()))
        self.assertEqual(fn(x, WithoutFlag()), compiled_fn(x, WithoutFlag()))

    def test_torch_objects_as_keys(self):
        remap = {torch.float16: torch.float32}

        def fn():
            return torch.randn(3, dtype=remap[torch.float16])

        opt = torch.compile(fn, backend="eager")
        opt()

    def test_tracing_nested_py_tree_mixed_all(self):
        def fn(xs):
            flat_xs, spec = python_pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            return python_pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsa = (xs, xs)
        xsb = {"aa": xsa, "ab": xs}
        xsl = {
            "a": xs,
            "b": xsa,
            "c": xsb,
        }

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 18)

    def test_reconstruct_set_across_graph_break(self):
        def foo(x, y):
            setty = set()
            for t in x:
                setty.add(t)
            print("Break!")
            return y * len(setty)

        x = torch.randn(10, 10)
        y = torch.randn(2, 2)

        counter = CompileCounter()
        foo = torch.compile(foo, backend=counter)
        result = foo([x, x, x, x, y], y)

    def test_sourceless_namedtuple(self):
        from collections import namedtuple

        CustomDtype = namedtuple("CustomDtype", ["dtype", "higher_dtype"])

        class CustomTensor(torch.Tensor):
            _data: torch.Tensor
            custom_dtype: CustomDtype
            __torch_function__ = torch._C._disabled_torch_function_impl
            __slots__ = [
                "_data",
                "custom_dtype",
            ]

            def __new__(
                cls,
                data: torch.Tensor,
                custom_dtype: CustomDtype,
            ):
                self = torch.Tensor._make_wrapper_subclass(
                    cls,
                    data.size(),
                    strides=data.stride(),
                    storage_offset=data.storage_offset(),
                    dtype=custom_dtype.dtype,
                    layout=data.layout,
                    requires_grad=data.requires_grad,
                    device=data.device,
                )
                self._data = data
                self.custom_dtype = custom_dtype
                return self

            def __tensor_flatten__(self):
                meta = {
                    "custom_dtype": self.custom_dtype,
                }
                return ["_data"], meta

            @staticmethod
            def __tensor_unflatten__(
                inner_tensors: dict, metadata, outer_size, outer_stride
            ):
                return CustomTensor(
                    inner_tensors["_data"],
                    metadata["custom_dtype"],
                )

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs={}):
                return func(*args, **kwargs)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = CustomTensor(x, CustomDtype(torch.float32, torch.bfloat16))
            return y, y.custom_dtype

        fn(torch.ones(2, 2, device="cpu"))

    def test_derpy_nn_module_usage(self):
        def ff1(x):
            self = mod1
            return torch.sigmoid(self.mod2(x) + self.param1)

        def ff2(x):
            self = mod2
            return torch.cos(torch.sin(x) * self.param2 + 10)

        mod1 = torch.nn.Module()
        mod2 = torch.nn.Module()
        mod1.register_module("mod2", mod2)
        mod1.register_parameter("param1", torch.nn.Parameter(torch.randn(10)))
        mod1.forward = ff1
        mod2.register_parameter("param2", torch.nn.Parameter(torch.randn(10)))
        mod2.forward = ff2
        mod1.eval()

        x = torch.randn(10)
        expected = mod1(x)
        counter = CompileCounter()
        actual = torch.compile(mod1, backend=counter, fullgraph=True)(x)
        self.assertEqual(actual, expected)
        self.assertEqual(counter.op_count, 6)

    def test_flat_name_to_original_fqn(self):
        class FooBarModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter("0", torch.nn.Parameter(torch.randn(3, 4)))
                self.test_buf = torch.nn.Buffer(torch.randn(3, 4))
                self.register_parameter(
                    "test_param", torch.nn.Parameter(torch.randn(3, 4))
                )

            def forward(self, x):
                return ((x + self.test_buf) * getattr(self, "0")) / self.test_param

        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo_bar = FooBarModule()
                self.register_parameter(
                    "test_param", torch.nn.Parameter(torch.randn(3, 4))
                )
                self.test_buf = torch.nn.Buffer(torch.randn(3, 4))

            def forward(self, x):
                return (self.foo_bar(x) + self.test_param) * self.test_buf

        gm, _ = torch._dynamo.export(TestModule(), torch.randn(3, 4))
        self.assertIn("dynamo_flat_name_to_original_fqn", gm.meta)
        expected_fqn = {
            "L__self___test_param": "test_param",
            "L__self___test_buf": "test_buf",
            "L__self___foo_bar_0": "foo_bar.0",
            "L__self___foo_bar_test_param": "foo_bar.test_param",
            "L__self___foo_bar_test_buf": "foo_bar.test_buf",
        }
        self.assertEqual(expected_fqn, gm.meta["dynamo_flat_name_to_original_fqn"])

    def test_proxy_frozen_dataclass(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor

        @allow_in_graph
        def inner_fn(dc):
            return dc.x + dc.y

        def fn(x, y):
            dc = TestDataClass(x, y)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_reconstruct_frozen_dataclass(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor

        def fn(x, y):
            dc = TestDataClass(x, y)
            torch._dynamo.graph_break()
            return dc.x + dc.y

        fn_opt = torch.compile(backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

    def test_frozen_dataclass_default_value(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor
            z: int = dataclasses.field(default=5)
            a: int = 6

        @allow_in_graph
        def inner_fn(dc):
            return dc.x + dc.y + dc.z + dc.a

        def fn(x, y):
            dc = TestDataClass(x, y)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_frozen_dataclass_default_factory(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor
            z: int = dataclasses.field(default_factory=list)
            a: int = dataclasses.field(default_factory=lambda: [5])

        @allow_in_graph
        def inner_fn(dc):
            return dc.x + dc.y + dc.a[0]

        def fn(x, y):
            dc = TestDataClass(x, y)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_frozen_dataclass_kw_only(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor
            z: int = dataclasses.field(kw_only=True)
            a: int = dataclasses.field(kw_only=True)

        @allow_in_graph
        def inner_fn(dc):
            return dc.x + dc.y + dc.a + dc.z

        def fn(x, y):
            dc = TestDataClass(x, y, z=5, a=2)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_frozen_dataclass_attr_access(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor
            z: int
            a: int

        def inner_fn(dc):
            return dc.x + dc.y + dc.a + dc.z

        def fn(x, y):
            dc = TestDataClass(x, y, z=5, a=2)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_frozen_dataclass_hashable(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: float
            y: float
            z: int
            a: int

        def inner_fn(dc, x, y):
            d = {}
            d[dc] = 2
            return dc.x + dc.y + d[dc] + x + y

        def fn(x, y):
            dc = TestDataClass(x=3.2, y=2.5, z=5, a=2)
            return inner_fn(dc, x, y)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)
        self.assertEqual(actual, expected)

    def test_frozen_dataclass_in_compile(self):
        from torch.utils._pytree import MappingKey, SequenceKey

        def fn(x):
            path = (MappingKey("a"), SequenceKey(0))
            msg = f"path={path}"
            return x * 2, msg

        x = torch.randn(4, 4)
        eager_result = fn(x)
        compiled_result = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(eager_result[0], compiled_result[0])
        self.assertEqual(eager_result[1], compiled_result[1])

    def test_frozen_dataclass_treespec_method_and_fields(self):
        from torch.utils._pytree import tree_flatten

        def fn(x):
            d = {"a": x, "b": [x * 2, x * 3]}
            flat, spec = tree_flatten(d)
            is_leaf = spec.is_leaf()
            return sum(flat), spec.num_leaves, spec.num_nodes, is_leaf

        x = torch.randn(4)
        eager_result = fn(x)
        compiled_result = torch.compile(fn, fullgraph=True)(x)
        for i in range(4):
            self.assertEqual(eager_result[i], compiled_result[i])

    def _test_compile_model_free(self, model_inp_ctr, weakref_watch):
        """
        Args:
        model_inp_ctr
            - constructor that returns a new model and inputs to that model
        weakref_watch
            - function that returns a layer of the model for weakref to
              finalize on, so we can check that the layer is freed after
              the model goes out of scope
        """
        cleared = False

        def finalize():
            nonlocal cleared
            cleared = True

        def run():
            mod, inp = model_inp_ctr()
            weakref.finalize(weakref_watch(mod), finalize)
            torch.compile(mod, backend="eager")(inp)

        run()
        gc.collect()
        self.assertTrue(cleared)

    def test_custom_module_free(self):
        """Test that a model is freed when it goes out of scope"""

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super(Mod, self).__init__()
                self.fc = torch.nn.Linear(100, 100)

            def forward(self, out):
                return self.fc(out)

        self._test_compile_model_free(
            lambda: (Mod(), torch.randn(100, 100)),
            lambda mod: mod.fc,
        )

    def test_sequential_module_free(self):
        self._test_compile_model_free(
            lambda: (
                torch.nn.Sequential(
                    torch.nn.Linear(100, 100),
                    torch.nn.ReLU(),
                ),
                torch.randn(100, 100),
            ),
            lambda mod: mod[0],
        )

    def test_linear_module_free(self):
        self._test_compile_model_free(
            lambda: (torch.nn.Linear(100, 100), torch.randn(100, 100)),
            lambda mod: mod,
        )

    def test_outside_linear_module_free(self):
        # Compared to test_linear_module_free, the linear
        # layer is not the code object that is directly compiled.

        # This test does not use _test_compile_model_free because of difficulty
        # in handling variable fc.

        cleared = False

        def finalize():
            nonlocal cleared
            cleared = True

        def run():
            fc = torch.nn.Linear(100, 100)

            class Mod(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.fc_ref = fc

                def forward(self, x):
                    return self.fc_ref(x)

            mod = Mod()
            inp = torch.randn(100, 100)
            weakref.finalize(fc, finalize)
            torch.compile(mod, backend="eager")(inp)

        run()
        # del fc  # This should delete all the references
        gc.collect()
        self.assertTrue(cleared)

    def test_parameter_free(self):
        def model_inp_ctr():
            param = torch.nn.Parameter(torch.randn(100, 100))

            class Mod(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.param = param

                def forward(self, x):
                    return self.param * x[0]

            # return param to keep it alive in _test_compile_model_free
            return Mod(), (torch.randn(100, 100), param)

        self._test_compile_model_free(model_inp_ctr, lambda mod: mod.param)

    def test_super_after_graph_break(self):
        class Foo(torch.nn.Sequential):
            def __init__(self, layers):
                torch._dynamo.graph_break()
                super().__init__(*layers)

        def fn(x):
            layers = [torch.nn.Linear(3, 3) for _ in range(3)]
            mod = Foo(layers)
            return mod(x)

        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.randn(3, 3))

    def test_inspect_signature_bind(self):
        import inspect

        def inner(a, b, *ar, c=10, d=11, **kw):
            pass

        def fn(x, apply_defaults):
            sig = inspect.signature(inner)
            bound = sig.bind(1, 2, 3, d=12, e=15)
            bound.arguments["d"] = 13
            if apply_defaults:
                bound.apply_defaults()
            return (
                sig,
                bound.signature,
                bound,
                bound.arguments,
                bound.args,
                bound.kwargs,
                x + 1,
            )

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        for apply_defaults in (True, False):
            _, _, bound0, arguments0, args0, kwargs0, _ = fn(
                torch.ones(3, 3), apply_defaults
            )
            _, _, bound1, arguments1, args1, kwargs1, _ = opt_fn(
                torch.ones(3, 3), apply_defaults
            )

            self.assertEqual(bound0, bound1)
            self.assertEqual(arguments0, arguments1)
            self.assertEqual(args0, args1)
            self.assertEqual(kwargs0, kwargs1)
            self.assertTrue(args1)
            self.assertTrue(kwargs1)

    def test_inspect_signature_bind_non_user_function(self):
        import inspect

        class Foo:
            def __init__(self, a, b, *ar, c=10, d=11, **kw):
                pass

        def fn(x):
            sig = inspect.signature(Foo)
            bound = sig.bind(1, 2, 3, d=12, e=15)
            return bound, x + 1

        opt_fn = torch.compile(fn, backend="eager")
        bound0, _ = fn(torch.ones(3, 3))
        bound1, _ = opt_fn(torch.ones(3, 3))

        self.assertEqual(bound0, bound1)

        import traceback

        # choose a function that is skipped but has defaults
        self.assertTrue(hasattr(traceback.print_exc, "__kwdefaults__"))
        self.assertIs(
            torch._dynamo.trace_rules.lookup(traceback.print_exc),
            torch._dynamo.variables.UserFunctionVariable,
        )

        def gn(x):
            sig = inspect.signature(traceback.print_exc)
            bound = sig.bind()
            return bound, x + 1

        opt_gn = torch.compile(gn, backend="eager", fullgraph=True)
        bound0, _ = gn(torch.ones(3, 3))
        bound1, _ = opt_gn(torch.ones(3, 3))

        self.assertEqual(bound0, bound1)

    def test_sourceless_mapping_proxy(self):
        # Test that Dynamo can handle a sourceless MappingProxyType.
        # This occurs when type.__dict__['__dict__'].__get__ is called
        # and returns a mappingproxy that was created during tracing.
        _get_dunder_dict = type.__dict__["__dict__"].__get__

        class MyClass:
            a = 1
            b = 2

        def fn(x):
            d = _get_dunder_dict(MyClass)
            return x + len(d)

        t = torch.randn(3)
        ref = fn(t)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(t)
        self.assertEqual(ref, res)

    def test_sourceless_inspect_parameter(self):
        import inspect

        class MyClass:
            param = inspect.Parameter(
                "x", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=42
            )

        _get_dunder_dict = type.__dict__["__dict__"].__get__

        def fn(x):
            d = _get_dunder_dict(MyClass)
            p = d["param"]
            return x + p.default

        t = torch.randn(3)
        ref = fn(t)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(t)
        self.assertEqual(ref, res)

    def test_inspect_signature_parameters(self):
        import inspect

        def fn(x, gn):
            d = inspect.signature(gn).parameters
            if d["a"].default is inspect.Parameter.empty:
                return torch.sin(x + 1)
            else:
                return torch.cos(x + 1)

        def gn(a: torch.Tensor, b: int) -> torch.Tensor:
            return a + b

        x = torch.randn(2, 3)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        self.assertEqual(fn(x, gn), opt_fn(x, gn))

    def test_inspect_signature_caching(self):
        """Test that inspect.signature results are cached for repeated calls."""
        import inspect
        from unittest.mock import patch

        def target_func(a, b, c=3):
            return a + b + c

        def other_func(x, y):
            return x * y

        def fn():
            results = []
            for _ in range(10):
                sig1 = inspect.signature(target_func)
                sig2 = inspect.signature(other_func)
                results.append(len(sig1.parameters))
                results.append(len(sig2.parameters))
            return sum(results)

        from torch._dynamo.output_graph import OutputGraph

        original_cleanup = OutputGraph.cleanup
        unique_calls = 0

        def capturing_cleanup(self):
            nonlocal unique_calls
            unique_calls = len(self.signature_cache)
            return original_cleanup(self)

        with patch.object(OutputGraph, "cleanup", capturing_cleanup):
            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            result = compiled_fn()

        # 10 iterations * (3 params from target_func + 2 params from other_func) = 50
        self.assertEqual(result, 50)
        # signature_cache should have exactly 2 entries (one per unique function)
        self.assertEqual(unique_calls, 2)

    def test_inspect_signature_caching_methods(self):
        """Test that inspect.signature caching works for methods."""
        import inspect
        from unittest.mock import patch

        class MyClass:
            def method_a(self, x, y, z):
                return x + y + z

            def method_b(self, a):
                return a * 2

        obj = MyClass()

        def fn():
            results = []
            for _ in range(10):
                sig1 = inspect.signature(obj.method_a)
                sig2 = inspect.signature(obj.method_b)
                # Note: bound methods don't include 'self' in signature
                results.append(len(sig1.parameters))
                results.append(len(sig2.parameters))
            return sum(results)

        from torch._dynamo.output_graph import OutputGraph

        original_cleanup = OutputGraph.cleanup
        unique_calls = 0

        def capturing_cleanup(self):
            nonlocal unique_calls
            unique_calls = len(self.signature_cache)
            return original_cleanup(self)

        with patch.object(OutputGraph, "cleanup", capturing_cleanup):
            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            result = compiled_fn()

        # 10 iterations * (3 params from method_a + 1 param from method_b) = 40
        self.assertEqual(result, 40)
        # signature_cache should have exactly 2 entries (one per unique method)
        self.assertEqual(unique_calls, 2)

    def test_inspect_variable_redirect(self):
        """Test that InspectVariable is used and redirects property accesses."""
        import inspect
        from unittest.mock import patch

        from torch._dynamo.variables.user_defined import InspectVariable

        redirected_attrs = []
        original_var_getattr = InspectVariable.var_getattr

        def tracking_var_getattr(self, tx, name):
            redirects = self._PROPERTY_REDIRECTS.get(type(self.value), {})
            if name in redirects:
                redirected_attrs.append(name)
            return original_var_getattr(self, tx, name)

        def fn(x, gn):
            sig = inspect.signature(gn)
            params = sig.parameters
            param = params["a"]
            return x + param.kind + len(param.name)

        def gn(a: torch.Tensor, b: int) -> torch.Tensor:
            return a + b

        x = torch.randn(2, 3)
        with patch.object(InspectVariable, "var_getattr", tracking_var_getattr):
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            result = opt_fn(x, gn)

        self.assertEqual(result, fn(x, gn))
        self.assertIn("parameters", redirected_attrs)
        self.assertIn("kind", redirected_attrs)
        self.assertIn("name", redirected_attrs)

    def test_custom_dict(self):
        class MyDict(dict):
            pass

        d = {
            "foo": 1,
            "bar": 2,
        }

        d = MyDict(d)

        @torch.compile(backend="eager")
        def fn(x, d):
            return x * d["foo"] * d["bar"]

        fn(torch.randn(4), d)
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(4), d)

    def test_descriptor(self):
        class lazy_property:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __get__(self, instance, obj_type=None):
                value = self.wrapped(instance)
                setattr(instance, self.wrapped.__name__, value)
                return value

        class UserDefined:
            def __init__(self) -> None:
                self.a = 3

            @lazy_property
            def length(self):
                return 3

            def run(self, x):
                return x * self.length

        obj = UserDefined()

        def fn(x):
            return obj.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        # Opt_fn is deliberately called first to trigger the __get__ function.
        # Otherwise, the setattr removes the lazy property.
        ref = opt_fn(x)
        res = fn(x)
        self.assertEqual(ref, res)
        ref = opt_fn(x)
        res = fn(x)
        self.assertEqual(ref, res)

    def test_descriptor_side_effect(self):
        # This pattern (readonly descriptor but writable value in `__dict__`) is
        # from scipy `_make_tuple_bunch`:
        # https://github.com/scipy/scipy/blob/maintenance/1.9.x/scipy/_lib/_bunch.py#L32-L226
        def fget(obj):
            return obj.__dict__["field"]

        class MyClass:
            def __init__(self, n):
                self.__dict__["field"] = n

            field = property(fget)

        def fn(x):
            obj = MyClass(42)
            return x + obj.field, obj

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref_t, ref_obj = fn(x)
        res_t, res_obj = opt_fn(x)
        self.assertEqual(ref_t, res_t)
        self.assertEqual(ref_obj.field, res_obj.field)

    def test_instance_dict_priority_over_non_data_descriptor(self):
        # CPython: instance dict values take priority over non-data
        # descriptors (those with only __get__, no __set__/__delete__).
        class Desc:
            def __init__(self, val):
                self.val = val

            def __get__(self, obj, cls):
                return self.val * 100

        class Foo:
            x = Desc(7)

        foo = Foo()
        # Instance dict value should shadow the non-data descriptor.
        foo.__dict__["x"] = 10

        def fn(t):
            return t + foo.x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.randn(4)
        ref = fn(t)
        res = opt_fn(t)
        self.assertEqual(ref, res)
        # CPython: instance dict wins → foo.x is 10, not 700
        self.assertEqual(ref, t + 10)

    def test_user_defined_data_descriptor(self):
        # A user-defined data descriptor (has __get__ + __set__) on the type
        # should be invoked even when the same name exists in the instance dict.
        class ValidatedAttr:
            def __set_name__(self, owner, name):
                self.storage_name = "_" + name

            def __get__(self, obj, cls):
                if obj is None:
                    return self
                return getattr(obj, self.storage_name)

            def __set__(self, obj, value):
                setattr(obj, self.storage_name, value)

        class Foo:
            x = ValidatedAttr()

            def __init__(self, x):
                self.x = x

        foo = Foo(10)

        def fn(t):
            return t + foo.x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.randn(4)
        ref = fn(t)
        res = opt_fn(t)
        self.assertEqual(ref, res)
        self.assertEqual(ref, t + 10)

    def test_property_setter(self):
        class Box:
            def __init__(self, value):
                self._value = torch.tensor([value], dtype=torch.float32)

            @property
            def value(self):
                return self._value + 1

            @value.setter
            def value(self, new_value):
                self._value = new_value * 2

        def fn(b):
            b.value = 5
            return b.value

        b = Box(0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(Box(0))
        res = opt_fn(b)
        self.assertEqual(ref, res)

    def test_property_setter_in_init(self):
        class Clamped:
            def __init__(self, val):
                self.val = val

            @property
            def val(self):
                return self._val

            @val.setter
            def val(self, v):
                self._val = torch.clamp(v, 0, 100)

        def fn(x):
            obj = Clamped(x)
            return obj.val

        x = torch.tensor([200.0])
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_frozen_dict(self):
        # A pattern from StableDiffusion
        class FrozenDict(collections.OrderedDict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                for key, value in self.items():
                    setattr(self, key, value)

                self.__frozen = True

            def __delitem__(self, *args, **kwargs):
                raise Exception(
                    f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
                )

            def setdefault(self, *args, **kwargs):
                raise Exception(
                    f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
                )

            def pop(self, *args, **kwargs):
                raise Exception(
                    f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
                )

            def update(self, *args, **kwargs):
                raise Exception(
                    f"You cannot use ``update`` on a {self.__class__.__name__} instance."
                )

            def __setattr__(self, name, value):
                if hasattr(self, "__frozen") and self.__frozen:
                    raise Exception(
                        f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance."
                    )
                super().__setattr__(name, value)

            def __setitem__(self, name, value):
                if hasattr(self, "__frozen") and self.__frozen:
                    raise Exception(
                        f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance."
                    )
                super().__setitem__(name, value)

        d = {"a": 1}
        frozen_d = FrozenDict(d)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            dict(frozen_d).items()
            return torch.sin(x)

        fn(torch.randn(4))

    def test_namedtuple_class(self):
        import collections

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            updated_x = []
            for v in x:
                updated_x.append(v + 1)
            return x.__class__(*updated_x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        d1 = torch.zeros(2, 2)
        d2 = torch.ones(2, 2)
        point = collections.namedtuple("Point", ["x", "y"])
        p = point(d1, d2)

        r = opt_fn(p)
        self.assertEqual(r.__class__, point)
        self.assertEqual(r.x, torch.ones(2, 2))
        self.assertEqual(r.y, torch.ones(2, 2) + 1)
        self.assertEqual(cnts.frame_count, 1)

    def test_getattrvariable_as_python_constant(self):
        from torch._dynamo.variables.misc import GetAttrVariable

        @torch.compile(backend="eager")
        def fn(x, rand1):
            random.Random().setstate(rand1.getstate())
            return x + rand1.random()

        def get_rng():
            rand1 = random.Random(1)
            orig_random = rand1.random
            rand1.random = lambda: orig_random()
            return rand1

        x = torch.randn(3, 3)
        expected = fn.__wrapped__(x, get_rng())

        with patch.object(GetAttrVariable, "as_python_constant", autospec=True) as po:
            actual = fn(x, get_rng())

        self.assertEqual(expected, actual)
        self.assertGreater(po.call_count, 0)

    def test_dataclass(self):
        @dataclasses.dataclass(frozen=True)
        class Foo:
            x: int

        @torch.compile(backend="eager", fullgraph=True)
        def run(x, foo0):
            if dataclasses.is_dataclass(foo0):
                foo1 = dataclasses.replace(foo0, **{"x": 1})
                return x + 1, foo1
            return x + 2, foo0

        res, foo = run(torch.zeros(1), Foo(0))
        self.assertTrue(res, torch.ones(1))
        self.assertEqual(foo.x, 1)

    def test_overridden_getattribute(self):
        class Bar:
            def __init__(self, v):
                self.v = v

        class Foo:
            attribute_map = {}

            def __init__(self):
                self.attribute_map = {
                    "a_premap": "a",
                }
                # `bar` attribute requires propagating sources correctly through
                # object.__getattribute__
                self.bar = Bar(5)

            def __setattr__(self, key, value):
                if key in super().__getattribute__("attribute_map"):
                    key = super().__getattribute__("attribute_map")[key]
                super().__setattr__(key, value)

            def __getattribute__(self, key):
                if key == "sentinel":
                    raise AttributeError()
                if key != "attribute_map" and key in super().__getattribute__(
                    "attribute_map"
                ):
                    key = super().__getattribute__("attribute_map")[key]
                return super().__getattribute__(key)

            def __getattr__(self, key):
                if key == "sentinel":
                    return 5
                raise AttributeError()

        def get_foo():
            f = Foo()
            f.a_premap = 2
            f.b = 3
            return f

        def fn(x, f):
            return x * f.a_premap * f.a * f.b * f.sentinel * f.bar.v

        x = torch.randn(4)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, get_foo()), opt_fn(x, get_foo()))

    def test_dunder_weakref(self):
        class Foo:
            pass

        def fn(x):
            foo = Foo()
            # tests isgetsetdescriptor
            if foo.__weakref__:
                return torch.cos(x)
            return torch.sin(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_compiled_class_graph_break(self):
        counter = CompileCounter()

        @torch.compile(backend=counter, fullgraph=False)
        def f(x):
            x += 1

            class C:
                pass

            return x.sin()

        x = torch.randn(3)
        f(x)
        self.assertEqual(counter.frame_count, 2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
