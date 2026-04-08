# Owner(s): ["module: dynamo"]
# flake8: noqa: E721, F403, F405, F823, F841
# ruff: noqa: E721,F403,F405,F823,F841,UP004,UP008,UP028
try:
    from ._test_misc_common import *
except ImportError:
    from _test_misc_common import *


class MiscTestsPart3:
    def test_out_variants_with_resizing_on_graph_inputs(self):
        def fn(x, y):
            return torch.cosh(x, out=y) + 1

        x = torch.rand(2, 3)
        y = torch.rand(4)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(fn(x, y), opt_fn(x.clone(), y.clone())))
        self.assertEqual(cnts.frame_count, 1)

    def test_out_variants_with_resizing_on_graph_inputs_with_dynamic(self):
        # https://github.com/pytorch/pytorch/issues/120482
        class CustomModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, inputs):
                return torch.outer(**inputs)

        compile_fn = torch.compile(CustomModel(), backend="eager", fullgraph=True)

        shapes = [(2, 1), (6, 1), (4, 1)]
        for shape in shapes:
            vec1, vec2 = shape
            input_tensor1 = torch.randn(vec1)
            input_tensor2 = torch.randn(vec2)
            out_tensor = torch.empty(shape)
            args = {"input": input_tensor1, "vec2": input_tensor2, "out": out_tensor}
            res = compile_fn(args)
            opt_res = res.clone()  # cuz this is out and we mutate it
            res = CustomModel()(args)
            self.assertEqual(res, opt_res)

    def test_out_variants_with_resizing_on_graph_inputs_with_dynamic1(self):
        mv_op = torch.mv

        def mv_out_op(a, b, c):
            torch.mv(b, c, out=a)
            return a

        def fn(op, *args):
            return op(*args)

        opt_fn = torch.compile(fn, backend="eager")

        ref = fn(mv_op, torch.ones(3, 3), torch.ones(3))
        res = opt_fn(mv_op, torch.ones(3, 3), torch.ones(3))
        self.assertEqual(ref, res)

        ref = fn(mv_out_op, torch.empty(0), torch.ones(3, 3), torch.ones(3))
        res = opt_fn(mv_out_op, torch.empty(0), torch.ones(3, 3), torch.ones(3))
        self.assertEqual(ref, res)

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

    def test_nested_wraps(self):
        def foo(x, y):
            def add(x, y):
                return x + y

            @functools.wraps(add)
            def wrapped_call(x, y):
                return add(x, y)

            return wrapped_call(x, y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        o = torch.compile(foo, fullgraph=True, backend="eager")(x, y)
        self.assertEqual(o, x + y)

        def foo(x, y):
            def nested_call(x, y):
                def mul(x, y):
                    return x * y

                @functools.wraps(mul)
                def double_nested_call(x, y):
                    return mul(x, y)

                return double_nested_call(x, y)

            return nested_call(x, y)

        o = torch.compile(foo, fullgraph=True, backend="eager")(x, y)
        self.assertEqual(o, x * y)

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

    def test_id_of_container_as_dict_key(self):
        MY_DICT = {"a": 1, "b": 2}

        def fn(x):
            memo = {}
            memo[id(MY_DICT)] = True
            if id(MY_DICT) in memo:
                return x + 1.0
            return x + 2.0

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(result, correct)

    def test_id_of_list_as_dict_key(self):
        MY_LIST = [1.0, 2.0]

        def fn(x):
            memo = {}
            memo[id(MY_LIST)] = True
            if id(MY_LIST) in memo:
                return x + 1.0
            return x + 2.0

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(result, correct)

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

    def test_global_state_guard_serialization(self):
        GlobalStateGuard = torch._C._dynamo.guards.GlobalStateGuard
        guards = GlobalStateGuard()
        serialized_guards = guards.__getstate__()
        json_guards = json.loads(serialized_guards)

        samples = []
        # Test on non autocast state and autocast cache states.
        self.assertIn("autocast_state", json_guards)
        for key, value in json_guards.items():
            if type(value) is int:
                variant = value + 1
            elif type(value) is bool:
                variant = not value
            elif isinstance(value, dict) and key == "autocast_state":
                variant = value.copy()
                variant["cached_enabled"] = not variant["cached_enabled"]
                continue
            else:
                self.fail(f"Unknown global state type {key}: {value}")
            new_dict = json_guards.copy()
            new_dict[key] = variant
            samples.append(new_dict)

        for sample in samples:
            guards.__setstate__(json.dumps(sample))
            self.assertFalse(guards.check())

        guards.__setstate__(json.dumps(json_guards))
        self.assertTrue(guards.check())

        # Test on autocast states.
        def _test_autocast(dtype):
            with torch.autocast("cpu", dtype):
                guards = GlobalStateGuard()
                serialized_guards = guards.__getstate__()
                json_guards = json.loads(serialized_guards)

                for i, enabled in enumerate(json_guards["autocast_state"]["enabled"]):
                    if enabled:
                        self.assertEqual(
                            type(json_guards["autocast_state"]["dtype"][i]), int
                        )
                        json_guards["autocast_state"]["dtype"][i] += 1
                        guards.__setstate__(json.dumps(json_guards))
                        self.assertFalse(guards.check())

        _test_autocast(torch.float16)
        _test_autocast(torch.float32)
        _test_autocast(torch.float64)
        _test_autocast(torch.bfloat16)

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

    def test_setattr_mutation1(self):
        class MyObj:  # noqa: B903
            def __init__(self, a, b):
                self.a = a
                self.b = b

        def fn(obj):
            obj.c = obj.a * obj.b + 1
            obj.b = obj.a * obj.c + 2
            obj.a = obj.b * obj.c + 3
            obj.c = obj.a * obj.b + 4
            obj.b = obj.a * obj.c + 5
            obj.a = obj.b * obj.c + 6
            return obj

        x1 = torch.randn(10)
        x2 = torch.randn(10)
        obj1 = MyObj(x1, x2)
        obj2 = MyObj(x1, x2)
        fn(obj2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertIs(opt_fn(obj1), obj1)
        self.assertTrue(same(obj1.a, obj2.a))
        self.assertTrue(same(obj1.b, obj2.b))
        self.assertTrue(same(obj1.c, obj2.c))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 12)

    def test_setattr_mutation2(self):
        class MyObj:
            def __init__(self, x):
                self.a = x + 1
                self.b = x + 2

        def fn(x):
            x = x / 3.0
            obj = MyObj(x)
            obj.c = obj.a * obj.b + 1
            obj.b = obj.a * obj.c + 2
            obj.a = obj.b * obj.c + 3
            return obj

        x1 = torch.randn(10)
        obj2 = fn(x1)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        obj1 = opt_fn(x1)
        self.assertTrue(same(obj1.a, obj2.a))
        self.assertTrue(same(obj1.b, obj2.b))
        self.assertTrue(same(obj1.c, obj2.c))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 9)

    def test_setattr_mutation3(self):
        # TODO(jansel): dead code eliminate the object creation
        class MyObj:
            def __init__(self, x):
                super().__init__()
                self.a = x + 1
                self.b = x + 2

        def fn(x):
            x = x / 3.0
            obj = MyObj(x)
            obj.c = obj.a * obj.b + 1
            obj.b = obj.a * obj.c + 2
            obj.a = obj.b * obj.c + 3
            return obj.a, obj.b, obj.c

        x1 = torch.randn(10)
        obj2 = fn(x1)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        obj1 = opt_fn(x1)
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 9)

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

    def test_nested_closure(self):
        v0 = torch.randn(10)

        def fn1():
            v1 = torch.randn(10)

            def fn2(*args, **kwargs):
                assert len(args) == 1  # noqa: S101
                assert len(kwargs) == 1  # noqa: S101
                v2 = torch.randn(10) + args[0] + kwargs["b"]

                def fn3(v3=torch.randn(10)):
                    def fn4():
                        return v0 + v1 + v2 + v3 + 1

                    return fn4

                return fn3

            return fn2(1, b=2)()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch._dynamo.optimize_assert(cnts)(fn1)
        tmp1 = torch._dynamo.optimize_assert(cnts)(opt_fn1())
        tmp2 = torch._dynamo.optimize_assert(cnts)(opt_fn1())
        self.assertTrue(tmp1().shape, (10,))
        self.assertTrue(same(tmp1(), tmp1()))
        self.assertFalse(same(tmp1(), tmp2()))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 9)

    def test_nested_closure_mutation(self):
        def fn1():
            v1 = torch.randn(10)

            def fn2():
                v2 = torch.randn(10)

                def fn3():
                    nonlocal v1, v2
                    v1 += 1
                    v2 += 2
                    return v1 + v2

                return fn3

            rv = fn2()
            rv()
            rv()
            return rv

        torch.manual_seed(9000)
        counter1 = fn1()
        result1 = [counter1(), counter1(), counter1()]

        torch.manual_seed(9000)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch._dynamo.optimize_assert(cnts)(fn1)
        counter2 = torch._dynamo.optimize_assert(cnts)(opt_fn1())
        result2 = [counter2(), counter2(), counter2()]
        result1.append(counter1())
        result2.append(counter2())

        self.assertTrue(same(result1, result2))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 11)

    def test_write_to_closures_in_inlining(self):
        out = []
        for use_dynamo in [False, True]:

            def make_counter():
                x = torch.randn(10)

                def counter():
                    nonlocal x
                    x = x + 1
                    return x

                return counter

            torch.manual_seed(0)
            counter = make_counter()
            if not use_dynamo:
                out.append(counter() + counter())
            else:
                cnts = torch._dynamo.testing.CompileCounter()

                @torch.compile(backend=cnts, fullgraph=True)
                def fn(counter):
                    return counter() + counter()

                out.append(fn(counter))
                self.assertEqual(cnts.frame_count, 1)
                self.assertEqual(cnts.op_count, 3)
                self.assertFalse(same(counter() + counter(), out[-1]))

        self.assertTrue(same(out[0], out[1]))

    # When we unspecialize float, we wobble this test by changing
    # the op count since previously we would just specialize and constant
    # fold floats into the graph, whereas when we unspecialize we will have
    # ops for item, add, and all other tensorified operations. Since this
    # test really isn't testing that, we purposely specialize floats here.
    @torch._dynamo.config.patch(specialize_float=True)
    def test_closure_out_of_scope_cell(self):
        cell1 = torch.rand(1).item()
        cell2 = torch.rand(3, 3)

        def indirect():
            return direct()

        def direct():
            def inner():
                return cell1 + 1, cell2 + 3

            return inner()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(indirect, backend=cnts)
        result1, result2 = opt_fn()
        self.assertAlmostEqual(cell1 + 1, result1)
        self.assertTrue(torch.allclose(cell2 + 3, result2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    # When we unspecialize float, we wobble this test by changing
    # the op count since previously we would just specialize and constant
    # fold floats into the graph, whereas when we unspecialize we will have
    # ops for item, add, and all other tensorified operations. Since this
    # test really isn't testing that, we purposely specialize floats here.
    @torch._dynamo.config.patch(specialize_float=True)
    def test_closure_out_of_scope_cell_with_mutation(self):
        cell1 = torch.rand(1).item()
        orig1 = cell1
        cell2 = torch.rand(3, 3)
        orig2 = cell2.clone()

        def indirect():
            return direct()

        def direct():
            def inner():
                nonlocal cell1, cell2
                x = cell2 + 1
                cell1 += 1
                cell2 += 10
                x = x + cell2
                return cell1, cell2, x

            return inner()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(indirect, backend=cnts, fullgraph=True)
        for i in range(1, 4):
            result1, result2, _ = opt_fn()
            self.assertAlmostEqual(orig1 + 1 * i, result1)
            self.assertTrue(torch.allclose(orig2 + 10 * i, result2))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(cnts.op_count, 3)
            cnts.clear()

    def test_closure_with_mutation_and_graph_break(self):
        def fn():
            x = torch.zeros(1)

            def subfunc():
                x[0] = backup

            if x[0] >= -1e5:
                pass

            backup = 1
            subfunc()
            return x

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        expected = fn()
        actual = opt_fn()
        self.assertTrue(same(expected, actual))
        self.assertEqual(cnts.frame_count, 2)

    def test_closure_out_of_scope_cell_with_cond(self):
        # Test closure with out-of-scope cell variable, used in a cond
        # where the two branches read different closure variables
        from functorch.experimental.control_flow import cond

        def g(x):
            return x

        class ModuleCondDeep(torch.nn.Module):
            def forward(self, pred, x):
                return self._indirection(pred, x)

            def _indirection(self, pred, x):
                return self.indirection(pred, x)

            def indirection(self, pred, x):
                def true_fn(y):
                    return y + 2

                def false_fn(y):
                    return y - 2

                def shallow(x):
                    return x * 2

                def deep(x):
                    # y = g(x)
                    y = x
                    return cond(
                        x[0][0] > 0,
                        true_fn,
                        false_fn,
                        [y],
                    )

                return cond(pred, shallow, deep, [x])

        mod = ModuleCondDeep()
        opt_mod = torch.compile(mod, backend="eager")
        inp = torch.randn(3, 3)
        exp1 = mod(torch.tensor(False), inp)
        actual1 = opt_mod(torch.tensor(False), inp)
        exp2 = mod(torch.tensor(True), inp)
        actual2 = opt_mod(torch.tensor(True), inp)
        self.assertTrue(torch.allclose(exp1, actual1))
        self.assertTrue(torch.allclose(exp2, actual2))

    def test_closure_write_across_functions(self):
        z = 1
        k = 2

        def create_fn():
            def fn(x):
                nonlocal k, z
                k = z

            return fn

        def update_z_and_run_fn(fn, x):
            nonlocal z
            z = 3
            fn(x)
            return x.cos()

        @torch.compile(backend="eager")
        def foo(x):
            fn = create_fn()
            return update_z_and_run_fn(fn, x)

        x = torch.randn(1)
        foo(x)
        self.assertEqual(3, z)
        self.assertEqual(3, k)

    def test_free_var_and_local_name_collision(self):
        x = 10

        def make_func():
            def func():
                return x

            return func

        @torch.compile(backend="eager")
        def root(t):
            x = 0
            func = make_func()
            res = func()
            return t + 1, x, res

        res = root(torch.ones(1))
        self.assertTrue(torch.allclose(torch.ones(1) + 1, res[0]))
        self.assertEqual(0, res[1])
        self.assertEqual(10, res[2])

    def test_cell_captured_by_existing_func_but_not_root_frame(self):
        x = torch.ones(1)

        def get_inner():
            def inner():
                return x + x

            # Calling `inner` so Dynamo won't skip this frame.
            return inner(), inner

        @torch.compile(backend="eager")
        def root():
            return get_inner()

        res, inner = root()
        self.assertTrue(torch.allclose(x + x, res))
        self.assertTrue(torch.allclose(inner(), res))

    def test_writes_to_cells_across_frames1(self):
        # This regression test was added when Dynamo accidentally had both
        # unboxed and normal modeling for pre-existing cells, and failed to
        # account for buffered writes when we read from the unboxed value.
        x = 0

        def inc_x():
            nonlocal x
            x += 1

        class MyObj:
            def inc_x_then_return_x(self, fn):
                fn()
                return x

        @torch.compile(backend="eager")
        def root(t):
            obj = MyObj()
            res = obj.inc_x_then_return_x(inc_x)
            return t + 1, res

        res = root(torch.zeros(1))
        self.assertTrue(torch.allclose(res[0], torch.ones(1)))
        self.assertEqual(res[1], 1)
        self.assertEqual(x, 1)

    def test_writes_to_cells_across_frames2(self):
        # This regression test was added when Dynamo didn't fully account for
        # already established `CellVariable` instance for pre-existing cell,
        # while encountering the same cell again (we should reuse the instance
        # rather than creating a new one). This caused buffered writes to escape
        # the newly created `CellVariable`.
        x = 0

        def inc_x_and_get_x(obj):
            nonlocal x
            x += 1
            return obj.get_x()

        class MyObj:
            def get_x(self):
                return x

        @torch.compile(backend="eager")
        def root(t):
            obj = MyObj()
            res = inc_x_and_get_x(obj)
            return t + 1, res

        res = root(torch.zeros(1))
        self.assertTrue(torch.allclose(res[0], torch.ones(1)))
        self.assertEqual(res[1], 1)
        self.assertEqual(x, 1)

    def test_write_to_cells_with_name_shadowing(self):
        x = 0
        y = x

        def make_x_get_set():
            # NOTE: this `x` is a different cell object than the outer `x`.
            x = y

            def set_x(v):
                nonlocal x
                x = v

            def get_x():
                return x

            return get_x, set_x

        get_x, set_x = make_x_get_set()

        @torch.compile(fullgraph=True, backend="eager")
        def fn(t):
            set_x(42)  # This sets the `x` created within `make_x_get_set`
            res = t + x  # This uses the `x` outside `make_x_get_set`.
            return res

        result = fn(torch.ones(1))
        inner_x = get_x()
        self.assertTrue(torch.allclose(result, torch.ones(1)))
        self.assertEqual(inner_x, 42)

    def test_existing_func_that_creates_capturing_nested_func(self):
        x = 0  # Captured by both `make_get_x` and `root`

        def make_get_x():
            def get_x():
                return x

            return get_x

        @torch.compile(backend="eager", fullgraph=True)
        def root(t):
            get_x = make_get_x()
            res = t + x
            return res, get_x

        res, get_x = root(torch.ones(1))
        self.assertTrue(torch.allclose(res, torch.ones(1)))
        self.assertEqual(0, get_x())
        x += 1
        self.assertEqual(1, get_x())

    def test_input_cell_mutation(self):
        def fn(x):
            x = x.cos()

            def inner():
                return x.sin()

            return inner()

        x = torch.ones(10)
        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(res, ref)

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

    def test_optimize_on_module(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = torch.nn.ReLU()

            def custom_member(self):
                # Just for checking that Dynamo returned mod object can redirect
                # to this method
                pass

            def forward(self, x):
                return self.relu(x)

        cnts1 = torch._dynamo.testing.CompileCounter()
        mod = MockModule()
        optimized_mod = torch.compile(mod, backend=cnts1, fullgraph=True)

        a = torch.randn(10)
        ref = mod(a)
        res = optimized_mod(a)

        optimized_mod.custom_member()

        self.assertTrue(same(ref, res))

    def test_nested_optimize_decorator(self):
        cnts2 = torch._dynamo.testing.CompileCounter()
        cnts3 = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.run()
        def fn1(x):
            return torch.sin(x) * 10

        @torch.compile(backend=cnts2, fullgraph=True)
        def fn2(x):
            return fn1(x) + 1

        @torch.compile(backend=cnts3, fullgraph=True)
        def fn3(x):
            return torch.relu(fn2(x))

        fn3(torch.randn(4, 5))
        self.assertEqual(cnts2.frame_count, 0)
        self.assertEqual(cnts3.frame_count, 1)
        self.assertEqual(cnts3.op_count, 4)

    def test_nested_optimize_run(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x):
            return torch.relu(torch.cos(x) + torch.sin(x))

        fn(torch.randn(4))
        self.assertEqual(cnts.frame_count, 1)

        fn(torch.randn(4, 4))
        self.assertEqual(cnts.frame_count, 2)

        # Test that run works on a decorated fn
        fn = torch._dynamo.run(fn)
        fn(torch.randn(4, 4, 4))
        self.assertEqual(cnts.frame_count, 2)

    def test_nested_optimize(self):
        cnts1 = torch._dynamo.testing.CompileCounter()
        cnts2 = torch._dynamo.testing.CompileCounter()

        def fn(x):
            return torch.relu(torch.cos(x) + torch.sin(x))

        fn1 = torch.compile(fn, backend=cnts1, fullgraph=True)
        fn2 = torch.compile(fn1, backend=cnts2, fullgraph=True)

        # The first optimize in the nesting should be ignored
        fn2(torch.randn(4))
        self.assertEqual(cnts2.frame_count, 1)
        self.assertEqual(cnts1.frame_count, 0)

        # Since the fn code object is already compiled, calling fn1 should
        # directly call the compiled_fn callable.
        torch._dynamo.run()(fn1)(torch.randn(4))
        self.assertEqual(cnts1.frame_count, 0)

        # Test same behavior by reversing the calls
        torch._dynamo.reset()
        cnts1 = torch._dynamo.testing.CompileCounter()
        cnts2 = torch._dynamo.testing.CompileCounter()
        fn1 = torch.compile(fn, backend=cnts1, fullgraph=True)
        fn2 = torch.compile(fn1, backend=cnts2, fullgraph=True)
        fn1(torch.randn(4))
        self.assertEqual(cnts1.frame_count, 1)
        torch._dynamo.run()(fn2)(torch.randn(4))
        self.assertEqual(cnts2.frame_count, 0)

    def test_torch_size(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            output_size = torch.Size([10, 10])
            x = x.view(*output_size)
            return (x,)

        x = torch.randn(100, requires_grad=True)
        x_clone = x.clone()
        ref = fn(x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x_clone)

        self.assertTrue(same(ref, res))

    def test_torch_size_numel(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn():
            return torch.Size([10, 8]).numel()

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        num = torch.Size([10, 8]).numel()
        self.assertEqual(opt_fn(), num)

    def test_torch_size_numel_dynamic(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            return x.size().numel()

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.rand(10, 1, 8, 1)
        expect = fn(x)
        self.assertEqual(opt_fn(x), expect)

    def test_shape_type(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            return x + (type(x.shape) == torch.Size)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.zeros(())
        self.assertEqual(opt_fn(x), fn(x))

    def test_size_dim(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, dim):
            return x.size(dim=dim)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.empty([4, 9, 8])
        self.assertEqual(opt_fn(x, 1), 9)
        self.assertEqual(opt_fn(x, -2), 9)

    def test_torch_size_tensor_index_scalar_constant(self):
        def fn(x):
            idx = torch.tensor(1)
            dim_size = x.shape[idx]
            return x.reshape(-1, dim_size)

        x = torch.randn(2, 3, 4)
        ref = fn(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x)

        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)

    def test_torch_size_tensor_index_non_scalar(self):
        def fn(x):
            idx = torch.tensor([1, 1])
            try:
                dim_size = x.shape[idx]
                return x * dim_size
            except TypeError:
                return x.sum()

        x = torch.randn(2, 3, 4)
        ref = fn(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x)

        self.assertTrue(same(ref, res))

    def test_stride_dim(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, dim):
            return x.stride(dim=dim)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.empty([4, 9, 8])
        self.assertEqual(opt_fn(x, 0), 72)
        self.assertEqual(opt_fn(x, -2), 8)

    def test_torch_seed(self):
        from torch._dynamo.utils import counters

        cnts = torch._dynamo.testing.CompileCounter()
        counters.clear()

        def fn(x):
            attention_seed = int(torch.seed() % sys.maxsize)
            torch.manual_seed(attention_seed)
            return (x,)

        x = torch.randn(10, requires_grad=True)
        ref = fn(x)

        # Python code is needed here, since torch.manual_seed graph-breaks.
        # Refs: https://github.com/pytorch/pytorch/issues/107187
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)
        res = opt_fn(x)

        self.assertTrue(same(ref, res))
        # Only the torch.seed call is turned into an FX graph.
        self.assertEqual(cnts.op_count, 1)
        self.assertEqual(cnts.frame_count, 1)
        # Graph breaks at manual_seed.
        self.assertEqual(len(counters["graph_break"]), 1)

    def test_torch_generator_manual_seed(self):
        from torch._dynamo.utils import counters

        cnts = torch._dynamo.testing.CompileCounter()
        counters.clear()

        def fn(x, gen):
            gen.manual_seed(3)
            return x + 1

        x = torch.randn(10)
        ref = fn(x, torch.Generator())

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)
        res = opt_fn(x, torch.Generator())

        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.op_count, 1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(len(counters["graph_break"]), 1)

    def test_torch_generator_initial_seed(self):
        from torch._dynamo.utils import counters

        cnts = torch._dynamo.testing.CompileCounter()
        counters.clear()

        def fn(x):
            return x + 1, torch.default_generator.initial_seed()

        x = torch.randn(10)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)
        res = opt_fn(x)

        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.op_count, 1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(len(counters["graph_break"]), 1)

    def test_torch_generator_get_state_fullgraph(self):
        def fn():
            return torch.default_generator.get_state()

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)()

    def test_is_tensor_like(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def f(x):
            if torch.overrides.is_tensor_like(x):
                return (x * 2,)
            return (torch.ones(10) + x,)

        x = torch.randn(10)
        ref0 = f(x)
        ref1 = f(4)
        opt_f = torch.compile(f, backend=cnts, fullgraph=True)
        res0 = opt_f(x)
        res1 = opt_f(4)
        self.assertTrue(same(ref0, res0))
        self.assertTrue(same(ref1, res1))

    def test_is_tensor_like2(self):
        class MyTensor:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}

                if func is torch.max:
                    return torch.tensor(123)
                return func(*args, **kwargs)

        def fn(x):
            if torch.overrides.is_tensor_like(x):
                return torch.max(x)
            else:
                return torch.zeros(1)

        x = MyTensor()
        ref0 = fn(x)
        ref1 = fn(4)
        opt_fn = torch.compile(fn, backend="eager")
        res0 = opt_fn(x)
        res1 = opt_fn(4)
        self.assertTrue(same(ref0, res0))
        self.assertTrue(same(ref1, res1))

    def test_tensor_data(self):
        def fn(x, y):
            return x[y.data]

        x = torch.rand(8)
        y = torch.ones(8).to(torch.int)
        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_tensor_layout(self):
        def fn(x):
            return torch.zeros(
                [x.size()[0], x.size()[1]],
                dtype=x.dtype,
                layout=x.layout,
                device=x.device,
            )

        x = torch.rand(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_version_ci(self):
        # temporary test to check that the ci torch version is set correctly
        self.assertTrue(hasattr(torch, "_subclasses"))

    def test_slice_input(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def getitem(a, idx):
            if isinstance(idx, slice):
                return (
                    torch.zeros(1),
                    a[idx]
                    + [
                        100,
                    ],
                )
            else:
                return (torch.zeros(1), a[idx])

        layers = list(range(10))
        ref0 = getitem(layers, slice(0, 2, 1))
        ref1 = getitem(layers, 2)
        ref2 = getitem(layers, slice(3, 8, 2))
        opt_getitem = torch.compile(getitem, backend=cnts, fullgraph=True)
        res0 = opt_getitem(layers, slice(0, 2, 1))
        res1 = opt_getitem(layers, 2)
        res2 = opt_getitem(layers, slice(3, 8, 2))

        self.assertTrue(ref0 == res0)
        self.assertTrue(ref1 == res1)
        self.assertTrue(ref2 == res2)

    def test_grad(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(a, b):
            out = a * b
            out.sum().backward()
            real_out = torch.sigmoid(a.grad + b)
            return real_out

        inps = [torch.randn(4, requires_grad=True) for _ in range(2)]
        for inp in inps:
            inp.grad = None
        ref = fn(*inps)

        for inp in inps:
            inp.grad = None
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(*inps)

        self.assertTrue(same(ref, res))

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_source_non_input_grad_access(self):
        # This test creates a model, and accesses the grads
        # from its parameter. This means that within dynamo,
        # the tensor we are reading the grad from HAS a source,
        # but is not known to graphargs.
        cnts = torch._dynamo.testing.CompileCounter()

        class TrivialModel(torch.nn.Module):
            def __init__(self) -> None:
                super(TrivialModel, self).__init__()
                self.linear = torch.nn.Linear(2, 1)

            def forward(self, x):
                return self.linear(x)

        def fn(a, b):
            outs = []
            for param in model.parameters():
                outs.append(torch.ones(param.grad.size()))
            return outs, param.grad + 1

        model = TrivialModel()
        # Eager
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        out = model(a)
        out_sum = out.sum()
        out_sum.backward()
        ref = fn(a, b)

        # Compiled
        model = TrivialModel()
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        out = model(a)
        out_sum = out.sum()
        out_sum.backward()

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(a, b)

        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    def test_intermediary_tensor_grad_access(self):
        # This test creates a model, and accesses the grads
        # from its parameters and an entirely intermediary tensor.
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(a, b):
            intermediary = torch.ones(2, 2)
            c = a + intermediary
            outs = []
            outs.append(intermediary.grad)
            return outs

        # Eager
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        ref = fn(a, b)

        # Compiled
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(a, b)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_clone_sparse_input(self):
        for layout in [
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        ]:
            for sparse_input in self.generate_simple_inputs(
                layout,
                device="cpu",
                dtype=torch.float64,
                index_dtype=torch.int64,
            ):
                # Invoke the dynamo clone input method directly.
                sparse_copy = torch._dynamo.utils.clone_input(sparse_input)
                # Make sure sparse clone is successful.
                self.assertEqual(sparse_input, sparse_copy)

    def test_tensor_is_contiguous(self):
        def fn(x):
            input = torch.randn((1, 16, 1, 1))
            weight = torch.randn((8, 16, 3, 3))
            weight = weight.to(memory_format=x)
            output = torch.conv2d(input, weight, None, (2, 1), (1, 1), (1, 1), 1)
            return output.is_contiguous(memory_format=x)

        opt_fn = torch.compile(fn, backend="eager")
        for x in [torch.contiguous_format, torch.channels_last]:
            self.assertEqual(fn(x), opt_fn(x))

    def test_python_slice(self):
        def f1(input):
            y = 0
            for i, x in enumerate(input[2:], 1):
                y = y + x
            return y

        def f2(input):
            y = 0
            for i, x in enumerate(input.shape[2:], 1):
                y = y + x
            return y

        cnts = torch._dynamo.testing.CompileCounter()
        opt_f1 = torch.compile(f1, backend=cnts)
        opt_f2 = torch.compile(f2, backend=cnts)
        res1 = opt_f1([1, 2, 3, 5])
        res2 = opt_f2(torch.rand([2, 3, 4, 5]))

        self.assertEqual(res1, 8)
        self.assertEqual(res2, 9)

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

    def test_builtin_subclasses_as_method_on_class_type(self):
        class Foo:
            def __init__(self, name):
                self.ame_ = name

            def get_name(self):
                return "Foo " + self.name_

        class Bar(Foo):
            def __init__(self, name):
                self.name_ = name

            def get_name(self):
                return "Bar " + self.name_

        class Baz(Foo):
            def __init__(self, name):  # noqa: B903
                self.name_ = name

            def get_name(self):
                return "Baz " + self.name_

        subs_of_foo_reg = Foo.__subclasses__()

        counter = CompileCounter()

        @torch._dynamo.optimize_assert(counter)
        def fn():
            return Foo.__subclasses__()

        subs_of_foo_optim = fn()

        self.assertEqual(len(subs_of_foo_reg), 2)
        self.assertEqual(subs_of_foo_reg, subs_of_foo_optim)
