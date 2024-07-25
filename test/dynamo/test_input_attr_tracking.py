# Owner(s): ["module: dynamo"]
# flake8: noqa
import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import (
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    normalize_gm,
)


class TestInputAttrTracking(torch._dynamo.test_case.TestCase):
    def test_tensor_property_on_tensor(self):
        def fn(x):
            return x * x.y

        x_ = torch.randn([2, 2])
        y_ = torch.randn([2, 2])
        x_.y = y_

        eager_result = fn(x_)

        graph = None

        def grab_graph_backend(gm, inps):
            nonlocal graph
            graph = gm
            return gm

        fn = torch._dynamo.optimize(grab_graph_backend, nopython=True)(fn)
        compile_result = fn(x_)
        self.assertEqual(eager_result, compile_result)

        placeholder_cnt = 0
        for node in graph.graph.nodes:
            if node.op == "placeholder":
                placeholder_cnt += 1

        # We want to be very sure that this lifts y to inputs!
        self.assertEqual(placeholder_cnt, 2)

    def test_tensor_property_assigned_on_tensor(self):
        def fn(x, y):
            x.y = y
            return x * x.y

        x_ = torch.randn([2, 2])
        y_ = torch.randn([2, 2])

        eager_result = fn(x_, y_)

        graph = None

        def grab_graph_backend(gm, inps):
            nonlocal graph
            graph = gm
            return gm

        fn = torch._dynamo.optimize(grab_graph_backend, nopython=True)(fn)
        compile_result = fn(x_, y_)
        self.assertEqual(eager_result, compile_result)

        placeholder_cnt = 0
        for node in graph.graph.nodes:
            if node.op == "placeholder":
                placeholder_cnt += 1

        # y is already an input
        self.assertEqual(placeholder_cnt, 2)

    def test_const_property_on_tensor(self):
        def fn(x):
            return x * x.y

        x_ = torch.randn([2, 2])
        y_ = 4
        x_.y = y_

        eager_result = fn(x_)

        graph = None

        def grab_graph_backend(gm, inps):
            nonlocal graph
            graph = gm
            return gm

        fn = torch._dynamo.optimize(grab_graph_backend, nopython=True)(fn)
        compile_result = fn(x_)
        self.assertEqual(eager_result, compile_result)

        placeholder_cnt = 0
        for node in graph.graph.nodes:
            if node.op == "placeholder":
                placeholder_cnt += 1

        # We want to be very sure that this does not lifts y to inputs, as its a const
        self.assertEqual(placeholder_cnt, 1)

    def test_const_property_assigned_on_tensor(self):
        def fn(x, y):
            x.y = y
            return x * x.y

        x_ = torch.randn([2, 2])
        y_ = 4

        eager_result = fn(x_, y_)

        fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        compile_result = fn(x_, y_)
        self.assertEqual(eager_result, compile_result)

    def test_guards_correctly_property_assigned_on_tensor_type_change(self):
        def fn(x, y):
            x.y = y
            return x * x.y

        x_ = torch.randn([2, 2])

        fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        compile_result_const = fn(x_, 4)
        self.assertEqual(compile_result_const, x_ * 4)

        y = torch.randn([2, 2])
        compile_result_tensor = fn(x_, y)
        self.assertEqual(compile_result_tensor, x_ * y)

    def test_guards_correctly_property_assigned_on_tensor_type_change_inductor(self):
        def fn(x, y):
            x.y = y
            return x * x.y

        x_ = torch.randn([2, 2])

        fn = torch._dynamo.optimize("inductor", nopython=True)(fn)
        compile_result_const = fn(x_, 4)
        self.assertEqual(compile_result_const, x_ * 4)

        y = torch.randn([2, 2])
        compile_result_tensor = fn(x_, y)
        self.assertEqual(compile_result_tensor, x_ * y)

    def test_complex_attr_access_without_graph_breaks(self):
        def fn(x, y, z):
            for t in x:
                t.y = y
                t.z = y * z

            new_y = 1
            new_z = 1
            for t in x:
                new_y = t.y * new_y
                new_z = t.z * new_z

            return new_y, new_z

        x_0 = torch.randn([2, 2])
        x_1 = torch.randn([2, 2])
        x_2 = torch.randn([2, 2])
        x = [x_0, x_1, x_2]

        y = torch.randn([2, 2])
        z = 5

        eager_result = fn(x, y, z)

        counter = CompileCounter()
        fn = torch._dynamo.optimize(counter, nopython=True)(fn)

        compile_result = fn(x, y, z)
        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 9)
        # Graph for reference
        #         -------------  ------  -----------------------  ------------------------------------  --------
        # placeholder    l_y_    L_y_                     ()                                    {}
        # call_function  mul     <built-in function mul>  (l_y_, 5)                             {}
        # call_function  mul_1   <built-in function mul>  (l_y_, 5)                             {}
        # call_function  mul_2   <built-in function mul>  (l_y_, 5)                             {}
        # call_function  mul_3   <built-in function mul>  (l_y_, 1)                             {}
        # call_function  mul_4   <built-in function mul>  (mul, 1)                              {}
        # call_function  mul_5   <built-in function mul>  (l_y_, mul_3)                         {}
        # call_function  mul_6   <built-in function mul>  (mul_1, mul_4)                        {}
        # call_function  mul_7   <built-in function mul>  (l_y_, mul_5)                         {}
        # call_function  mul_8   <built-in function mul>  (mul_2, mul_6)                        {}
        # output         output  output                   ((mul_7, mul_8, mul, mul_1, mul_2),)  {}

    def test_complex_attr_access_with_graph_breaks(self):
        def fn(x, y, z):
            for t in x:
                t.y = y
                t.z = y * z

            print("Break!")

            new_y = 1
            new_z = 1
            for t in x:
                new_y = t.y * new_y
                new_z = t.z * new_z

            return new_y, new_z

        x_0 = torch.randn([2, 2])
        x_1 = torch.randn([2, 2])
        x_2 = torch.randn([2, 2])
        x = [x_0, x_1, x_2]

        y = torch.randn([2, 2])
        z = 5

        eager_result = fn(x, y, z)

        counter = CompileCounter()
        fn = torch._dynamo.optimize(counter, nopython=False)(fn)

        compile_result = fn(x, y, z)
        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 2)
        self.assertEqual(counter.op_count, 9)
        # Graph for reference
        # -------------  ------  -----------------------  ----------------------  --------
        # placeholder    l_y_    L_y_                     ()                      {}
        # call_function  mul     <built-in function mul>  (l_y_, 5)               {}
        # call_function  mul_1   <built-in function mul>  (l_y_, 5)               {}
        # call_function  mul_2   <built-in function mul>  (l_y_, 5)               {}
        # output         output  output                   ((mul, mul_1, mul_2),)  {}
        # [GRAPH BREAK!]
        # -------------  -------  -----------------------  -----------------  --------
        # placeholder    l_x_0_y  L_x_0_y                  ()                 {}
        # placeholder    l_x_0_z  L_x_0_z                  ()                 {}
        # placeholder    l_x_1_y  L_x_1_y                  ()                 {}
        # placeholder    l_x_1_z  L_x_1_z                  ()                 {}
        # placeholder    l_x_2_y  L_x_2_y                  ()                 {}
        # placeholder    l_x_2_z  L_x_2_z                  ()                 {}
        # call_function  mul      <built-in function mul>  (l_x_0_y, 1)       {}
        # call_function  mul_1    <built-in function mul>  (l_x_0_z, 1)       {}
        # call_function  mul_2    <built-in function mul>  (l_x_1_y, mul)     {}
        # call_function  mul_3    <built-in function mul>  (l_x_1_z, mul_1)   {}
        # call_function  mul_4    <built-in function mul>  (l_x_2_y, mul_2)   {}
        # call_function  mul_5    <built-in function mul>  (l_x_2_z, mul_3)   {}
        # output         output   output                   ((mul_4, mul_5),)  {}

    def test_complex_attr_access_with_inline_reconstruct(self):
        def inline_test_fn(x, y, z):
            print("f")
            return x.a + y.a + z.a

        def fn(x, y, z):
            x.a = 1
            y.a = 2
            z.a = 3

            mult = inline_test_fn(x, y, z)
            y = y * mult
            x = x * mult
            return x, y

        x = torch.randn([2, 2])
        y = torch.randn([2, 2])
        z = torch.randn([2, 2])

        eager_result = fn(x, y, z)

        counter = CompileCounter()

        fn = torch._dynamo.optimize(counter, nopython=False)(fn)

        compile_result = fn(x, y, z)
        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 2)
        # Graph for reference
        # __compiled_fn_2 <eval_with_key>.0 opcode         name    target                   args             kwargs
        # -------------  ------  -----------------------  ---------------  --------
        # placeholder    l_x_    L_x_                     ()               {}
        # placeholder    l_y_    L_y_                     ()               {}
        # call_function  mul     <built-in function mul>  (l_y_, 6)        {}
        # call_function  mul_1   <built-in function mul>  (l_x_, 6)        {}
        # output         output  output                   ((mul_1, mul),)  {}

    def test_set_data_on_input_tensor(self):
        def fn(x, y):
            x.data = y.data
            if x.size() == y.size():
                return x * y
            else:
                return y * y

        x = torch.randn([5, 5])
        y = torch.randn([2, 2])

        eager_result = fn(x, y)

        eager_and_record = EagerAndRecordGraphs()

        counter = CompileCounterWithBackend(eager_and_record)

        fn = torch._dynamo.optimize(counter, nopython=True)(fn)

        compile_result = fn(x, y)

        graph = eager_and_record.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 6)
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_y_: "f32[2, 2]", L_x_: "f32[2, 2]"):
        l_y_ = L_y_
        l_x_ = L_x_

        _get_data_attr: "f32[2, 2]" = torch._C._autograd._get_data_attr(l_y_)

        _set_grad_enabled = torch._C._set_grad_enabled(False)

        set_: "f32[2, 2]" = torch_Tensor_set_(l_x_, _get_data_attr);  _get_data_attr = None

        _set_grad_enabled_1 = torch._C._set_grad_enabled(True)

        _lower_version_count_by_1 = torch__dynamo_variables_builtin__lower_version_count_by_1(set_);  set_ = None

        mul: "f32[2, 2]" = l_x_ * l_y_;  l_x_ = l_y_ = None
        return (mul,)
""",
        )

    # Note - this does not actually get captured in the graph yet.
    # The plan of record is to introduce a set_data op, entirely subsume the operation into a call_function
    # in the fx graph, and let aot_autograd handle it.
    def test_set_data_on_scoped_tensor(self):
        def fn(x):
            z = torch.zeros([4, 4])
            z.data = x.data
            if x.size() == z.size():
                return z * x
            else:
                return x

        x = torch.randn([5, 5])

        eager_result = fn(x)

        counter = CompileCounter()

        fn = torch._dynamo.optimize(counter, nopython=False)(fn)

        compile_result = fn(x)
        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 2)
        self.assertEqual(counter.op_count, 3)

    def test_set_data_on_user_defined_class_input_tensor(self):
        class MyUserDefinedClass:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def do_some_setattr_stuff(self):
                self.z = x * y
                self.a = x + x
                return self.z * self.a

        x = torch.randn([5, 5])
        y = torch.randn([5, 5])
        mudc_1 = MyUserDefinedClass(x, y)

        eager_result = mudc_1.do_some_setattr_stuff()

        counter = CompileCounter()

        mudc_2 = MyUserDefinedClass(x, y)
        do_some_setattr_stuff = torch._dynamo.optimize(counter, nopython=True)(
            mudc_2.do_some_setattr_stuff
        )

        compile_result = do_some_setattr_stuff()
        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 3)
        # Graph for reference
        #  __compiled_fn_0 <eval_with_key>.0 opcode         name    target                   args                  kwargs
        # -------------  ------  -----------------------  --------------------  --------
        # placeholder    l_x_    L_x_                     ()                    {}
        # placeholder    l_y_    L_y_                     ()                    {}
        # call_function  mul     <built-in function mul>  (l_x_, l_y_)          {}
        # call_function  add     <built-in function add>  (l_x_, l_x_)          {}
        # call_function  mul_1   <built-in function mul>  (mul, add)            {}
        # output         output  output                   ((mul_1, mul, add),)  {}


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
